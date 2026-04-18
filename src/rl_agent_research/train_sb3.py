from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_agent_research.baselines import compute_trade_metrics
from rl_agent_research.env_trading import make_trading_env


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(cfg: dict) -> str:
    wanted = str(cfg.get("device", "auto")).lower()
    if wanted == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if wanted == "cuda" and not torch.cuda.is_available():
        print("[train] cuda requested but unavailable; fallback to cpu")
        return "cpu"
    return wanted


def build_model(config: dict, env: gym.Env, device: str):
    algo = str(config.get("algorithm", "PPO")).upper()
    seed = int(config.get("seed", 42))
    policy_kwargs = config.get("policy_kwargs", None)

    if algo == "PPO":
        ppo_cfg = config.get("ppo", {})
        return PPO(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=1,
            device=device,
            policy_kwargs=policy_kwargs,
            n_steps=int(ppo_cfg.get("n_steps", 512)),
            batch_size=int(ppo_cfg.get("batch_size", 64)),
            learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
            ent_coef=float(ppo_cfg.get("ent_coef", 0.0)),
            gamma=float(ppo_cfg.get("gamma", 0.99)),
        )

    if algo == "DQN":
        dqn_cfg = config.get("dqn", {})
        return DQN(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=1,
            device=device,
            policy_kwargs=policy_kwargs,
            buffer_size=int(dqn_cfg.get("buffer_size", 50000)),
            learning_rate=float(dqn_cfg.get("learning_rate", 1e-4)),
            exploration_fraction=float(dqn_cfg.get("exploration_fraction", 0.1)),
            exploration_final_eps=float(dqn_cfg.get("exploration_final_eps", 0.05)),
            learning_starts=int(dqn_cfg.get("learning_starts", 100)),
            train_freq=int(dqn_cfg.get("train_freq", 4)),
            target_update_interval=int(dqn_cfg.get("target_update_interval", 500)),
            gamma=float(dqn_cfg.get("gamma", 0.99)),
        )

    raise ValueError(f"Unsupported algorithm: {algo}. Use PPO or DQN.")


def make_env(config: dict, split: str = "train") -> gym.Env:
    task = str(config.get("task", "classic_control")).lower()
    if task == "trading":
        return make_trading_env(config, split=split)
    return Monitor(gym.make(config["env_id"]))


def evaluate_trading_model(model, env) -> dict:
    obs, _ = env.reset()
    terminated = False
    truncated = False
    actions: list[int] = []
    returns: list[float] = []
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        actions.append(int(action))
        returns.append(float(info.get("daily_return", 0.0)))

    metrics = compute_trade_metrics(
        actions=np.asarray(actions, dtype=int),
        returns=np.asarray(returns, dtype=float),
        trading_cost=float(config_or_zero(env, "trading_cost")),
    )
    metrics["final_equity"] = float(env.equity)
    return metrics


def config_or_zero(env, field: str) -> float:
    return float(getattr(env, field, 0.0))


def train(config: dict) -> Path:
    env = make_env(config, split="train")
    device = resolve_device(config)
    model = build_model(config, env, device)

    model.learn(total_timesteps=int(config["total_timesteps"]))

    artifact_dir = Path(config["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    algo = str(config.get("algorithm", "PPO")).lower()
    model_path = artifact_dir / f"{config['experiment_name']}_{algo}.zip"
    model.save(str(model_path))

    task = str(config.get("task", "classic_control")).lower()
    if task == "trading":
        eval_env = make_env(config, split="eval")
        trading_metrics = evaluate_trading_model(model, eval_env)
        mean_reward = trading_metrics["total_return"]
        std_reward = trading_metrics["max_drawdown"]

        report_dir = Path(config.get("report_dir", "artifacts/reports"))
        report_dir.mkdir(parents=True, exist_ok=True)
        training_report = report_dir / f"{config['experiment_name']}_{algo}_train_metrics.json"
        training_report.write_text(json.dumps(trading_metrics, indent=2), encoding="utf-8")
        print(f"[train] trading_report={training_report}")
        eval_env.close()
    else:
        mean_reward, std_reward = evaluate_policy(
            model,
            env,
            n_eval_episodes=int(config["eval_episodes"]),
            deterministic=True,
        )
    print(f"[train] algorithm={config.get('algorithm', 'PPO')} device={device}")
    print(f"[train] model={model_path}")
    print(f"[train] eval_mean_reward={mean_reward:.2f}, eval_std={std_reward:.2f}")
    env.close()
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SB3 baseline model (PPO/DQN).")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    train(config)


if __name__ == "__main__":
    main()
