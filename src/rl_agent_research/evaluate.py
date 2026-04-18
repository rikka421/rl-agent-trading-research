from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_agent_research.baselines import buy_and_hold_metrics, compute_trade_metrics, moving_average_metrics
from rl_agent_research.env_trading import make_trading_env


def make_env(config: dict, split: str = "eval") -> gym.Env:
    task = str(config.get("task", "classic_control")).lower()
    if task == "trading":
        return make_trading_env(config, split=split)
    return Monitor(gym.make(config["env_id"]))


def evaluate_trading(model, env, config: dict) -> dict:
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

    returns_arr = np.asarray(returns, dtype=float)
    actions_arr = np.asarray(actions, dtype=int)
    trading_cfg = config.get("trading_env", {})
    rl_metrics = compute_trade_metrics(actions_arr, returns_arr, trading_cost=float(trading_cfg.get("trading_cost", 0.001)))
    rl_metrics["final_equity"] = float(env.equity)
    rl_metrics["episode_length"] = len(actions)

    start = env.start_index
    end = min(env.end_index, len(env.prices) - 1)
    prices = env.prices[start:end]
    aligned_returns = env.returns[start:end]
    buy_hold = buy_and_hold_metrics(aligned_returns)
    moving_average = moving_average_metrics(
        prices=prices,
        returns=aligned_returns,
        short_window=int(trading_cfg.get("moving_average_short", 5)),
        long_window=int(trading_cfg.get("moving_average_long", 20)),
        trading_cost=float(trading_cfg.get("trading_cost", 0.001)),
        allow_short=bool(trading_cfg.get("moving_average_allow_short", False)),
    )

    return {
        "task": "trading",
        "algorithm": str(config.get("algorithm", "PPO")).upper(),
        "rl": rl_metrics,
        "buy_and_hold": buy_hold,
        "moving_average": moving_average,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None, help="Path to model zip")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env = make_env(config, split="eval")
    algo = str(config.get("algorithm", "PPO")).upper()
    default_name = f"{config['experiment_name']}_{algo.lower()}.zip"
    model_path = Path(args.model) if args.model else Path(config["artifact_dir"]) / default_name

    if algo == "PPO":
        model = PPO.load(str(model_path), env=env)
    elif algo == "DQN":
        model = DQN.load(str(model_path), env=env)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}. Use PPO or DQN.")

    task = str(config.get("task", "classic_control")).lower()
    print(f"[eval] algorithm={algo}")
    print(f"[eval] model={model_path}")
    if task == "trading":
        report = evaluate_trading(model, env, config)
        report_dir = Path(config.get("report_dir", "artifacts/reports"))
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{config['experiment_name']}_{algo.lower()}_eval.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[eval] report={report_path}")
        print(f"[eval] rl_total_return={report['rl']['total_return']:.4f}")
        print(f"[eval] rl_max_drawdown={report['rl']['max_drawdown']:.4f}")
        print(f"[eval] buy_hold_total_return={report['buy_and_hold']['total_return']:.4f}")
        print(f"[eval] ma_total_return={report['moving_average']['total_return']:.4f}")
    else:
        mean_reward, std_reward = evaluate_policy(
            model,
            env,
            n_eval_episodes=int(config["eval_episodes"]),
            deterministic=True,
        )
        print(f"[eval] mean_reward={mean_reward:.2f}, std_reward={std_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()
