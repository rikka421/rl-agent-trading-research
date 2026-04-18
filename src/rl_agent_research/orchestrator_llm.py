from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import requests
import yaml


def call_deepseek_plan(config: dict) -> str:
    ds_cfg = config.get("deepseek", {})
    api_key = ds_cfg.get("api_key")
    if not api_key:
        api_key = __import__("os").environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return "DeepSeek API key not found. Skipping LLM planning. Set DEEPSEEK_API_KEY to enable."

    api_base = ds_cfg.get("api_base", "https://api.deepseek.com").rstrip("/")
    model = ds_cfg.get("model", "deepseek-chat")

    prompt = {
        "env_id": config.get("env_id"),
        "algorithm": config.get("algorithm", "PPO"),
        "total_timesteps": config.get("total_timesteps"),
        "goal": "Give 5 lightweight experiment suggestions for a laptop GPU, each with one-line rationale.",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an RL research assistant. Keep output concise and practical."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        "temperature": 0.3,
    }

    resp = requests.post(
        f"{api_base}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def run_step(name: str, cmd: list[str]) -> None:
    print(f"\n[orchestrator-llm] step={name}")
    print("[orchestrator-llm] cmd=", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"Step failed: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-assisted RL orchestrator.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--plan-only", action="store_true", help="Only generate planning suggestions.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and run evaluate only.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    python = sys.executable

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("[orchestrator-llm] planning suggestions:")
    try:
        plan = call_deepseek_plan(config)
    except Exception as exc:
        plan = f"DeepSeek planning failed: {exc}"
    print(plan)

    if args.plan_only:
        print("\n[orchestrator-llm] plan-only mode done")
        return

    if not args.skip_train:
        run_step(
            "train",
            [python, str(root / "src" / "rl_agent_research" / "train_sb3.py"), "--config", args.config],
        )
    run_step(
        "evaluate",
        [python, str(root / "src" / "rl_agent_research" / "evaluate.py"), "--config", args.config],
    )
    print("\n[orchestrator-llm] done")


if __name__ == "__main__":
    main()
