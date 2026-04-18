from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(name: str, cmd: list[str]) -> None:
    print(f"\n[orchestrator] step={name}")
    print("[orchestrator] cmd=", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"Step failed: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stable train -> evaluate pipeline.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    python = sys.executable

    run_step(
        "train",
        [python, str(root / "src" / "rl_agent_research" / "train_sb3.py"), "--config", args.config],
    )
    run_step(
        "evaluate",
        [python, str(root / "src" / "rl_agent_research" / "evaluate.py"), "--config", args.config],
    )
    print("\n[orchestrator] done")


if __name__ == "__main__":
    main()
