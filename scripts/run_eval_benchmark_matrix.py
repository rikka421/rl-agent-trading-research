from __future__ import annotations

import csv
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml


def run_cmd(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    python = sys.executable

    out_dir = project_root / "docs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_cfg_dir = project_root / "artifacts" / "generated_configs"
    generated_cfg_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        {"name": "ma_5_20_long_only", "short": 5, "long": 20, "allow_short": False},
        {"name": "ma_10_40_long_short", "short": 10, "long": 40, "allow_short": True},
        {"name": "ma_20_60_long_short", "short": 20, "long": 60, "allow_short": True},
    ]

    model_runs = [
        {
            "model_name": "tuned_dqn_spy_dqn",
            "algorithm": "DQN",
            "base_config": project_root / "configs" / "tuned_dqn.yaml",
            "model_path": project_root / "artifacts" / "models" / "tuned_dqn_spy_dqn.zip",
        },
        {
            "model_name": "tuned_ppo_spy_ppo",
            "algorithm": "PPO",
            "base_config": project_root / "configs" / "tuned_ppo.yaml",
            "model_path": project_root / "artifacts" / "models" / "tuned_ppo_spy_ppo.zip",
        },
    ]

    rows: list[dict] = []

    for run in model_runs:
        base_cfg = load_yaml(run["base_config"])
        for sc in scenarios:
            cfg = deepcopy(base_cfg)
            cfg["algorithm"] = run["algorithm"]
            cfg["experiment_name"] = f"eval_{run['model_name']}_{sc['name']}"
            trading_env = cfg.setdefault("trading_env", {})
            trading_env["moving_average_short"] = int(sc["short"])
            trading_env["moving_average_long"] = int(sc["long"])
            trading_env["moving_average_allow_short"] = bool(sc["allow_short"])

            cfg_path = generated_cfg_dir / f"{cfg['experiment_name']}.yaml"
            write_yaml(cfg_path, cfg)

            cmd = [
                python,
                str(project_root / "src" / "rl_agent_research" / "evaluate.py"),
                "--config",
                str(cfg_path),
                "--model",
                str(run["model_path"]),
            ]
            print(f"[matrix] evaluating {run['model_name']} under {sc['name']}")
            run_cmd(cmd, cwd=project_root)

            report_path = project_root / cfg["report_dir"] / f"{cfg['experiment_name']}_{run['algorithm'].lower()}_eval.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            rl = report["rl"]
            ma = report["moving_average"]
            bh = report["buy_and_hold"]

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "suite": f"{run['model_name']}::{sc['name']}",
                "algorithm": run["algorithm"],
                "seed": -1,
                "timesteps": 0,
                "rl_total_return": float(rl["total_return"]),
                "rl_max_drawdown": float(rl["max_drawdown"]),
                "rl_win_rate": float(rl["win_rate"]),
                "rl_num_trades": int(rl["num_trades"]),
                "ma_total_return": float(ma["total_return"]),
                "buy_hold_total_return": float(bh["total_return"]),
                "delta_vs_ma": float(rl["total_return"] - ma["total_return"]),
                "delta_vs_buy_hold": float(rl["total_return"] - bh["total_return"]),
                "report_path": str(report_path.relative_to(project_root)).replace("\\", "/"),
            }
            rows.append(row)

    output_json = out_dir / "experiment_suite_results.json"
    output_csv = out_dir / "experiment_suite_results.csv"

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project": "rl_agent_research",
        "experiments": rows,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[matrix] wrote {output_json}")
    print(f"[matrix] wrote {output_csv}")


if __name__ == "__main__":
    main()
