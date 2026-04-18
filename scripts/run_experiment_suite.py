from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print(f"[suite] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {' '.join(cmd)}")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def collect_metrics(report_path: Path) -> dict:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "rl_total_return": float(data["rl"]["total_return"]),
        "rl_max_drawdown": float(data["rl"]["max_drawdown"]),
        "rl_win_rate": float(data["rl"]["win_rate"]),
        "rl_num_trades": int(data["rl"]["num_trades"]),
        "ma_total_return": float(data["moving_average"]["total_return"]),
        "buy_hold_total_return": float(data["buy_and_hold"]["total_return"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-seed RL trading experiment suite.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    python = sys.executable

    out_dir = project_root / "docs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_cfg_dir = project_root / "artifacts" / "generated_configs"
    generated_cfg_dir.mkdir(parents=True, exist_ok=True)

    suite = [
        {
            "name": "dqn_v3",
            "base_config": project_root / "configs" / "tuned_dqn_v3.yaml",
            "algorithm": "DQN",
            "seeds": [42, 7],
            "timesteps": 80000,
        },
        {
            "name": "ppo_v4",
            "base_config": project_root / "configs" / "tuned_ppo_v4.yaml",
            "algorithm": "PPO",
            "seeds": [42],
            "timesteps": 100000,
        },
    ]

    rows: list[dict] = []
    for item in suite:
        base = load_yaml(item["base_config"])
        for seed in item["seeds"]:
            cfg = deepcopy(base)
            cfg["seed"] = int(seed)
            cfg["total_timesteps"] = int(item["timesteps"])
            cfg["experiment_name"] = f"{item['name']}_seed{seed}"

            generated_cfg = generated_cfg_dir / f"{item['name']}_seed{seed}.yaml"
            write_yaml(generated_cfg, cfg)

            run_cmd(
                [
                    python,
                    str(project_root / "src" / "rl_agent_research" / "orchestrator.py"),
                    "--config",
                    str(generated_cfg),
                ],
                cwd=project_root,
            )

            algo_lower = str(cfg["algorithm"]).lower()
            report_path = project_root / cfg["report_dir"] / f"{cfg['experiment_name']}_{algo_lower}_eval.json"
            metrics = collect_metrics(report_path)
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "suite": item["name"],
                "algorithm": item["algorithm"],
                "seed": int(seed),
                "timesteps": int(item["timesteps"]),
                **metrics,
                "delta_vs_ma": metrics["rl_total_return"] - metrics["ma_total_return"],
                "delta_vs_buy_hold": metrics["rl_total_return"] - metrics["buy_hold_total_return"],
                "report_path": str(report_path.relative_to(project_root)).replace("\\", "/"),
            }
            rows.append(row)
            print(
                "[suite] completed "
                f"{item['name']} seed={seed} rl={row['rl_total_return']:.4f} "
                f"ma={row['ma_total_return']:.4f} delta_vs_ma={row['delta_vs_ma']:.4f}"
            )

    if not rows:
        raise RuntimeError("No experiment rows generated.")

    summary_json = out_dir / "experiment_suite_results.json"
    summary_csv = out_dir / "experiment_suite_results.csv"

    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project": "rl_agent_research",
        "experiments": rows,
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    csv_fields = list(rows[0].keys())
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[suite] wrote {summary_json}")
    print(f"[suite] wrote {summary_csv}")


if __name__ == "__main__":
    main()
