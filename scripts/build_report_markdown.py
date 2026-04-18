from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Build markdown summary from experiment suite JSON.")
    parser.add_argument("--summary-json", default="docs/reports/experiment_suite_results.json")
    parser.add_argument("--output-md", default="docs/reports/experiment_report_2026-04-18.md")
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    output_path = Path(args.output_md)

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload["experiments"]

    by_algo: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_algo[row["algorithm"]].append(row)

    lines = []
    lines.append("# RL Trading Experiment Report (2026-04-18)")
    lines.append("")
    lines.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Total runs: {len(rows)}")
    lines.append("")
    lines.append("## Aggregate Results")
    lines.append("")
    lines.append("| Algorithm | N | RL Mean | RL Std | MA Mean | Buy&Hold Mean | Delta vs MA |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for algo, items in sorted(by_algo.items()):
        rl = np.array([float(x["rl_total_return"]) for x in items], dtype=float)
        ma = np.array([float(x["ma_total_return"]) for x in items], dtype=float)
        bh = np.array([float(x["buy_hold_total_return"]) for x in items], dtype=float)
        lines.append(
            "| "
            f"{algo} | {len(items)} | {np.mean(rl):.4f} | {np.std(rl):.4f} | {np.mean(ma):.4f} | {np.mean(bh):.4f} | {np.mean(rl - ma):.4f} |"
        )

    lines.append("")
    lines.append("## Run-Level Results")
    lines.append("")
    lines.append("| Suite | Algorithm | Seed | Timesteps | RL Return | MA Return | Buy&Hold Return | Delta vs MA | Max Drawdown | Trades |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        lines.append(
            "| "
            f"{row['suite']} | {row['algorithm']} | {row['seed']} | {row['timesteps']} | "
            f"{float(row['rl_total_return']):.4f} | {float(row['ma_total_return']):.4f} | {float(row['buy_hold_total_return']):.4f} | "
            f"{float(row['delta_vs_ma']):.4f} | {float(row['rl_max_drawdown']):.4f} | {int(row['rl_num_trades'])} |"
        )

    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append("- Use this run as reproducible benchmark evidence, not final deployment proof.")
    lines.append("- Next: walk-forward windows, richer features, and longer horizon retraining.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] wrote {output_path}")


if __name__ == "__main__":
    main()
