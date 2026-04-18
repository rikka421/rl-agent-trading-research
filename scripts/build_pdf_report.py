from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas


def aggregate_by_algo(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["algorithm"], []).append(row)

    agg = []
    for algo, items in grouped.items():
        rl = np.array([float(r["rl_total_return"]) for r in items], dtype=float)
        ma = np.array([float(r["ma_total_return"]) for r in items], dtype=float)
        bh = np.array([float(r["buy_hold_total_return"]) for r in items], dtype=float)
        dd = np.array([float(r["rl_max_drawdown"]) for r in items], dtype=float)
        delta = rl - ma
        agg.append(
            {
                "algorithm": algo,
                "n": len(items),
                "rl_mean": float(np.mean(rl)),
                "rl_std": float(np.std(rl)),
                "ma_mean": float(np.mean(ma)),
                "bh_mean": float(np.mean(bh)),
                "dd_mean": float(np.mean(dd)),
                "delta_vs_ma_mean": float(np.mean(delta)),
                "win_rate_vs_ma": float(np.mean(rl > ma)),
            }
        )
    return sorted(agg, key=lambda x: x["rl_mean"], reverse=True)


def create_pdf(summary_json: Path, output_pdf: Path) -> None:
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    rows = payload["experiments"]
    agg = aggregate_by_algo(rows)

    best = max(rows, key=lambda r: float(r["rl_total_return"]))
    worst = min(rows, key=lambda r: float(r["rl_total_return"]))

    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(output_pdf), pagesize=A4)
    width, height = A4
    y = height - 2 * cm

    c.setFont("Helvetica-Bold", 16)
    c.drawString(2 * cm, y, "RL Trading Experiment Report")
    y -= 1.0 * cm

    c.setFont("Helvetica", 10)
    header_lines = [
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Source: {summary_json.as_posix()}",
        f"Total runs: {len(rows)}",
        f"Algorithms: {', '.join(sorted({r['algorithm'] for r in rows}))}",
        f"Best run: {best['suite']} (RL={best['rl_total_return']:.4f}, MA={best['ma_total_return']:.4f})",
        f"Worst run: {worst['suite']} (RL={worst['rl_total_return']:.4f}, MA={worst['ma_total_return']:.4f})",
    ]
    for line in header_lines:
        c.drawString(2 * cm, y, line)
        y -= 0.55 * cm

    y -= 0.3 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Per-Algorithm Aggregate")
    y -= 0.7 * cm
    c.setFont("Helvetica", 10)
    for a in agg:
        c.drawString(
            2 * cm,
            y,
            f"{a['algorithm']}: RL mean={a['rl_mean']:.4f}, std={a['rl_std']:.4f}, delta_vs_MA={a['delta_vs_ma_mean']:.4f}, win% vs MA={100.0 * a['win_rate_vs_ma']:.1f}%",
        )
        y -= 0.55 * cm

    y -= 0.3 * cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2 * cm, y, "Run-Level Results")
    y -= 0.7 * cm
    c.setFont("Helvetica", 9)
    for idx, r in enumerate(rows, start=1):
        line = (
            f"{idx}. {r['suite']} | RL={float(r['rl_total_return']):.4f} | "
            f"MA={float(r['ma_total_return']):.4f} | BH={float(r['buy_hold_total_return']):.4f} | "
            f"Delta={float(r['delta_vs_ma']):.4f} | MDD={float(r['rl_max_drawdown']):.4f}"
        )
        if y < 2 * cm:
            c.showPage()
            c.setFont("Helvetica", 9)
            y = height - 2 * cm
        c.drawString(2 * cm, y, line[:150])
        y -= 0.5 * cm

    c.save()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PDF report from experiment suite summary JSON.")
    parser.add_argument(
        "--summary-json",
        default="docs/reports/experiment_suite_results.json",
    )
    parser.add_argument(
        "--output-pdf",
        default="docs/reports/rl_trading_experiment_report_2026-04-18.pdf",
    )
    args = parser.parse_args()

    summary_json = Path(args.summary_json)
    output_pdf = Path(args.output_pdf)
    create_pdf(summary_json, output_pdf)
    print(f"[report] wrote {output_pdf}")


if __name__ == "__main__":
    main()
