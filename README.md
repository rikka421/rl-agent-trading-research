# RL-Agent Trading Research

This project is a trading-oriented RL + Agent pipeline built to support a resume-ready story:
- custom OHLCV trading environment
- DQN and PPO training entry points
- automated train -> evaluate workflow
- baseline comparison against Buy-and-Hold and Moving Average
- report artifacts for interview discussion

Current default mode:
- task: trading
- default config: `configs/default.yaml`
- default algorithm: DQN
- execution mode: stable orchestrator first, optional LLM-assisted orchestrator second

## 1) Structure

- `configs/default.yaml`: trading-first DQN config
- `configs/trading_ppo.yaml`: PPO comparison config
- `configs/tuned_dqn_v3.yaml`: stronger DQN config for multi-seed study
- `configs/tuned_ppo_v4.yaml`: stronger PPO config for multi-seed study
- `src/rl_agent_research/env_trading.py`: custom trading environment with reward/cost/drawdown logic
- `src/rl_agent_research/baselines.py`: Buy-and-Hold and Moving Average evaluation helpers
- `src/rl_agent_research/train_sb3.py`: SB3 training entry for classic control and trading tasks
- `src/rl_agent_research/evaluate.py`: evaluation and baseline comparison report generation
- `src/rl_agent_research/orchestrator.py`: stable one-command pipeline
- `src/rl_agent_research/orchestrator_llm.py`: optional DeepSeek-assisted pipeline
- `scripts/download_market_data.py`: market data fetch with fallback generation when public sources fail
- `scripts/run_experiment_suite.py`: reproducible multi-seed runner
- `scripts/build_pdf_report.py`: generate a PDF report from suite outputs
- `scripts/quick_start.ps1`: complete bootstrap script
- `docs/trading_results_2026-04-18.md`: current run summary and limitations
- `docs/interview_notes_trading_rl.md`: interview-oriented explanation notes
- `docs/`: survey, study plan, and research notes

## 2) Quick Start (Windows PowerShell)

```powershell
cd projects/rl_agent_research
.\scripts\quick_start.ps1
```

Expected outputs:
- model file in `artifacts/models/`
- evaluation report in `artifacts/reports/`
- downloaded or fallback-generated market CSV in `data/raw/`
- console metrics for RL / Buy-and-Hold / Moving Average
- laptop-friendly baseline training

## 3) Manual Setup

```powershell
cd projects/rl_agent_research
c:/Users/22130/Desktop/worksapce/.venv/Scripts/python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_market_data.py --config configs/default.yaml
python src/rl_agent_research/orchestrator.py --config configs/default.yaml

# PPO comparison
python src/rl_agent_research/orchestrator.py --config configs/trading_ppo.yaml

# optional: LLM-assisted experiment suggestions via DeepSeek
# set DEEPSEEK_API_KEY first
python src/rl_agent_research/orchestrator_llm.py --config configs/default.yaml

# plan-only mode (no training)
python src/rl_agent_research/orchestrator_llm.py --config configs/default.yaml --plan-only

# evaluate-only mode (skip training)
python src/rl_agent_research/orchestrator_llm.py --config configs/default.yaml --skip-train

# multi-seed experiment suite + PDF report
python scripts/run_experiment_suite.py --project-root .
python scripts/build_pdf_report.py --summary-json docs/reports/experiment_suite_results.json --output-pdf docs/reports/rl_trading_experiment_report_2026-04-18.pdf
```

## 4) Current Validated State

Validated on 2026-04-18:
- custom trading environment is implemented
- DQN and PPO trading pipelines run end-to-end
- evaluation report includes `total_return`, `max_drawdown`, `win_rate`, `num_trades`, and `action_distribution`
- baseline comparison includes Buy-and-Hold and Moving Average

Current limitation:
- the latest run used synthetic OHLCV fallback because public market APIs were rate-limited
- current PPO baseline collapses to a no-trade policy under the present reward/data setup
- treat current numbers as pipeline validation, not final research evidence

See `docs/trading_results_2026-04-18.md` for the current run summary.

## 5) Suggested Next Milestones

1. Tune reward scaling and turnover penalties to reduce DQN under-trading and PPO no-trade collapse.
2. Add multi-seed evaluation with aggregate metrics.
3. Add walk-forward validation and richer features.
4. Replace synthetic fallback with a stable real-market data source.
5. Add experiment tracker and auto-generated markdown report.

## 6) Notes

- This repo is intentionally minimal and trading-focused for interview storytelling.
- GPU is auto-detected (`device: auto`), but workload remains lightweight for laptop stability.

## 7) Resume Date Alignment

- For resume consistency, the RL-Agent project period is standardized as 2025.10 - 2026.01.
- Timeline clarification: the core RL-Agent project work started in 2025, while the current repository version mainly reflects later optimization, refactoring, and incremental improvements.
