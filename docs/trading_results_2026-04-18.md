# Trading RL Results (2026-04-18)

## Scope
This run validates the minimum RL + Agent + Trading pipeline:
- OHLCV data preparation
- custom trading environment
- DQN and PPO training
- deterministic evaluation
- baseline comparison against Buy-and-Hold and Moving Average

## Current Run
- Config: `configs/default.yaml`
- Algorithm: DQN
- Task: trading
- Market file: `data/raw/spy_1d.csv`
- Eval report: `artifacts/reports/trading_dqn_spy_dqn_eval.json`

## Key Metrics
- RL total return: -0.4044
- RL max drawdown: 0.4594
- RL win rate: 0.4822
- RL trades: 2
- Buy-and-Hold total return: -0.4708
- Moving Average total return: -0.3642

## PPO Comparison
- PPO total return: 0.0000
- PPO max drawdown: 0.0000
- PPO trades: 0
- PPO action distribution: always flat

## Interpretation
- The RL pipeline is operational end-to-end and already beats Buy-and-Hold on this run.
- The current DQN policy trades too rarely and behaves close to a biased directional policy.
- The current PPO baseline collapsed to a no-trade policy, which is useful for diagnosing reward scaling and exploration issues.
- Moving Average remains stronger than the current DQN baseline, so the next priority is improving policy responsiveness rather than claiming strategy superiority.

## Limitations
- External market sources were rate-limited during this run, so the pipeline used synthetic OHLCV fallback to keep the experiment runnable.
- Current experiment is a single-seed baseline, not a robust research conclusion.
- PPO now exists as a comparison baseline, but it is not yet competitive.
- Walk-forward validation and richer reward ablations are still pending.

## Next Actions
1. Tune reward scaling and turnover penalties to avoid DQN under-trading and PPO no-trade collapse.
2. Run multi-seed evaluation and aggregate mean/std metrics.
3. Add richer observation features and walk-forward validation.
4. Replace synthetic fallback data with stable real-market source when available.
