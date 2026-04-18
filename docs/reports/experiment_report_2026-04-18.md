# RL Trading Experiment Report (2026-04-18)

Generated at: 2026-04-18T21:26:12
Total runs: 6

## Aggregate Results

| Algorithm | N | RL Mean | RL Std | MA Mean | Buy&Hold Mean | Delta vs MA |
|---|---:|---:|---:|---:|---:|---:|
| DQN | 3 | -0.0540 | 0.0000 | 0.0002 | -0.4708 | -0.0542 |
| PPO | 3 | -0.0429 | 0.0000 | 0.0002 | -0.4708 | -0.0431 |

## Run-Level Results

| Suite | Algorithm | Seed | Timesteps | RL Return | MA Return | Buy&Hold Return | Delta vs MA | Max Drawdown | Trades |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tuned_dqn_spy_dqn::ma_5_20_long_only | DQN | -1 | 0 | -0.0540 | -0.1773 | -0.4708 | 0.1233 | 0.2417 | 93 |
| tuned_dqn_spy_dqn::ma_10_40_long_short | DQN | -1 | 0 | -0.0540 | -0.0636 | -0.4708 | 0.0096 | 0.2417 | 93 |
| tuned_dqn_spy_dqn::ma_20_60_long_short | DQN | -1 | 0 | -0.0540 | 0.2415 | -0.4708 | -0.2955 | 0.2417 | 93 |
| tuned_ppo_spy_ppo::ma_5_20_long_only | PPO | -1 | 0 | -0.0429 | -0.1773 | -0.4708 | 0.1344 | 0.0776 | 22 |
| tuned_ppo_spy_ppo::ma_10_40_long_short | PPO | -1 | 0 | -0.0429 | -0.0636 | -0.4708 | 0.0207 | 0.0776 | 22 |
| tuned_ppo_spy_ppo::ma_20_60_long_short | PPO | -1 | 0 | -0.0429 | 0.2415 | -0.4708 | -0.2844 | 0.0776 | 22 |

## Conclusion

- Use this run as reproducible benchmark evidence, not final deployment proof.
- Next: walk-forward windows, richer features, and longer horizon retraining.
