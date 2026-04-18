from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


def max_drawdown(equity_curve: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = 1.0 - equity_curve / np.clip(peaks, 1e-8, None)
    return float(np.max(drawdowns))


def compute_trade_metrics(actions: np.ndarray, returns: np.ndarray, trading_cost: float) -> dict:
    positions = actions.astype(int) - 1
    prev_positions = np.concatenate([[0], positions[:-1]])
    pnl = prev_positions * returns
    turnover = np.abs(positions - prev_positions)
    costs = turnover * trading_cost
    daily = pnl - costs
    equity = np.cumprod(1.0 + daily)
    wins = float(np.mean(daily > 0)) if len(daily) else 0.0
    action_counts = Counter(actions.tolist())
    return {
        "total_return": float(equity[-1] - 1.0) if len(equity) else 0.0,
        "max_drawdown": max_drawdown(equity) if len(equity) else 0.0,
        "win_rate": wins,
        "num_trades": int(np.sum(turnover > 0)),
        "turnover": float(np.sum(turnover)),
        "action_distribution": {str(k): int(v) for k, v in sorted(action_counts.items())},
        "equity_curve": equity.tolist(),
    }


def buy_and_hold_metrics(returns: np.ndarray) -> dict:
    actions = np.full(shape=len(returns), fill_value=2, dtype=int)
    return compute_trade_metrics(actions, returns, trading_cost=0.0)


def moving_average_metrics(
    prices: np.ndarray,
    returns: np.ndarray,
    short_window: int = 5,
    long_window: int = 20,
    trading_cost: float = 0.001,
    allow_short: bool = False,
) -> dict:
    if len(prices) <= long_window:
        return compute_trade_metrics(np.full(len(returns), 1, dtype=int), returns, trading_cost=trading_cost)

    # Use causal rolling means to avoid look-ahead leakage in the baseline.
    price_series = pd.Series(prices)
    short_ma = price_series.rolling(window=short_window, min_periods=short_window).mean().to_numpy()
    long_ma = price_series.rolling(window=long_window, min_periods=long_window).mean().to_numpy()

    actions = np.full(len(returns), 1, dtype=int)
    valid = ~np.isnan(short_ma) & ~np.isnan(long_ma)
    if allow_short:
        actions[valid] = np.where(short_ma[valid] > long_ma[valid], 2, 0).astype(int)
    else:
        actions[valid] = np.where(short_ma[valid] > long_ma[valid], 2, 1).astype(int)

    return compute_trade_metrics(actions, returns, trading_cost=trading_cost)
