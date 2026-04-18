from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


@dataclass
class TradingDataset:
    prices: np.ndarray
    returns: np.ndarray


def load_market_prices(csv_path: str | Path) -> TradingDataset:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Market data is empty: {csv_path}")

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    prices = df[price_col].astype(float).to_numpy()
    if len(prices) < 64:
        raise ValueError("Not enough market rows for trading environment.")

    returns = np.zeros_like(prices, dtype=np.float32)
    returns[1:] = (prices[1:] - prices[:-1]) / np.clip(prices[:-1], 1e-8, None)
    return TradingDataset(prices=prices.astype(np.float32), returns=returns.astype(np.float32))


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: np.ndarray,
        returns: np.ndarray,
        window_size: int = 20,
        start_index: int = 20,
        end_index: int | None = None,
        trading_cost: float = 0.001,
        drawdown_penalty: float = 0.0,
        holding_penalty: float = 0.0,
        trend_reward_coef: float = 0.0,
        episode_length: int | None = None,
        randomize_start: bool = False,
        reward_scale: float = 100.0,
    ) -> None:
        super().__init__()
        self.prices = prices
        self.returns = returns
        self.window_size = window_size
        self._data_start = max(start_index, window_size)
        self._data_end = end_index if end_index is not None else len(prices) - 1
        self.trading_cost = float(trading_cost)
        self.drawdown_penalty = float(drawdown_penalty)
        self.holding_penalty = float(holding_penalty)
        self.trend_reward_coef = float(trend_reward_coef)
        self.episode_length = episode_length
        self.randomize_start = randomize_start
        self.reward_scale = float(reward_scale)

        self.action_space = spaces.Discrete(3)  # 0 short, 1 flat, 2 long
        obs_size = window_size + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        self.start_index = self._data_start
        self.end_index = self._data_end
        self._idx = self.start_index
        self.position = 0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.prev_drawdown = 0.0
        self.action_history: list[int] = []
        self.reward_history: list[float] = []
        self.equity_curve: list[float] = []

    def _get_obs(self) -> np.ndarray:
        window = self.returns[self._idx - self.window_size : self._idx]
        price_window = self.prices[self._idx - self.window_size : self._idx]
        short_ma = float(price_window[-5:].mean() / max(float(price_window[-1]), 1e-8) - 1.0)
        long_ma = float(price_window.mean() / max(float(price_window[-1]), 1e-8) - 1.0)
        obs = np.concatenate(
            [
                window.astype(np.float32),
                np.array([self.position, short_ma, long_ma], dtype=np.float32),
            ]
        )
        return obs.clip(-10.0, 10.0)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.randomize_start and self.episode_length is not None:
            max_start = max(self._data_start, self._data_end - self.episode_length - 1)
            self.start_index = int(self.np_random.integers(self._data_start, max(self._data_start + 1, max_start)))
        else:
            self.start_index = self._data_start

        if self.episode_length is not None:
            self.end_index = min(self.start_index + self.episode_length, self._data_end)
        else:
            self.end_index = self._data_end

        self._idx = self.start_index
        self.position = 0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.prev_drawdown = 0.0
        self.action_history = []
        self.reward_history = []
        self.equity_curve = [self.equity]
        return self._get_obs(), {}

    def step(self, action: int):
        action = int(action)
        new_position = action - 1
        daily_ret = float(self.returns[self._idx])
        pnl = self.position * daily_ret
        trade_cost = self.trading_cost * abs(new_position - self.position)

        next_equity = self.equity * (1.0 + pnl - trade_cost)
        next_equity = max(next_equity, 1e-8)
        next_peak = max(self.peak_equity, next_equity)
        drawdown = max(0.0, 1.0 - next_equity / next_peak)
        dd_penalty = self.drawdown_penalty * max(0.0, drawdown - self.prev_drawdown)
        # holding_penalty charges for non-flat positions — incentivises going flat when uncertain
        h_penalty = self.holding_penalty * abs(self.position)
        # trend_alignment: reward position aligned with recent 5-day trend, penalise opposite
        trend_5 = float(self.returns[max(0, self._idx - 5) : self._idx].mean())
        trend_reward = self.trend_reward_coef * self.position * np.sign(trend_5) if trend_5 != 0 else 0.0
        reward = (pnl - trade_cost - dd_penalty - h_penalty + trend_reward) * self.reward_scale

        self.position = new_position
        self.equity = next_equity
        self.peak_equity = next_peak
        self.prev_drawdown = drawdown
        self._idx += 1

        self.action_history.append(action)
        self.reward_history.append(reward / max(self.reward_scale, 1.0))  # store unscaled
        self.equity_curve.append(self.equity)

        terminated = self._idx >= self.end_index
        truncated = False
        info = {
            "equity": self.equity,
            "drawdown": drawdown,
            "daily_return": daily_ret,
            "position": self.position,
        }
        return self._get_obs(), float(reward), terminated, truncated, info


def make_trading_env(config: dict, split: str = "train") -> TradingEnv:
    market_cfg = config["market_data"]
    trading_cfg = config.get("trading_env", {})
    dataset = load_market_prices(market_cfg["output_csv"])

    train_ratio = float(trading_cfg.get("train_ratio", 0.8))
    split_index = int(len(dataset.prices) * train_ratio)
    window_size = int(trading_cfg.get("window_size", 20))
    episode_length = trading_cfg.get("episode_length", None)
    if episode_length is not None:
        episode_length = int(episode_length)
    reward_scale = float(trading_cfg.get("reward_scale", 100.0))
    drawdown_penalty = float(trading_cfg.get("drawdown_penalty", 0.0))
    holding_penalty = float(trading_cfg.get("holding_penalty", 0.0))
    trend_reward_coef = float(trading_cfg.get("trend_reward_coef", 0.0))

    if split == "train":
        start_index = window_size
        end_index = max(split_index, window_size + 10)
        randomize_start = bool(trading_cfg.get("randomize_start", True))
    else:
        start_index = max(split_index, window_size)
        end_index = len(dataset.prices) - 1
        randomize_start = False
        episode_length = None  # full eval episode

    return TradingEnv(
        prices=dataset.prices,
        returns=dataset.returns,
        window_size=window_size,
        start_index=start_index,
        end_index=end_index,
        trading_cost=float(trading_cfg.get("trading_cost", 0.001)),
        drawdown_penalty=drawdown_penalty,
        holding_penalty=holding_penalty,
        trend_reward_coef=trend_reward_coef,
        episode_length=episode_length,
        randomize_start=randomize_start,
        reward_scale=reward_scale,
    )
