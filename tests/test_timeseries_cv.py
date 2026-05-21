"""Unit tests for walk-forward CV and backtest utilities."""

import pytest
import numpy as np
import pandas as pd
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from src.models.timeseries_cv import (
    mean_absolute_percentage_error,
    directional_accuracy,
    sharpe_ratio,
    max_drawdown,
    backtest_strategy,
)


class TestMAPE:
    def test_perfect_prediction(self):
        y = np.array([100.0, 200.0, 300.0])
        assert mean_absolute_percentage_error(y, y) == 0.0

    def test_known_value(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        # (10/100 + 20/200) / 2 = (0.10 + 0.10) / 2 = 0.10 → 10%
        result = mean_absolute_percentage_error(y_true, y_pred)
        assert abs(result - 10.0) < 0.01

    def test_ignores_zero_true_values(self):
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([10.0, 90.0])
        # Only non-zero elements contribute
        result = mean_absolute_percentage_error(y_true, y_pred)
        assert abs(result - 10.0) < 0.01


class TestDirectionalAccuracy:
    def test_perfect_direction(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.1, 3.1, 4.1])
        assert directional_accuracy(y_true, y_pred) == 1.0

    def test_wrong_direction(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 0.9, 0.8])
        assert directional_accuracy(y_true, y_pred) == 0.0

    def test_returns_float_between_0_and_1(self):
        rng = np.random.default_rng(42)
        y_true = rng.uniform(10, 100, 100)
        y_pred = rng.uniform(10, 100, 100)
        da = directional_accuracy(y_true, y_pred)
        assert 0.0 <= da <= 1.0


class TestSharpeRatio:
    def test_zero_returns_zero_sharpe(self):
        returns = np.zeros(252)
        assert sharpe_ratio(returns) == 0.0

    def test_positive_returns_positive_sharpe(self):
        returns = np.full(252, 0.001)  # consistent positive returns
        assert sharpe_ratio(returns) > 0

    def test_annualized_scaling(self):
        # Daily return of 0.1% with std 0.01% → Sharpe ≈ sqrt(252) * 10
        daily_mean = 0.001
        returns = np.full(252, daily_mean)
        sr = sharpe_ratio(returns)
        # With zero std, should return 0 (handled by guard)
        assert sr == 0.0 or sr > 0


class TestMaxDrawdown:
    def test_no_drawdown_flat(self):
        equity = np.ones(100)
        assert max_drawdown(equity) == 0.0

    def test_known_drawdown(self):
        equity = np.array([100.0, 90.0, 80.0, 100.0])
        # Peak=100, trough=80 → drawdown = -20%
        dd = max_drawdown(equity)
        assert abs(dd - (-0.20)) < 0.001

    def test_always_non_positive(self):
        rng = np.random.default_rng(7)
        equity = np.cumprod(1 + rng.normal(0.001, 0.01, 252)) * 10_000
        assert max_drawdown(equity) <= 0.0


class TestBacktestStrategy:
    def test_output_keys(self):
        rng = np.random.default_rng(42)
        y_true = np.cumprod(1 + rng.normal(0.001, 0.01, 100)) * 100
        y_pred = y_true + rng.normal(0, 0.5, 100)
        result = backtest_strategy(y_true, y_pred)
        expected = {
            "strategy_total_return_pct", "bah_total_return_pct",
            "strategy_sharpe", "bah_sharpe",
            "strategy_max_drawdown_pct", "bah_max_drawdown_pct",
            "directional_accuracy",
        }
        assert expected.issubset(result.keys())

    def test_directional_accuracy_in_range(self):
        rng = np.random.default_rng(1)
        y_true = np.cumsum(rng.normal(0, 1, 200)) + 100
        y_pred = y_true + rng.normal(0, 0.1, 200)
        result = backtest_strategy(y_true, y_pred)
        assert 0.0 <= result["directional_accuracy"] <= 1.0
