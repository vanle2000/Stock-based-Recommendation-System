"""
Correct time-series cross-validation for stock price prediction.

The standard train/test split is INVALID for time-series:
- It allows future data to leak into training
- It overstates R² (often near 1.0) because the model sees tomorrow's price
  reflected in today's technical indicators computed on overlapping windows

This module implements walk-forward (rolling-window) validation, which is
the correct methodology for any sequential financial data.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE — the honest accuracy metric for price prediction."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    What fraction of predicted price *directions* (up/down) are correct?
    This matters more than RMSE for trading signals.
    """
    actual_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return (actual_dir == pred_dir).mean()


def walk_forward_validate(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model,
    n_splits: int = 5,
    gap: int = 1,
) -> pd.DataFrame:
    """
    Walk-forward (expanding window) cross-validation for time-series.

    Parameters
    ----------
    df : DataFrame sorted by date, with feature and target columns
    feature_cols : list of input feature names
    target_col : name of the target column (e.g., 'Close')
    model : sklearn-compatible estimator
    n_splits : number of CV folds
    gap : number of rows to skip between train and test to prevent leakage
          from indicators computed on overlapping windows

    Returns
    -------
    DataFrame with per-fold metrics
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    X = df[feature_cols].values
    y = df[target_col].values

    results = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        dir_acc = directional_accuracy(y_test, y_pred)

        results.append({
            "fold": fold + 1,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "rmse": rmse,
            "mae": mae,
            "mape_pct": mape,
            "r2": r2,
            "directional_accuracy": dir_acc,
        })

        logger.info(
            "Fold %d | RMSE=%.3f | MAPE=%.1f%% | R²=%.3f | Dir.Acc=%.1f%%",
            fold + 1, rmse, mape, r2, dir_acc * 100,
        )

    results_df = pd.DataFrame(results)
    print("\n=== Walk-Forward CV Results ===")
    print(results_df.round(4).to_string(index=False))
    print("\n=== Mean ± Std ===")
    summary = results_df[["rmse", "mae", "mape_pct", "r2", "directional_accuracy"]]
    print((summary.mean().round(4).to_frame("mean")
           .join(summary.std().round(4).to_frame("std"))))
    return results_df


def build_linear_svr_pipeline() -> Pipeline:
    """LinearSVR with proper scaling — the best-performing model from Phase 2."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearSVR(
            C=0.01, epsilon=0.01,
            loss="squared_epsilon_insensitive",
            max_iter=10_000,
            random_state=42,
        )),
    ])


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Annualized Sharpe ratio for a daily return series.
    Assumes 252 trading days per year.
    """
    excess = returns - risk_free_rate / 252
    if excess.std() == 0:
        return 0.0
    return np.sqrt(252) * excess.mean() / excess.std()


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of a cumulative equity curve."""
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()


def backtest_strategy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    initial_capital: float = 10_000.0,
) -> dict:
    """
    Simple long/flat strategy: buy if model predicts price increase, hold cash otherwise.
    Returns portfolio-level metrics comparable to a buy-and-hold benchmark.
    """
    pred_dir = np.sign(np.diff(y_pred))  # +1 = buy, -1 or 0 = hold cash
    actual_returns = np.diff(y_true) / y_true[:-1]

    strategy_returns = np.where(pred_dir > 0, actual_returns, 0.0)
    bah_returns = actual_returns  # buy-and-hold benchmark

    strategy_equity = initial_capital * np.cumprod(1 + strategy_returns)
    bah_equity = initial_capital * np.cumprod(1 + bah_returns)

    return {
        "strategy_total_return_pct": (strategy_equity[-1] / initial_capital - 1) * 100,
        "bah_total_return_pct": (bah_equity[-1] / initial_capital - 1) * 100,
        "strategy_sharpe": sharpe_ratio(strategy_returns),
        "bah_sharpe": sharpe_ratio(bah_returns),
        "strategy_max_drawdown_pct": max_drawdown(strategy_equity) * 100,
        "bah_max_drawdown_pct": max_drawdown(bah_equity) * 100,
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }
