"""Returns-based evaluation for the price model.

Why this module exists
-----------------------
The original project reported R2 = 0.997 for "next-day price prediction." That
number is a measurement artefact, not a result. When the target is the price
*level* (Close_t+1), the previous close Close_t is an almost perfect predictor
on its own, because adjacent daily closes are ~0.999 correlated. Any model that
is handed features derived from Close_t inherits that correlation and scores
R2 -> 1.0 while having learned nothing about tomorrow that you did not already
know today.

The honest question is whether the model predicts the *change* -- the next-day
return r_t+1 = Close_t+1 / Close_t - 1. On the return scale the trivial
information in the previous price is removed, so R2 is no longer inflated and
you can see what signal (if any) actually exists.

This module therefore:

* reframes the target as next-day return,
* scores every model against two baselines a reviewer will expect --
  "no change" (r_hat = 0) and "persistence" (r_hat = r_t),
* reports directional accuracy against the only honest coin-flip, 50%,
* and exposes a leakage demonstration that reproduces the inflated R2 so the
  artefact is documented rather than hidden.

Everything here is deterministic given a seed, so the numbers in the README can
be regenerated with `python -m src.models.returns_prediction`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# targets and features
# --------------------------------------------------------------------------

def add_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Attach next-day return as the target and simple lagged-return features.

    The feature set is deliberately built only from information available at
    the close of day t: lagged returns and rolling statistics of past returns.
    None of them peek at Close_t+1.
    """
    out = df.copy()
    out["ret"] = out[price_col].pct_change()
    # target: the return realised over the next day
    out["target_ret"] = out["ret"].shift(-1)
    # features known at time t
    for lag in (1, 2, 3, 5, 10):
        out[f"ret_lag{lag}"] = out["ret"].shift(lag - 1)
    out["ret_mean5"] = out["ret"].rolling(5).mean()
    out["ret_std5"] = out["ret"].rolling(5).std()
    out["ret_mean10"] = out["ret"].rolling(10).mean()
    return out.dropna().reset_index(drop=True)


RETURN_FEATURES = [
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5", "ret_lag10",
    "ret_mean5", "ret_std5", "ret_mean10",
]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def directional_accuracy_returns(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of next-day *signs* called correctly (zeros count as a miss)."""
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman-free IC: Pearson correlation of predicted vs realised returns.

    This is the number a systematic desk actually tracks. An IC a few hundredths
    above zero is already a tradable signal; an IC indistinguishable from zero
    means the model has found nothing.
    """
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


@dataclass
class FoldResult:
    fold: int
    model_r2: float
    naive_r2: float
    persistence_r2: float
    directional_acc: float
    info_coef: float


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def _pipeline(model) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def walk_forward_returns(
    df: pd.DataFrame,
    model=None,
    n_splits: int = 5,
    gap: int = 1,
) -> pd.DataFrame:
    """Walk-forward evaluation on the *return* target with explicit baselines.

    For each expanding-window fold we compare the model against:

    * naive "no change":  r_hat = 0        (R2 is 0 by construction on the mean,
                                             so this exposes negative R2 clearly)
    * persistence:        r_hat = r_t      (today's return repeated)

    A model only earns its place if it beats *both* on out-of-sample R2 and
    clears 50% directional accuracy by a margin that survives the fold spread.
    """
    if model is None:
        model = LinearSVR(C=0.01, epsilon=0.0, max_iter=10_000, random_state=42)

    X = df[RETURN_FEATURES].to_numpy()
    y = df["target_ret"].to_numpy()
    r_today = df["ret_lag1"].to_numpy()  # ret at time t, the persistence guess

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    rows: list[FoldResult] = []

    for i, (tr, te) in enumerate(tscv.split(X), start=1):
        pipe = _pipeline(model)
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])

        naive = np.zeros_like(y[te])
        persistence = r_today[te]

        rows.append(
            FoldResult(
                fold=i,
                model_r2=r2_score(y[te], pred),
                naive_r2=r2_score(y[te], naive),
                persistence_r2=r2_score(y[te], persistence),
                directional_acc=directional_accuracy_returns(y[te], pred),
                info_coef=information_coefficient(y[te], pred),
            )
        )

    res = pd.DataFrame(r.__dict__ for r in rows)
    logger.info("walk-forward returns complete over %d folds", n_splits)
    return res


def demonstrate_level_leakage(df: pd.DataFrame, price_col: str = "Close") -> dict:
    """Reproduce the inflated R2 to document the artefact, not to claim it.

    Regressing the next-day *price* on the current price (and nothing else)
    scores R2 ~ 0.99+ purely because adjacent closes are almost identical. This
    function returns that number next to the correlation that drives it, so the
    README can show *why* 0.997 was meaningless.
    """
    price = df[price_col].to_numpy()
    x_today = price[:-1]
    y_tomorrow = price[1:]
    r2_level = r2_score(y_tomorrow, x_today)  # "predict tomorrow = today"
    adj_corr = float(np.corrcoef(x_today, y_tomorrow)[0, 1])
    return {"r2_level_naive": r2_level, "adjacent_close_corr": adj_corr}


# --------------------------------------------------------------------------
# reproducible demo
# --------------------------------------------------------------------------

def simulate_prices(n: int = 2500, seed: int = 42) -> pd.DataFrame:
    """Geometric Brownian motion -- a market with *no* predictable structure.

    This is deliberate. The point of the rebuild is to show that on data with
    no learnable signal, level-R2 is still ~1.0 (the artefact) while return-R2
    collapses to ~0 and directional accuracy sits at a coin flip. A model that
    looked brilliant on the level target is exposed on the return target.
    """
    rng = np.random.default_rng(seed)
    mu, sigma = 0.0003, 0.012
    shocks = rng.normal(mu, sigma, n)
    price = 100.0 * np.exp(np.cumsum(shocks))
    return pd.DataFrame({"Close": price})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raw = simulate_prices()

    leak = demonstrate_level_leakage(raw)
    print("\n=== The artefact: predicting price LEVELS ===")
    print(f"Adjacent-close correlation : {leak['adjacent_close_corr']:.4f}")
    print(f"R2 of 'tomorrow = today'   : {leak['r2_level_naive']:.4f}")
    print("Any model fed the current price inherits this R2 for free.\n")

    feats = add_returns(raw)
    print(f"=== The honest question: predicting next-day RETURNS "
          f"({len(feats)} rows) ===")
    for name, mdl in (
        ("LinearSVR", LinearSVR(C=0.01, epsilon=0.0, max_iter=10_000,
                                random_state=42)),
        ("Ridge", Ridge(alpha=1.0)),
    ):
        res = walk_forward_returns(feats, model=mdl)
        m = res.mean(numeric_only=True)
        print(f"\n-- {name} --")
        print(f"  out-of-sample R2 (model)       : {m['model_r2']:+.4f}")
        print(f"  out-of-sample R2 (naive r=0)   : {m['naive_r2']:+.4f}")
        print(f"  out-of-sample R2 (persistence) : {m['persistence_r2']:+.4f}")
        print(f"  directional accuracy           : {m['directional_acc']*100:5.1f}%  "
              f"(coin flip = 50.0%)")
        print(f"  information coefficient        : {m['info_coef']:+.4f}")


if __name__ == "__main__":
    main()
