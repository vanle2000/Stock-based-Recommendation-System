"""
Offline evaluation framework for the stock recommendation system.

The autoencoder + cosine similarity has NO built-in quality guarantee.
These functions define what "a good recommendation" means and measure it.

Core question: given a seed stock, do the top-K recommendations have
higher forward-looking return correlation than a random baseline?

Evaluation metrics:
  1. Precision@K    — fraction of top-K recs with return correlation > threshold
  2. NDCG@K         — ranked quality (highly similar stocks ranked higher = better)
  3. Correlation lift — mean correlation of top-K vs mean correlation of random K
  4. Sector hit rate — what fraction of recs are in the same sector (sanity check)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ── Ground truth: return-based similarity ────────────────────────────────────

def compute_return_correlation_matrix(
    price_df: pd.DataFrame,
    window_days: int = 30,
) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlation of forward daily returns.

    This is our ground truth: stocks that move together are "truly similar"
    regardless of what the autoencoder thinks.

    Parameters
    ----------
    price_df : DataFrame with columns = ticker symbols, index = date (sorted)
    window_days : forward-looking window for correlation (default 30 days)

    Returns
    -------
    Symmetric DataFrame of pairwise return correlations
    """
    returns = price_df.pct_change().dropna()
    corr = returns.iloc[-window_days:].corr()
    return corr


def compute_latent_similarity_matrix(
    latent_vectors: np.ndarray,
    tickers: list[str],
) -> pd.DataFrame:
    """
    Compute cosine similarity matrix from autoencoder latent representations.

    Parameters
    ----------
    latent_vectors : (n_tickers, latent_dim) array of encoded stock representations
    tickers : ticker symbols in the same order as rows

    Returns
    -------
    DataFrame indexed and columned by ticker
    """
    sim = cosine_similarity(latent_vectors)
    np.fill_diagonal(sim, 0.0)  # exclude self-similarity
    return pd.DataFrame(sim, index=tickers, columns=tickers)


# ── Recommendation evaluation metrics ────────────────────────────────────────

def precision_at_k(
    seed_ticker: str,
    similarity_matrix: pd.DataFrame,
    ground_truth_corr: pd.DataFrame,
    k: int = 5,
    correlation_threshold: float = 0.50,
) -> float:
    """
    Precision@K: fraction of top-K model recommendations that are "truly similar"
    (i.e., have forward return correlation above threshold with the seed stock).

    Parameters
    ----------
    correlation_threshold : stocks with corr > this are considered "relevant"
    """
    if seed_ticker not in similarity_matrix.index:
        return float("nan")
    if seed_ticker not in ground_truth_corr.index:
        return float("nan")

    # Model top-K (excluding self)
    scores = similarity_matrix.loc[seed_ticker].drop(seed_ticker, errors="ignore")
    top_k = scores.nlargest(k).index.tolist()

    # Ground truth relevance
    true_corr = ground_truth_corr.loc[seed_ticker]
    relevant = true_corr[true_corr >= correlation_threshold].index.tolist()

    hits = len(set(top_k) & set(relevant))
    return hits / k


def ndcg_at_k(
    seed_ticker: str,
    similarity_matrix: pd.DataFrame,
    ground_truth_corr: pd.DataFrame,
    k: int = 5,
) -> float:
    """
    NDCG@K: Normalised Discounted Cumulative Gain.

    Uses forward return correlation as the relevance score.
    A model that ranks the most correlated stocks highest gets NDCG close to 1.

    Returns float in [0, 1]. Higher is better.
    """
    if seed_ticker not in similarity_matrix.index:
        return float("nan")

    scores = similarity_matrix.loc[seed_ticker].drop(seed_ticker, errors="ignore")
    top_k_tickers = scores.nlargest(k).index.tolist()

    available = [t for t in top_k_tickers if t in ground_truth_corr.columns]
    if not available:
        return float("nan")

    # Relevance = forward return correlation (clipped to [0,1])
    relevance = ground_truth_corr.loc[seed_ticker, available].clip(lower=0).values

    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))

    # Ideal: sort by true correlation
    ideal_corr = ground_truth_corr.loc[seed_ticker].drop(seed_ticker, errors="ignore")
    ideal_top_k = ideal_corr.nlargest(k).values.clip(min=0)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_top_k))

    return dcg / idcg if idcg > 0 else 0.0


def correlation_lift_vs_random(
    seed_ticker: str,
    similarity_matrix: pd.DataFrame,
    ground_truth_corr: pd.DataFrame,
    k: int = 5,
    n_random_trials: int = 1000,
    random_state: int = 42,
) -> dict:
    """
    Compare mean return correlation of model's top-K vs random K picks.

    This answers: "Is the model's similarity measure better than chance?"

    Returns
    -------
    dict with keys: model_mean_corr, random_mean_corr, lift, p_value
    """
    if seed_ticker not in similarity_matrix.index:
        return {}

    all_tickers = [t for t in similarity_matrix.columns
                   if t != seed_ticker and t in ground_truth_corr.columns]

    if len(all_tickers) < k:
        return {}

    # Model recommendations
    scores = similarity_matrix.loc[seed_ticker].drop(seed_ticker, errors="ignore")
    top_k = [t for t in scores.nlargest(k).index if t in ground_truth_corr.columns]
    model_corr = ground_truth_corr.loc[seed_ticker, top_k].mean() if top_k else float("nan")

    # Random baseline (Monte Carlo)
    rng = np.random.default_rng(random_state)
    random_corrs = []
    for _ in range(n_random_trials):
        sample = rng.choice(all_tickers, size=k, replace=False).tolist()
        random_corrs.append(ground_truth_corr.loc[seed_ticker, sample].mean())

    random_mean = float(np.mean(random_corrs))
    random_std = float(np.std(random_corrs))

    # One-sample t-test: is model_corr significantly better than random?
    from scipy import stats
    t_stat = (model_corr - random_mean) / (random_std / np.sqrt(n_random_trials)) if random_std > 0 else 0.0
    p_value = stats.t.sf(t_stat, df=n_random_trials - 1)  # one-tailed

    return {
        "seed_ticker": seed_ticker,
        "k": k,
        "model_mean_corr": round(float(model_corr), 4),
        "random_mean_corr": round(random_mean, 4),
        "lift": round(float(model_corr) - random_mean, 4),
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
    }


def sector_hit_rate(
    seed_ticker: str,
    similarity_matrix: pd.DataFrame,
    ticker_metadata: pd.DataFrame,
    k: int = 5,
) -> float:
    """
    Sanity check: what fraction of top-K recs are in the same sector?
    A latent space that learns sector structure should score > random (~1/13 sectors).
    """
    if seed_ticker not in similarity_matrix.index:
        return float("nan")

    meta = ticker_metadata.set_index("ticker") if "ticker" in ticker_metadata.columns else ticker_metadata
    if seed_ticker not in meta.index or "Sector" not in meta.columns:
        return float("nan")

    seed_sector = meta.loc[seed_ticker, "Sector"]
    scores = similarity_matrix.loc[seed_ticker].drop(seed_ticker, errors="ignore")
    top_k = scores.nlargest(k).index.tolist()

    in_sector = sum(
        1 for t in top_k
        if t in meta.index and meta.loc[t, "Sector"] == seed_sector
    )
    return in_sector / k


# ── Aggregate evaluation over a sample of seed stocks ────────────────────────

def evaluate_recommender(
    similarity_matrix: pd.DataFrame,
    ground_truth_corr: pd.DataFrame,
    ticker_metadata: Optional[pd.DataFrame] = None,
    k: int = 5,
    n_seed_stocks: int = 100,
    correlation_threshold: float = 0.50,
    random_state: int = 42,
) -> dict:
    """
    Run full offline evaluation over a sample of seed stocks.

    Returns aggregate metrics suitable for logging to MLflow or a JSON report.
    """
    common_tickers = list(
        set(similarity_matrix.index) & set(ground_truth_corr.index)
    )
    if len(common_tickers) < k + 1:
        raise ValueError(
            f"Too few tickers in common between similarity matrix and ground truth: "
            f"{len(common_tickers)}"
        )

    rng = np.random.default_rng(random_state)
    seeds = rng.choice(common_tickers, size=min(n_seed_stocks, len(common_tickers)), replace=False)

    p_at_k, ndcg, lifts, sector_rates = [], [], [], []

    for seed in seeds:
        p_at_k.append(precision_at_k(seed, similarity_matrix, ground_truth_corr, k, correlation_threshold))
        ndcg.append(ndcg_at_k(seed, similarity_matrix, ground_truth_corr, k))
        lift_result = correlation_lift_vs_random(seed, similarity_matrix, ground_truth_corr, k)
        if lift_result:
            lifts.append(lift_result["lift"])
        if ticker_metadata is not None:
            sector_rates.append(sector_hit_rate(seed, similarity_matrix, ticker_metadata, k))

    def _mean(vals):
        clean = [v for v in vals if not np.isnan(v)]
        return round(float(np.mean(clean)), 4) if clean else None

    results = {
        f"precision_at_{k}": _mean(p_at_k),
        f"ndcg_at_{k}": _mean(ndcg),
        "correlation_lift_vs_random": _mean(lifts),
        "sector_hit_rate": _mean(sector_rates) if sector_rates else None,
        "n_seeds_evaluated": len(seeds),
        "k": k,
        "correlation_threshold": correlation_threshold,
    }

    logger.info("=== Recommender Offline Evaluation ===")
    for key, val in results.items():
        logger.info("  %s: %s", key, val)

    return results
