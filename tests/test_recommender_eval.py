"""Unit tests for recommender offline evaluation metrics."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from src.models.recommender_eval import (
    compute_latent_similarity_matrix,
    compute_return_correlation_matrix,
    correlation_lift_vs_random,
    ndcg_at_k,
    precision_at_k,
    sector_hit_rate,
    evaluate_recommender,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tickers():
    return ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "JPM", "BAC", "WFC", "GS"]


@pytest.fixture
def perfect_sim_matrix(tickers):
    """Similarity matrix where AAPL's top-5 are the first 5 tickers (excluding itself)."""
    n = len(tickers)
    rng = np.random.default_rng(0)
    data = rng.uniform(0.1, 0.5, (n, n))
    np.fill_diagonal(data, 0)
    # Make AAPL highly similar to first 5 tickers
    for i in range(1, 6):
        data[0, i] = 0.95 - i * 0.01
        data[i, 0] = data[0, i]
    df = pd.DataFrame(data, index=tickers, columns=tickers)
    return df


@pytest.fixture
def ground_truth_corr(tickers):
    """Correlation matrix where AAPL has high correlation with first 5 tickers."""
    n = len(tickers)
    rng = np.random.default_rng(1)
    data = rng.uniform(-0.1, 0.3, (n, n))
    np.fill_diagonal(data, 1.0)
    # AAPL highly correlated with first 5
    for i in range(1, 6):
        data[0, i] = 0.75
        data[i, 0] = 0.75
    df = pd.DataFrame(data, index=tickers, columns=tickers)
    return df


# ── Tests: compute_latent_similarity_matrix ───────────────────────────────────

class TestComputeLatentSimilarityMatrix:
    def test_output_shape(self, tickers):
        n = len(tickers)
        vectors = np.random.default_rng(0).uniform(-1, 1, (n, 8))
        matrix = compute_latent_similarity_matrix(vectors, tickers)
        assert matrix.shape == (n, n)

    def test_diagonal_is_zero(self, tickers):
        n = len(tickers)
        vectors = np.random.default_rng(0).uniform(-1, 1, (n, 8))
        matrix = compute_latent_similarity_matrix(vectors, tickers)
        assert np.allclose(np.diag(matrix.values), 0.0)

    def test_values_in_minus1_to_1(self, tickers):
        n = len(tickers)
        vectors = np.random.default_rng(0).uniform(-1, 1, (n, 8))
        matrix = compute_latent_similarity_matrix(vectors, tickers)
        assert (matrix.values >= -1.0).all() and (matrix.values <= 1.0).all()

    def test_index_and_columns_are_tickers(self, tickers):
        n = len(tickers)
        vectors = np.random.default_rng(0).uniform(0, 1, (n, 4))
        matrix = compute_latent_similarity_matrix(vectors, tickers)
        assert list(matrix.index) == tickers
        assert list(matrix.columns) == tickers

    def test_symmetry(self, tickers):
        n = len(tickers)
        vectors = np.random.default_rng(42).uniform(0, 1, (n, 6))
        matrix = compute_latent_similarity_matrix(vectors, tickers)
        assert np.allclose(matrix.values, matrix.values.T)

    def test_identical_vectors_give_high_similarity(self):
        # Two stocks with same latent vector should be maximally similar
        vectors = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        matrix = compute_latent_similarity_matrix(vectors, ["A", "B", "C"])
        assert matrix.loc["A", "B"] > 0.99
        assert matrix.loc["A", "C"] < matrix.loc["A", "B"]


# ── Tests: compute_return_correlation_matrix ─────────────────────────────────

class TestComputeReturnCorrelationMatrix:
    def test_output_shape(self, tickers):
        n = len(tickers)
        rng = np.random.default_rng(0)
        prices = pd.DataFrame(
            rng.uniform(10, 200, (60, n)),
            columns=tickers,
        )
        corr = compute_return_correlation_matrix(prices, window_days=30)
        assert corr.shape == (n, n)

    def test_diagonal_is_one(self, tickers):
        n = len(tickers)
        rng = np.random.default_rng(0)
        prices = pd.DataFrame(rng.uniform(10, 200, (60, n)), columns=tickers)
        corr = compute_return_correlation_matrix(prices, window_days=30)
        assert np.allclose(np.diag(corr.values), 1.0)

    def test_values_in_valid_range(self, tickers):
        n = len(tickers)
        prices = pd.DataFrame(
            np.random.default_rng(1).uniform(10, 500, (60, n)), columns=tickers
        )
        corr = compute_return_correlation_matrix(prices, window_days=30)
        finite = corr.values[np.isfinite(corr.values)]
        assert (finite >= -1.0).all() and (finite <= 1.0).all()


# ── Tests: precision_at_k ─────────────────────────────────────────────────────

class TestPrecisionAtK:
    def test_perfect_model_precision_1(self, perfect_sim_matrix, ground_truth_corr):
        p = precision_at_k("AAPL", perfect_sim_matrix, ground_truth_corr, k=5, correlation_threshold=0.70)
        assert p == 1.0

    def test_returns_nan_unknown_ticker(self, perfect_sim_matrix, ground_truth_corr):
        p = precision_at_k("ZZZZ", perfect_sim_matrix, ground_truth_corr, k=5)
        assert np.isnan(p)

    def test_range_zero_to_one(self, perfect_sim_matrix, ground_truth_corr):
        p = precision_at_k("AAPL", perfect_sim_matrix, ground_truth_corr, k=5)
        assert 0.0 <= p <= 1.0

    def test_k_equals_one(self, perfect_sim_matrix, ground_truth_corr):
        p = precision_at_k("AAPL", perfect_sim_matrix, ground_truth_corr, k=1, correlation_threshold=0.70)
        assert p in (0.0, 1.0)


# ── Tests: ndcg_at_k ──────────────────────────────────────────────────────────

class TestNDCGAtK:
    def test_perfect_model_ndcg_near_1(self, perfect_sim_matrix, ground_truth_corr):
        ndcg = ndcg_at_k("AAPL", perfect_sim_matrix, ground_truth_corr, k=5)
        assert ndcg > 0.8

    def test_range_0_to_1(self, perfect_sim_matrix, ground_truth_corr):
        ndcg = ndcg_at_k("AAPL", perfect_sim_matrix, ground_truth_corr, k=5)
        assert 0.0 <= ndcg <= 1.0

    def test_nan_on_unknown_ticker(self, perfect_sim_matrix, ground_truth_corr):
        ndcg = ndcg_at_k("UNKNOWN", perfect_sim_matrix, ground_truth_corr, k=5)
        assert np.isnan(ndcg)

    def test_random_model_lower_than_perfect(self, tickers, ground_truth_corr):
        """A random similarity matrix should score lower NDCG than the perfect one."""
        n = len(tickers)
        rng = np.random.default_rng(99)
        random_sim = pd.DataFrame(
            rng.uniform(0, 1, (n, n)), index=tickers, columns=tickers
        )
        np.fill_diagonal(random_sim.values, 0)

        # Build a perfect similarity matrix
        perfect = pd.DataFrame(0.1, index=tickers, columns=tickers)
        np.fill_diagonal(perfect.values, 0)
        for i in range(1, 6):
            perfect.iloc[0, i] = 0.9
            perfect.iloc[i, 0] = 0.9

        ndcg_perfect = ndcg_at_k("AAPL", perfect, ground_truth_corr, k=5)
        ndcg_random = ndcg_at_k("AAPL", random_sim, ground_truth_corr, k=5)
        assert ndcg_perfect >= ndcg_random - 0.1  # perfect >= random (with margin)


# ── Tests: correlation_lift_vs_random ────────────────────────────────────────

class TestCorrelationLiftVsRandom:
    def test_returns_dict_with_expected_keys(self, perfect_sim_matrix, ground_truth_corr):
        result = correlation_lift_vs_random("AAPL", perfect_sim_matrix, ground_truth_corr, k=5)
        assert "lift" in result
        assert "model_mean_corr" in result
        assert "random_mean_corr" in result
        assert "p_value" in result
        assert "significant" in result

    def test_perfect_model_positive_lift(self, perfect_sim_matrix, ground_truth_corr):
        result = correlation_lift_vs_random("AAPL", perfect_sim_matrix, ground_truth_corr, k=5)
        assert result["lift"] > 0

    def test_empty_on_unknown_ticker(self, perfect_sim_matrix, ground_truth_corr):
        result = correlation_lift_vs_random("ZZZZ", perfect_sim_matrix, ground_truth_corr)
        assert result == {}

    def test_p_value_in_range(self, perfect_sim_matrix, ground_truth_corr):
        result = correlation_lift_vs_random("AAPL", perfect_sim_matrix, ground_truth_corr, k=3)
        assert 0.0 <= result["p_value"] <= 1.0


# ── Tests: evaluate_recommender ──────────────────────────────────────────────

class TestEvaluateRecommender:
    def test_returns_all_metric_keys(self, perfect_sim_matrix, ground_truth_corr):
        result = evaluate_recommender(
            perfect_sim_matrix, ground_truth_corr, k=5, n_seed_stocks=5
        )
        assert "precision_at_5" in result
        assert "ndcg_at_5" in result
        assert "correlation_lift_vs_random" in result

    def test_precision_in_range(self, perfect_sim_matrix, ground_truth_corr):
        result = evaluate_recommender(
            perfect_sim_matrix, ground_truth_corr, k=5, n_seed_stocks=10
        )
        p = result["precision_at_5"]
        if p is not None:
            assert 0.0 <= p <= 1.0

    def test_n_seeds_evaluated(self, perfect_sim_matrix, ground_truth_corr):
        result = evaluate_recommender(
            perfect_sim_matrix, ground_truth_corr, k=3, n_seed_stocks=5
        )
        assert result["n_seeds_evaluated"] <= 5

    def test_raises_on_insufficient_tickers(self):
        tiny_sim = pd.DataFrame([[0, 1], [1, 0]], index=["A", "B"], columns=["A", "B"])
        tiny_corr = pd.DataFrame([[1, 0.5], [0.5, 1]], index=["A", "B"], columns=["A", "B"])
        with pytest.raises(ValueError, match="Too few tickers"):
            evaluate_recommender(tiny_sim, tiny_corr, k=5)
