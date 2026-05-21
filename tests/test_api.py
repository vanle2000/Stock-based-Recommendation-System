"""
Integration tests for the FastAPI recommendation service.

Uses TestClient (sync) — no live server needed.
Models are mocked so tests run without trained artifacts.
"""

import pathlib
import sys
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))


# ── Mock model state ──────────────────────────────────────────────────────────

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]


def _mock_similarity_matrix():
    rng = np.random.default_rng(0)
    data = rng.uniform(0.1, 0.9, (len(TICKERS), len(TICKERS)))
    np.fill_diagonal(data, 0)
    return pd.DataFrame(data, index=TICKERS, columns=TICKERS)


def _mock_price_model():
    m = MagicMock()
    m.predict.return_value = np.array([150.25])
    return m


def _mock_ticker_index():
    return pd.DataFrame({
        "ticker": TICKERS,
        "Sector": ["Technology", "Technology", "Technology", "Consumer Discretionary", "Technology"],
        "Name": [f"{t} Inc." for t in TICKERS],
    })


@pytest.fixture(scope="module")
def client():
    from src.api.app import app, _state

    # Inject mock state directly — bypass the lifespan loader
    _state["price_model"] = _mock_price_model()
    _state["similarity_matrix"] = _mock_similarity_matrix()
    _state["ticker_index"] = _mock_ticker_index()
    _state["offline_metrics"] = {
        "precision_at_5": 0.72,
        "ndcg_at_5": 0.81,
        "correlation_lift_vs_random": 0.14,
        "evaluation_date": "2024-01-15",
    }
    _state["startup_time"] = 0.0

    with TestClient(app) as c:
        yield c

    _state.clear()


# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_ok(self, client):
        assert r := client.get("/health").json()
        assert r["status"] == "ok"

    def test_models_loaded(self, client):
        r = client.get("/health").json()
        assert r["price_model_loaded"] is True
        assert r["similarity_matrix_loaded"] is True

    def test_n_tickers_indexed(self, client):
        r = client.get("/health").json()
        assert r["n_tickers_indexed"] == len(TICKERS)


# ── Metrics endpoint ──────────────────────────────────────────────────────────

class TestMetricsEndpoint:
    def test_returns_200(self, client):
        assert client.get("/metrics").status_code == 200

    def test_precision_at_5_present(self, client):
        r = client.get("/metrics").json()
        assert r["precision_at_5"] == 0.72

    def test_ndcg_present(self, client):
        r = client.get("/metrics").json()
        assert r["ndcg_at_5"] == 0.81


# ── Recommend endpoint ────────────────────────────────────────────────────────

class TestRecommendEndpoint:
    def test_valid_ticker_returns_200(self, client):
        r = client.post("/recommend", json={"ticker": "AAPL", "top_n": 3})
        assert r.status_code == 200

    def test_lowercase_ticker_normalised(self, client):
        r = client.post("/recommend", json={"ticker": "aapl", "top_n": 3})
        assert r.status_code == 200
        assert r.json()["seed_ticker"] == "AAPL"

    def test_correct_number_of_recommendations(self, client):
        r = client.post("/recommend", json={"ticker": "AAPL", "top_n": 3})
        assert len(r.json()["recommendations"]) == 3

    def test_recommendation_schema(self, client):
        r = client.post("/recommend", json={"ticker": "AAPL", "top_n": 1})
        rec = r.json()["recommendations"][0]
        assert "ticker" in rec
        assert "similarity_score" in rec
        assert 0.0 <= rec["similarity_score"] <= 1.0

    def test_seed_not_in_recommendations(self, client):
        r = client.post("/recommend", json={"ticker": "AAPL", "top_n": 4})
        tickers = [rec["ticker"] for rec in r.json()["recommendations"]]
        assert "AAPL" not in tickers

    def test_unknown_ticker_returns_404(self, client):
        r = client.post("/recommend", json={"ticker": "ZZZZ", "top_n": 5})
        assert r.status_code == 404

    def test_top_n_above_max_rejected(self, client):
        r = client.post("/recommend", json={"ticker": "AAPL", "top_n": 99})
        assert r.status_code == 422  # Pydantic validation

    def test_top_n_zero_rejected(self, client):
        r = client.post("/recommend", json={"ticker": "AAPL", "top_n": 0})
        assert r.status_code == 422

    def test_latency_ms_present(self, client):
        r = client.post("/recommend", json={"ticker": "AAPL", "top_n": 2})
        assert "latency_ms" in r.json()
        assert r.json()["latency_ms"] >= 0


# ── Predict endpoint ──────────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_valid_request_returns_200(self, client):
        r = client.post("/predict", json={
            "ticker": "AAPL",
            "features": [0.1, -0.5, 0.3, 1.2, -0.8],
        })
        assert r.status_code == 200

    def test_prediction_is_float(self, client):
        r = client.post("/predict", json={
            "ticker": "AAPL",
            "features": [0.1, -0.5, 0.3, 1.2, -0.8],
        })
        assert isinstance(r.json()["predicted_close"], float)

    def test_wrong_feature_length_rejected(self, client):
        r = client.post("/predict", json={
            "ticker": "AAPL",
            "features": [0.1, 0.2],  # wrong length
        })
        assert r.status_code == 422

    def test_ticker_normalised(self, client):
        r = client.post("/predict", json={
            "ticker": "msft",
            "features": [0.0, 0.0, 0.0, 0.0, 0.0],
        })
        assert r.json()["ticker"] == "MSFT"
