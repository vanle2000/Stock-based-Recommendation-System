"""
FastAPI production service for the Stock Recommendation System.

Endpoints:
  POST /recommend        → top-N similar stocks for a given ticker
  POST /predict-price    → next-day price prediction for a ticker
  GET  /health           → service health + model metadata
  GET  /metrics          → latest offline evaluation scores

Run locally:
  uvicorn src.api.app:app --reload --port 8000

Docker:
  docker build -t stock-recommender .
  docker run -p 8000:8000 stock-recommender
"""

from __future__ import annotations

import logging
import pathlib
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

MODELS_DIR = pathlib.Path(__file__).parents[2] / "models"
REPORTS_DIR = pathlib.Path(__file__).parents[2] / "reports"
PROCESSED_DIR = pathlib.Path(__file__).parents[2] / "data" / "processed"

# ── State shared across requests ─────────────────────────────────────────────

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup; release on shutdown."""
    logger.info("Loading models...")
    _state["price_model"] = _load_price_model()
    _state["similarity_matrix"] = _load_similarity_matrix()
    _state["ticker_index"] = _load_ticker_index()
    _state["offline_metrics"] = _load_offline_metrics()
    _state["startup_time"] = time.time()
    logger.info("Models loaded. Service ready.")
    yield
    _state.clear()
    logger.info("Models released.")


def _load_price_model():
    path = MODELS_DIR / "linear_svr_price_model.joblib"
    if path.exists():
        return joblib.load(path)
    logger.warning("Price model not found at %s  -  prediction endpoint unavailable", path)
    return None


def _load_similarity_matrix() -> Optional[pd.DataFrame]:
    path = PROCESSED_DIR / "similarity_matrix.parquet"
    if path.exists():
        return pd.read_parquet(path)
    logger.warning("Similarity matrix not found  -  recommend endpoint unavailable")
    return None


def _load_ticker_index() -> Optional[pd.DataFrame]:
    path = PROCESSED_DIR / "ticker_metadata.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def _load_offline_metrics() -> dict:
    path = REPORTS_DIR / "offline_eval.json"
    if path.exists():
        import json
        return json.loads(path.read_text())
    return {}


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Stock Recommendation API",
    description=(
        "Production REST API for stock price prediction and content-based "
        "recommendation via autoencoder latent-space similarity search."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    ticker: str = Field(..., description="NASDAQ ticker symbol, e.g. 'AAPL'")
    top_n: int = Field(5, ge=1, le=20, description="Number of recommendations (1–20)")
    exclude_same_sector: bool = Field(
        False, description="If True, exclude stocks in the same sector as the seed ticker"
    )

    @field_validator("ticker")
    @classmethod
    def normalise_ticker(cls, v: str) -> str:
        return v.strip().upper()


class RecommendedStock(BaseModel):
    ticker: str
    similarity_score: float = Field(..., description="Cosine similarity in latent space (0–1)")
    sector: Optional[str] = None
    name: Optional[str] = None


class RecommendResponse(BaseModel):
    seed_ticker: str
    recommendations: list[RecommendedStock]
    model_version: str = "autoencoder-v1"
    latency_ms: float


class PredictRequest(BaseModel):
    ticker: str
    features: list[float] = Field(
        ...,
        description="5 PCA-compressed technical indicator values (PC1–PC5)",
        min_length=5,
        max_length=5,
    )

    @field_validator("ticker")
    @classmethod
    def normalise_ticker(cls, v: str) -> str:
        return v.strip().upper()


class PredictResponse(BaseModel):
    ticker: str
    predicted_close: float
    model_version: str = "linear-svr-v1"
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    price_model_loaded: bool
    similarity_matrix_loaded: bool
    n_tickers_indexed: Optional[int]


class MetricsResponse(BaseModel):
    precision_at_5: Optional[float]
    ndcg_at_5: Optional[float]
    correlation_lift_vs_random: Optional[float]
    evaluation_date: Optional[str]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        uptime_seconds=time.time() - _state.get("startup_time", time.time()),
        price_model_loaded=_state.get("price_model") is not None,
        similarity_matrix_loaded=_state.get("similarity_matrix") is not None,
        n_tickers_indexed=(
            len(_state["similarity_matrix"])
            if _state.get("similarity_matrix") is not None else None
        ),
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    m = _state.get("offline_metrics", {})
    return MetricsResponse(
        precision_at_5=m.get("precision_at_5"),
        ndcg_at_5=m.get("ndcg_at_5"),
        correlation_lift_vs_random=m.get("correlation_lift_vs_random"),
        evaluation_date=m.get("evaluation_date"),
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    t0 = time.time()
    sim_matrix: pd.DataFrame | None = _state.get("similarity_matrix")
    ticker_idx: pd.DataFrame | None = _state.get("ticker_index")

    if sim_matrix is None:
        raise HTTPException(503, "Recommendation model not loaded")

    if req.ticker not in sim_matrix.index:
        raise HTTPException(
            404,
            f"Ticker '{req.ticker}' not in index. "
            f"Available tickers: {list(sim_matrix.index[:10])}...",
        )

    scores = sim_matrix.loc[req.ticker].drop(req.ticker)

    if req.exclude_same_sector and ticker_idx is not None and "Sector" in ticker_idx.columns:
        seed_sector = ticker_idx.loc[ticker_idx["ticker"] == req.ticker, "Sector"]
        if not seed_sector.empty:
            same_sector = ticker_idx[ticker_idx["Sector"] == seed_sector.iloc[0]]["ticker"]
            scores = scores.drop(labels=same_sector, errors="ignore")

    top = scores.nlargest(req.top_n)

    recs = []
    for ticker, score in top.items():
        meta = {}
        if ticker_idx is not None:
            row = ticker_idx[ticker_idx["ticker"] == ticker]
            if not row.empty:
                meta = row.iloc[0].to_dict()
        recs.append(RecommendedStock(
            ticker=ticker,
            similarity_score=round(float(score), 4),
            sector=meta.get("Sector"),
            name=meta.get("Name"),
        ))

    return RecommendResponse(
        seed_ticker=req.ticker,
        recommendations=recs,
        latency_ms=round((time.time() - t0) * 1000, 2),
    )


@app.post("/predict", response_model=PredictResponse)
def predict_price(req: PredictRequest):
    t0 = time.time()
    model = _state.get("price_model")

    if model is None:
        raise HTTPException(503, "Price prediction model not loaded")

    X = np.array(req.features).reshape(1, -1)
    pred = float(model.predict(X)[0])

    return PredictResponse(
        ticker=req.ticker,
        predicted_close=round(pred, 4),
        latency_ms=round((time.time() - t0) * 1000, 2),
    )
