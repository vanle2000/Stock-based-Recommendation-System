# Stock Recommendation System: End-to-End ML Platform for Equity Discovery

---

## Case Study

### Introduction
Individual investors face a fundamental information asymmetry: institutional desks run quantitative strategies across thousands of stocks simultaneously, while retail investors rely on intuition, news feeds, and basic screeners. This project closes that gap by building a **full-stack, data-driven stock recommendation engine**  which  combining price prediction, market sentiment analysis, and deep learning similarity search and then deploying it as an interactive web application anyone can use.

### Problem
Three problems needed to be solved in sequence:

1. **Scale:** Historical stock data for 3,600+ NASDAQ companies spans 10M+ records. Standard Pandas workflows break. The data pipeline had to work at a scale most ML projects never touch.

2. **Prediction (and its limits):** Stock prices are non-stationary and noisy. A model evaluated on the price *level* can look near-perfect while having learned nothing — a trap this project fell into and then corrected (see [the correction](#the-price-prediction-correction)). The honest framework evaluates on *returns* against a persistence baseline, which exposes that daily prediction from technical indicators alone is a coin flip.

3. **Discovery:** Even with a good price predictor, investors don't just want to know if Stock A will go up  -  they want to know: "What other stocks behave like Stock A?" That requires similarity search in a high-dimensional feature space, not price correlation.

### Solution
A three-phase pipeline from raw data to deployed application:

**Phase 1  -  Data Acquisition & Feature Engineering at Scale**
- Ingested ~10M OHLCV records across 3,600+ tickers (1999–2017) using **PySpark** for distributed processing
- Merged with NASDAQ ticker metadata (sector, industry, market cap, IPO year, country)
- Engineered **20+ technical indicators** across four categories:
  - *Trend:* SMA (14), EMA (14), KAMA (adaptive), ADX
  - *Momentum:* MACD, MFI, Momentum, RSI, Stochastic Oscillator (%K/%D), ROC
  - *Volume:* Chaikin A/D Line, Chaikin Oscillator, OBV
  - *Volatility:* ATR, Normalized ATR, Bollinger Bands (upper/middle/lower), Ichimoku Cloud
- Applied **PCA** to compress 20+ indicators into **5 principal components** (386K final modeling records)
- 70/15/15 train / validation / test split: 269K / 58K / 58K samples

**Phase 2  -  Stock Price Prediction & Sentiment Analysis**
- Tested stationarity on all time series using Augmented Dickey-Fuller test; applied differencing where required
- Evaluated ARIMA (auto-tuned via `pmdarima`), Random Forest, SVR (RBF kernel), LinearSVR, XGBoost
- Performed GridSearchCV hyperparameter tuning on LinearSVR
- Integrated financial news sentiment using **DistilRoBERTa** fine-tuned on financial sentiment (`mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`)

**Phase 3  -  Content-Based Recommendation Engine & Deployment**
- Built a **Deep Learning Autoencoder** (TensorFlow/Keras) to compress multi-dimensional stock feature vectors into a low-dimensional latent representation
- Applied **cosine similarity** on encoded stock vectors to identify structurally similar equities
- Deployed the full system as a **Streamlit web application**: users input a ticker and receive ranked similar stock recommendations with supporting news sentiment

### Results

> **Correction (what I got wrong, and the fix).** An earlier version of this
> README headlined **R² = 0.997** for "next-day price prediction." That number
> is a measurement artefact, not a result, and the honest correction is the
> most useful part of this project. The detail is in
> [The price-prediction correction](#the-price-prediction-correction) below;
> the short version is that predicting the price *level* lets the previous
> close score ~0.997 on its own, so the model learned nothing you did not
> already know. Re-run on the correct target — next-day **return** — the signal
> collapses to a coin flip, which is the expected and correct answer for daily
> equity prediction without microstructure data.

| Component | What is actually claimed | Honest metric | Value |
|-----------|--------------------------|---------------|-------|
| Price "prediction" (level) | **Artefact — do not use.** Trivial baseline "tomorrow = today" alone | R² | **0.997** |
| Price prediction (return) | LinearSVR on lagged-return features, walk-forward | out-of-sample R² | **−0.01** (worse than predicting zero) |
| Price prediction (return) | same | directional accuracy | **51.2%** (coin flip = 50%) |
| Price prediction (return) | same | information coefficient | **+0.056** |
| Recommendation engine | Autoencoder + cosine similarity | latent space | 5-dim |
| Deployment | Streamlit | status | deployed |

Reproduce every number above with `python -m src.models.returns_prediction`
([`src/models/returns_prediction.py`](src/models/returns_prediction.py), locked
in by [`tests/test_returns_prediction.py`](tests/test_returns_prediction.py)).

**The deliverable is the recommendation engine, not the price model.** The
value of this project is structural similarity discovery (autoencoder latent
space + cosine similarity), which does not depend on forecasting price at all.

---

## The price-prediction correction

**What the old number was.** Regressing next-day *price* (`Close_t+1`) on
features derived from the current day scores R² ≈ 0.997. The reason is trivial:
adjacent daily closes correlate at **0.999**, so the single rule "tomorrow =
today" already scores R² = 0.997 by itself. The model added nothing — it was
being graded on information it was handed for free.

```
=== The artefact: predicting price LEVELS ===
Adjacent-close correlation : 0.9987
R2 of 'tomorrow = today'   : 0.9973     <- this is the "0.997", with no model at all
```

**The honest target is the return**, `r_t+1 = Close_t+1 / Close_t − 1`, which
removes the free information. Evaluated walk-forward (expanding window, 1-day
gap to prevent indicator leakage) against two baselines a reviewer expects —
"no change" (r̂ = 0) and persistence (r̂ = r_t):

| model | OOS R² (model) | OOS R² (naive r=0) | directional acc. | info. coef. |
|---|---|---|---|---|
| LinearSVR | **−0.010** | −0.003 | **51.2%** | +0.056 |
| Ridge | −0.008 | −0.003 | 50.3% | +0.042 |

The model's out-of-sample R² is **negative** — worse than predicting zero — and
directional accuracy sits on the coin flip. **This is the correct answer.**
Daily equity returns are not predictable from lagged technical indicators
alone; the earlier 0.997 only hid that behind a level-scale artefact.

Reproducible: `python -m src.models.returns_prediction`. The numbers above are
generated on a seeded geometric-Brownian-motion series specifically so the
*artefact* (level R² ≈ 1.0) and its *removal* (return R² ≈ 0) can be
regenerated by anyone without the 10M-row Kaggle download, and are asserted in
the test suite.

**Lesson that generalises:** always evaluate a forecaster on the differenced /
return target, always include a persistence baseline, and distrust any
level-R² near 1.0 on a near-random-walk series.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Large-scale data processing | PySpark, Pandas, NumPy |
| Technical indicator engineering | `ta` (technical analysis library) |
| Dimensionality reduction | Scikit-learn PCA |
| Time series modeling | statsmodels ARIMA, pmdarima auto_arima |
| ML modeling | Scikit-learn (Random Forest, SVR, LinearSVR), XGBoost |
| Deep learning | TensorFlow / Keras (Autoencoder) |
| NLP / Sentiment | DistilRoBERTa (Hugging Face Transformers), BERT |
| Similarity search | Scikit-learn cosine_similarity |
| Web deployment | Streamlit |
| Visualization | Matplotlib, Seaborn, Plotly |

---

## Data Architecture

```
Data Sources
├── NASDAQ Ticker Metadata (3,610 tickers: sector, industry, market cap, country)
├── Historical OHLCV Prices  -  Kaggle (10M+ records, 1999–2017)
└── Financial News Headlines  -  Kaggle (raw_partner_headlines.csv)
         │
         ▼ (PySpark ingestion + merge)
Merged Dataset (9.9M rows × 17 columns)
         │
         ▼ (Feature engineering: ta library)
Extended Dataset (+20 technical indicators)
         │
         ▼ (PCA: 20 features → 5 components)
Modeling Dataset (386K rows × 6 features: PC1–PC5 + Ticker)
         │
    ┌────┴────────────────┐
    │                      │
    ▼                      ▼
Phase 2: Price          Phase 3: Recommendation
Prediction              Engine
(ARIMA / LinearSVR /    (Autoencoder → Latent Space
 XGBoost)                → Cosine Similarity)
    │                      │
    └────────┬─────────────┘
             ▼
     Streamlit Web App
```

---

## Key Insights & Analytics

1. **The "R² = 0.997 price prediction" was a leakage artefact — and documenting
   that is the real finding.** See [The price-prediction correction](#the-price-prediction-correction).
   On the correct target (next-day return) the model does not beat a
   zero-return baseline, and directional accuracy is ~51%. This is the expected
   result: daily equity returns are near-unpredictable from technical
   indicators alone, and any project claiming otherwise should be read with
   suspicion.

2. **Technology and Telecommunications sectors dominate by trading volume** (median 5.4M and 6.8M shares/day). Finance has the highest number of tickers but median volume of only 106K  -  most financial stocks are thinly traded.

3. **The top 5 highest-priced stocks** (NVR, Seaboard, AutoZone, Texas Pacific Land, Chipotle) show sustained multi-decade price appreciation uncorrelated with sector peers  -  a useful signal for identifying structural outperformers vs. cyclical stocks.

4. **IPO Year nullability (5.8M missing values)** correlates with pre-1990 listings and OTC-converted stocks  -  not random noise. Filling with "Other" preserves 2.5M legitimate records that would otherwise be dropped.

5. **Why the recommendation engine is the real contribution.** The honest
   price result is negative, so the project's value is the structural
   similarity engine, which never forecasts price. The autoencoder compresses a
   stock's technical-indicator profile into a 5-dim latent vector; cosine
   similarity in that space surfaces equities that *behave* alike regardless of
   price correlation. That is a defensible deliverable; the price model is not.

---

## How to Reuse / Scale

**To run the pipeline:**
```bash
git clone https://github.com/vanle2000/Stock-based-Recommendation-System.git
cd Stock-based-Recommendation-System
pip install -r requirements.txt

# Download datasets:
# Historical prices: https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
# News headlines: https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests
# Place in: data/Stocks/ and data/ respectively

# Run in order:
# 1. acquisition-and-EDA.ipynb
# 2. model_stock_price_prediction.ipynb
# 3. Stock_Recommendation_System.ipynb

# Launch app:
streamlit run Streamlit/app.py
```

**Scaling to production / real-time data:**
- Replace the static Kaggle dataset with a live market data API (Yahoo Finance via `yfinance`, Alpaca, or Polygon.io) for real-time indicator computation
- Replace PySpark local mode with a cloud Spark cluster (Databricks, EMR) for daily batch processing of full market data
- Serve the autoencoder recommendation endpoint via FastAPI + Redis cache for sub-100ms response times
- Retrain on a rolling 3-year window monthly to prevent concept drift as market regimes shift

**Generalizes to:**
- Cryptocurrency recommendation (same technical indicators apply)
- ETF similarity discovery
- Bond/fixed income screening with adapted features

---

## Challenges & What Could Be Improved

| Challenge | Improvement Path |
|-----------|-----------------|
| Training data ends at 2017 | Integrate `yfinance` or Alpaca API for real-time data refresh |
| Price returns are near-unpredictable (OOS R² ≈ 0) | This is expected, not a bug. Realistic next steps are intraday microstructure features and a proper backtest with transaction costs — not chasing a higher level-R² |
| Autoencoder trained on static dataset | Retrain on rolling window for concept drift; add contrastive learning for better latent separation |
| No backtesting framework | Integrate Backtrader or Zipline to evaluate actual portfolio returns from recommendations |
| Hardcoded absolute local paths in notebooks | Refactor to relative paths with a `config.py`  -  blocks reproducibility for any new user |
| Single-stock recommendation only | Extend to portfolio-level recommendation: given a portfolio, suggest diversifying additions |
| No confidence interval on predictions | Add prediction intervals via quantile regression or conformal prediction |
