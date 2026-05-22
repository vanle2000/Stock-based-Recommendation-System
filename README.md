# Stock Recommendation System: End-to-End ML Platform for Equity Discovery

---

## Case Study

### Introduction
Individual investors face a fundamental information asymmetry: institutional desks run quantitative strategies across thousands of stocks simultaneously, while retail investors rely on intuition, news feeds, and basic screeners. This project closes that gap by building a **full-stack, data-driven stock recommendation engine**  -  combining price prediction, market sentiment analysis, and deep learning similarity search  -  then deploying it as an interactive web application anyone can use.

### Problem
Three problems needed to be solved in sequence:

1. **Scale:** Historical stock data for 3,600+ NASDAQ companies spans 10M+ records. Standard Pandas workflows break. The data pipeline had to work at a scale most ML projects never touch.

2. **Prediction:** Stock prices are non-stationary, noisy, and notoriously difficult to forecast. A model that simply follows the trend will look accurate on paper but be useless in practice. We needed a framework that distinguishes signal from noise.

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

| Component | Model | Metric | Value |
|-----------|-------|--------|-------|
| Price prediction | LinearSVR (tuned) | R² | **0.997** |
| Price prediction | LinearSVR (tuned) | RMSE | **7.30** |
| Price prediction | LinearSVR (tuned) | MAE | **1.14** |
| Price prediction | LinearSVR (tuned) | MAPE | **13.1%** |
| Recommendation engine | Autoencoder + cosine similarity | Architecture | 5-dim latent space |
| Deployment | Streamlit | Status | Deployed |

Best params (GridSearchCV): `C=0.01, epsilon=0.01, loss=squared_epsilon_insensitive`

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

1. **LinearSVR (tuned) achieves R² = 0.997** on the test set  -  explaining 99.7% of variance in next-day closing price when using PCA-compressed technical indicators as features. The key insight: PCA denoising is what makes the prediction tractable; raw indicators with multicollinearity yield much weaker models.

2. **Technology and Telecommunications sectors dominate by trading volume** (median 5.4M and 6.8M shares/day). Finance has the highest number of tickers but median volume of only 106K  -  most financial stocks are thinly traded.

3. **The top 5 highest-priced stocks** (NVR, Seaboard, AutoZone, Texas Pacific Land, Chipotle) show sustained multi-decade price appreciation uncorrelated with sector peers  -  a useful signal for identifying structural outperformers vs. cyclical stocks.

4. **IPO Year nullability (5.8M missing values)** correlates with pre-1990 listings and OTC-converted stocks  -  not random noise. Filling with "Other" preserves 2.5M legitimate records that would otherwise be dropped.

5. **MAPE of 13.1% is the honest accuracy.** R² near 1.0 can reflect scale effects in stock price data. MAPE of 13.1% means the model is off by ~$1.14 per dollar of price on average  -  useful for trend direction but not precise enough for high-frequency trading signals.

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
| MAPE of 13.1% limits trading use | Add order book microstructure features and macroeconomic signals (yield curve, VIX) |
| Autoencoder trained on static dataset | Retrain on rolling window for concept drift; add contrastive learning for better latent separation |
| No backtesting framework | Integrate Backtrader or Zipline to evaluate actual portfolio returns from recommendations |
| Hardcoded absolute local paths in notebooks | Refactor to relative paths with a `config.py`  -  blocks reproducibility for any new user |
| Single-stock recommendation only | Extend to portfolio-level recommendation: given a portfolio, suggest diversifying additions |
| No confidence interval on predictions | Add prediction intervals via quantile regression or conformal prediction |
