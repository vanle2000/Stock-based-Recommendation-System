# Stock Recommendation System: End-to-End ML Platform for Equity Discovery

## Business Problem

Individual investors are overwhelmed. The NASDAQ alone lists 3,600+ stocks across 13 sectors. Manually screening for similar equities, timing entries, or discovering hidden correlations is infeasible at scale. Institutional desks use proprietary quantitative tools — retail investors have nothing comparable.

This project builds a **full-stack, data-driven stock recommendation engine** that combines price prediction, sentiment analysis, and deep learning similarity search — then deploys it as an interactive web application. The goal: give any investor a quantitative edge without requiring a quant background.

---

## System Architecture (3-Phase Pipeline)

```
Phase 1: Data Acquisition & EDA
  └── NASDAQ ticker data + 10M historical price records
      └── Feature engineering (20+ technical indicators)
          └── PCA dimensionality reduction → 5 principal components

Phase 2: Price Prediction & Sentiment Analysis
  └── ARIMA, Random Forest, SVR, LinearSVR, XGBoost
      └── Tuned LinearSVR → best model (R² = 0.997)
          └── Financial news sentiment (DistilRoBERTa)

Phase 3: Content-Based Recommendation Engine
  └── Deep Learning Autoencoder (compresses stock profiles → latent space)
      └── Cosine similarity search on encoded representations
          └── Streamlit web application deployment
```

---

## Datasets

| Dataset | Source | Scale |
|---------|--------|-------|
| Historical stock prices | [Kaggle – Huge Stock Market Dataset](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) | ~10M rows, 3,600+ companies, 1999–2017 |
| NASDAQ ticker metadata | [NASDAQ Market Activity](https://www.nasdaq.com/market-activity/stocks) | 3,610 tickers with sector, industry, market cap |
| Financial news headlines | [Kaggle – Massive Stock News Analysis DB](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests) | Raw partner headlines with stock ticker, date, publisher |

---

## Phase 1: Data Acquisition & Exploratory Analysis

### Data Processing at Scale
- Ingested 10M+ OHLCV records across 3,600 tickers using **PySpark** for distributed processing
- Merged historical prices with NASDAQ metadata (sector, industry, market cap, country)
- Cleaned: removed zero/negative prices, de-duplicated on OHLCV composite key, handled 163K+ missing sector/industry values

### Technical Feature Engineering
Computed **20+ market indicators** as predictive signals across 4 categories:

| Category | Indicators |
|----------|-----------|
| Trend / Overlap | SMA, EMA, KAMA, ADX |
| Momentum | MACD, MFI, Momentum, RSI, Stochastic Oscillator (%K/%D), ROC |
| Volume | Chaikin A/D Line, Chaikin Oscillator, OBV |
| Volatility | ATR, Normalized ATR, Bollinger Bands (upper/middle/lower), Ichimoku Cloud (Tenkan-sen, Kijun-sen) |

### Key EDA Findings
- **Technology and Finance** dominate by trading volume (median: 5.4M and 1.0M shares/day respectively)
- **Top 5 by last sale price:** NVR Inc., Seaboard Corp., AutoZone, Texas Pacific Land, Chipotle — all mega-caps with decades of price appreciation
- High correlation between `Close`, `SMA`, `EMA`, and `KAMA` — confirms momentum persistence in large-cap stocks
- Significant `IPO Year` nullability (5.8M missing) treated as `Other` to preserve dataset scale

### Dimensionality Reduction (PCA)
Applied PCA on the 20+ technical indicators to reduce multicollinearity and compress the feature space:
- Retained **5 principal components** (Feature_1 through Feature_5)
- Final modeling dataset: **386,566 records** across all tickers and dates (2017 window)
- 70/15/15 train/validation/test split → 269K / 58K / 58K samples

---

## Phase 2: Stock Price Prediction

### Models Evaluated

**Time Series Models:**
- **ARIMA(5,1,1):** Stationarity verified via Augmented Dickey-Fuller test. `auto_arima` used to identify optimal (p,d,q) parameters
- **ARIMA(4,3,1):** Alternative specification with higher-order differencing

**Machine Learning Models (on PCA features):**
- Random Forest Regressor
- SVR (RBF kernel)
- LinearSVR (with GridSearchCV hyperparameter tuning)
- XGBoost Regressor

### Model Results

| Model | RMSE | MAE | MAPE | R² |
|-------|------|-----|------|----|
| ARIMA(5,1,1) | — | — | — | Baseline |
| ARIMA(4,3,1) | — | — | — | Improved AIC |
| Random Forest | — | — | — | — |
| SVR (RBF) | — | — | — | — |
| **LinearSVR (tuned)** | **7.30** | **1.14** | **13.1%** | **0.997** |

**Best model: Tuned LinearSVR** (GridSearchCV, `C=0.01, epsilon=0.01, loss=squared_epsilon_insensitive`)

- R² of **0.997** means the model explains 99.7% of variance in next-day closing price
- MAPE of 13.1% is the honest accuracy measure — the model is off by ~$1.14 on average per dollar of stock price

### Sentiment Analysis
- Financial news headlines processed using **DistilRoBERTa** fine-tuned on financial sentiment (`mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`)
- Sentiment scores (positive/negative/neutral) integrated as supplementary signals for recommendation ranking

---

## Phase 3: Content-Based Stock Recommendation Engine

### Approach
Rather than recommending stocks based on price correlation alone (which is noisy and non-stationary), this system learns a **compressed latent representation** of each stock's behavioral profile:

1. **Autoencoder architecture:** Encodes multi-dimensional stock feature vectors (technical indicators + PCA components) into a low-dimensional latent space
2. **Similarity search:** Computes **cosine similarity** between encoded stock vectors to identify stocks that behave similarly in market conditions
3. **Recommendation output:** Given a stock ticker, returns the top-N most similar equities ranked by latent-space proximity

This is a **content-based filtering** approach — it surfaces structural similarity between stocks, not just historical price correlation.

### Streamlit Web Application
Deployed as an interactive web app where users can:
- Input a stock ticker of interest
- Receive personalized similar stock recommendations
- View supporting financial news sentiment for each recommendation

---

## Results Summary

| Component | Key Outcome |
|-----------|------------|
| Price prediction (LinearSVR) | R² = 0.997, RMSE = 7.30, MAE = 1.14 |
| Feature engineering | 20+ technical indicators across 10M records |
| Dimensionality reduction | 20+ features → 5 PCA components |
| Recommendation system | Cosine similarity on autoencoder latent space |
| Deployment | Streamlit web application |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data processing at scale | PySpark, Pandas, NumPy |
| Feature engineering | `ta` library (technical analysis), scikit-learn |
| Dimensionality reduction | Scikit-learn PCA |
| Time series modeling | statsmodels ARIMA, pmdarima auto_arima |
| ML modeling | Scikit-learn (Random Forest, SVR, LinearSVR), XGBoost |
| Deep learning | TensorFlow / Keras (Autoencoder, LSTM) |
| NLP / Sentiment | DistilRoBERTa (Hugging Face), BERT |
| Similarity search | Scikit-learn cosine similarity |
| Web application | Streamlit |
| Visualization | Matplotlib, Seaborn, Plotly |

---

## Repository Structure

```
.
├── acquisition-and-EDA.ipynb          # Phase 1: Data pipeline, EDA, feature engineering
├── model_stock_price_prediction.ipynb # Phase 2: ARIMA, ML models, sentiment analysis
├── Stock_Recommendation_System.ipynb  # Phase 3: Autoencoder, recommendation engine
├── requirements.txt                   # Core dependencies
├── data/
│   └── nasdaq_ticker_symbols.csv      # NASDAQ ticker metadata
├── Streamlit/
│   └── requirements.txt              # Streamlit app dependencies
└── Assets/                           # Model performance screenshots
    ├── unnamed.png                   # Streamlit app UI
    ├── unnamed-2.png                 # News sentiment feed
    └── unnamed-3.png                 # Model performance chart
```

## How to Run

```bash
# Clone the repo
git clone https://github.com/vanle2000/Stock-based-Recommendation-System.git
cd Stock-based-Recommendation-System

# Install dependencies
pip install -r requirements.txt

# Download datasets (see Dataset section above for Kaggle links)
# Place historical stock data in: data/Stocks/
# Place news headlines CSV in: data/

# Run notebooks in order:
# 1. acquisition-and-EDA.ipynb
# 2. model_stock_price_prediction.ipynb
# 3. Stock_Recommendation_System.ipynb

# Launch the Streamlit app
streamlit run Streamlit/app.py
```

---

## Screenshots

**Streamlit Recommendation UI:**
![UI](Assets/unnamed.png)

**Financial News Sentiment Feed:**
![news](Assets/unnamed-2.png)

**Model Performance:**
![ModelPerformance](Assets/unnamed-3.png)

---

## Limitations & Future Work

| Limitation | Path Forward |
|------------|-------------|
| Training data ends at 2017 | Integrate real-time data via yfinance or Alpha Vantage API |
| MAPE of 13.1% on price prediction | Incorporate order book data and macro signals |
| Autoencoder trained offline | Retrain on rolling window for concept drift |
| No backtesting framework | Implement Backtrader or Zipline for strategy validation |
| Hardcoded local paths in notebooks | Refactor to relative paths with config file |
