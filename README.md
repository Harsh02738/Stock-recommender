# Momenty — Quantitative Equity Strategy for Indian Markets

A systematic, factor-driven portfolio construction engine for the Indian equity market (NIFTY 500 + Microcap 250). Combines technical analysis, Fama-French factor models, and unsupervised machine learning to identify high-momentum stocks and build optimised monthly portfolios — with a live Streamlit dashboard for real-time signals.

**Backtested 2021–2026 · 3 strategies · Monthly rebalancing · NIFTY 500 benchmark**

---

## How It Works

The pipeline runs in five stages every month:

```
Daily OHLCV data (yfinance)
        │
        ▼
Feature Engineering  ──  17 features per stock
(Garman-Klass Vol, RSI, Bollinger Bands, ATR, MACD,
 Dollar Volume, 6 momentum lags × 1/2/3/6/9/12m)
        │
        ▼
Fama-French Rolling Betas  ──  RollingOLS on IIM-A factor data
(SMB, HML, WML, MF — 24-month window, lagged 1m)
        │
        ▼
KMeans Clustering (k=4)  ──  Best cluster by composite score
(0.4 × momentum z-score + 0.4 × RSI z-score − 0.2 × vol z-score)
        │
        ▼
Portfolio Optimisation  ──  3 parallel strategies
(Max Sharpe L2 · Hierarchical Risk Parity · Max Quadratic Utility)
        │
        ▼
Live Dashboard  ──  Picks + prices + FF betas + targets
```

---

## Features

- **17-feature engineering** per stock: momentum across 6 horizons, RSI, Bollinger Bands, ATR, MACD, Garman-Klass volatility, dollar volume
- **Fama-French 4-factor model** (IIM Ahmedabad dataset): rolling OLS betas for SMB, HML, WML, MF used both as clustering features and as the expected-returns input to portfolio optimisation
- **KMeans momentum clustering**: selects the investable universe each month by identifying the highest-scoring risk/return cluster
- **Three portfolio strategies**: Max Sharpe (L2 regularised), Hierarchical Risk Parity, Max Quadratic Utility — all using Ledoit-Wolf covariance shrinkage
- **Bull/bear regime filter**: NIFTY 500 50-day vs 200-day MA golden/death cross shown on dashboard
- **Streamlit dashboard**: live signals tab, backtest tab (Plotly charts + full metrics), factor insights tab (beta heatmaps, rolling premiums)
- **Auto-refreshing caches**: OHLCV and price data refresh automatically after 20 hours

---

## Dashboard

Three tabs:

**📊 Live Signals** — current month's portfolio for each strategy with weight %, live price, 1M return, RSI, and all four FF betas

**📈 Backtest** — cumulative return curves vs NIFTY 500 benchmark, plus a full metrics table: Annual Return, Volatility, Sharpe, Max Drawdown, Calmar Ratio

**🔬 Factor Insights** — rolling 12-month factor premiums (SMB/HML/WML/MF), average factor exposure by strategy, per-stock beta heatmap

---

## Backtest Results (2021–2026)

| Strategy | Ann. Return | Sharpe | vs NIFTY 500 |
|---|---|---|---|
| Max Sharpe L2 | ~28–35% | ~1.4 | +15–20 pp |
| Hierarchical Risk Parity | ~24–30% | ~1.2 | +10–15 pp |
| Max Quadratic Utility | ~26–32% | ~1.3 | +12–18 pp |
| NIFTY 500 Benchmark | ~13% | ~0.7 | — |

*Results vary by run date and universe composition. Past performance is not indicative of future results.*

---

## Project Structure

```
momenty/
├── dashboard.py          # Streamlit app — run this
├── pipeline.py           # All core logic (importable)
├── main.py               # CLI backtest entry point
├── FFdata.csv            # IIM Ahmedabad Fama-French factor data (monthly)
├── requirements.txt      # Python dependencies
├── req.txt               # Same (legacy)
└── .gitignore            # Excludes large parquet caches and images
```

**Generated at runtime (not in repo):**
- `expandeduniversedata.parquet` — 5 years of daily OHLCV for ~750 stocks (~21 MB)
- `pricedatadaily.parquet` — daily close prices for backtest (~23 MB)
- `result.png` — backtest chart (CLI)
- `portfolio_weights.xlsx` — latest weights export (CLI)

---

## Setup

**Requirements:** Python 3.10+

```bash
git clone https://github.com/Harsh02738/Stock-recommender.git
cd Stock-recommender
pip install -r requirements.txt
```

### Run the dashboard
```bash
streamlit run dashboard.py
```

On first launch the app downloads ~750 tickers × 5 years of data from yfinance (~5 minutes). Subsequent loads use the local cache.

### Run the CLI backtest
```bash
python main.py
```

Saves `result.png` (cumulative return chart) and prints a full metrics table to the terminal.

---

## Data Sources

| Data | Source | Update frequency |
|---|---|---|
| NIFTY 500 constituents | Wikipedia (scraped) | On cache refresh |
| NIFTY Microcap 250 constituents | NSE India API | On cache refresh |
| Daily OHLCV prices | [yfinance](https://github.com/ranaroussi/yfinance) | Auto-refreshes every 20h |
| Fama-French factors (India) | [IIM Ahmedabad](https://faculty.iima.ac.in/~iffm/Indian-Fama-French-Momentum/) | Bundled (`FFdata.csv`) |

---

## Fama-French Implementation

Factor betas are estimated via rolling OLS with a 24-month window and lagged one period to prevent look-ahead bias:

```
R_i,t = α + β_SMB · SMB_t + β_HML · HML_t + β_WML · WML_t + β_MF · MF_t + ε
```

Factor-model expected returns are used directly in portfolio construction:

```
E[R_i] = β_SMB · premium_SMB + β_HML · premium_HML + β_WML · premium_WML + β_MF · premium_MF
```

where premiums are the trailing 36-month average of each factor. This replaces naive historical mean returns with theoretically grounded estimates — stocks with high WML beta (momentum) receive higher expected returns; stocks with high HML beta (value) receive the value premium.

---

## Portfolio Strategies

| Strategy | Method | Objective |
|---|---|---|
| **Max Sharpe L2** | `EfficientFrontier` + L2 regularisation (γ=2.0) | Maximise Sharpe, L2 penalty prevents concentration |
| **HRP** | `HRPOpt` (hierarchical clustering on returns) | Minimise cluster fragmentation, no expected-returns assumption |
| **Max Utility** | `EfficientFrontier` + L2 (γ=1.5), risk aversion=10 | Maximise quadratic utility, more return-oriented than Sharpe |

All strategies use Ledoit-Wolf covariance shrinkage.

---

## Deployment

### Streamlit Community Cloud (recommended)

1. Fork or push to your GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. **New app** → select this repo → branch `main` → main file: `dashboard.py`
4. Deploy — the app installs `requirements.txt` automatically

The parquet caches are excluded from the repo and built on first run.

---

## Disclaimer

This project is for educational and research purposes only. Nothing in this repository constitutes financial advice. All backtest results are hypothetical and subject to data-snooping and survivorship bias. Always do your own research before making investment decisions.

---

## Dependencies

`pandas` · `numpy` · `yfinance` · `statsmodels` · `scikit-learn` · `PyPortfolioOpt` · `pandas_ta` · `streamlit` · `plotly` · `pyarrow` · `scipy` · `lxml` · `requests`
