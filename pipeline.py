from statsmodels.regression.rolling import RollingOLS
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import statsmodels.api as sm
from io import StringIO
import os
import time
import pandas_ta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns as pf_expected_returns, HRPOpt
from pypfopt.objective_functions import L2_reg
import warnings
warnings.filterwarnings('ignore')

# ── Constants ──────────────────────────────────────────────────────────────────
CACHE_FILE     = "expandeduniversedata.parquet"
PRICE_CACHE    = "pricedatadaily.parquet"
FF_DATA_FILE   = "FFdata.csv"
outlier_cutoff = 0.005
GAMMA_L2       = 2.0
GAMMA_UTILITY  = 1.5
RISK_AVERSION  = 10.0
CLUSTER_W_MOM  = 0.4
CLUSTER_W_RSI  = 0.4
CLUSTER_W_VOL  = 0.2
TRADING_DAYS   = 252


# ── Universe helpers ───────────────────────────────────────────────────────────
def fetch_nifty500_wikipedia():
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get("https://en.wikipedia.org/wiki/NIFTY_500", headers=headers)
    tables = pd.read_html(StringIO(response.text))
    tbl = tables[4]
    tbl.columns = ['Slno', 'Company Name', 'Industry', 'Symbol', 'Series', 'ISIN Code']
    symbols = [s for s in tbl['Symbol'].tolist()[1:] if isinstance(s, str)]
    return [s + '.NS' for s in symbols]


def fetch_nse_index(index_name):
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        api_headers = {**headers, "Referer": "https://www.nseindia.com"}
        url = f"https://www.nseindia.com/api/equity-stockIndices?index={requests.utils.quote(index_name)}"
        resp = session.get(url, headers=api_headers, timeout=15)
        resp.raise_for_status()
        rows = resp.json().get('data', [])
        return [r['symbol'] + '.NS' for r in rows if r.get('symbol') and r['symbol'] != index_name]
    except Exception as e:
        print(f"Warning: Could not fetch '{index_name}' from NSE API: {e}")
        return []


# ── Data loading ───────────────────────────────────────────────────────────────
def load_price_data(force_refresh=False, max_age_hours=20):
    """Download or load cached NIFTY 500 + Microcap 250 daily OHLCV data."""
    if not force_refresh and os.path.exists(CACHE_FILE):
        age_hours = (time.time() - os.path.getmtime(CACHE_FILE)) / 3600
        if age_hours <= max_age_hours:
            return pd.read_parquet(CACHE_FILE)

    nifty500_syms = fetch_nifty500_wikipedia()
    microcap_syms = fetch_nse_index("NIFTY MICROCAP 250")
    seen, symbolslist = set(), []
    for s in nifty500_syms + microcap_syms:
        if s not in seen:
            seen.add(s)
            symbolslist.append(s)

    print(f"Expanded universe: {len(symbolslist)} unique tickers "
          f"({len(nifty500_syms)} NIFTY 500 + {len(microcap_syms)} Microcap 250 before dedup)")

    enddate   = pd.Timestamp.today()
    startdate = enddate - pd.DateOffset(years=5)
    df = yf.download(tickers=symbolslist, start=startdate, end=enddate).stack(future_stack=True)
    df.index.names = ['date', 'ticker']
    df.columns     = df.columns.str.lower()
    df.to_parquet(CACHE_FILE)
    return df


def load_price_cache(all_tickers, start_date, force_refresh=False, max_age_hours=20):
    """Download or load cached daily close prices for all tickers."""
    cache_stale = False
    if os.path.exists(PRICE_CACHE):
        age_hours = (time.time() - os.path.getmtime(PRICE_CACHE)) / 3600
        if age_hours > max_age_hours:
            cache_stale = True

    if not force_refresh and not cache_stale and os.path.exists(PRICE_CACHE):
        return pd.read_parquet(PRICE_CACHE)

    newdf = yf.download(tickers=all_tickers, start=start_date)
    newdf.to_parquet(PRICE_CACHE)
    return newdf


# ── Feature engineering ────────────────────────────────────────────────────────
def _compute_atr(data):
    high  = data['high'].astype(float)
    low   = data['low'].astype(float)
    close = data['close'].astype(float)
    tr    = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr   = tr.ewm(span=14, adjust=False).mean()
    return atr.sub(atr.mean()).div(atr.std())


def _compute_macd(close):
    macd = pandas_ta.macd(close=close).iloc[:, 0]
    return macd.sub(macd.mean()).div(macd.std())


def _compute_returns(df):
    for lag in [1, 2, 3, 6, 9, 12]:
        df[f'return_{lag}m'] = df['close'].pct_change(lag).pipe(
            lambda x: x.clip(lower=x.quantile(outlier_cutoff), upper=x.quantile(1 - outlier_cutoff))
        ).add(1).pow(1 / lag).sub(1)
    return df


def compute_features(df):
    """Add technical indicators; return monthly-resampled feature DataFrame."""
    df = df.copy()
    df['garman_klass_vol'] = (
        ((np.log(df['high']) - np.log(df['low']))**2) / 2
        - (2*np.log(2) - 1) * ((np.log(df['close']) - np.log(df['open']))**2)
    )
    df['rsi']     = df.groupby(level=1)['close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))
    df['bb_low']  = df.groupby(level=1)['close'].transform(
        lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 0])
    df['bb_mid']  = df.groupby(level=1)['close'].transform(
        lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 1])
    df['bb_high'] = df.groupby(level=1)['close'].transform(
        lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 2])
    df['atr']     = df.groupby(level=1, group_keys=False).apply(_compute_atr)
    df['macd']    = df.groupby(level=1, group_keys=False)['close'].apply(_compute_macd)
    df['rs_vol']  = (df['volume'] * df['close']) / 1e6

    lastcols = [c for c in df.columns.unique(0) if c not in ['rs_vol', 'open', 'volume', 'high', 'low', 'adj close']]
    features = df.unstack()[lastcols].resample('ME').last().stack('ticker')
    vol      = df.unstack('ticker')['rs_vol'].resample('ME').mean().stack('ticker').to_frame('rs_vol')
    data     = pd.concat([features, vol], axis=1).dropna()
    data['rs_vol'] = data['rs_vol'].unstack('ticker').rolling(5*12).mean().stack()
    data = data.drop(['rs_vol'], axis=1)
    data = data.groupby(level=1, group_keys=False).apply(_compute_returns).dropna()
    data = data.drop('close', axis=1)
    return data


# ── Fama-French ────────────────────────────────────────────────────────────────
def load_ffdata():
    """Load and normalise the IIM Ahmedabad Fama-French factor CSV."""
    ffdata = pd.read_csv(FF_DATA_FILE, parse_dates=['Date']).drop('RF', axis=1).set_index('Date')
    ffdata = ffdata.resample('ME').last().div(100)
    ffdata.index.name = 'date'
    return ffdata


def compute_ff_betas(data, ffdata):
    """
    Add rolling SMB/HML/WML/MF betas to data.
    Betas are lagged one month to prevent look-ahead bias.
    """
    factordata = ffdata.join(data['return_1m']).sort_index()
    valid      = factordata.groupby(level=1).size()
    valid      = valid[valid >= 10]
    factordata = factordata[factordata.index.get_level_values('ticker').isin(valid.index)]

    betas = factordata.groupby(level=1, group_keys=False).apply(
        lambda x: RollingOLS(
            endog=x['return_1m'],
            exog=sm.add_constant(x.drop('return_1m', axis=1)),
            window=min(24, x.shape[0]),
            min_nobs=len(x.columns) + 1
        ).fit(params_only=True).params.drop('const', axis=1)
    )
    data = data.join(betas.groupby('ticker').shift())
    factors = ['SMB', 'HML', 'WML', 'MF']
    data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(
        lambda x: x.fillna(x.mean())
    )
    return data


def compute_ff_expected_returns(best_stocks, ffdata, universe_tickers=None):
    """
    Factor-model expected returns: E[R_i] = beta_i @ factor_premiums * 12.

    Uses trailing 36-month average of each factor as the premium estimate.
    Provides theoretically grounded expected returns instead of naive historical means.
    """
    factors = ['SMB', 'HML', 'WML', 'MF']
    factor_premiums = ffdata[factors].tail(36).mean()

    betas = best_stocks[factors]
    if universe_tickers is not None:
        betas = betas.reindex(universe_tickers)

    ff_er = betas.dot(factor_premiums) * 12  # annualised
    return ff_er.dropna()


# ── Clustering ─────────────────────────────────────────────────────────────────
def _get_clusters(df):
    scaler    = StandardScaler()
    feat_cols = [c for c in df.columns if c != 'cluster']
    scaled    = scaler.fit_transform(df[feat_cols])
    df = df.copy()
    df['cluster'] = KMeans(n_clusters=4, random_state=0, init='random').fit(scaled).labels_
    return df


def _score_cluster(cluster_slice):
    summary = cluster_slice.groupby('cluster').agg(
        mom1  = ('return_1m',        'mean'),
        mom6  = ('return_6m',        'mean'),
        rsi   = ('rsi',              'mean'),
        gkvol = ('garman_klass_vol', 'mean'),
    )
    def zscore(s):
        std = s.std()
        return (s - s.mean()) / std if std > 0 else s * 0
    summary['z_mom']      = zscore(summary['mom1'] * 0.5 + summary['mom6'] * 0.5)
    summary['z_rsi']      = zscore(summary['rsi'])
    summary['z_gkvol']    = zscore(summary['gkvol'])
    summary['composite']  = (
        CLUSTER_W_MOM * summary['z_mom']
        + CLUSTER_W_RSI * summary['z_rsi']
        - CLUSTER_W_VOL * summary['z_gkvol']
    )
    return summary['composite'].idxmax()


def run_clustering(data):
    """Add 'cluster', 'best_cluster', and 'is_best' columns to data."""
    data = data.dropna().groupby('date', group_keys=False).apply(_get_clusters)
    best_cluster = data.groupby(level=0).apply(_score_cluster).rename('best_cluster')
    data = data.join(best_cluster, on='date')
    data['is_best'] = data['cluster'] == data['best_cluster']
    return data


# ── Regime filter ──────────────────────────────────────────────────────────────
def get_regime():
    """Bull/bear indicator via NIFTY 500 50-day vs 200-day moving average."""
    try:
        nifty = yf.download('^CRSLDX', period='1y', progress=False)['Close']
        if isinstance(nifty, pd.DataFrame):
            nifty = nifty.iloc[:, 0]
        ma50  = float(nifty.rolling(50).mean().iloc[-1])
        ma200 = float(nifty.rolling(200).mean().iloc[-1])
        return 'bull' if ma50 > ma200 else 'bear'
    except Exception:
        return 'unknown'


# ── Portfolio optimisation ─────────────────────────────────────────────────────
def _resolve_mu(prices, mu):
    """Reindex mu to match prices.columns; fill gaps with historical mean."""
    if mu is None:
        return pf_expected_returns.mean_historical_return(prices=prices, frequency=TRADING_DAYS)
    mu = mu.reindex(prices.columns)
    if mu.isna().any():
        hist_mu = pf_expected_returns.mean_historical_return(prices=prices, frequency=TRADING_DAYS)
        mu = mu.fillna(hist_mu)
    return mu


def _optimize_max_sharpe_l2(prices, mu=None):
    mu = _resolve_mu(prices, mu)
    S  = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    ef = EfficientFrontier(expected_returns=mu, cov_matrix=S, weight_bounds=(0, 1), solver='SCS')
    ef.add_objective(L2_reg, gamma=GAMMA_L2)
    ef.max_sharpe()
    return ef.clean_weights()


def _optimize_hrp(prices, mu=None):
    log_ret = np.log(prices / prices.shift(1)).dropna()
    hrp = HRPOpt(returns=log_ret)
    hrp.optimize()
    return hrp.clean_weights()


def _optimize_max_utility(prices, mu=None):
    mu = _resolve_mu(prices, mu)
    S  = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    ef = EfficientFrontier(expected_returns=mu, cov_matrix=S, weight_bounds=(0, 1), solver='SCS')
    ef.add_objective(L2_reg, gamma=GAMMA_UTILITY)
    ef.max_quadratic_utility(risk_aversion=RISK_AVERSION)
    return ef.clean_weights()


STRATEGY_FNS = {
    'max_sharpe_l2': _optimize_max_sharpe_l2,
    'hrp':           _optimize_hrp,
    'max_utility':   _optimize_max_utility,
}


def run_optimization(prices, mu=None):
    """Run all 3 strategies. Returns {name: weights_dict}."""
    results = {}
    for name, fn in STRATEGY_FNS.items():
        try:
            results[name] = fn(prices, mu=mu)
        except Exception as e:
            print(f"  [{name}] optimisation failed: {e}")
            results[name] = None
    return results


# ── Performance metrics ────────────────────────────────────────────────────────
def compute_metrics(ret_series):
    """Annual return, vol, sharpe, max drawdown, and calmar for a daily return series."""
    ann_ret     = ret_series.mean() * TRADING_DAYS
    ann_vol     = ret_series.std()  * np.sqrt(TRADING_DAYS)
    sharpe      = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cum         = (1 + ret_series).cumprod()
    rolling_max = cum.cummax()
    max_dd      = ((cum - rolling_max) / rolling_max).min()
    calmar      = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    return {
        'annual_return': ann_ret,
        'annual_vol':    ann_vol,
        'sharpe':        sharpe,
        'max_drawdown':  max_dd,
        'calmar':        calmar,
    }


# ── Backtest ───────────────────────────────────────────────────────────────────
def run_backtest(data, prices_raw, ffdata):
    """
    Monthly walk-forward backtest using FF-model expected returns.

    data       : monthly feature + beta DataFrame with 'is_best' column
    prices_raw : yf.download() output (MultiLevel columns, 'Close' level)
    ffdata     : monthly FF factor DataFrame from load_ffdata()

    Returns {strategy_name: daily_return_series}.
    """
    filterdf = data[data['is_best']].copy()
    filterdf = filterdf.reset_index(level=1)
    filterdf.index = filterdf.index + pd.DateOffset(1)
    filterdf = filterdf.reset_index().set_index(['date', 'ticker'])
    dates      = filterdf.index.get_level_values('date').unique().tolist()
    fixeddates = {date: filterdf.xs(date, level=0).index.to_list() for date in dates}

    returnsdf      = np.log(prices_raw['Close']).diff()
    strategy_results = {}

    for strat_name, strat_fn in STRATEGY_FNS.items():
        print(f"\n{'='*50}\nRunning backtest: {strat_name}\n{'='*50}")
        portfoliodf = pd.DataFrame()

        for startdate in fixeddates.keys():
            enddate   = pd.to_datetime(startdate) + pd.offsets.MonthEnd(0)
            cols      = fixeddates[startdate]
            opt_start = pd.to_datetime(startdate) - pd.DateOffset(months=12)
            opt_end   = pd.to_datetime(startdate) - pd.DateOffset(days=1)

            available = [c for c in cols if c in prices_raw['Close'].columns]
            if len(available) < 2:
                continue
            optdf = prices_raw[opt_start:opt_end]['Close'][available]
            threshold = int(0.8 * len(optdf))
            optdf = optdf.dropna(axis=1, thresh=threshold).ffill().bfill()
            if optdf.shape[1] < 2:
                continue

            # Build FF expected returns for this rebalancing date
            mu = None
            try:
                month_data = filterdf.xs(startdate, level=0)
                ff_er = compute_ff_expected_returns(
                    month_data, ffdata, universe_tickers=optdf.columns.tolist()
                )
                if len(ff_er) >= 2:
                    mu = ff_er
            except Exception:
                pass

            try:
                weights = strat_fn(optdf, mu=mu)
            except Exception as e:
                print(f"  skipped {pd.to_datetime(startdate).date()}: {e}")
                continue

            weights = pd.DataFrame(weights, index=pd.Series(0))
            tempdf  = returnsdf[startdate:enddate]
            tempdf  = (tempdf.stack(future_stack=True)
                             .rename_axis(['date', 'ticker'])
                             .to_frame('return')
                             .reset_index(level=0))
            tempdf.index.name = 'ticker'
            w = weights.stack().to_frame('weight')
            w.index = w.index.droplevel(0)
            w.index.name = 'ticker'
            tempdf = tempdf.join(w).reset_index().set_index(['date', 'ticker'])
            tempdf['weighted_return'] = tempdf['return'] * tempdf['weight']
            portfoliodf = pd.concat([portfoliodf, tempdf], axis=0)

        if portfoliodf.empty:
            print(f"  No results for {strat_name}")
            continue

        portfoliodf = portfoliodf.dropna()
        strategy_results[strat_name] = portfoliodf.groupby(level=0)['weighted_return'].sum()

        latest   = portfoliodf.index.get_level_values('date').max()
        latest_w = portfoliodf.loc[latest].dropna()[['weight']]
        latest_w = latest_w[latest_w['weight'] > 0.001].sort_values('weight', ascending=False)
        print(f"\n[{strat_name}] Latest weights ({pd.to_datetime(latest).date()}):")
        print(latest_w.to_string())

    return strategy_results


# ── Live signals ───────────────────────────────────────────────────────────────
def get_live_signals(force_refresh=False):
    """
    Run the pipeline on latest data and return current recommendations.

    Returns {strategy_name: DataFrame} where each DataFrame has columns:
    ticker, weight, price, return_1m, rsi, smb_beta, hml_beta, wml_beta, mf_beta
    """
    df     = load_price_data(force_refresh=force_refresh)
    ffdata = load_ffdata()
    data   = compute_features(df)
    data   = compute_ff_betas(data, ffdata)
    data   = run_clustering(data)

    latest_month = data.index.get_level_values('date').max()
    cluster_data = data.xs(latest_month, level='date')
    best_stocks  = cluster_data[cluster_data['is_best']].copy()
    tickers      = best_stocks.index.tolist()

    if len(tickers) < 2:
        return {}

    all_tickers = data.index.get_level_values('ticker').unique().tolist()
    start_date  = data.index.get_level_values('date').unique()[0] - pd.DateOffset(months=12)
    prices_raw  = load_price_cache(all_tickers, start_date, force_refresh=force_refresh)

    opt_start = pd.Timestamp.today() - pd.DateOffset(months=12)
    optdf     = prices_raw[opt_start:]['Close'][tickers]
    threshold = int(0.8 * len(optdf))
    optdf     = optdf.dropna(axis=1, thresh=threshold).ffill().bfill()
    tickers   = optdf.columns.tolist()

    ff_er = compute_ff_expected_returns(best_stocks, ffdata, universe_tickers=tickers)
    mu    = ff_er if len(ff_er) >= 2 else None

    weights_by_strat = run_optimization(optdf, mu=mu)

    latest_prices = prices_raw['Close'].iloc[-1]

    signals = {}
    for strat_name, w_dict in weights_by_strat.items():
        if w_dict is None:
            continue
        w = pd.Series(w_dict, name='weight')
        w = w[w > 0.001].sort_values(ascending=False)
        if w.empty:
            continue
        out = pd.DataFrame({'weight': w})
        out['price']     = latest_prices.reindex(out.index)
        out['return_1m'] = best_stocks['return_1m'].reindex(out.index)
        out['rsi']       = best_stocks['rsi'].reindex(out.index)
        out['smb_beta']  = best_stocks['SMB'].reindex(out.index)
        out['hml_beta']  = best_stocks['HML'].reindex(out.index)
        out['wml_beta']  = best_stocks['WML'].reindex(out.index)
        out['mf_beta']   = best_stocks['MF'].reindex(out.index)
        out.index.name   = 'ticker'
        signals[strat_name] = out.reset_index()

    return signals
