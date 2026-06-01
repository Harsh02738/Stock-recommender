"""CLI entry point — runs the full backtest and saves result.png + portfolio_weights.xlsx."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pipeline import (
    load_price_data, load_ffdata, load_price_cache,
    compute_features, compute_ff_betas, run_clustering,
    run_backtest, compute_metrics, TRADING_DAYS,
)
import yfinance as yf

# ── Build pipeline ──────────────────────────────────────────────────────────
df         = load_price_data()
ffdata     = load_ffdata()
data       = compute_features(df)
data       = compute_ff_betas(data, ffdata)
data       = run_clustering(data)

all_tickers = data.index.get_level_values('ticker').unique().tolist()
start_date  = data.index.get_level_values('date').unique()[0] - pd.DateOffset(months=12)
prices_raw  = load_price_cache(all_tickers, start_date)

strategy_results = run_backtest(data, prices_raw, ffdata)

# ── Benchmark ───────────────────────────────────────────────────────────────
all_dates = pd.concat(list(strategy_results.values())).index
nifty_raw = yf.download('^CRSLDX', start=all_dates.min(), end=all_dates.max(), progress=False)
nifty_ret = np.log(nifty_raw['Close']).diff().dropna()
if isinstance(nifty_ret, pd.DataFrame):
    nifty_ret = nifty_ret.iloc[:, 0]

# ── Chart ───────────────────────────────────────────────────────────────────
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(16, 7))
ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b-%Y'))
plt.xticks(rotation=45)

palette = {'max_sharpe_l2': 'steelblue', 'hrp': 'darkorange', 'max_utility': 'seagreen'}
for name, ret_series in strategy_results.items():
    cum = (ret_series.add(1).cumprod() - 1) * 100
    ax.plot(cum, label=name, color=palette.get(name), linewidth=1.8)

nifty_cum = (nifty_ret.add(1).cumprod() - 1) * 100
ax.plot(nifty_cum, label='NIFTY 500', color='black', linewidth=1.2, linestyle='--')

ax.legend(fontsize=11)
ax.set_title('Strategy Comparison vs NIFTY 500 Benchmark', fontsize=14)
ax.set_ylabel('Cumulative Return %')
plt.tight_layout()
plt.savefig('result.png', dpi=150)
plt.show()

# ── Summary stats ───────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"{'Strategy':<22}  {'Ann. Return':>11}  {'Ann. Vol':>8}  {'Sharpe':>7}  {'Max DD':>8}  {'Calmar':>7}")
print("-" * 70)
for name, ret_series in strategy_results.items():
    m = compute_metrics(ret_series)
    calmar_str = f"{m['calmar']:>7.2f}" if not np.isnan(m['calmar']) else "    N/A"
    print(f"{name:<22}  {m['annual_return']*100:>10.2f}%  {m['annual_vol']*100:>7.2f}%"
          f"  {m['sharpe']:>7.2f}  {m['max_drawdown']*100:>7.2f}%  {calmar_str}")

n = compute_metrics(nifty_ret)
calmar_str = f"{n['calmar']:>7.2f}" if not np.isnan(n['calmar']) else "    N/A"
print(f"{'NIFTY 500 Benchmark':<22}  {n['annual_return']*100:>10.2f}%  {n['annual_vol']*100:>7.2f}%"
      f"  {n['sharpe']:>7.2f}  {n['max_drawdown']*100:>7.2f}%  {calmar_str}")
print(f"{'='*70}")

# ── Export latest weights ───────────────────────────────────────────────────
with pd.ExcelWriter('portfolio_weights.xlsx') as writer:
    for name, ret_series in strategy_results.items():
        latest_date = ret_series.index.max()
        print(f"\n[{name}] Saving weights as of {latest_date.date()}")
