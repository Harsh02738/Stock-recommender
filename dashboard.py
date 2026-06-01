"""Streamlit dashboard — run with: streamlit run dashboard.py"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from pipeline import (
    load_price_data, load_ffdata, load_price_cache,
    compute_features, compute_ff_betas, run_clustering,
    get_regime, get_live_signals, run_backtest, compute_metrics,
    TRADING_DAYS,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Momenty",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stMetric label { font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Cached heavy computations ──────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _get_regime():
    return get_regime()


@st.cache_data(ttl=3600, show_spinner=False)
def _get_live_signals(force_refresh=False):
    return get_live_signals(force_refresh=force_refresh)


@st.cache_data(ttl=86400, show_spinner=False)
def _load_ffdata():
    return load_ffdata()


# ── Header ─────────────────────────────────────────────────────────────────────
col_title, col_regime, col_date, col_refresh = st.columns([3, 1.5, 1.5, 1])
with col_title:
    st.title("📈 Momenty")
with col_regime:
    regime = _get_regime()
    badge  = "🟢 **BULL**" if regime == 'bull' else ("🔴 **BEAR**" if regime == 'bear' else "⚪ Unknown")
    st.markdown(f"**Market Regime**<br>{badge}", unsafe_allow_html=True)
with col_date:
    st.markdown(f"**As of**<br>{pd.Timestamp.today().strftime('%d %b %Y')}", unsafe_allow_html=True)
with col_refresh:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.session_state['force_refresh'] = True
        st.rerun()

if regime == 'bear':
    st.warning("⚠️  Bear market detected (NIFTY 500 50d MA < 200d MA). Proceed with caution.")

st.divider()

# ── Load live signals (used across tabs) ────────────────────────────────────────
with st.spinner("Computing live signals from latest prices..."):
    _force = st.session_state.pop('force_refresh', False)
    signals = _get_live_signals(force_refresh=_force)

ffdata = _load_ffdata()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Live Signals", "📈 Backtest", "🔬 Factor Insights"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Signals
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Current Portfolio Recommendations")
    st.caption("Stocks selected by the best momentum cluster, optimised with Fama-French expected returns.")

    if not signals:
        st.error("Could not compute signals. Ensure data caches exist or run `python main.py` first.")
    else:
        STRATEGY_LABELS = {
            'max_sharpe_l2': '🎯 Max Sharpe  (L2 Regularised)',
            'hrp':           '⚖️  Hierarchical Risk Parity',
            'max_utility':   '📐 Max Quadratic Utility',
        }

        for strat_name, df in signals.items():
            label = STRATEGY_LABELS.get(strat_name, strat_name)
            with st.expander(label, expanded=True):
                if df.empty:
                    st.info("No stocks selected for this strategy.")
                    continue

                top_ticker  = df.iloc[0]['ticker']
                top_weight  = df.iloc[0]['weight'] * 100
                n_stocks    = len(df)
                avg_rsi     = df['rsi'].mean() if 'rsi' in df.columns else float('nan')
                avg_wml     = df['wml_beta'].mean() if 'wml_beta' in df.columns else float('nan')

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Stocks Selected", n_stocks)
                m2.metric("Top Holding", top_ticker)
                m3.metric("Top Weight", f"{top_weight:.1f}%")
                m4.metric("Avg RSI", f"{avg_rsi:.1f}" if not np.isnan(avg_rsi) else "—")
                m5.metric("Avg WML β", f"{avg_wml:.2f}" if not np.isnan(avg_wml) else "—")

                display = df.copy()
                display['weight']    = (display['weight']    * 100).round(2)
                display['return_1m'] = (display['return_1m'] * 100).round(2)
                for col in ['price', 'rsi', 'smb_beta', 'hml_beta', 'wml_beta', 'mf_beta']:
                    if col in display.columns:
                        display[col] = display[col].round(3)

                display = display.rename(columns={
                    'ticker':    'Ticker',
                    'weight':    'Weight %',
                    'price':     'Price (₹)',
                    'return_1m': '1M Return %',
                    'rsi':       'RSI',
                    'smb_beta':  'SMB β',
                    'hml_beta':  'HML β',
                    'wml_beta':  'WML β',
                    'mf_beta':   'MF β',
                })

                st.dataframe(
                    display.style.background_gradient(subset=['Weight %'], cmap='Greens', vmin=0),
                    use_container_width=True,
                    hide_index=True,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Backtest
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Backtest Performance vs NIFTY 500")
    st.caption("Walk-forward monthly rebalancing with FF-model expected returns.")

    run_col, _ = st.columns([2, 6])
    with run_col:
        run_bt = st.button("▶ Run Full Backtest", type="primary",
                           help="Takes 3–8 minutes. Results are cached for the session.")

    if run_bt:
        with st.spinner("Loading data and running backtest (this can take several minutes)..."):
            _df         = load_price_data()
            _ffdata     = load_ffdata()
            _data       = compute_features(_df)
            _data       = compute_ff_betas(_data, _ffdata)
            _data       = run_clustering(_data)
            _all_t      = _data.index.get_level_values('ticker').unique().tolist()
            _start_date = _data.index.get_level_values('date').unique()[0] - pd.DateOffset(months=12)
            _prices     = load_price_cache(_all_t, _start_date)
            _results    = run_backtest(_data, _prices, _ffdata)
            st.session_state['backtest_results'] = _results
        st.success("Backtest complete.")

    if 'backtest_results' not in st.session_state:
        st.info("Click **▶ Run Full Backtest** to generate performance results.")
    else:
        strategy_results = st.session_state['backtest_results']

        all_dates  = pd.concat(list(strategy_results.values())).index
        _nifty_raw = yf.download('^CRSLDX', start=all_dates.min(), end=all_dates.max(), progress=False)
        nifty_ret  = np.log(_nifty_raw['Close']).diff().dropna()
        if isinstance(nifty_ret, pd.DataFrame):
            nifty_ret = nifty_ret.iloc[:, 0]

        # Cumulative return chart
        palette = {'max_sharpe_l2': '#4472C4', 'hrp': '#ED7D31', 'max_utility': '#70AD47'}
        fig = go.Figure()
        for name, ret_series in strategy_results.items():
            cum = (ret_series.add(1).cumprod() - 1) * 100
            fig.add_trace(go.Scatter(
                x=cum.index, y=cum.values.round(2),
                name=name,
                line=dict(color=palette.get(name, '#888'), width=2),
                hovertemplate='%{y:.1f}%<extra>' + name + '</extra>',
            ))
        nifty_cum = (nifty_ret.add(1).cumprod() - 1) * 100
        fig.add_trace(go.Scatter(
            x=nifty_cum.index, y=nifty_cum.values.round(2),
            name='NIFTY 500',
            line=dict(color='black', width=1.5, dash='dash'),
            hovertemplate='%{y:.1f}%<extra>NIFTY 500</extra>',
        ))
        fig.update_layout(
            title='Cumulative Return % — Strategies vs NIFTY 500',
            xaxis_title='Date', yaxis_title='Cumulative Return %',
            height=500, hovermode='x unified', legend=dict(orientation='h', y=1.08),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Metrics table
        rows = []
        for name, ret_series in strategy_results.items():
            m = compute_metrics(ret_series)
            rows.append({
                'Strategy':       name,
                'Ann. Return %':  round(m['annual_return'] * 100, 2),
                'Ann. Vol %':     round(m['annual_vol']    * 100, 2),
                'Sharpe':         round(m['sharpe'],              2),
                'Max Drawdown %': round(m['max_drawdown']  * 100, 2),
                'Calmar':         round(m['calmar'], 2) if not np.isnan(m['calmar']) else None,
            })
        nm = compute_metrics(nifty_ret)
        rows.append({
            'Strategy':       'NIFTY 500 Benchmark',
            'Ann. Return %':  round(nm['annual_return'] * 100, 2),
            'Ann. Vol %':     round(nm['annual_vol']    * 100, 2),
            'Sharpe':         round(nm['sharpe'],              2),
            'Max Drawdown %': round(nm['max_drawdown']  * 100, 2),
            'Calmar':         round(nm['calmar'], 2) if not np.isnan(nm['calmar']) else None,
        })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Factor Insights
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Fama-French Factor Insights")

    # Factor premium trend
    st.subheader("Rolling 12-Month Factor Premiums (Annualised %)")
    fig_trend = go.Figure()
    factor_colors = {'SMB': '#4472C4', 'HML': '#ED7D31', 'WML': '#70AD47', 'MF': '#9B59B6'}
    for factor, color in factor_colors.items():
        rolling = ffdata[factor].rolling(12).mean() * 12 * 100
        fig_trend.add_trace(go.Scatter(
            x=rolling.index, y=rolling.round(2).values,
            name=factor, line=dict(color=color, width=1.8),
            hovertemplate='%{y:.1f}%<extra>' + factor + '</extra>',
        ))
    fig_trend.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1)
    fig_trend.update_layout(
        xaxis_title='Date', yaxis_title='Annualised Premium %',
        height=380, hovermode='x unified',
        legend=dict(orientation='h', y=1.08),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    if not signals:
        st.info("Live signals not available. Check Tab 1.")
    else:
        st.subheader("Average Factor Exposure by Strategy")

        # Bar chart: avg beta per strategy
        avg_rows = []
        for strat, df in signals.items():
            avg_rows.append({
                'Strategy': strat,
                'SMB': df['smb_beta'].mean(),
                'HML': df['hml_beta'].mean(),
                'WML': df['wml_beta'].mean(),
                'MF':  df['mf_beta'].mean(),
            })
        avg_df = pd.DataFrame(avg_rows).set_index('Strategy')

        fig_bar = go.Figure()
        for factor, color in factor_colors.items():
            fig_bar.add_trace(go.Bar(
                name=factor,
                x=avg_df.index.tolist(),
                y=avg_df[factor].round(3).tolist(),
                marker_color=color,
            ))
        fig_bar.update_layout(
            barmode='group',
            xaxis_title='Strategy', yaxis_title='Average Beta',
            height=350,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Heatmap per strategy
        st.subheader("Per-Stock Factor Beta Heatmap")
        selected_strat = st.selectbox("Strategy", list(signals.keys()))
        hdf = signals[selected_strat][['ticker', 'smb_beta', 'hml_beta', 'wml_beta', 'mf_beta']].copy()
        hdf = hdf.set_index('ticker')
        hdf.columns = ['SMB', 'HML', 'WML', 'MF']
        hdf = hdf.round(3)

        fig_heat = px.imshow(
            hdf,
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0,
            aspect='auto',
            height=max(350, len(hdf) * 22),
            title=f'FF Beta Heatmap — {selected_strat}',
            text_auto='.2f',
        )
        fig_heat.update_xaxes(side='top')
        st.plotly_chart(fig_heat, use_container_width=True)

        # Latest factor premiums table
        st.subheader("Current Factor Premiums (Trailing 36-Month Average)")
        factors = ['SMB', 'HML', 'WML', 'MF']
        prems = ffdata[factors].tail(36).mean() * 12 * 100  # annualised
        prem_df = pd.DataFrame({
            'Factor': factors,
            'Trailing 36M Premium (Ann. %)': prems.round(2).values,
        })
        st.dataframe(prem_df, use_container_width=False, hide_index=True)
