import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from mftool import Mftool
import yfinance as yf
import quantstats as qs
import plotly.express as px
import statsmodels.api as sm

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(layout="wide", page_title="MF Quant Lab Pro")
st.title("🚀 Mutual Fund- Scheme Comparison")

mf = Mftool()

# -------------------------------------------------------
# LOAD SCHEMES
# -------------------------------------------------------
@st.cache_data
def load_schemes():
    return mf.get_scheme_codes()

scheme_codes = load_schemes()
scheme_list = list(scheme_codes.values())

selected_funds = st.multiselect(
    "Select up to 5 Funds",
    scheme_list,
    max_selections=5
)

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime(2015, 1, 1))
with col2:
    end_date = st.date_input("End Date", datetime.today())

# Use TRI benchmark (better than price index)
benchmark_symbol = "^NSEI"

# -------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------
if selected_funds:

    nav_data = {}
    return_data = pd.DataFrame()
    summary = []

    # ---------------------------------------------------
    # BENCHMARK DOWNLOAD (FIXED VERSION)
    # ---------------------------------------------------
    benchmark_df = yf.download(
        benchmark_symbol,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    # Flatten MultiIndex if exists
    if isinstance(benchmark_df.columns, pd.MultiIndex):
        benchmark_df.columns = benchmark_df.columns.get_level_values(0)

    # Prefer Adj Close if available
    if "Adj Close" in benchmark_df.columns:
        benchmark_series = benchmark_df["Adj Close"]
    else:
        benchmark_series = benchmark_df["Close"]

    benchmark_returns = benchmark_series.pct_change().dropna()

    # ---------------------------------------------------
    # PROCESS EACH FUND
    # ---------------------------------------------------
    for fund in selected_funds:

        code = list(scheme_codes.keys())[list(scheme_codes.values()).index(fund)]
        hist = mf.get_scheme_historical_nav(code, as_Dataframe=True)

        hist.columns = hist.columns.str.lower().str.strip()

        if "date" not in hist.columns:
            hist = hist.reset_index()

        hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
        hist["nav"] = pd.to_numeric(hist["nav"], errors="coerce")
        hist = hist.dropna().sort_values("date")

        hist = hist[(hist["date"] >= pd.to_datetime(start_date)) &
                    (hist["date"] <= pd.to_datetime(end_date))]

        hist.set_index("date", inplace=True)

        nav_series = hist["nav"]

        if len(nav_series) < 100:
            st.warning(f"Not enough data for {fund}")
            continue

        nav_data[fund] = nav_series

        returns = nav_series.pct_change().dropna()
        return_data[fund] = returns

        # CAGR
        years = (nav_series.index[-1] - nav_series.index[0]).days / 365
        cagr = ((nav_series.iloc[-1] / nav_series.iloc[0]) ** (1 / years) - 1)

        # Volatility
        volatility = returns.std() * np.sqrt(252)

        # Sharpe Ratio
        sharpe = (returns.mean() * 252) / volatility if volatility != 0 else 0

        # Max Drawdown
        max_dd = qs.stats.max_drawdown(returns)

        # Alpha & Beta
        combined = pd.concat([returns, benchmark_returns], axis=1).dropna()
        combined.columns = ["fund", "benchmark"]

        if len(combined) > 50:
            X = sm.add_constant(combined["benchmark"])
            model = sm.OLS(combined["fund"], X).fit()
            alpha = model.params["const"] * 252
            beta = model.params["benchmark"]
        else:
            alpha = 0
            beta = 0

        summary.append([
            fund,
            round(cagr * 100, 2),
            round(volatility * 100, 2),
            round(sharpe, 2),
            round(max_dd * 100, 2),
            round(alpha * 100, 2),
            round(beta, 2)
        ])

    # ---------------------------------------------------
    # TABS UI
    # ---------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Performance",
        "📈 Rolling Returns",
        "📉 Drawdown",
        "📌 Risk Analytics",
        "🔗 Correlation"
    ])

    # ---------------------------------------------------
    # PERFORMANCE TAB
    # ---------------------------------------------------
    with tab1:

        df = pd.DataFrame(summary,
            columns=["Fund", "CAGR %", "Volatility %", "Sharpe",
                     "Max Drawdown %", "Alpha %", "Beta"])

        st.dataframe(df, use_container_width=True)

        fig = px.scatter(
            df,
            x="Volatility %",
            y="CAGR %",
            size="Sharpe",
            text="Fund",
            title="Risk vs Return Quadrant"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------
    # ROLLING RETURNS
    # ---------------------------------------------------
    with tab2:

        rolling_period = st.radio(
            "Rolling Window",
            ["1Y", "3Y", "5Y"],
            horizontal=True
        )

        window_map = {"1Y": 252, "3Y": 756, "5Y": 1260}
        window = window_map[rolling_period]

        rolling_df = pd.DataFrame()

        for fund in return_data.columns:
            roll = (
                return_data[fund]
                .rolling(window)
                .apply(lambda x: (np.prod(1 + x)) ** (252/window) - 1)
            )
            rolling_df[fund] = roll

        fig = px.line(rolling_df, title="Rolling CAGR Comparison")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------
    # DRAWDOWN TAB
    # ---------------------------------------------------
    with tab3:

        dd_df = pd.DataFrame()

        for fund in return_data.columns:
            dd = qs.stats.to_drawdown_series(return_data[fund])
            dd_df[fund] = dd

        fig = px.line(dd_df, title="Underwater (Drawdown) Chart")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------
    # RISK ANALYTICS TAB
    # ---------------------------------------------------
    with tab4:

        for fund in return_data.columns:
            st.subheader(fund)

            st.write("Sortino:", round(qs.stats.sortino(return_data[fund]), 2))
            st.write("Calmar:", round(qs.stats.calmar(return_data[fund]), 2))
            st.write("Win Rate:", round(qs.stats.win_rate(return_data[fund]) * 100, 2), "%")
            st.write("Skew:", round(qs.stats.skew(return_data[fund]), 2))
            st.write("Kurtosis:", round(qs.stats.kurtosis(return_data[fund]), 2))
            st.markdown("---")

    # ---------------------------------------------------
    # CORRELATION TAB
    # ---------------------------------------------------
    with tab5:

        corr = return_data.corr()

        fig = px.imshow(
            corr,
            text_auto=True,
            title="Correlation Matrix",
            aspect="auto"
        )

        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Select funds to begin comparison.")