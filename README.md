"""
BRENT CRUDE OIL CTA STRATEGY — Backtest Platform
TSMOM + Carry + OVX Macro Filter
Scotiabank QIS Desk
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime, date
import os

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Brent CTA Backtest",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load Bloomberg data from xlsx or csv."""
    if os.path.exists("bloomberg_data.xlsx"):
        df = pd.read_excel("bloomberg_data.xlsx")
    elif os.path.exists("bloomberg_data.csv"):
        df = pd.read_csv("bloomberg_data.csv")
    else:
        return None

    # Find the date column (case-insensitive)
    date_col = None
    for col in df.columns:
        if col.strip().lower() == "date":
            date_col = col
            break
    if date_col is None:
        # Assume first column is date
        date_col = df.columns[0]

    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Standardize column names — handle various Bloomberg export formats
    col_map = {}
    for col in df.columns:
        cl = col.strip().upper()
        if cl in ["CO1", "CO1 COMDTY", "BRENT", "CO1_CLOSE"]:
            col_map[col] = "CO1"
        elif cl in ["CO12", "CO12 COMDTY", "CO12_CLOSE"]:
            col_map[col] = "CO12"
        elif cl in ["OVX", "OVX INDEX", "OVX_CLOSE"]:
            col_map[col] = "OVX"
        elif cl in ["VIX", "VIX INDEX", "VIX_CLOSE"]:
            col_map[col] = "VIX"
    df = df.rename(columns=col_map)

    # Ensure required columns exist
    required = ["Date", "CO1", "CO12", "OVX", "VIX"]
    for col in required:
        if col not in df.columns:
            st.error(f"Missing column: {col}. Expected columns: {required}")
            st.stop()

    # Convert to numeric, coerce errors
    for col in ["CO1", "CO12", "OVX", "VIX"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Forward-fill then backfill missing values (OVX before June 2011)
    df[["CO1", "CO12", "OVX", "VIX"]] = df[["CO1", "CO12", "OVX", "VIX"]].ffill().bfill()

    # Drop any remaining NaN rows
    df = df.dropna(subset=["CO1", "CO12", "OVX", "VIX"]).reset_index(drop=True)

    return df


# ──────────────────────────────────────────────
# SIGNAL COMPUTATION
# ──────────────────────────────────────────────
def compute_signals(df_daily, tsmom_weight, carry_weight, lookbacks,
                    ovx_vix_threshold, vix_caution, vix_flatten):
    """
    Compute signals at month-end using daily data.
    Returns a monthly DataFrame with signals.
    """
    df = df_daily.copy()
    df = df.set_index("Date")

    # Identify month-end dates (last trading day of each month)
    month_end_mask = df.index.to_period("M") != df.index.to_period("M").shift(-1, freq="M")
    # Simpler: group by year-month, take last date
    df["YM"] = df.index.to_period("M")
    month_ends = df.groupby("YM").tail(1).copy()

    # TSMOM: for each lookback, compute trailing return and take sign
    tsmom_components = []
    for lb in lookbacks:
        # Trailing return over lb trading days
        ret = df["CO1"].pct_change(lb)
        # Sample at month-ends
        month_end_ret = ret.loc[month_ends.index]
        tsmom_components.append(np.sign(month_end_ret))

    if len(tsmom_components) > 0:
        tsmom_df = pd.concat(tsmom_components, axis=1)
        month_ends["TSMOM"] = tsmom_df.mean(axis=1).values
    else:
        month_ends["TSMOM"] = 0.0

    # Carry: (CO1 - CO12) / CO1 → sign
    month_ends["Carry_Spread"] = (month_ends["CO1"] - month_ends["CO12"]) / month_ends["CO1"]
    month_ends["Carry"] = np.sign(month_ends["Carry_Spread"])

    # OVX Macro Filter
    month_ends["OVX_VIX_Ratio"] = month_ends["OVX"] / month_ends["VIX"].replace(0, np.nan)
    month_ends["OVX_VIX_Ratio"] = month_ends["OVX_VIX_Ratio"].fillna(1.0)

    def macro_scale(row):
        if row["OVX_VIX_Ratio"] > ovx_vix_threshold:
            return 1.0  # Oil-specific shock — stay full size
        if row["VIX"] > vix_flatten:
            return 0.0  # Systemic crisis — flatten
        if row["VIX"] > vix_caution:
            return 0.5  # Caution — half size
        return 1.0  # Normal

    month_ends["Macro_Scale"] = month_ends.apply(macro_scale, axis=1)

    # Composite signal
    raw = tsmom_weight * month_ends["TSMOM"] + carry_weight * month_ends["Carry"]
    month_ends["Composite"] = (raw * month_ends["Macro_Scale"]).clip(-1, 1)

    # Signal contributions for breakdown chart
    month_ends["TSMOM_Contribution"] = tsmom_weight * month_ends["TSMOM"]
    month_ends["Carry_Contribution"] = carry_weight * month_ends["Carry"]

    # Clean up
    month_ends = month_ends.drop(columns=["YM"], errors="ignore")
    month_ends = month_ends.reset_index()
    month_ends = month_ends.rename(columns={"index": "Date"} if "index" in month_ends.columns else {})

    # Make sure Date is in columns
    if "Date" not in month_ends.columns and month_ends.index.name == "Date":
        month_ends = month_ends.reset_index()

    return month_ends


# ──────────────────────────────────────────────
# BACKTEST ENGINE
# ──────────────────────────────────────────────
def run_backtest(df_daily, df_signals, tc,
                 vol_target_enabled, vol_target,
                 trailing_stop_enabled, trailing_stop_threshold):
    """
    Run daily backtest using monthly signals.
    """
    df = df_daily.copy()
    df = df.set_index("Date") if "Date" in df.columns else df.copy()

    # Daily CO1 returns
    df["Market_Ret"] = df["CO1"].pct_change()

    # Map monthly signals to daily: each day gets the signal from the prior month-end
    df_signals_indexed = df_signals.set_index("Date")
    signal_series = df_signals_indexed["Composite"]

    # For each trading day, find the most recent month-end signal BEFORE that day
    df["Signal"] = np.nan
    for i, sig_date in enumerate(signal_series.index):
        sig_val = signal_series.iloc[i]
        # Apply this signal to all days AFTER this month-end until the next month-end
        if i + 1 < len(signal_series.index):
            next_date = signal_series.index[i + 1]
            mask = (df.index > sig_date) & (df.index <= next_date)
        else:
            mask = df.index > sig_date
        df.loc[mask, "Signal"] = sig_val

    # Lag by one more period: signal computed at month-end M applied to month M+1
    # The loop above already does this (applies signal to days AFTER the month-end)
    df["Position"] = df["Signal"].ffill().fillna(0)

    # Vol targeting
    if vol_target_enabled and vol_target > 0:
        realized_vol = df["Market_Ret"].rolling(21, min_periods=5).std() * np.sqrt(252)
        vol_scalar = (vol_target / realized_vol).clip(0.25, 2.0)
        df["Position"] = df["Position"] * vol_scalar

    # Strategy returns
    df["Strat_Gross"] = df["Position"] * df["Market_Ret"]

    # Transaction costs: charge on first day of month where signal changed
    df["Month"] = df.index.to_period("M")
    monthly_positions = df.groupby("Month")["Position"].first()
    position_changed = monthly_positions.diff().abs() > 0.01
    tc_months = position_changed[position_changed].index

    df["TC"] = 0.0
    for m in tc_months:
        month_days = df[df["Month"] == m].index
        if len(month_days) > 0:
            df.loc[month_days[0], "TC"] = tc

    df["Strat_Net"] = df["Strat_Gross"] - df["TC"]

    # Cumulative
    df["Cumulative"] = (1 + df["Strat_Net"].fillna(0)).cumprod()

    # Trailing stop
    if trailing_stop_enabled and trailing_stop_threshold < 0:
        running_max = df["Cumulative"].cummax()
        drawdown = df["Cumulative"] / running_max - 1
        stopped = False
        current_month = None

        for idx in df.index:
            month = idx.to_period("M")
            if month != current_month:
                # New month — re-enter
                stopped = False
                current_month = month

            if stopped:
                df.loc[idx, "Strat_Net"] = 0.0
            elif drawdown.loc[idx] < trailing_stop_threshold:
                stopped = True
                df.loc[idx, "Strat_Net"] = 0.0

        # Recompute cumulative
        df["Cumulative"] = (1 + df["Strat_Net"].fillna(0)).cumprod()

    # Buy & hold
    df["BH_Ret"] = df["Market_Ret"].fillna(0)
    df["BH_Cumulative"] = (1 + df["BH_Ret"]).cumprod()

    # Drawdown
    running_max = df["Cumulative"].cummax()
    df["Drawdown"] = df["Cumulative"] / running_max - 1

    df = df.drop(columns=["Month"], errors="ignore")
    df = df.reset_index()

    return df


# ──────────────────────────────────────────────
# STATS COMPUTATION
# ──────────────────────────────────────────────
def compute_stats(df_bt, return_col="Strat_Net", cum_col="Cumulative"):
    """Compute performance statistics from daily backtest results."""
    rets = df_bt[return_col].dropna()
    n_days = len(rets)
    if n_days == 0:
        return {k: 0 for k in ["ann_ret", "ann_vol", "sharpe", "max_dd", "calmar", "hit_rate"]}

    final_cum = df_bt[cum_col].iloc[-1]
    total_ret = final_cum - 1
    ann_ret = final_cum ** (252 / n_days) - 1 if final_cum > 0 else -1
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    cumulative = df_bt[cum_col]
    running_max = cumulative.cummax()
    drawdowns = cumulative / running_max - 1
    max_dd = drawdowns.min()

    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # Hit rate: % of months with positive return
    df_temp = df_bt.copy()
    df_temp["YM"] = pd.to_datetime(df_temp["Date"]).dt.to_period("M")
    monthly_rets = df_temp.groupby("YM")[return_col].sum()
    hit_rate = (monthly_rets > 0).sum() / len(monthly_rets) if len(monthly_rets) > 0 else 0

    return {
        "total_ret": total_ret,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
        "hit_rate": hit_rate,
        "avg_daily": rets.mean(),
        "best_day": rets.max(),
        "worst_day": rets.min(),
        "n_days": n_days,
        "n_months": len(monthly_rets),
    }


# ──────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────
def main():

    # Title
    st.markdown("# Brent Crude Oil CTA Strategy — Backtest Platform")
    st.markdown("**TSMOM + Carry + OVX Macro Filter  |  Scotiabank QIS**")

    # Load data
    df_raw = load_data()
    if df_raw is None:
        st.error("❌ **bloomberg_data.xlsx** (or .csv) not found. Place your Bloomberg data export in the same directory as this app.")
        st.stop()

    if len(df_raw) < 252:
        st.error(f"Only {len(df_raw)} rows in data. Need at least 252 for the 12-month lookback.")
        st.stop()

    min_date = df_raw["Date"].min().date()
    max_date = df_raw["Date"].max().date()

    # ── SIDEBAR ──
    with st.sidebar:
        st.header("📊 Backtest Parameters")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=max(date(2011, 1, 1), min_date), min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

        # Data info
        df_filtered = df_raw[(df_raw["Date"] >= pd.Timestamp(start_date)) & (df_raw["Date"] <= pd.Timestamp(end_date))].copy()
        n_days = len(df_filtered)
        n_months = df_filtered["Date"].dt.to_period("M").nunique()
        st.caption(f"📅 Data: {start_date} → {end_date}  ({n_days:,} trading days, {n_months} months)")

        st.divider()
        st.subheader("Signal Weights")
        tsmom_w = st.slider("TSMOM Weight", 0.0, 1.0, 0.55, 0.05)
        carry_w = st.slider("Carry Weight", 0.0, 1.0, 0.45, 0.05)
        total_w = tsmom_w + carry_w
        if abs(total_w - 1.0) < 0.001:
            st.success(f"Total Signal Weight: {total_w:.2f} ✓")
        else:
            st.warning(f"Total Signal Weight: {total_w:.2f} (ideally = 1.0)")

        st.divider()
        st.subheader("TSMOM Settings")
        lookback_options = {
            "21d (~1M)": 21, "42d (~2M)": 42, "63d (~3M)": 63,
            "126d (~6M)": 126, "189d (~9M)": 189, "252d (~12M)": 252,
        }
        selected_labels = st.multiselect(
            "Lookback Windows",
            options=list(lookback_options.keys()),
            default=["21d (~1M)", "63d (~3M)", "252d (~12M)"],
        )
        lookbacks = [lookback_options[l] for l in selected_labels]
        if len(lookbacks) == 0:
            st.error("Select at least one lookback window.")
            st.stop()

        st.divider()
        st.subheader("OVX Macro Filter")
        ovx_vix_threshold = st.slider(
            "OVX/VIX Ratio Threshold", 1.0, 3.0, 1.8, 0.1,
            help="Above this = oil-specific event, stay full size"
        )
        vix_caution = st.number_input(
            "VIX Caution Level", min_value=15, max_value=50, value=30,
            help="Half size when VIX exceeds this AND OVX/VIX below threshold"
        )
        vix_flatten = st.number_input(
            "VIX Flatten Level", min_value=20, max_value=80, value=40,
            help="Flatten when VIX exceeds this AND OVX/VIX below threshold"
        )

        st.divider()
        st.subheader("Execution")
        tc_bps = st.number_input("Transaction Cost (bps)", min_value=0, max_value=50, value=5)
        tc = tc_bps / 10000

        st.divider()
        st.subheader("Enhancements")
        vol_target_on = st.checkbox("Enable Vol Targeting", value=False)
        vol_target = 0.15
        if vol_target_on:
            vol_target_pct = st.slider("Vol Target (%)", 5, 30, 15, 1)
            vol_target = vol_target_pct / 100

        trailing_stop_on = st.checkbox("Enable Trailing Stop", value=False)
        trailing_stop_thresh = -0.10
        if trailing_stop_on:
            trailing_stop_pct = st.slider("Stop Threshold (%)", -25, -5, -10, 1)
            trailing_stop_thresh = trailing_stop_pct / 100

    # ── COMPUTE ──
    # Need extra lookback data before start_date
    max_lookback = max(lookbacks) if lookbacks else 252
    lookback_start = pd.Timestamp(start_date) - pd.Timedelta(days=int(max_lookback * 1.6))
    df_with_lookback = df_raw[df_raw["Date"] >= lookback_start].copy()

    if len(df_with_lookback) < max_lookback:
        st.error(f"Not enough data for {max_lookback}-day lookback. Extend your start date or add more data.")
        st.stop()

    # Compute signals
    df_signals = compute_signals(
        df_with_lookback, tsmom_w, carry_w, lookbacks,
        ovx_vix_threshold, vix_caution, vix_flatten
    )

    # Filter signals to backtest range
    df_signals_bt = df_signals[df_signals["Date"] <= pd.Timestamp(end_date)].copy()

    if len(df_signals_bt) < 3:
        st.error("Not enough signal months. Widen your date range.")
        st.stop()

    # Run backtest on daily data within range
    df_daily_bt = df_with_lookback[
        (df_with_lookback["Date"] >= pd.Timestamp(start_date)) &
        (df_with_lookback["Date"] <= pd.Timestamp(end_date))
    ].copy()

    df_bt = run_backtest(
        df_daily_bt, df_signals_bt, tc,
        vol_target_on, vol_target,
        trailing_stop_on, trailing_stop_thresh
    )

    # Filter out any NaN rows at start
    df_bt = df_bt.dropna(subset=["Strat_Net", "Cumulative"]).reset_index(drop=True)

    if len(df_bt) == 0:
        st.error("No backtest results. Check your date range and data.")
        st.stop()

    # Compute stats
    strat_stats = compute_stats(df_bt, "Strat_Net", "Cumulative")
    bh_stats = compute_stats(df_bt, "BH_Ret", "BH_Cumulative")

    # ── SECTION 1: STRATEGY OVERVIEW ──
    with st.expander("📖 Strategy Overview", expanded=False):
        st.markdown("""
        **This strategy combines three signals on Brent crude oil futures:**

        - **TSMOM (Time-Series Momentum):** Go long if Brent has been trending up over multiple lookback windows, short if trending down. Blends short, medium, and long-term momentum equally.
        - **Carry (Term Structure):** Go long in backwardation (CO1 > CO12 = positive roll yield), short in contango (CO1 < CO12 = negative roll yield).
        - **OVX Macro Filter:** Uses the OVX/VIX ratio to distinguish oil-specific shocks (stay invested) from systemic financial crises (reduce exposure). Prevents the strategy from being full-size during liquidity crises where momentum signals break down, while staying aggressive during geopolitical supply shocks where crude trends hardest.

        **Composite = (TSMOM_w × TSMOM + Carry_w × Carry) × Macro Scale.** Rebalanced monthly, executed on prior month's signal.
        """)

    # ── SECTION 2: PERFORMANCE SUMMARY ──
    st.markdown("---")
    st.subheader("📈 Performance Summary")

    def sharpe_color(s):
        if s >= 1.0:
            return "normal"
        elif s >= 0.5:
            return "off"
        else:
            return "inverse"

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Ann. Return", f"{strat_stats['ann_ret']:.1%}")
    with col2:
        st.metric("Ann. Vol", f"{strat_stats['ann_vol']:.1%}")
    with col3:
        st.metric("Sharpe Ratio", f"{strat_stats['sharpe']:.2f}",
                   delta=f"{'✓ Above 1.0' if strat_stats['sharpe'] >= 1.0 else ''}")
    with col4:
        st.metric("Max Drawdown", f"{strat_stats['max_dd']:.1%}")
    with col5:
        st.metric("Calmar Ratio", f"{strat_stats['calmar']:.2f}")
    with col6:
        st.metric("Hit Rate", f"{strat_stats['hit_rate']:.1%}")

    # Buy & hold comparison
    st.caption("**Buy & Hold (Brent Long Only)**")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Ann. Return", f"{bh_stats['ann_ret']:.1%}",
                   delta=f"{strat_stats['ann_ret'] - bh_stats['ann_ret']:.1%} vs B&H",
                   delta_color="normal")
    with col2:
        st.metric("Ann. Vol", f"{bh_stats['ann_vol']:.1%}")
    with col3:
        st.metric("Sharpe Ratio", f"{bh_stats['sharpe']:.2f}")
    with col4:
        st.metric("Max Drawdown", f"{bh_stats['max_dd']:.1%}")
    with col5:
        st.metric("Calmar Ratio", f"{bh_stats['calmar']:.2f}")
    with col6:
        st.metric("Hit Rate", f"{bh_stats['hit_rate']:.1%}")

    # ── SECTION 3: CUMULATIVE RETURNS ──
    st.markdown("---")
    st.subheader("📊 Cumulative Returns")

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=df_bt["Date"], y=df_bt["Cumulative"],
        name="Strategy", mode="lines",
        line=dict(color="#00D4AA", width=2.5),
    ))
    fig_cum.add_trace(go.Scatter(
        x=df_bt["Date"], y=df_bt["BH_Cumulative"],
        name="Buy & Hold", mode="lines",
        line=dict(color="#888888", width=1.5, dash="dash"),
    ))
    fig_cum.add_hline(y=1.0, line_dash="dot", line_color="rgba(255,255,255,0.3)", line_width=1)
    fig_cum.update_layout(
        template="plotly_dark",
        yaxis_title="Growth of $1",
        xaxis_title="Date",
        height=500,
        margin=dict(l=60, r=30, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # ── SECTION 4: DRAWDOWN ──
    st.subheader("🔻 Drawdown")

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=df_bt["Date"], y=df_bt["Drawdown"],
        fill="tozeroy", mode="lines",
        line=dict(color="#FF4B4B", width=1),
        fillcolor="rgba(255,75,75,0.3)",
        name="Drawdown",
    ))
    fig_dd.update_layout(
        template="plotly_dark",
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        xaxis_title="Date",
        height=350,
        margin=dict(l=60, r=30, t=30, b=40),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # ── SECTION 5: SIGNAL ANALYSIS ──
    st.markdown("---")
    st.subheader("🎯 Signal Analysis")

    # Filter signals to backtest date range for display
    df_sig_display = df_signals_bt[
        (df_signals_bt["Date"] >= pd.Timestamp(start_date)) &
        (df_signals_bt["Date"] <= pd.Timestamp(end_date))
    ].copy()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Composite Signal Over Time**")
        colors = ["#00D4AA" if x > 0 else "#FF4B4B" if x < 0 else "#888888"
                  for x in df_sig_display["Composite"]]
        fig_sig = go.Figure()
        fig_sig.add_trace(go.Bar(
            x=df_sig_display["Date"], y=df_sig_display["Composite"],
            marker_color=colors, name="Signal",
        ))
        fig_sig.update_layout(
            template="plotly_dark",
            yaxis_title="Signal [-1, +1]",
            height=400,
            margin=dict(l=60, r=30, t=30, b=40),
        )
        st.plotly_chart(fig_sig, use_container_width=True)

    with col_right:
        st.markdown("**Signal Contribution Breakdown**")
        fig_contrib = make_subplots(specs=[[{"secondary_y": True}]])
        fig_contrib.add_trace(go.Bar(
            x=df_sig_display["Date"],
            y=df_sig_display["TSMOM_Contribution"],
            name="TSMOM",
            marker_color="#4A90D9",
        ), secondary_y=False)
        fig_contrib.add_trace(go.Bar(
            x=df_sig_display["Date"],
            y=df_sig_display["Carry_Contribution"],
            name="Carry",
            marker_color="#F5A623",
        ), secondary_y=False)
        fig_contrib.add_trace(go.Scatter(
            x=df_sig_display["Date"],
            y=df_sig_display["Macro_Scale"],
            name="Macro Scale",
            mode="lines",
            line=dict(color="#FF4B4B", width=2, dash="dot"),
        ), secondary_y=True)
        fig_contrib.update_layout(
            template="plotly_dark",
            barmode="relative",
            height=400,
            margin=dict(l=60, r=60, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_contrib.update_yaxes(title_text="Signal Contribution", secondary_y=False)
        fig_contrib.update_yaxes(title_text="Macro Scale", range=[0, 1.1], secondary_y=True)
        st.plotly_chart(fig_contrib, use_container_width=True)

    # ── SECTION 6: DETAIL TABLE ──
    st.markdown("---")
    st.subheader("📋 Position & Macro Filter Detail")

    # Build monthly summary table
    df_monthly_detail = df_sig_display[[
        "Date", "CO1", "CO12", "Carry_Spread", "TSMOM", "Carry",
        "OVX", "VIX", "OVX_VIX_Ratio", "Macro_Scale", "Composite"
    ]].copy()

    # Add monthly strategy and market returns
    df_bt_temp = df_bt.copy()
    df_bt_temp["YM"] = pd.to_datetime(df_bt_temp["Date"]).dt.to_period("M")
    monthly_strat_ret = df_bt_temp.groupby("YM")["Strat_Net"].apply(lambda x: (1 + x).prod() - 1)
    monthly_mkt_ret = df_bt_temp.groupby("YM")["Market_Ret"].apply(lambda x: (1 + x).prod() - 1)
    monthly_cum = (1 + monthly_strat_ret).cumprod()
    monthly_bh_cum = (1 + monthly_mkt_ret).cumprod()

    df_monthly_detail["YM"] = pd.to_datetime(df_monthly_detail["Date"]).dt.to_period("M")
    df_monthly_detail = df_monthly_detail.merge(
        monthly_strat_ret.rename("Strat_Ret_Monthly"),
        left_on="YM", right_index=True, how="left"
    )
    df_monthly_detail = df_monthly_detail.merge(
        monthly_mkt_ret.rename("Market_Ret_Monthly"),
        left_on="YM", right_index=True, how="left"
    )
    df_monthly_detail = df_monthly_detail.merge(
        monthly_cum.rename("Cum_Strategy"),
        left_on="YM", right_index=True, how="left"
    )
    df_monthly_detail = df_monthly_detail.merge(
        monthly_bh_cum.rename("Cum_BuyHold"),
        left_on="YM", right_index=True, how="left"
    )

    # Format for display
    display_df = df_monthly_detail.drop(columns=["YM"]).copy()
    display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")

    # Rename for readability
    display_df = display_df.rename(columns={
        "Carry_Spread": "Carry Spread",
        "OVX_VIX_Ratio": "OVX/VIX",
        "Macro_Scale": "Macro Scale",
        "Strat_Ret_Monthly": "Strat Ret",
        "Market_Ret_Monthly": "Mkt Ret",
        "Cum_Strategy": "Cum Strat",
        "Cum_BuyHold": "Cum B&H",
    })

    styled_df = display_df.style.format({
        "CO1": "${:.2f}", "CO12": "${:.2f}",
        "Carry Spread": "{:.2%}", "TSMOM": "{:+.3f}", "Carry": "{:+.3f}",
        "OVX": "{:.1f}", "VIX": "{:.1f}", "OVX/VIX": "{:.2f}",
        "Macro Scale": "{:.2f}", "Composite": "{:+.3f}",
        "Strat Ret": "{:.2%}", "Mkt Ret": "{:.2%}",
        "Cum Strat": "{:.4f}", "Cum B&H": "{:.4f}",
    })
    try:
        # pandas >= 2.1
        styled_df = styled_df.map(
            lambda x: "background-color: rgba(255,75,75,0.15)" if isinstance(x, (int, float)) and x < 1.0 else "",
            subset=["Macro Scale"]
        )
    except AttributeError:
        # pandas < 2.1
        styled_df = styled_df.applymap(
            lambda x: "background-color: rgba(255,75,75,0.15)" if isinstance(x, (int, float)) and x < 1.0 else "",
            subset=["Macro Scale"]
        )
    st.dataframe(styled_df, use_container_width=True, height=400)

    st.download_button(
        "📥 Download Monthly Detail as CSV",
        display_df.to_csv(index=False),
        "brent_cta_monthly_detail.csv",
        "text/csv",
    )

    # ── SECTION 7: MONTHLY RETURNS HEATMAP ──
    st.markdown("---")
    st.subheader("🗓️ Monthly Returns Heatmap")

    df_bt_hm = df_bt.copy()
    df_bt_hm["Year"] = pd.to_datetime(df_bt_hm["Date"]).dt.year
    df_bt_hm["Month"] = pd.to_datetime(df_bt_hm["Date"]).dt.month

    # Aggregate daily returns to monthly
    monthly_returns = df_bt_hm.groupby(["Year", "Month"])["Strat_Net"].apply(
        lambda x: (1 + x).prod() - 1
    ).reset_index()
    monthly_returns.columns = ["Year", "Month", "Return"]

    # Pivot for heatmap
    heatmap_data = monthly_returns.pivot(index="Year", columns="Month", values="Return")
    heatmap_data = heatmap_data.reindex(columns=range(1, 13))

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Text values for display
    try:
        text_vals = heatmap_data.map(lambda x: f"{x:.1%}" if pd.notna(x) else "").values
    except AttributeError:
        text_vals = heatmap_data.applymap(lambda x: f"{x:.1%}" if pd.notna(x) else "").values

    fig_hm = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=month_labels,
        y=heatmap_data.index.astype(str),
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="Return", tickformat=".0%"),
    ))
    fig_hm.update_layout(
        template="plotly_dark",
        height=max(300, len(heatmap_data) * 35 + 100),
        margin=dict(l=60, r=30, t=30, b=40),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── SECTION 8: ROLLING STATISTICS ──
    st.markdown("---")
    st.subheader("📉 Rolling Statistics (252-Day Window)")

    rolling_ret = df_bt["Strat_Net"].rolling(252, min_periods=126)
    rolling_ann_ret = rolling_ret.mean() * 252
    rolling_ann_vol = rolling_ret.std() * np.sqrt(252)
    rolling_sharpe = rolling_ann_ret / rolling_ann_vol

    fig_rolling = make_subplots(specs=[[{"secondary_y": True}]])
    fig_rolling.add_trace(go.Scatter(
        x=df_bt["Date"], y=rolling_sharpe,
        name="Rolling Sharpe (252d)",
        mode="lines",
        line=dict(color="#00D4AA", width=2),
    ), secondary_y=False)
    fig_rolling.add_trace(go.Scatter(
        x=df_bt["Date"], y=rolling_ann_ret,
        name="Rolling Ann. Return (252d)",
        mode="lines",
        line=dict(color="#F5A623", width=1.5, dash="dash"),
    ), secondary_y=True)
    fig_rolling.add_hline(y=1.0, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                          line_width=1, secondary_y=False)
    fig_rolling.add_hline(y=0.0, line_dash="dot", line_color="rgba(255,255,255,0.2)",
                          line_width=1, secondary_y=False)
    fig_rolling.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=60, r=60, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_rolling.update_yaxes(title_text="Sharpe Ratio", secondary_y=False)
    fig_rolling.update_yaxes(title_text="Ann. Return", tickformat=".0%", secondary_y=True)
    st.plotly_chart(fig_rolling, use_container_width=True)

    # ── SECTION 9: DAILY P&L DISTRIBUTION ──
    st.markdown("---")
    st.subheader("📊 Daily P&L Distribution")

    daily_rets = df_bt["Strat_Net"].dropna()
    ret_mean = daily_rets.mean()
    ret_median = daily_rets.median()
    ret_skew = stats.skew(daily_rets)
    ret_kurt = stats.kurtosis(daily_rets)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=daily_rets,
        nbinsx=100,
        marker_color="rgba(0,212,170,0.6)",
        marker_line=dict(color="rgba(0,212,170,1)", width=0.5),
        name="Daily Returns",
    ))
    fig_dist.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.5)", line_width=1)
    fig_dist.add_vline(x=ret_mean, line_color="#F5A623", line_width=2,
                       annotation_text=f"Mean: {ret_mean:.4%}",
                       annotation_position="top right")
    fig_dist.update_layout(
        template="plotly_dark",
        xaxis_title="Daily Return",
        yaxis_title="Frequency",
        height=400,
        margin=dict(l=60, r=30, t=30, b=40),
        annotations=[
            dict(x=0.98, y=0.95, xref="paper", yref="paper",
                 text=f"Mean: {ret_mean:.4%}<br>Median: {ret_median:.4%}<br>Skew: {ret_skew:.2f}<br>Kurtosis: {ret_kurt:.2f}",
                 showarrow=False, font=dict(size=12, color="white"),
                 bgcolor="rgba(0,0,0,0.6)", bordercolor="rgba(255,255,255,0.3)",
                 borderwidth=1, borderpad=8,
                 align="left"),
        ],
    )
    fig_dist.update_xaxes(tickformat=".1%")
    st.plotly_chart(fig_dist, use_container_width=True)

    # ── FOOTER ──
    st.markdown("---")
    st.caption("Scotiabank QIS Desk — Brent CTA Strategy Backtest Platform. "
               "Built with Streamlit. Data source: Bloomberg Terminal.")


if __name__ == "__main__":
    main()
