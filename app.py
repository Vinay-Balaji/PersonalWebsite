"""
BRENT CRUDE OIL CTA STRATEGY — Backtest Platform
TSMOM + Carry + Vol Targeting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
import os

st.set_page_config(page_title="Brent CTA Backtest", layout="wide", initial_sidebar_state="expanded")


@st.cache_data
def load_data():
    if os.path.exists("bloomberg_data.xlsx"):
        df = pd.read_excel("bloomberg_data.xlsx", header=None)
    elif os.path.exists("bloomberg_data.csv"):
        df = pd.read_csv("bloomberg_data.csv")
    else:
        return None

    n_cols = len(df.columns)

    # Detect raw BDH: 4+ columns, cols 0/2 are dates, cols 1/3 are prices
    if n_cols >= 4:
        test_dates = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        if test_dates.notna().sum() > len(df) * 0.5:
            result = pd.DataFrame({
                "Date": pd.to_datetime(df.iloc[:, 0], errors="coerce"),
                "CO1":  pd.to_numeric(df.iloc[:, 1], errors="coerce"),
            })
            # CO12 in column D (index 3) if available
            if n_cols >= 4:
                result["CO12"] = pd.to_numeric(df.iloc[:, 3], errors="coerce")
            else:
                result["CO12"] = result["CO1"]  # fallback
            result = result.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
            result[["CO1","CO12"]] = result[["CO1","CO12"]].ffill().bfill()
            result = result.dropna(subset=["CO1","CO12"]).reset_index(drop=True)
            return result

    # Fallback: named columns
    df.columns = [str(c).strip() if not isinstance(c, datetime) else "Date" for c in df.columns]
    date_col = None
    for col in df.columns:
        if str(col).strip().lower() == "date":
            date_col = col
            break
    if date_col is None:
        date_col = df.columns[0]
    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    col_map = {}
    for col in df.columns:
        cl = str(col).strip().upper()
        if cl in ["CO1","CO1 COMDTY","BRENT"]: col_map[col] = "CO1"
        elif cl in ["CO12","CO12 COMDTY"]: col_map[col] = "CO12"
    df = df.rename(columns=col_map)
    for col in ["CO1","CO12"]:
        if col not in df.columns:
            st.error(f"Missing column: {col}. Need CO1 and CO12.")
            st.stop()
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[["CO1","CO12"]] = df[["CO1","CO12"]].ffill().bfill()
    df = df.dropna(subset=["CO1","CO12"]).reset_index(drop=True)
    return df


def compute_signals(df_daily, tsmom_weight, carry_weight, lookbacks):
    """Compute TSMOM and Carry signals at month-end."""
    df = df_daily.copy().set_index("Date")
    df["YM"] = df.index.to_period("M")
    month_ends = df.groupby("YM").tail(1).copy()

    # TSMOM
    tsmom_components = []
    for lb in lookbacks:
        ret = df["CO1"].pct_change(lb)
        tsmom_components.append(np.sign(ret.loc[month_ends.index]))
    if len(tsmom_components) > 0:
        month_ends["TSMOM"] = pd.concat(tsmom_components, axis=1).mean(axis=1).values
    else:
        month_ends["TSMOM"] = 0.0

    # Carry
    month_ends["Carry_Spread"] = (month_ends["CO1"] - month_ends["CO12"]) / month_ends["CO1"]
    month_ends["Carry"] = np.sign(month_ends["Carry_Spread"])

    # Composite (before vol targeting — vol targeting applied in backtest)
    raw = tsmom_weight * month_ends["TSMOM"] + carry_weight * month_ends["Carry"]
    month_ends["Composite"] = raw.clip(-1, 1)
    month_ends["TSMOM_Contribution"] = tsmom_weight * month_ends["TSMOM"]
    month_ends["Carry_Contribution"] = carry_weight * month_ends["Carry"]

    month_ends = month_ends.drop(columns=["YM"], errors="ignore").reset_index()
    month_ends = month_ends.rename(columns={"index": "Date"} if "index" in month_ends.columns else {})
    if "Date" not in month_ends.columns and month_ends.index.name == "Date":
        month_ends = month_ends.reset_index()
    return month_ends


def run_backtest(df_daily, df_signals, tc, vol_target, vol_lookback):
    """Run daily backtest with vol targeting."""
    df = df_daily.copy()
    df = df.set_index("Date") if "Date" in df.columns else df.copy()
    df["Market_Ret"] = df["CO1"].pct_change()

    # Map monthly signals to daily
    df_signals_indexed = df_signals.set_index("Date")
    signal_series = df_signals_indexed["Composite"]
    df["Signal"] = np.nan
    for i, sig_date in enumerate(signal_series.index):
        sig_val = signal_series.iloc[i]
        if i + 1 < len(signal_series.index):
            mask = (df.index > sig_date) & (df.index <= signal_series.index[i + 1])
        else:
            mask = df.index > sig_date
        df.loc[mask, "Signal"] = sig_val
    df["Raw_Position"] = df["Signal"].ffill().fillna(0)

    # Vol targeting: scale position so strategy vol stays near target
    realized_vol = df["Market_Ret"].rolling(vol_lookback, min_periods=max(5, vol_lookback // 4)).std() * np.sqrt(252)
    df["Realized_Vol"] = realized_vol
    df["Vol_Scalar"] = (vol_target / realized_vol).clip(0.25, 2.0).fillna(1.0)
    df["Position"] = df["Raw_Position"] * df["Vol_Scalar"]

    # Strategy returns
    df["Strat_Gross"] = df["Position"] * df["Market_Ret"]

    # Transaction costs
    df["Month"] = df.index.to_period("M")
    monthly_pos = df.groupby("Month")["Raw_Position"].first()
    pos_changed = monthly_pos.diff().abs() > 0.01
    tc_months = pos_changed[pos_changed].index
    df["TC"] = 0.0
    for m in tc_months:
        md = df[df["Month"] == m].index
        if len(md) > 0:
            df.loc[md[0], "TC"] = tc

    df["Strat_Net"] = df["Strat_Gross"] - df["TC"]
    df["Cumulative"] = (1 + df["Strat_Net"].fillna(0)).cumprod()

    # Buy & hold
    df["BH_Ret"] = df["Market_Ret"].fillna(0)
    df["BH_Cumulative"] = (1 + df["BH_Ret"]).cumprod()

    # Drawdown
    df["Drawdown"] = df["Cumulative"] / df["Cumulative"].cummax() - 1

    df = df.drop(columns=["Month"], errors="ignore").reset_index()
    return df


def compute_stats(df_bt, return_col="Strat_Net", cum_col="Cumulative", rf_annual=0.0):
    """
    Compute stats using Bloomberg methodology:
    Ann Return = mean(daily returns) * 252
    Ann Vol = std(daily returns) * sqrt(252)
    Sharpe = (Ann Return - Rf) / Ann Vol
    """
    rets = df_bt[return_col].dropna()
    n_days = len(rets)
    if n_days == 0:
        return {k: 0 for k in ["ann_ret","ann_vol","sharpe","max_dd","hit_rate","total_ret"]}

    # Bloomberg-style: arithmetic annualization
    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = (ann_ret - rf_annual) / ann_vol if ann_vol > 0 else 0

    total_ret = df_bt[cum_col].iloc[-1] - 1
    max_dd = (df_bt[cum_col] / df_bt[cum_col].cummax() - 1).min()

    df_temp = df_bt.copy()
    df_temp["YM"] = pd.to_datetime(df_temp["Date"]).dt.to_period("M")
    monthly_rets = df_temp.groupby("YM")[return_col].apply(lambda x: (1+x).prod()-1)
    hit_rate = (monthly_rets > 0).sum() / len(monthly_rets) if len(monthly_rets) > 0 else 0

    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
            "max_dd": max_dd, "hit_rate": hit_rate, "total_ret": total_ret, "n_days": n_days}


def main():
    st.markdown("# Brent Crude Oil CTA — Backtest Platform")
    st.markdown("**TSMOM + Carry + Vol Targeting**")

    df_raw = load_data()
    if df_raw is None:
        st.error("**bloomberg_data.xlsx** (or .csv) not found. Place your Bloomberg data export in the same directory as this app.")
        st.stop()
    if len(df_raw) < 252:
        st.error(f"Only {len(df_raw)} rows. Need at least 252 for 12-month lookback.")
        st.stop()

    min_date = df_raw["Date"].min().date()
    max_date = df_raw["Date"].max().date()

    # ── SIDEBAR ──
    with st.sidebar:
        st.header("Backtest Settings")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start", value=max(date(2011, 1, 1), min_date), min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End", value=max_date, min_value=min_date, max_value=max_date)

        df_filt = df_raw[(df_raw["Date"] >= pd.Timestamp(start_date)) & (df_raw["Date"] <= pd.Timestamp(end_date))]
        st.caption(f"{start_date} to {end_date} — {len(df_filt):,} days, {df_filt['Date'].dt.to_period('M').nunique()} months")

        # Signal 1
        st.divider()
        st.subheader("Signal 1 — TSMOM")
        st.caption("Long if trending up, short if down.")
        tsmom_w = st.slider("Weight", 0.0, 1.0, 0.55, 0.05, key="tw")
        lb_opts = {"21d (~1M)": 21, "42d (~2M)": 42, "63d (~3M)": 63,
                   "126d (~6M)": 126, "189d (~9M)": 189, "252d (~12M)": 252}
        sel = st.multiselect("Lookback Windows", list(lb_opts.keys()), default=["21d (~1M)", "63d (~3M)", "252d (~12M)"])
        lookbacks = [lb_opts[l] for l in sel]
        if not lookbacks:
            st.error("Select at least one lookback.")
            st.stop()

        # Signal 2
        st.divider()
        st.subheader("Signal 2 — Carry")
        st.caption("Long in backwardation, short in contango.")
        carry_w = st.slider("Weight", 0.0, 1.0, 0.45, 0.05, key="cw")

        total_w = tsmom_w + carry_w
        if abs(total_w - 1.0) < 0.001:
            st.success(f"Total weight: {total_w:.2f}")
        else:
            st.warning(f"Total weight: {total_w:.2f} (should be 1.0)")

        # Signal 3
        st.divider()
        st.subheader("Signal 3 — Vol Targeting")
        st.caption("Scales position size to maintain constant risk.")
        vol_target_pct = st.slider("Target Volatility (%)", 5, 30, 15, 1)
        vol_target = vol_target_pct / 100
        vol_lookback = st.select_slider("Vol Lookback (days)", options=[10, 15, 21, 42, 63], value=21)

        # Execution
        st.divider()
        st.subheader("Execution")
        tc_bps = st.number_input("Transaction Cost (bps)", 0, 50, 5)
        tc = tc_bps / 10000
        rf_pct = st.number_input("Risk-Free Rate (% annual)", 0.0, 10.0, 4.5, 0.1,
                                 help="Used for Sharpe calculation. Match Bloomberg's assumption.")
        rf = rf_pct / 100

    # ── COMPUTE ──
    max_lb = max(lookbacks)
    lb_start = pd.Timestamp(start_date) - pd.Timedelta(days=int(max_lb * 1.6))
    df_lb = df_raw[df_raw["Date"] >= lb_start].copy()
    if len(df_lb) < max_lb:
        st.error(f"Not enough data for {max_lb}-day lookback.")
        st.stop()

    df_signals = compute_signals(df_lb, tsmom_w, carry_w, lookbacks)
    df_signals_bt = df_signals[df_signals["Date"] <= pd.Timestamp(end_date)].copy()
    if len(df_signals_bt) < 3:
        st.error("Not enough signal months. Widen your date range.")
        st.stop()

    df_daily_bt = df_lb[(df_lb["Date"] >= pd.Timestamp(start_date)) & (df_lb["Date"] <= pd.Timestamp(end_date))].copy()
    df_bt = run_backtest(df_daily_bt, df_signals_bt, tc, vol_target, vol_lookback)
    df_bt = df_bt.dropna(subset=["Strat_Net","Cumulative"]).reset_index(drop=True)
    if len(df_bt) == 0:
        st.error("No backtest results.")
        st.stop()

    ss = compute_stats(df_bt, "Strat_Net", "Cumulative", rf)
    bh = compute_stats(df_bt, "BH_Ret", "BH_Cumulative", rf)

    # ── OVERVIEW ──
    with st.expander("Strategy Overview", expanded=False):
        st.markdown(f"""
        **Two signals + vol targeting on Brent crude oil futures (CO1 Comdty):**
        - **TSMOM:** Long if Brent trending up across multiple lookback windows, short if down.
        - **Carry:** Long in backwardation (CO1 > CO12), short in contango (CO1 < CO12).
        - **Vol Targeting:** Scales position daily so the strategy runs at ~{vol_target_pct}% annualized volatility. When Brent vol is low, lever up. When vol spikes, scale down.

        **Composite = (TSMOM_w × TSMOM + Carry_w × Carry) × Vol Scalar.** Signals rebalanced monthly. Vol scalar updated daily.

        **Sharpe Methodology:** Matches Bloomberg — arithmetic annualization (mean × 252 / std × √252), excess return over {rf_pct:.1f}% risk-free rate.
        Buy & Hold uses raw CO1 front-month returns which include roll gaps; Bloomberg's reported B&H Sharpe uses a roll-adjusted index.
        """)

    # ── PERFORMANCE ──
    st.markdown("---")
    st.subheader("Performance Summary")
    st.caption("**Strategy**")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ann. Return", f"{ss['ann_ret']:.1%}")
    c2.metric("Ann. Vol", f"{ss['ann_vol']:.1%}")
    badge = " 🟢" if ss['sharpe'] >= 1.0 else (" 🟡" if ss['sharpe'] >= 0.5 else " 🔴")
    c3.metric("Sharpe" + badge, f"{ss['sharpe']:.2f}")
    c4.metric("Max Drawdown", f"{ss['max_dd']:.1%}")
    c5.metric("Hit Rate", f"{ss['hit_rate']:.1%}")

    st.caption("**Buy & Hold**")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ann. Return", f"{bh['ann_ret']:.1%}", delta=f"{ss['ann_ret']-bh['ann_ret']:+.1%} vs B&H")
    c2.metric("Ann. Vol", f"{bh['ann_vol']:.1%}")
    c3.metric("Sharpe", f"{bh['sharpe']:.2f}")
    c4.metric("Max Drawdown", f"{bh['max_dd']:.1%}")
    c5.metric("Hit Rate", f"{bh['hit_rate']:.1%}")

    # ── CUMULATIVE ──
    st.markdown("---")
    st.subheader("Cumulative Returns")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_bt["Date"], y=df_bt["Cumulative"], name="Strategy",
                             line=dict(color="#00D4AA", width=2.5)))
    fig.add_trace(go.Scatter(x=df_bt["Date"], y=df_bt["BH_Cumulative"], name="Buy & Hold",
                             line=dict(color="#888", width=1.5, dash="dash")))
    fig.add_hline(y=1.0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    fig.update_layout(template="plotly_dark", yaxis_title="Growth of $1", height=500,
                      margin=dict(l=60,r=30,t=30,b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig, use_container_width=True)

    # ── DRAWDOWN ──
    st.subheader("Drawdown")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=df_bt["Date"], y=df_bt["Drawdown"], fill="tozeroy",
                                line=dict(color="#FF4B4B", width=1), fillcolor="rgba(255,75,75,0.3)"))
    fig_dd.update_layout(template="plotly_dark", yaxis_title="Drawdown", yaxis_tickformat=".0%",
                         height=350, margin=dict(l=60,r=30,t=30,b=40))
    st.plotly_chart(fig_dd, use_container_width=True)

    # ── VOL TARGETING IN ACTION ──
    st.markdown("---")
    st.subheader("Vol Targeting")
    fig_vol = make_subplots(specs=[[{"secondary_y": True}]])
    fig_vol.add_trace(go.Scatter(x=df_bt["Date"], y=df_bt["Realized_Vol"],
                                 name="Realized Vol (ann.)", line=dict(color="#F5A623", width=1.5)),
                      secondary_y=False)
    fig_vol.add_hline(y=vol_target, line_dash="dash", line_color="#00D4AA", line_width=2,
                      annotation_text=f"Target: {vol_target_pct}%", secondary_y=False)
    fig_vol.add_trace(go.Scatter(x=df_bt["Date"], y=df_bt["Vol_Scalar"],
                                 name="Vol Scalar", line=dict(color="#4A90D9", width=1, dash="dot")),
                      secondary_y=True)
    fig_vol.update_layout(template="plotly_dark", height=400,
                          margin=dict(l=60,r=60,t=30,b=40),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig_vol.update_yaxes(title_text="Annualized Vol", tickformat=".0%", secondary_y=False)
    fig_vol.update_yaxes(title_text="Position Scalar", secondary_y=True)
    st.plotly_chart(fig_vol, use_container_width=True)

    # ── SIGNALS ──
    st.markdown("---")
    st.subheader("Signal Analysis")
    df_sig_disp = df_signals_bt[(df_signals_bt["Date"] >= pd.Timestamp(start_date)) &
                                (df_signals_bt["Date"] <= pd.Timestamp(end_date))].copy()

    cl, cr = st.columns(2)
    with cl:
        st.markdown("**Composite Signal**")
        colors = ["#00D4AA" if x > 0 else "#FF4B4B" if x < 0 else "#888" for x in df_sig_disp["Composite"]]
        fig_s = go.Figure(go.Bar(x=df_sig_disp["Date"], y=df_sig_disp["Composite"], marker_color=colors))
        fig_s.update_layout(template="plotly_dark", yaxis_title="Signal [-1,+1]", height=400,
                            margin=dict(l=60,r=30,t=30,b=40))
        st.plotly_chart(fig_s, use_container_width=True)

    with cr:
        st.markdown("**Signal Contribution Breakdown**")
        fig_c = go.Figure()
        fig_c.add_trace(go.Bar(x=df_sig_disp["Date"], y=df_sig_disp["TSMOM_Contribution"],
                               name="TSMOM", marker_color="#4A90D9"))
        fig_c.add_trace(go.Bar(x=df_sig_disp["Date"], y=df_sig_disp["Carry_Contribution"],
                               name="Carry", marker_color="#F5A623"))
        fig_c.update_layout(template="plotly_dark", barmode="relative", height=400,
                            margin=dict(l=60,r=30,t=30,b=40),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            yaxis_title="Contribution")
        st.plotly_chart(fig_c, use_container_width=True)

    # ── DETAIL TABLE ──
    st.markdown("---")
    st.subheader("Monthly Detail")
    dmd = df_sig_disp[["Date","CO1","CO12","Carry_Spread","TSMOM","Carry","Composite"]].copy()

    bt_t = df_bt.copy()
    bt_t["YM"] = pd.to_datetime(bt_t["Date"]).dt.to_period("M")
    m_sr = bt_t.groupby("YM")["Strat_Net"].apply(lambda x: (1+x).prod()-1)
    m_mr = bt_t.groupby("YM")["Market_Ret"].apply(lambda x: (1+x).prod()-1)
    m_sc = (1+m_sr).cumprod()
    m_bc = (1+m_mr).cumprod()
    # Average vol scalar per month
    m_vs = bt_t.groupby("YM")["Vol_Scalar"].mean()

    dmd["YM"] = pd.to_datetime(dmd["Date"]).dt.to_period("M")
    dmd = dmd.merge(m_vs.rename("Avg Vol Scalar"), left_on="YM", right_index=True, how="left")
    dmd = dmd.merge(m_sr.rename("Strat Ret"), left_on="YM", right_index=True, how="left")
    dmd = dmd.merge(m_mr.rename("Mkt Ret"), left_on="YM", right_index=True, how="left")
    dmd = dmd.merge(m_sc.rename("Cum Strat"), left_on="YM", right_index=True, how="left")
    dmd = dmd.merge(m_bc.rename("Cum B&H"), left_on="YM", right_index=True, how="left")
    dmd = dmd.drop(columns=["YM"])
    dmd["Date"] = dmd["Date"].dt.strftime("%Y-%m-%d")
    dmd = dmd.rename(columns={"Carry_Spread": "Carry Spread"})

    fmt = {"CO1":"${:.2f}", "CO12":"${:.2f}", "Carry Spread":"{:.2%}",
           "TSMOM":"{:+.3f}", "Carry":"{:+.3f}", "Composite":"{:+.3f}",
           "Avg Vol Scalar":"{:.2f}",
           "Strat Ret":"{:.2%}", "Mkt Ret":"{:.2%}",
           "Cum Strat":"{:.4f}", "Cum B&H":"{:.4f}"}
    st.dataframe(dmd.style.format(fmt), use_container_width=True, height=400)
    st.download_button("Download Monthly Detail", dmd.to_csv(index=False), "monthly_detail.csv", "text/csv")

    # ── FOOTER ──
    st.markdown("---")
    st.caption("Brent CTA Backtest Platform. Built with Streamlit.")


if __name__ == "__main__":
    main()
