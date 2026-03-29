"""
BRENT CRUDE OIL CTA STRATEGY — Backtest Platform
TSMOM + Carry + OVX Macro Filter
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

st.set_page_config(page_title="Brent CTA Backtest", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_data():
    if os.path.exists("bloomberg_data.xlsx"):
        df = pd.read_excel("bloomberg_data.xlsx")
    elif os.path.exists("bloomberg_data.csv"):
        df = pd.read_csv("bloomberg_data.csv")
    else:
        return None

    # Fix column names — convert datetime objects to strings (Bloomberg Excel export bug)
    df.columns = [str(c).strip() if not isinstance(c, datetime) else "Date" for c in df.columns]

    # Find the date column
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

    # Standardize column names
    col_map = {}
    for col in df.columns:
        cl = str(col).strip().upper()
        if cl in ["CO1", "CO1 COMDTY", "BRENT", "CO1_CLOSE"]: col_map[col] = "CO1"
        elif cl in ["CO12", "CO12 COMDTY", "CO12_CLOSE"]: col_map[col] = "CO12"
        elif cl in ["OVX", "OVX INDEX", "OVX_CLOSE"]: col_map[col] = "OVX"
        elif cl in ["VIX", "VIX INDEX", "VIX_CLOSE"]: col_map[col] = "VIX"
    df = df.rename(columns=col_map)

    required = ["Date", "CO1", "CO12", "OVX", "VIX"]
    for col in required:
        if col not in df.columns:
            st.error(f"Missing column: {col}. Expected: {required}")
            st.stop()

    for col in ["CO1", "CO12", "OVX", "VIX"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[["CO1", "CO12", "OVX", "VIX"]] = df[["CO1", "CO12", "OVX", "VIX"]].ffill().bfill()
    df = df.dropna(subset=["CO1", "CO12", "OVX", "VIX"]).reset_index(drop=True)
    return df


def compute_signals(df_daily, tsmom_weight, carry_weight, lookbacks,
                    ovx_vix_threshold, vix_caution, vix_flatten):
    df = df_daily.copy().set_index("Date")
    df["YM"] = df.index.to_period("M")
    month_ends = df.groupby("YM").tail(1).copy()

    tsmom_components = []
    for lb in lookbacks:
        ret = df["CO1"].pct_change(lb)
        tsmom_components.append(np.sign(ret.loc[month_ends.index]))

    if len(tsmom_components) > 0:
        month_ends["TSMOM"] = pd.concat(tsmom_components, axis=1).mean(axis=1).values
    else:
        month_ends["TSMOM"] = 0.0

    month_ends["Carry_Spread"] = (month_ends["CO1"] - month_ends["CO12"]) / month_ends["CO1"]
    month_ends["Carry"] = np.sign(month_ends["Carry_Spread"])

    month_ends["OVX_VIX_Ratio"] = month_ends["OVX"] / month_ends["VIX"].replace(0, np.nan)
    month_ends["OVX_VIX_Ratio"] = month_ends["OVX_VIX_Ratio"].fillna(1.0)

    def macro_scale(row):
        if row["OVX_VIX_Ratio"] > ovx_vix_threshold: return 1.0
        if row["VIX"] > vix_flatten: return 0.0
        if row["VIX"] > vix_caution: return 0.5
        return 1.0

    month_ends["Macro_Scale"] = month_ends.apply(macro_scale, axis=1)
    raw = tsmom_weight * month_ends["TSMOM"] + carry_weight * month_ends["Carry"]
    month_ends["Composite"] = (raw * month_ends["Macro_Scale"]).clip(-1, 1)
    month_ends["TSMOM_Contribution"] = tsmom_weight * month_ends["TSMOM"]
    month_ends["Carry_Contribution"] = carry_weight * month_ends["Carry"]

    month_ends = month_ends.drop(columns=["YM"], errors="ignore").reset_index()
    month_ends = month_ends.rename(columns={"index": "Date"} if "index" in month_ends.columns else {})
    if "Date" not in month_ends.columns and month_ends.index.name == "Date":
        month_ends = month_ends.reset_index()
    return month_ends


def run_backtest(df_daily, df_signals, tc):
    df = df_daily.copy()
    df = df.set_index("Date") if "Date" in df.columns else df.copy()
    df["Market_Ret"] = df["CO1"].pct_change()

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

    df["Position"] = df["Signal"].ffill().fillna(0)
    df["Strat_Gross"] = df["Position"] * df["Market_Ret"]

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
    df["Cumulative"] = (1 + df["Strat_Net"].fillna(0)).cumprod()
    df["BH_Ret"] = df["Market_Ret"].fillna(0)
    df["BH_Cumulative"] = (1 + df["BH_Ret"]).cumprod()
    df["Drawdown"] = df["Cumulative"] / df["Cumulative"].cummax() - 1
    df = df.drop(columns=["Month"], errors="ignore").reset_index()
    return df


def compute_stats(df_bt, return_col="Strat_Net", cum_col="Cumulative"):
    rets = df_bt[return_col].dropna()
    n_days = len(rets)
    if n_days == 0:
        return {k: 0 for k in ["ann_ret", "ann_vol", "sharpe", "max_dd", "hit_rate"]}

    final_cum = df_bt[cum_col].iloc[-1]
    ann_ret = final_cum ** (252 / n_days) - 1 if final_cum > 0 else -1
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (df_bt[cum_col] / df_bt[cum_col].cummax() - 1).min()

    df_temp = df_bt.copy()
    df_temp["YM"] = pd.to_datetime(df_temp["Date"]).dt.to_period("M")
    monthly_rets = df_temp.groupby("YM")[return_col].sum()
    hit_rate = (monthly_rets > 0).sum() / len(monthly_rets) if len(monthly_rets) > 0 else 0

    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
            "max_dd": max_dd, "hit_rate": hit_rate, "n_days": n_days}


def main():
    st.markdown("# Brent Crude Oil CTA — Backtest Platform")
    st.markdown("**TSMOM + Carry + OVX Macro Filter**")

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
        st.subheader("Signal 3 — OVX Macro Filter")
        st.caption("Scales exposure. Separates oil shocks from financial crises.")
        ovx_thresh = st.slider("OVX/VIX Ratio Threshold", 1.0, 3.0, 1.8, 0.1,
                               help="Above = oil-specific event, stay full size")
        vix_caution = st.number_input("VIX Caution Level", 15, 50, 30,
                                      help="Half size when VIX exceeds this (if not oil-specific)")
        vix_flatten = st.number_input("VIX Flatten Level", 20, 80, 40,
                                      help="Flatten when VIX exceeds this (if not oil-specific)")

        st.divider()
        st.subheader("Execution")
        tc_bps = st.number_input("Transaction Cost (bps)", 0, 50, 5)
        tc = tc_bps / 10000

    # ── COMPUTE ──
    max_lb = max(lookbacks)
    lb_start = pd.Timestamp(start_date) - pd.Timedelta(days=int(max_lb * 1.6))
    df_lb = df_raw[df_raw["Date"] >= lb_start].copy()
    if len(df_lb) < max_lb:
        st.error(f"Not enough data for {max_lb}-day lookback.")
        st.stop()

    df_signals = compute_signals(df_lb, tsmom_w, carry_w, lookbacks, ovx_thresh, vix_caution, vix_flatten)
    df_signals_bt = df_signals[df_signals["Date"] <= pd.Timestamp(end_date)].copy()
    if len(df_signals_bt) < 3:
        st.error("Not enough signal months. Widen your date range.")
        st.stop()

    df_daily_bt = df_lb[(df_lb["Date"] >= pd.Timestamp(start_date)) & (df_lb["Date"] <= pd.Timestamp(end_date))].copy()
    df_bt = run_backtest(df_daily_bt, df_signals_bt, tc)
    df_bt = df_bt.dropna(subset=["Strat_Net", "Cumulative"]).reset_index(drop=True)
    if len(df_bt) == 0:
        st.error("No backtest results.")
        st.stop()

    ss = compute_stats(df_bt, "Strat_Net", "Cumulative")
    bh = compute_stats(df_bt, "BH_Ret", "BH_Cumulative")

    # ── OVERVIEW ──
    with st.expander("Strategy Overview", expanded=False):
        st.markdown("""
        **Three signals on Brent crude oil futures (CO1 Comdty):**
        - **TSMOM:** Long if Brent trending up across multiple lookback windows, short if down.
        - **Carry:** Long in backwardation (CO1 > CO12), short in contango (CO1 < CO12).
        - **OVX Filter:** High OVX with normal VIX = oil-specific shock → stay full size. High VIX with normal OVX = financial crisis → scale down.

        **Composite = (TSMOM_w × TSMOM + Carry_w × Carry) × Macro Scale.** Monthly rebalance on prior month's signal.
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
        fig_c = make_subplots(specs=[[{"secondary_y": True}]])
        fig_c.add_trace(go.Bar(x=df_sig_disp["Date"], y=df_sig_disp["TSMOM_Contribution"],
                               name="TSMOM", marker_color="#4A90D9"), secondary_y=False)
        fig_c.add_trace(go.Bar(x=df_sig_disp["Date"], y=df_sig_disp["Carry_Contribution"],
                               name="Carry", marker_color="#F5A623"), secondary_y=False)
        fig_c.add_trace(go.Scatter(x=df_sig_disp["Date"], y=df_sig_disp["Macro_Scale"],
                                   name="Macro Scale", line=dict(color="#FF4B4B", width=2, dash="dot")),
                        secondary_y=True)
        fig_c.update_layout(template="plotly_dark", barmode="relative", height=400,
                            margin=dict(l=60,r=60,t=30,b=40),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig_c.update_yaxes(title_text="Contribution", secondary_y=False)
        fig_c.update_yaxes(title_text="Macro Scale", range=[0,1.1], secondary_y=True)
        st.plotly_chart(fig_c, use_container_width=True)

    # ── DETAIL TABLE ──
    st.markdown("---")
    st.subheader("Position & Macro Filter Detail")
    dmd = df_sig_disp[["Date","CO1","CO12","Carry_Spread","TSMOM","Carry",
                        "OVX","VIX","OVX_VIX_Ratio","Macro_Scale","Composite"]].copy()
    bt_t = df_bt.copy()
    bt_t["YM"] = pd.to_datetime(bt_t["Date"]).dt.to_period("M")
    m_sr = bt_t.groupby("YM")["Strat_Net"].apply(lambda x: (1+x).prod()-1)
    m_mr = bt_t.groupby("YM")["Market_Ret"].apply(lambda x: (1+x).prod()-1)
    m_sc = (1+m_sr).cumprod()
    m_bc = (1+m_mr).cumprod()
    dmd["YM"] = pd.to_datetime(dmd["Date"]).dt.to_period("M")
    dmd = dmd.merge(m_sr.rename("Strat Ret"), left_on="YM", right_index=True, how="left")
    dmd = dmd.merge(m_mr.rename("Mkt Ret"), left_on="YM", right_index=True, how="left")
    dmd = dmd.merge(m_sc.rename("Cum Strat"), left_on="YM", right_index=True, how="left")
    dmd = dmd.merge(m_bc.rename("Cum B&H"), left_on="YM", right_index=True, how="left")
    dmd = dmd.drop(columns=["YM"])
    dmd["Date"] = dmd["Date"].dt.strftime("%Y-%m-%d")
    dmd = dmd.rename(columns={"Carry_Spread":"Carry Spread","OVX_VIX_Ratio":"OVX/VIX","Macro_Scale":"Macro Scale"})

    fmt = {"CO1":"${:.2f}","CO12":"${:.2f}","Carry Spread":"{:.2%}","TSMOM":"{:+.3f}","Carry":"{:+.3f}",
           "OVX":"{:.1f}","VIX":"{:.1f}","OVX/VIX":"{:.2f}","Macro Scale":"{:.2f}","Composite":"{:+.3f}",
           "Strat Ret":"{:.2%}","Mkt Ret":"{:.2%}","Cum Strat":"{:.4f}","Cum B&H":"{:.4f}"}
    sdf = dmd.style.format(fmt)
    try:
        sdf = sdf.map(lambda x: "background-color: rgba(255,75,75,0.15)" if isinstance(x,(int,float)) and x<1.0 else "", subset=["Macro Scale"])
    except AttributeError:
        sdf = sdf.applymap(lambda x: "background-color: rgba(255,75,75,0.15)" if isinstance(x,(int,float)) and x<1.0 else "", subset=["Macro Scale"])
    st.dataframe(sdf, use_container_width=True, height=400)
    st.download_button("Download Monthly Detail", dmd.to_csv(index=False), "monthly_detail.csv", "text/csv")

    # ── HEATMAP ──
    st.markdown("---")
    st.subheader("Monthly Returns Heatmap")
    hm = df_bt.copy()
    hm["Year"] = pd.to_datetime(hm["Date"]).dt.year
    hm["Month"] = pd.to_datetime(hm["Date"]).dt.month
    mr = hm.groupby(["Year","Month"])["Strat_Net"].apply(lambda x: (1+x).prod()-1).reset_index()
    mr.columns = ["Year","Month","Return"]
    hd = mr.pivot(index="Year", columns="Month", values="Return").reindex(columns=range(1,13))
    ml = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    try:
        tv = hd.map(lambda x: f"{x:.1%}" if pd.notna(x) else "").values
    except AttributeError:
        tv = hd.applymap(lambda x: f"{x:.1%}" if pd.notna(x) else "").values
    fig_hm = go.Figure(go.Heatmap(z=hd.values, x=ml, y=hd.index.astype(str), text=tv,
                                  texttemplate="%{text}", textfont=dict(size=11),
                                  colorscale="RdYlGn", zmid=0,
                                  colorbar=dict(title="Return", tickformat=".0%")))
    fig_hm.update_layout(template="plotly_dark", height=max(300,len(hd)*35+100),
                         margin=dict(l=60,r=30,t=30,b=40), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── ROLLING STATS ──
    st.markdown("---")
    st.subheader("Rolling Statistics (252-Day)")
    rr = df_bt["Strat_Net"].rolling(252, min_periods=126)
    r_ann = rr.mean() * 252
    r_vol = rr.std() * np.sqrt(252)
    r_sh = r_ann / r_vol
    fig_r = make_subplots(specs=[[{"secondary_y": True}]])
    fig_r.add_trace(go.Scatter(x=df_bt["Date"], y=r_sh, name="Rolling Sharpe",
                               line=dict(color="#00D4AA", width=2)), secondary_y=False)
    fig_r.add_trace(go.Scatter(x=df_bt["Date"], y=r_ann, name="Rolling Ann. Return",
                               line=dict(color="#F5A623", width=1.5, dash="dash")), secondary_y=True)
    fig_r.add_hline(y=1.0, line_dash="dot", line_color="rgba(255,255,255,0.3)", secondary_y=False)
    fig_r.add_hline(y=0.0, line_dash="dot", line_color="rgba(255,255,255,0.2)", secondary_y=False)
    fig_r.update_layout(template="plotly_dark", height=400, margin=dict(l=60,r=60,t=30,b=40),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig_r.update_yaxes(title_text="Sharpe", secondary_y=False)
    fig_r.update_yaxes(title_text="Ann. Return", tickformat=".0%", secondary_y=True)
    st.plotly_chart(fig_r, use_container_width=True)

    # ── DISTRIBUTION ──
    st.markdown("---")
    st.subheader("Daily P&L Distribution")
    dr = df_bt["Strat_Net"].dropna()
    rm, rmed, rsk, rku = dr.mean(), dr.median(), stats.skew(dr), stats.kurtosis(dr)
    fig_d = go.Figure()
    fig_d.add_trace(go.Histogram(x=dr, nbinsx=100, marker_color="rgba(0,212,170,0.6)",
                                 marker_line=dict(color="rgba(0,212,170,1)", width=0.5)))
    fig_d.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.5)")
    fig_d.add_vline(x=rm, line_color="#F5A623", line_width=2,
                    annotation_text=f"Mean: {rm:.4%}", annotation_position="top right")
    fig_d.update_layout(template="plotly_dark", xaxis_title="Daily Return", yaxis_title="Frequency",
                        height=400, margin=dict(l=60,r=30,t=30,b=40),
                        annotations=[dict(x=0.98, y=0.95, xref="paper", yref="paper",
                                         text=f"Mean: {rm:.4%}<br>Median: {rmed:.4%}<br>Skew: {rsk:.2f}<br>Kurtosis: {rku:.2f}",
                                         showarrow=False, font=dict(size=12, color="white"),
                                         bgcolor="rgba(0,0,0,0.6)", bordercolor="rgba(255,255,255,0.3)",
                                         borderwidth=1, borderpad=8, align="left")])
    fig_d.update_xaxes(tickformat=".1%")
    st.plotly_chart(fig_d, use_container_width=True)

    st.markdown("---")
    st.caption("Brent CTA Backtest Platform. Built with Streamlit.")

if __name__ == "__main__":
    main()
