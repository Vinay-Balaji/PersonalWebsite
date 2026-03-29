"""
BRENT CRUDE OIL CTA STRATEGY — Backtest Platform
TSMOM + Carry + Vol Targeting + Ichimoku + Triangular MA
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
    if n_cols >= 4:
        test_dates = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        if test_dates.notna().sum() > len(df) * 0.5:
            result = pd.DataFrame({
                "Date": pd.to_datetime(df.iloc[:, 0], errors="coerce"),
                "CO1":  pd.to_numeric(df.iloc[:, 1], errors="coerce"),
            })
            if n_cols >= 4:
                result["CO12"] = pd.to_numeric(df.iloc[:, 3], errors="coerce")
            else:
                result["CO12"] = result["CO1"]
            result = result.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
            result[["CO1","CO12"]] = result[["CO1","CO12"]].ffill().bfill()
            result = result.dropna(subset=["CO1","CO12"]).reset_index(drop=True)
            return result

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


def compute_signals(df_daily, signals_config):
    df = df_daily.copy().set_index("Date")
    df["YM"] = df.index.to_period("M")
    month_ends = df.groupby("YM").tail(1).copy()
    prices = df["CO1"]
    signal_columns = {}

    if signals_config["tsmom"]["enabled"]:
        lookbacks = signals_config["tsmom"]["lookbacks"]
        components = []
        for lb in lookbacks:
            ret = prices.pct_change(lb)
            components.append(np.sign(ret.loc[month_ends.index]))
        if components:
            month_ends["TSMOM"] = pd.concat(components, axis=1).mean(axis=1).values
        else:
            month_ends["TSMOM"] = 0.0
        signal_columns["TSMOM"] = signals_config["tsmom"]["weight"]

    if signals_config["carry"]["enabled"]:
        month_ends["Carry_Spread"] = (month_ends["CO1"] - month_ends["CO12"]) / month_ends["CO1"]
        month_ends["Carry"] = np.sign(month_ends["Carry_Spread"])
        signal_columns["Carry"] = signals_config["carry"]["weight"]

    if signals_config["ichimoku"]["enabled"]:
        tp = signals_config["ichimoku"]["tenkan"]
        kp = signals_config["ichimoku"]["kijun"]
        sp = signals_config["ichimoku"]["senkou"]
        tenkan = (prices.rolling(tp).max() + prices.rolling(tp).min()) / 2
        kijun = (prices.rolling(kp).max() + prices.rolling(kp).min()) / 2
        span_a = ((tenkan + kijun) / 2).shift(kp)
        span_b = ((prices.rolling(sp).max() + prices.rolling(sp).min()) / 2).shift(kp)
        cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
        cloud_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)
        ichi_signal = pd.Series(0.0, index=prices.index)
        ichi_signal[prices > cloud_top] = 1.0
        ichi_signal[prices < cloud_bot] = -1.0
        month_ends["Ichimoku"] = ichi_signal.loc[month_ends.index].values
        signal_columns["Ichimoku"] = signals_config["ichimoku"]["weight"]

    if signals_config["tma"]["enabled"]:
        period = signals_config["tma"]["period"]
        first_sma = prices.rolling(period, min_periods=period // 2).mean()
        tma = first_sma.rolling(period, min_periods=period // 2).mean()
        tma_signal = np.sign(prices - tma)
        month_ends["TMA"] = tma_signal.loc[month_ends.index].values
        signal_columns["TMA"] = signals_config["tma"]["weight"]

    if not signal_columns:
        month_ends["Composite"] = 0.0
    else:
        total_w = sum(signal_columns.values())
        if total_w == 0:
            total_w = 1.0
        composite = pd.Series(0.0, index=month_ends.index)
        for sig_name, raw_weight in signal_columns.items():
            norm_w = raw_weight / total_w
            month_ends[f"{sig_name}_Contribution"] = norm_w * month_ends[sig_name]
            composite += month_ends[f"{sig_name}_Contribution"]
        month_ends["Composite"] = composite.clip(-1, 1)

    month_ends = month_ends.drop(columns=["YM"], errors="ignore").reset_index()
    month_ends = month_ends.rename(columns={"index": "Date"} if "index" in month_ends.columns else {})
    if "Date" not in month_ends.columns and month_ends.index.name == "Date":
        month_ends = month_ends.reset_index()
    return month_ends, list(signal_columns.keys())


def run_backtest(df_daily, df_signals, vol_config):
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
    df["Raw_Position"] = df["Signal"].ffill().fillna(0)

    if vol_config["enabled"]:
        vt = vol_config["target"]
        vl = vol_config["lookback"]
        realized_vol = df["Market_Ret"].rolling(vl, min_periods=max(5, vl // 4)).std() * np.sqrt(252)
        df["Realized_Vol"] = realized_vol
        df["Vol_Scalar"] = (vt / realized_vol).clip(0.25, 2.0).fillna(1.0)
        df["Position"] = df["Raw_Position"] * df["Vol_Scalar"]
    else:
        df["Realized_Vol"] = df["Market_Ret"].rolling(21, min_periods=5).std() * np.sqrt(252)
        df["Vol_Scalar"] = 1.0
        df["Position"] = df["Raw_Position"]

    df["Strat_Gross"] = df["Position"] * df["Market_Ret"]
    tc = 0.0005
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
    df["BH_Ret"] = df["Market_Ret"].fillna(0)
    df["BH_Cumulative"] = (1 + df["BH_Ret"]).cumprod()
    df = df.drop(columns=["Month"], errors="ignore").reset_index()
    return df


def compute_stats(df_bt, return_col="Strat_Net", cum_col="Cumulative"):
    rets = df_bt[return_col].dropna()
    n_days = len(rets)
    if n_days == 0:
        return {k: 0 for k in ["ann_ret","ann_vol","sharpe","max_dd","hit_rate"]}
    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (df_bt[cum_col] / df_bt[cum_col].cummax() - 1).min()
    df_temp = df_bt.copy()
    df_temp["YM"] = pd.to_datetime(df_temp["Date"]).dt.to_period("M")
    monthly_rets = df_temp.groupby("YM")[return_col].apply(lambda x: (1+x).prod()-1)
    hit_rate = (monthly_rets > 0).sum() / len(monthly_rets) if len(monthly_rets) > 0 else 0
    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
            "max_dd": max_dd, "hit_rate": hit_rate, "n_days": n_days}


def main():
    st.markdown("# Wes Vidhya and Vinay Backtest Platform")
    st.markdown("**Brent Crude Oil CTA — Systematic Long/Short Strategy**")

    df_raw = load_data()
    if df_raw is None:
        st.error("**bloomberg_data.xlsx** (or .csv) not found. Place your Bloomberg data export in the same directory.")
        st.stop()
    if len(df_raw) < 252:
        st.error(f"Only {len(df_raw)} rows. Need at least 252.")
        st.stop()

    min_date = df_raw["Date"].min().date()
    max_date = df_raw["Date"].max().date()

    with st.sidebar:
        st.header("Backtest Settings")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start", value=max(date(2011, 1, 1), min_date), min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End", value=max_date, min_value=min_date, max_value=max_date)
        df_filt = df_raw[(df_raw["Date"] >= pd.Timestamp(start_date)) & (df_raw["Date"] <= pd.Timestamp(end_date))]
        st.caption(f"{start_date} to {end_date} — {len(df_filt):,} days, {df_filt['Date'].dt.to_period('M').nunique()} months")

        st.divider()
        tsmom_on = st.toggle("Signal 1 — TSMOM", value=True, key="tsmom_on")
        if tsmom_on:
            st.caption("Rides the trend. Long if up, short if down.")
            tsmom_w = st.slider("Weight", 0.0, 1.0, 0.55, 0.05, key="tw")
            lb_opts = {"21d (~1M)": 21, "42d (~2M)": 42, "63d (~3M)": 63,
                       "126d (~6M)": 126, "189d (~9M)": 189, "252d (~12M)": 252}
            sel = st.multiselect("Lookback Windows", list(lb_opts.keys()),
                                 default=["21d (~1M)", "63d (~3M)", "252d (~12M)"])
            lookbacks = [lb_opts[l] for l in sel]
            if not lookbacks:
                st.error("Select at least one lookback.")
                st.stop()
        else:
            tsmom_w = 0; lookbacks = [21, 63, 252]

        st.divider()
        carry_on = st.toggle("Signal 2 — Carry", value=True, key="carry_on")
        if carry_on:
            st.caption("Trades the curve. Long in backwardation, short in contango.")
            carry_w = st.slider("Weight", 0.0, 1.0, 0.45, 0.05, key="cw")
        else:
            carry_w = 0

        st.divider()
        vol_on = st.toggle("Signal 3 — Vol Targeting", value=True, key="vol_on")
        if vol_on:
            st.caption("Adjusts bet size to keep risk constant.")
            vol_target_pct = st.slider("Target Volatility (%)", 5, 30, 15, 1)
            vol_target = vol_target_pct / 100
            vol_lookback = st.select_slider("Vol Lookback (days)", options=[10, 15, 21, 42, 63], value=21)
        else:
            vol_target = 0.15; vol_lookback = 21; vol_target_pct = 15

        st.divider()
        ichi_on = st.toggle("Signal 4 — Ichimoku Cloud", value=False, key="ichi_on")
        if ichi_on:
            st.caption("Japanese trend system. Long above cloud, short below.")
            ichi_w = st.slider("Weight", 0.0, 1.0, 0.30, 0.05, key="iw")
            ichi_tenkan = st.number_input("Tenkan Period", 5, 30, 9, key="it")
            ichi_kijun = st.number_input("Kijun Period", 10, 60, 26, key="ik")
            ichi_senkou = st.number_input("Senkou Period", 20, 120, 52, key="is")
        else:
            ichi_w = 0; ichi_tenkan = 9; ichi_kijun = 26; ichi_senkou = 52

        st.divider()
        tma_on = st.toggle("Signal 5 — Triangular MA", value=False, key="tma_on")
        if tma_on:
            st.caption("Double-smoothed MA. Long above, short below.")
            tma_w = st.slider("Weight", 0.0, 1.0, 0.25, 0.05, key="tmaw")
            tma_period = st.slider("Period (days)", 10, 200, 50, 5, key="tmap")
        else:
            tma_w = 0; tma_period = 50

        enabled = []
        if tsmom_on: enabled.append(("TSMOM", tsmom_w))
        if carry_on: enabled.append(("Carry", carry_w))
        if ichi_on: enabled.append(("Ichimoku", ichi_w))
        if tma_on: enabled.append(("TMA", tma_w))
        if not enabled:
            st.divider()
            st.error("Enable at least one directional signal.")
            st.stop()

        st.divider()
        total_w = sum(w for _, w in enabled)
        st.caption("**Active Signals:**")
        for name, w in enabled:
            norm_w = w / total_w if total_w > 0 else 0
            st.caption(f"  {name}: {w:.2f} → normalized {norm_w:.0%}")
        st.caption(f"  Vol Target: {vol_target_pct}%" if vol_on else "  Vol Target: off")

    signals_config = {
        "tsmom": {"enabled": tsmom_on, "weight": tsmom_w, "lookbacks": lookbacks},
        "carry": {"enabled": carry_on, "weight": carry_w},
        "ichimoku": {"enabled": ichi_on, "weight": ichi_w,
                     "tenkan": ichi_tenkan, "kijun": ichi_kijun, "senkou": ichi_senkou},
        "tma": {"enabled": tma_on, "weight": tma_w, "period": tma_period},
    }
    vol_config = {"enabled": vol_on, "target": vol_target, "lookback": vol_lookback}

    max_lb = max(lookbacks) if tsmom_on else 252
    if ichi_on: max_lb = max(max_lb, ichi_senkou + ichi_kijun + 10)
    if tma_on: max_lb = max(max_lb, tma_period * 2 + 10)
    lb_start = pd.Timestamp(start_date) - pd.Timedelta(days=int(max_lb * 1.6))
    df_lb = df_raw[df_raw["Date"] >= lb_start].copy()
    if len(df_lb) < max_lb:
        st.error(f"Not enough data for lookback. Need ~{max_lb} days before start date.")
        st.stop()

    df_signals, active_names = compute_signals(df_lb, signals_config)
    df_signals_bt = df_signals[df_signals["Date"] <= pd.Timestamp(end_date)].copy()
    if len(df_signals_bt) < 3:
        st.error("Not enough signal months. Widen your date range.")
        st.stop()

    df_daily_bt = df_lb[(df_lb["Date"] >= pd.Timestamp(start_date)) & (df_lb["Date"] <= pd.Timestamp(end_date))].copy()
    df_bt = run_backtest(df_daily_bt, df_signals_bt, vol_config)
    df_bt = df_bt.dropna(subset=["Strat_Net","Cumulative"]).reset_index(drop=True)
    if len(df_bt) == 0:
        st.error("No backtest results.")
        st.stop()

    ss = compute_stats(df_bt, "Strat_Net", "Cumulative")
    bh = compute_stats(df_bt, "BH_Ret", "BH_Cumulative")

    with st.expander("Strategy Overview — How Each Signal Works", expanded=False):
        st.markdown("""
        ### Signal 1 — TSMOM (Time-Series Momentum)

        Imagine watching the price of oil over the past few months. If it has been going **up**, this signal says
        "the trend is likely to continue — **go long**." If it has been going **down**, it says "the downtrend will
        probably keep going — **go short**."

        We look at three different time windows — for example, the past 1 month, 3 months, and 12 months — and
        check whether each one is positive or negative. Then we average them together. This way we catch both
        quick short-term moves AND long slow trends. If all three windows agree (all up or all down), the signal
        is strongest. If they disagree, the signal is weaker.

        **Why it works:** Markets tend to trend. When oil starts moving in one direction — whether because of
        OPEC cuts, a demand surge, or a geopolitical crisis — it usually keeps moving that way for weeks or months
        before reversing. TSMOM captures that persistence.

        ---

        ### Signal 2 — Carry (Term Structure)

        Oil futures trade at different prices depending on when the oil will be delivered. The "front month" (CO1)
        is oil for delivery soon. The "12-month deferred" (CO12) is oil for delivery a year from now.

        When today's oil costs **more** than oil a year out, the curve is in **backwardation**. This usually means
        physical supply is tight right now — refineries need oil urgently and are willing to pay a premium.
        That is bullish, so we **go long**.

        When today's oil costs **less** than oil a year out, the curve is in **contango**. This means there is
        plenty of supply sitting around — storage tanks are filling up. That is bearish, so we **go short**.

        **Why it works:** The shape of the futures curve reflects real supply and demand fundamentals. Backwardation
        also means you earn positive "roll yield" just by holding the position — the futures price converges up
        toward spot as expiry approaches.

        ---

        ### Signal 3 — Vol Targeting (Position Sizing)

        This is not a directional signal — it does not tell you to go long or short. Instead, it adjusts how
        **big** your position is based on how volatile the market is right now.

        Some months oil barely moves — maybe 0.5% per day. Other months it swings 3-5% daily. If you hold the
        same position size in both environments, your risk is wildly inconsistent. Vol targeting fixes this.

        It works by dividing your **target volatility** (e.g., 15%) by the **current realized volatility**.
        If realized vol is 30%, your position gets cut in half (15/30 = 0.5x). If realized vol is 10%,
        your position gets levered up (15/10 = 1.5x).

        **Why it works:** It prevents one crazy week from destroying your returns. Your good months get slightly
        smaller, but your blowup months get much smaller. The net effect is a smoother equity curve and better
        risk-adjusted returns (higher Sharpe ratio).

        ---

        ### Signal 4 — Ichimoku Cloud (GOC)

        The Ichimoku Cloud is a Japanese trend-following system developed in the 1960s. It creates a shaded
        "cloud" on the price chart using averages of recent highs and lows over different time windows.

        The cloud has a top edge and a bottom edge. When the price is **above** the entire cloud, the trend is
        clearly bullish — **go long**. When the price is **below** the entire cloud, the trend is bearish —
        **go short**. When the price is **inside** the cloud, the market is undecided and choppy — **stay flat**
        (signal = 0).

        The cloud uses three periods: a short window (Tenkan, default 9 days), a medium window (Kijun, default
        26 days), and a long window (Senkou, default 52 days). These create a forward-looking zone that acts
        like dynamic support and resistance.

        **Why it works:** The cloud combines trend direction, momentum, and support/resistance into one indicator.
        The "inside the cloud = flat" feature helps avoid whipsaws during choppy, trendless markets — which is
        where most pure momentum strategies lose money.

        ---

        ### Signal 5 — Triangular Moving Average (TMA)

        A Triangular Moving Average is a double-smoothed version of the price. It takes a Simple Moving Average
        (SMA) of the price, then takes another SMA of that result. This double-smoothing removes most of the
        day-to-day noise and gives you a very clean trend line.

        When the actual price is **above** this super-smooth line, momentum is up — **go long**.
        When the price is **below** it, momentum is down — **go short**.

        Compared to a regular moving average, the TMA reacts more slowly. This means fewer false signals and
        less whipsaw, but it also means slightly later entries and exits. It works best as a confirmation tool
        alongside faster signals like TSMOM.

        **Why it works:** By smoothing twice, the TMA filters out random noise and only responds to real,
        sustained price movements. It is especially effective in trending commodity markets where the underlying
        move lasts for months.
        """)

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

    if vol_on:
        st.markdown("---")
        st.subheader("Vol Targeting")
        fig_vol = make_subplots(specs=[[{"secondary_y": True}]])
        fig_vol.add_trace(go.Scatter(x=df_bt["Date"], y=df_bt["Realized_Vol"],
                                     name="Realized Vol", line=dict(color="#F5A623", width=1.5)),
                          secondary_y=False)
        fig_vol.add_hline(y=vol_target, line_dash="dash", line_color="#00D4AA", line_width=2,
                          annotation_text=f"Target: {vol_target_pct}%", secondary_y=False)
        fig_vol.add_trace(go.Scatter(x=df_bt["Date"], y=df_bt["Vol_Scalar"],
                                     name="Position Scalar", line=dict(color="#4A90D9", width=1, dash="dot")),
                          secondary_y=True)
        fig_vol.update_layout(template="plotly_dark", height=400,
                              margin=dict(l=60,r=60,t=30,b=40),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig_vol.update_yaxes(title_text="Annualized Vol", tickformat=".0%", secondary_y=False)
        fig_vol.update_yaxes(title_text="Scalar", secondary_y=True)
        st.plotly_chart(fig_vol, use_container_width=True)

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
        color_map = {"TSMOM": "#4A90D9", "Carry": "#F5A623", "Ichimoku": "#E74C3C", "TMA": "#9B59B6"}
        for sig_name in active_names:
            contrib_col = f"{sig_name}_Contribution"
            if contrib_col in df_sig_disp.columns:
                fig_c.add_trace(go.Bar(x=df_sig_disp["Date"], y=df_sig_disp[contrib_col],
                                       name=sig_name, marker_color=color_map.get(sig_name, "#888")))
        fig_c.update_layout(template="plotly_dark", barmode="relative", height=400,
                            margin=dict(l=60,r=30,t=30,b=40),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            yaxis_title="Contribution")
        st.plotly_chart(fig_c, use_container_width=True)

    st.markdown("---")
    st.caption("Brent CTA Backtest Platform. Built with Streamlit.")


if __name__ == "__main__":
    main()
