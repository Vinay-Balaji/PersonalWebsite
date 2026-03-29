I need you to build a Streamlit application for a Brent Crude Oil CTA (Commodity Trading Advisor) backtest platform. The project folder is called "Brent CTA Strategy" and contains a CSV file called "bloomberg_data.csv" with columns: Date, CO1, CO12, OVX, VIX. Date is in YYYY-MM-DD format, CO1 is the Brent front-month futures price (CO1 Comdty), CO12 is the Brent 12-month deferred futures price (CO12 Comdty), OVX is the CBOE Crude Oil Volatility Index, and VIX is the CBOE equity volatility index. The data is DAILY frequency, approximately 4,000 rows from January 2010 to present. OVX data may be missing before June 2011 — forward-fill then backfill any NaN values. Handle the Date column flexibly: use pd.to_datetime(df["Date"], format="mixed") in case Bloomberg exports dates in different formats. Drop any rows where Date is NaT. If the CSV has fewer than 252 rows, show an error that more data is needed for the 12-month lookback.

THE STRATEGY:

This is a systematic long/short strategy on Brent crude oil futures that combines three signals into a composite signal between -1 (full short) and +1 (full long). Signals are computed at month-end and held constant for the entire following month. The position each month is based on the PRIOR month's composite signal (lagged one period to avoid look-ahead bias). Daily returns are computed using the constant monthly signal multiplied by each day's CO1 return.

Signal 1 — TSMOM (Time-Series Momentum):
- At each month-end, compute the return of CO1 over three lookback windows: trailing 21 trading days (~1 month), trailing 63 trading days (~3 months), and trailing 252 trading days (~12 months).
- Take the sign of each return (+1 if positive, -1 if negative).
- Blend them equally: TSMOM = (sign(ret_21d) + sign(ret_63d) + sign(ret_252d)) / 3.
- This signal ranges from -1 to +1.
- The lookback windows should be configurable in the sidebar. The user selects which windows to include from a multi-select with options [21, 42, 63, 126, 189, 252] days. Default = [21, 63, 252]. All selected windows are blended equally.

Signal 2 — Carry (Term Structure):
- At each month-end, compute the carry spread: (CO1 - CO12) / CO1.
- Take the sign: if positive (backwardation = front above back), signal = +1 (long). If negative (contango = front below back), signal = -1 (short).
- The intuition is: backwardation means tight physical supply and positive roll yield. Contango means oversupply and negative roll yield.

Signal 3 — OVX Macro Filter (applied as a RISK SCALAR, not an additive signal):
- At each month-end, compute OVX / VIX ratio.
- If OVX/VIX > threshold (default 1.8), this is an oil-specific shock (e.g., Iran, Hormuz, OPEC surprise). The oil market is moving on its own fundamentals, not because of a financial crisis. In this case → macro scale = 1.0 (STAY FULL SIZE, do NOT reduce exposure).
- If OVX/VIX <= threshold AND VIX > flatten level (default 40) → systemic financial crisis where all assets correlate and momentum signals break down → macro scale = 0.0 (flatten position entirely).
- If OVX/VIX <= threshold AND VIX > caution level (default 30) → elevated systemic risk → macro scale = 0.5 (half position size).
- Otherwise → macro scale = 1.0 (normal conditions).
- The key insight: a high OVX with low VIX means the oil market is stressed but the financial system is fine — these are the best trending environments for crude (geopolitical supply shocks). A high VIX with normal OVX means the stress is in equities/credit and oil is getting dragged down by macro — these are the worst environments for momentum signals.

Composite signal = (TSMOM_weight × TSMOM + Carry_weight × Carry) × macro_scale, clamped to [-1, +1].

BACKTEST LOGIC:

- Resample daily data to month-end for signal computation. Use the last trading day of each month.
- Compute all three signals at month-end.
- The composite signal computed at month-end M is applied to ALL TRADING DAYS in month M+1.
- Daily strategy return = position (constant for the month) × daily CO1 return.
- Transaction cost (default 5bps) is charged on the FIRST trading day of any month where the position changed from the prior month (i.e., where abs(new_signal - old_signal) > 0.01).
- Cumulative returns: cumulative = cumprod(1 + daily_strategy_return).
- Buy-and-hold benchmark: just hold CO1 long every day, cumulative = cumprod(1 + daily_CO1_return).
- Drawdown series: drawdown = cumulative / cumulative.cummax() - 1.

OPTIONAL ENHANCEMENTS (implement as toggles in the sidebar):

1. Vol Targeting: if enabled, scale the daily position by (vol_target / realized_vol), where realized_vol is the trailing 21-day rolling standard deviation of daily CO1 returns, annualized by multiplying by sqrt(252). Clamp the vol scalar between 0.25 and 2.0. Recompute daily (not monthly). Default vol target = 15%. The slider should go from 5% to 30%.

2. Trailing Stop-Loss: if enabled, flatten the position (set to zero) whenever the cumulative strategy return drawdown exceeds the stop threshold. Remain flat until the next month-end signal rebalance. Default stop threshold = -10%. The slider should go from -5% to -25%.

STREAMLIT APP STRUCTURE:

Page config: st.set_page_config(page_title="Brent CTA Backtest", layout="wide")
Page title: "Brent Crude Oil CTA Strategy — Backtest Platform"
Subtitle: "TSMOM + Carry + OVX Macro Filter | Scotiabank QIS"

SIDEBAR — all inputs dynamically update the backtest when changed:

- Header: "Backtest Parameters"
- Date range: st.date_input for start date (default 2011-01-01) and end date (default = latest date in data).
- Show: "Data: {first_date} to {last_date} ({N} trading days, {M} signal months)"
- st.divider()
- Header: "Signal Weights"
- TSMOM Weight: st.slider from 0.0 to 1.0, default 0.55, step 0.05.
- Carry Weight: st.slider from 0.0 to 1.0, default 0.45, step 0.05.
- Show computed text: "Total Signal Weight: {TSMOM + Carry}" — display in green if = 1.0, orange if not.
- st.divider()
- Header: "TSMOM Settings"
- TSMOM Lookback Windows: st.multiselect with options [21, 42, 63, 126, 189, 252] days, default = [21, 63, 252]. Label them as "21d (~1M), 42d (~2M), 63d (~3M), 126d (~6M), 189d (~9M), 252d (~12M)".
- st.divider()
- Header: "OVX Macro Filter"
- OVX/VIX Ratio Threshold: st.slider 1.0 to 3.0, default 1.8, step 0.1. Add help text: "Above this ratio = oil-specific event, stay full size"
- VIX Caution Level: st.number_input, default 30, min 15, max 50. Help: "Half size when VIX exceeds this AND OVX/VIX is below threshold"
- VIX Flatten Level: st.number_input, default 40, min 20, max 80. Help: "Flatten when VIX exceeds this AND OVX/VIX is below threshold"
- st.divider()
- Header: "Execution"
- Transaction Cost (bps): st.number_input, default 5, min 0, max 50. Divide by 10000 to convert to decimal in code.
- st.divider()
- Header: "Enhancements"
- Enable Vol Targeting: st.checkbox, default False. If checked, show:
  - Vol Target: st.slider 5% to 30%, default 15%, step 1%, format as percentage.
- Enable Trailing Stop: st.checkbox, default False. If checked, show:
  - Stop Threshold: st.slider -5% to -25%, default -10%, step 1%, format as percentage.

MAIN PAGE — sections from top to bottom:

Section 1: "Strategy Overview"
Display a short expander (st.expander) with this text:
"This strategy combines three signals on Brent crude oil futures:
- **TSMOM**: Go long if Brent has been trending up over multiple lookback windows, short if trending down. Blends short, medium, and long-term momentum.
- **Carry**: Go long in backwardation (positive roll yield), short in contango (negative roll yield). Signal derived from CO1 vs CO12 spread.
- **OVX Macro Filter**: Uses the OVX/VIX ratio to distinguish oil-specific shocks (stay invested) from systemic financial crises (reduce exposure). This prevents the strategy from being full-size during liquidity crises where momentum signals break down, while staying aggressive during geopolitical supply shocks where crude trends hardest.
Composite signal = (TSMOM_w × TSMOM + Carry_w × Carry) × Macro Scale. Rebalanced monthly, executed on prior month's signal."

Section 2: "Performance Summary"
Use st.columns(6) to display metric cards side by side:
Row 1 (Strategy): Annualized Return, Annualized Vol, Sharpe Ratio, Max Drawdown, Calmar Ratio, Hit Rate.
Row 2 (Buy & Hold): same six metrics for comparison.
Label the rows clearly as "Strategy" and "Buy & Hold".
For the Sharpe Ratio metric, use delta_color: if Sharpe >= 1.0 show it in green, if >= 0.5 yellow, if < 0.5 red. Display Sharpe to 2 decimal places.
Annualized return = (final_cumulative)^(252/trading_days) - 1.
Annualized vol = daily_returns.std() * sqrt(252).
Sharpe = annualized_return / annualized_vol.
Max drawdown = drawdown_series.min().
Calmar = annualized_return / abs(max_drawdown).
Hit Rate = percentage of MONTHS with positive total return (aggregate daily returns by month first).

Section 3: "Cumulative Returns"
Plotly line chart with:
- Strategy cumulative (solid line, thicker, blue or teal color).
- Buy & hold cumulative (dashed line, thinner, gray).
- Horizontal line at y=1.0 (thin, dotted, white or light gray).
- X-axis = date, Y-axis = "Growth of $1".
- Dark/clean theme. Show hover values. Add range slider on x-axis.

Section 4: "Underwater / Drawdown"
Plotly area chart showing strategy drawdown over time.
- Fill area in red/salmon with some transparency.
- Y-axis should be negative (e.g., 0% to -40%).
- Same x-axis as above.

Section 5: "Signal Analysis"
Use st.columns(2) to put these side by side:

Left column — "Composite Signal Over Time":
Plotly bar chart of the MONTHLY composite signal.
Color bars green if signal > 0, red if signal < 0.
Y-axis = "Signal [-1, +1]".

Right column — "Signal Contribution Breakdown":
Plotly stacked bar chart showing the weighted contribution of each signal at each month-end:
- TSMOM component = TSMOM_weight × TSMOM_signal (one color, e.g., blue).
- Carry component = Carry_weight × Carry_signal (another color, e.g., orange).
- Overlay the Macro Scale as a line on a secondary y-axis (0 to 1 scale) so the user can see when the filter is reducing exposure.
This is the most important chart for understanding the strategy — it shows which signal is driving the position at any point in time and when the macro filter is active.

Section 6: "Position & Macro Filter Detail"
st.dataframe with the following MONTHLY data (one row per month-end):
Columns: Date, CO1, CO12, Carry Spread (%), TSMOM Signal, Carry Signal, OVX, VIX, OVX/VIX Ratio, Macro Scale, Composite Signal, Monthly Strategy Return (%), Monthly Market Return (%), Cumulative Strategy, Cumulative Buy&Hold.
Format: prices as $XX.XX, percentages as X.XX%, ratios as X.XX, signals as +X.XXX.
Make the dataframe sortable and scrollable. Highlight rows where Macro Scale < 1.0 in light red.
Allow the user to download this table as CSV with st.download_button.

Section 7: "Monthly Returns Heatmap"
A heatmap showing monthly returns by year (rows) and calendar month (columns, Jan-Dec).
Use Plotly heatmap (go.Heatmap) with:
- Green color scale for positive returns, red for negative.
- Use RdYlGn or RdGn diverging colorscale centered at zero.
- Display the actual percentage value as text in each cell.
- Round to 1 decimal place (e.g., "2.3%", "-1.1%").

Section 8: "Rolling Statistics"
Plotly chart with:
- Rolling 252-day (1 year) Sharpe ratio as a solid line, left y-axis.
- Rolling 252-day annualized return as a dashed line, right y-axis.
- Add a horizontal line at Sharpe = 1.0 for reference.
- Show both on the same time axis.

Section 9: "Daily P&L Distribution"
Plotly histogram of daily strategy returns with:
- ~100 bins.
- Vertical line at zero (dashed).
- Vertical line at the mean daily return (solid, colored).
- Add annotations showing: mean, median, skewness (scipy.stats.skew), kurtosis (scipy.stats.kurtosis).
- This helps visualize the return distribution shape and fat tails.

TECHNICAL REQUIREMENTS:
- Use pandas and numpy for all computations.
- Use plotly.graph_objects and plotly.express for all charts. No matplotlib.
- Use scipy.stats for skew and kurtosis in the distribution section.
- Load the CSV with pd.read_csv("bloomberg_data.csv", parse_dates=["Date"]).
- All computations should re-run reactively when any sidebar input changes.
- Use @st.cache_data on the data loading function only (not on computation functions, since they depend on sidebar inputs).
- Use st.set_page_config(layout="wide") at the very top.
- Clean, professional styling throughout. Use plotly template "plotly_dark" for all charts.
- No hardcoded data — everything comes from the CSV.
- If bloomberg_data.csv is not found, show st.error("bloomberg_data.csv not found. Place your Bloomberg data export in the same directory as this app.") and st.stop().
- Name the main file app.py.
- Include a requirements.txt with: streamlit, pandas, numpy, plotly, scipy.

IMPORTANT CODE QUALITY:
- Put signal computation in a function: compute_signals(df, tsmom_weight, carry_weight, lookbacks, ovx_vix_threshold, vix_caution, vix_flatten).
- Put backtest logic in a function: run_backtest(df_signals, tc, vol_target_enabled, vol_target, trailing_stop_enabled, trailing_stop_threshold).
- Put stats computation in a function: compute_stats(df_backtest).
- Keep the main Streamlit layout code clean and readable.
- Add brief comments explaining non-obvious logic.
- Do not truncate or abbreviate any code. Give me the complete app.py in a single code block with no placeholders or "... rest of code here" shortcuts. Every line of code must be present.
