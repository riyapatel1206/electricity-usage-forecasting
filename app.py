"""
Electricity Usage Forecaster — Web App
Run:  streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from forecaster import generate_synthetic, load_data, run_pipeline, DEFAULT_SLABS, FIXED_CHARGE

st.set_page_config(page_title="Electricity Usage Forecaster", page_icon="⚡", layout="wide")

ACTUAL_COLOR = "#2b6cb0"
FORECAST_COLOR = "#dd6b20"


# --------------------------------------------------------------------------- #
# Cached helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def cached_demo():
    return generate_synthetic()


@st.cache_data(show_spinner=False)
def cached_load(file_bytes):
    return load_data(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def cached_pipeline(df, horizon, folds):
    return run_pipeline(df, horizon, folds)


def slab_breakdown(units, slabs, fixed):
    """Returns a per-slab breakdown table and the total bill."""
    rows, prev = [], 0.0
    for limit, rate in slabs:
        if units <= prev:
            break
        used = min(units, limit) - prev
        label = f"{prev:.0f}–{limit:.0f}" if np.isfinite(limit) else f"Above {prev:.0f}"
        rows.append({"Slab (units)": label, "Units": round(used, 1),
                     "Rate (₹/unit)": rate, "Amount (₹)": round(used * rate, 2)})
        prev = limit
    rows.append({"Slab (units)": "Fixed charge", "Units": None,
                 "Rate (₹/unit)": None, "Amount (₹)": fixed})
    table = pd.DataFrame(rows)
    return table, float(table["Amount (₹)"].sum())


def bill(units, slabs, fixed):
    return slab_breakdown(units, slabs, fixed)[1]


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")

    source = st.radio("Data source", ["Demo data", "Upload CSV"], horizontal=True)
    uploaded = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("CSV with columns: date, kwh, temp (optional)", type="csv")
        template = pd.DataFrame({"date": ["2025-01-01", "2025-01-02"],
                                 "kwh": [7.4, 8.1], "temp": [18.2, 19.0]})
        st.download_button("Download CSV template", template.to_csv(index=False),
                           "usage_template.csv", "text/csv")

    st.subheader("Forecast")
    horizon = st.slider("Days to forecast", 7, 60, 14)
    folds = st.slider("Backtest folds", 3, 10, 5,
                      help="How many past periods to test the models on. More = more reliable "
                           "accuracy numbers, but slower.")

    st.subheader("Tariff")
    st.caption("Default rates are illustrative — enter your DISCOM's slab rates.")
    slab_df = st.data_editor(
        pd.DataFrame({"Up to units": [s[0] if np.isfinite(s[0]) else None for s in DEFAULT_SLABS],
                      "₹/unit": [s[1] for s in DEFAULT_SLABS]}),
        num_rows="dynamic", hide_index=True, width="stretch",
        column_config={
            "Up to units": st.column_config.NumberColumn(
                help="Leave the last row empty for 'and above'", width="small"),
            "₹/unit": st.column_config.NumberColumn(format="%.2f", width="small")},
    )
    st.caption("Empty 'Up to' = and above.")
    fixed = st.number_input("Fixed charge (₹/month)", 0.0, 2000.0, float(FIXED_CHARGE), 10.0)
    budget = st.number_input("Monthly budget (₹)", 0.0, 50000.0, 2000.0, 100.0,
                             help="You'll get a warning if the projected bill exceeds this.")

# Clean up the slab table
slab_df = slab_df.dropna(subset=["₹/unit"])
slabs = []
for _, r in slab_df.iterrows():
    lim = r["Up to units"]
    slabs.append((float(lim) if pd.notna(lim) else float("inf"), float(r["₹/unit"])))
slabs.sort(key=lambda s: s[0])
if not slabs or np.isfinite(slabs[-1][0]):
    last_rate = slabs[-1][1] if slabs else 0.0
    slabs.append((float("inf"), last_rate))

# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
st.title("⚡ Electricity Usage Forecaster")
st.caption("Forecast your daily electricity use, see how accurate the forecast is, "
           "and estimate next month's bill.")

if source == "Upload CSV":
    if uploaded is None:
        st.info("👈 Upload a CSV to get started, or switch to **Demo data**.")
        st.stop()
    try:
        df = cached_load(uploaded.getvalue())
    except Exception as e:
        st.error(f"Couldn't read that file: {e}")
        st.stop()
else:
    df = cached_demo()

min_needed = 28 + horizon * folds + 30
if len(df) < min_needed:
    st.error(f"Need at least {min_needed} days of data for a {horizon}-day forecast with "
             f"{folds} backtest folds (you have {len(df)}). Reduce the horizon or folds, "
             f"or add more data.")
    st.stop()

has_temp = "temp" in df.columns

with st.spinner("Training and testing models…"):
    res = cached_pipeline(df, horizon, folds)

bt, best, fc = res["backtest"], res["best_model"], res["forecast"]

# --------------------------------------------------------------------------- #
# KPIs
# --------------------------------------------------------------------------- #
last30 = df["kwh"].tail(30)
fc_daily = fc["forecast_kwh"].mean()
month_units = fc_daily * 30
month_bill = bill(month_units, slabs, fixed)
last30_bill = bill(last30.sum(), slabs, fixed)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Last 30 days", f"{last30.sum():.0f} units", f"≈ ₹{last30_bill:,.0f} bill",
          delta_color="off", delta_arrow="off")
c2.metric("Forecast avg / day", f"{fc_daily:.1f} kWh",
          f"{(fc_daily / last30.mean() - 1) * 100:+.1f}% vs last 30 days", delta_color="inverse")
c3.metric("Projected next 30 days", f"{month_units:.0f} units")
c4.metric("Estimated bill", f"₹{month_bill:,.0f}",
          f"{'-' if month_bill < last30_bill else '+'}₹{abs(month_bill - last30_bill):,.0f} "
          "vs last 30 days", delta_color="inverse")

if budget > 0 and month_bill > budget:
    # How many units/day to cut to get under budget
    lo_u, hi_u = 0.0, month_units
    for _ in range(40):
        mid = (lo_u + hi_u) / 2
        lo_u, hi_u = (mid, hi_u) if bill(mid, slabs, fixed) <= budget else (lo_u, mid)
    cut = (month_units - lo_u) / 30
    st.warning(f"⚠️ Projected bill ₹{month_bill:,.0f} is over your ₹{budget:,.0f} budget. "
               f"Cut about **{cut:.1f} kWh/day** to stay within it "
               f"(e.g. ~{cut / 1.5:.1f} fewer hours of a 1.5-ton AC).")
elif budget > 0:
    st.success(f"✅ On track: projected bill is ₹{budget - month_bill:,.0f} under your budget.")

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_fc, tab_model, tab_insights, tab_bill, tab_data = st.tabs(
    ["📈 Forecast", "🎯 Model accuracy", "🔍 Usage insights", "🧾 Bill breakdown", "📄 Data"])

with tab_fc:
    show_days = st.select_slider("History to show", [30, 60, 90, 180, 365], value=90)
    hist = df.tail(show_days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.concat([fc["date"], fc["date"][::-1]]),
        y=pd.concat([fc["high_80"], fc["low_80"][::-1]]),
        fill="toself", fillcolor="rgba(221,107,32,0.18)", line=dict(width=0),
        name="80% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["kwh"], name="Actual",
                             line=dict(color=ACTUAL_COLOR, width=1.8)))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast_kwh"], name="Forecast",
                             mode="lines+markers", line=dict(color=FORECAST_COLOR, width=2.5),
                             marker=dict(size=5)))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="kWh per day", hovermode="x unified",
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")

    st.caption(f"Model used: **{best}** (picked automatically — lowest error in backtesting). "
               "The shaded band shows where actual usage landed 80% of the time in testing."
               + ("" if not has_temp else " Future temperatures assume the same date last year."))

    show = fc.copy()
    show["date"] = show["date"].dt.strftime("%a, %d %b %Y")
    show = show.rename(columns={"date": "Date", "forecast_kwh": "Forecast (kWh)",
                                "low_80": "Low", "high_80": "High",
                                "assumed_temp": "Assumed temp (°C)"})
    st.dataframe(show, hide_index=True, width="stretch")
    st.download_button("⬇️ Download forecast CSV", fc.to_csv(index=False),
                       "forecast.csv", "text/csv")

with tab_model:
    st.markdown(
        f"Each model was trained on past data only, asked to forecast the next {horizon} days, "
        f"and scored against what actually happened, repeated over {folds} periods. "
        f"**{best}** had the lowest error, **{res['improvement_pct']:.0f}% better** than the "
        f"best simple baseline.")
    table = bt.rename(columns={"MAE": "Avg error (kWh/day)", "RMSE": "RMSE",
                               "MAPE%": "Avg error (%)"})
    st.dataframe(table, width="stretch")
    colors = [FORECAST_COLOR if n == best else
              ("#a0aec0" if ("naive" in n or "average" in n) else ACTUAL_COLOR) for n in bt.index]
    fig = go.Figure(go.Bar(x=bt["MAE"], y=bt.index, orientation="h", marker_color=colors,
                           text=bt["MAE"].round(2), textposition="outside"))
    fig.update_layout(height=280, margin=dict(l=10, r=40, t=10, b=10),
                      xaxis_title="Average error (kWh/day) — lower is better",
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    with st.expander("What do these models do?"):
        st.markdown(
            "- **Seasonal naive** — tomorrow = same weekday last week. A baseline any real model must beat.\n"
            "- **30-day average** — every future day = average of the last 30 days.\n"
            "- **Ridge regression** — linear model on calendar features, past usage and temperature.\n"
            "- **Gradient boosting** — tree ensemble that can capture non-linear effects "
            "(e.g. AC kicking in only above a certain temperature).\n\n"
            "Multi-day forecasts are made recursively: predict tomorrow, feed it back in, "
            "predict the day after, and so on.")

with tab_insights:
    d = df.copy()
    d["weekday"] = d["date"].dt.day_name()
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    col_a, col_b = st.columns(2)
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    wd = d.groupby("weekday")["kwh"].mean().reindex(order).reset_index()
    fig = px.bar(wd, x="weekday", y="kwh", title="Average use by weekday",
                 labels={"kwh": "kWh/day", "weekday": ""}, color_discrete_sequence=[ACTUAL_COLOR])
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    col_a.plotly_chart(fig, width="stretch")

    mo = d.groupby("month")["kwh"].sum().reset_index()
    fig = px.bar(mo, x="month", y="kwh", title="Total units per month",
                 labels={"kwh": "Units", "month": ""}, color_discrete_sequence=[ACTUAL_COLOR])
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    col_b.plotly_chart(fig, width="stretch")

    if has_temp:
        fig = px.scatter(d, x="temp", y="kwh", opacity=0.5,
                         title="Temperature vs daily use",
                         labels={"temp": "Avg temperature (°C)", "kwh": "kWh/day"},
                         color_discrete_sequence=[ACTUAL_COLOR])
        # Trend line: average use in 1 °C bins
        trend = d.groupby(d["temp"].round())["kwh"].mean()
        fig.add_trace(go.Scatter(x=trend.index, y=trend.values, name="Average",
                                 line=dict(color=FORECAST_COLOR, width=3)))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
        hot = d[d["temp"] >= 30]["kwh"].mean()
        mild = d[(d["temp"] >= 20) & (d["temp"] < 26)]["kwh"].mean()
        if np.isfinite(hot) and np.isfinite(mild):
            st.info(f"On hot days (≥30 °C) you use **{hot:.1f} kWh/day** vs **{mild:.1f}** "
                    f"on mild days (20–26 °C), {(hot / mild - 1) * 100:.0f}% more, "
                    "most likely from cooling.")
    else:
        st.caption("Add a `temp` column to your CSV to see how temperature drives your usage.")

    top = d.nlargest(5, "kwh")[["date", "kwh"]].assign(date=lambda x: x["date"].dt.strftime("%d %b %Y"))
    st.markdown("**Highest-use days**")
    st.dataframe(top.rename(columns={"date": "Date", "kwh": "kWh"}), hide_index=True)

with tab_bill:
    st.markdown("Bill estimate for the next 30 days at the forecast pace. "
                "Use the slider to see how the bill changes with usage.")
    what_if = st.slider("Monthly units", 0, int(max(month_units * 2, 100)),
                        int(round(month_units)))
    breakdown, total = slab_breakdown(what_if, slabs, fixed)
    c1, c2 = st.columns([2, 1])
    c1.dataframe(breakdown, hide_index=True, width="stretch")
    c2.metric("Total bill", f"₹{total:,.0f}")
    c2.metric("Effective rate", f"₹{total / max(what_if, 1):.2f}/unit")
    nxt = next((lim for lim, _ in slabs if np.isfinite(lim) and lim > what_if), None)
    if nxt is not None:
        c2.caption(f"{nxt - what_if:.0f} units until the next slab.")

    xs = np.arange(0, int(max(month_units * 2, 100)) + 1, 5)
    fig = go.Figure(go.Scatter(x=xs, y=[bill(x, slabs, fixed) for x in xs],
                               line=dict(color=ACTUAL_COLOR)))
    fig.add_vline(x=what_if, line_dash="dash", line_color=FORECAST_COLOR)
    if budget > 0:
        fig.add_hline(y=budget, line_dash="dot", line_color="#e53e3e",
                      annotation_text="Budget", annotation_position="top left")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Units per month", yaxis_title="Bill (₹)")
    st.plotly_chart(fig, width="stretch")

with tab_data:
    st.markdown(f"**{len(df)} days** · {df['date'].min():%d %b %Y} → {df['date'].max():%d %b %Y}"
                + (" · includes temperature" if has_temp else " · no temperature column"))
    st.dataframe(df.sort_values("date", ascending=False), hide_index=True, width="stretch")
    st.download_button("⬇️ Download cleaned data", df.to_csv(index=False),
                       "cleaned_usage.csv", "text/csv")
