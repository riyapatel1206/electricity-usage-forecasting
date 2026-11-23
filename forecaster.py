"""
Electricity Usage Forecaster
============================
Forecasts daily household electricity usage (kWh), backtests models against
naive baselines, gives prediction intervals, and estimates the upcoming bill
using slab-based tariffs.

Usage:
    python forecaster.py                          # demo on synthetic data
    python forecaster.py --csv my_usage.csv       # your data (columns: date,kwh[,temp])
    python forecaster.py --horizon 30 --folds 8

CSV format:
    date,kwh,temp
    2024-01-01,7.4,18.2
    ...
    `temp` (daily average temperature in °C) is optional but improves accuracy.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LAGS = [1, 2, 7, 14, 28]
ROLL_WINDOWS = [7, 30]

# Illustrative domestic slab tariff (Rs per unit). Replace with your DISCOM's rates.
DEFAULT_SLABS = [(100, 3.0), (200, 4.5), (400, 6.5), (float("inf"), 8.0)]
FIXED_CHARGE = 50.0  # Rs per month


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def generate_synthetic(days=900, seed=42):
    """Synthetic Indian-household data: AC-driven summer peaks, weekend bump,
    festival spikes, temperature dependence, and noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    doy = dates.dayofyear.values
    # Temperature: peak around May (day ~135), coolest in Jan
    temp = 27 + 9 * np.sin(2 * np.pi * (doy - 45) / 365) + rng.normal(0, 1.5, days)
    base = 6.0
    cooling = np.clip(temp - 26, 0, None) * 0.9          # AC load above 26°C
    heating = np.clip(16 - temp, 0, None) * 0.3          # heater/geyser in cold days
    weekend = np.where(dates.dayofweek >= 5, 1.2, 0.0)
    trend = np.linspace(0, 1.0, days)                    # slowly adding appliances
    festival = np.isin(dates.strftime("%m-%d"), ["10-31", "11-01", "11-12", "11-13"]) * 3
    noise = rng.normal(0, 0.7, days)
    kwh = np.clip(base + cooling + heating + weekend + trend + festival + noise, 1, None)
    return pd.DataFrame({"date": dates, "kwh": kwh.round(2), "temp": temp.round(1)})


def load_data(path):
    """Accepts a file path or file-like object (e.g. a Streamlit upload)."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "kwh" not in df.columns:
        raise ValueError("CSV needs columns: date,kwh (optional: temp)")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")
    df = df.dropna(subset=["date"])
    if "temp" in df.columns:
        df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
        if df["temp"].isna().all():
            df = df.drop(columns="temp")
    df = df.sort_values("date").drop_duplicates("date")
    # Fill missing days so lags line up; interpolate gaps
    df = df.set_index("date").asfreq("D")
    df["kwh"] = df["kwh"].interpolate(limit_direction="both")
    if "temp" in df.columns:
        df["temp"] = df["temp"].interpolate(limit_direction="both")
    return df.reset_index()


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def make_features(df):
    out = df.copy()
    d = out["date"]
    out["dow"] = d.dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["month"] = d.dt.month
    out["doy_sin"] = np.sin(2 * np.pi * d.dt.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * d.dt.dayofyear / 365.25)
    for lag in LAGS:
        out[f"lag_{lag}"] = out["kwh"].shift(lag)
    for w in ROLL_WINDOWS:
        out[f"roll_mean_{w}"] = out["kwh"].shift(1).rolling(w).mean()
    out["roll_std_7"] = out["kwh"].shift(1).rolling(7).std()
    if "temp" in out.columns:
        out["cooling_deg"] = np.clip(out["temp"] - 24, 0, None)
        out["heating_deg"] = np.clip(18 - out["temp"], 0, None)
    return out


def feature_cols(df):
    cols = ["dow", "is_weekend", "month", "doy_sin", "doy_cos", "roll_std_7"]
    cols += [f"lag_{l}" for l in LAGS] + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    if "temp" in df.columns:
        cols += ["temp", "cooling_deg", "heating_deg"]
    return cols


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class SeasonalNaive:
    """Forecast = value from the same weekday last week."""
    name = "Seasonal naive (last week)"

    def forecast(self, history, future):
        last_week = history["kwh"].values[-7:]
        return np.array([last_week[i % 7] for i in range(len(future))])


class MovingAverage:
    name = "30-day average"

    def forecast(self, history, future):
        return np.full(len(future), history["kwh"].values[-30:].mean())


class MLForecaster:
    """Recursive multi-step forecaster: predicts day t+1, appends it to the
    history, recomputes lag features, predicts t+2, and so on."""

    def __init__(self, kind="gbm"):
        self.kind = kind
        self.name = {"gbm": "Gradient boosting", "ridge": "Ridge regression"}[kind]

    def _new_model(self):
        if self.kind == "ridge":
            return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                             max_leaf_nodes=15, min_samples_leaf=10,
                                             random_state=0)

    def fit(self, history):
        feats = make_features(history).dropna()
        self.cols = feature_cols(history)
        X, y = feats[self.cols], feats["kwh"]
        self.model = self._new_model().fit(X, y)
        return self

    def forecast(self, history, future):
        """future: DataFrame with 'date' (and 'temp' if used) for the horizon."""
        hist = history.copy()
        preds = []
        for _, row in future.iterrows():
            new = {"date": row["date"], "kwh": np.nan}
            if "temp" in hist.columns:
                new["temp"] = row.get("temp", hist["temp"].iloc[-7:].mean())
            hist = pd.concat([hist, pd.DataFrame([new])], ignore_index=True)
            x = make_features(hist).iloc[[-1]][self.cols]
            p = float(self.model.predict(x)[0])
            preds.append(p)
            hist.loc[hist.index[-1], "kwh"] = p
        return np.array(preds)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def metrics(actual, pred):
    err = actual - pred
    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "MAPE%": np.mean(np.abs(err) / np.maximum(actual, 1e-6)) * 100,
    }


MODEL_FACTORIES = [SeasonalNaive, MovingAverage,
                   lambda: MLForecaster("ridge"), lambda: MLForecaster("gbm")]


def backtest(df, horizon=7, folds=6):
    """Rolling-origin backtest: train on data up to a cutoff, forecast the next
    `horizon` days, move the cutoff forward, repeat. No future data leaks in."""
    results, residuals, factories = {}, {}, {}
    n = len(df)
    cutoffs = [n - horizon * (folds - i) for i in range(folds)]
    for model_factory in MODEL_FACTORIES:
        all_a, all_p, name = [], [], None
        for c in cutoffs:
            train, test = df.iloc[:c], df.iloc[c:c + horizon]
            m = model_factory()
            name = m.name
            if hasattr(m, "fit"):
                m.fit(train)
            p = m.forecast(train, test[[c_ for c_ in test.columns if c_ != "kwh"]])
            all_a.extend(test["kwh"].values)
            all_p.extend(p)
        results[name] = metrics(np.array(all_a), np.array(all_p))
        residuals[name] = np.array(all_a) - np.array(all_p)
        factories[name] = model_factory
    table = pd.DataFrame(results).T.round(3).sort_values("MAE")
    return table, residuals, factories


# --------------------------------------------------------------------------- #
# Billing
# --------------------------------------------------------------------------- #
def slab_bill(units, slabs=DEFAULT_SLABS, fixed=FIXED_CHARGE):
    bill, prev = fixed, 0
    for limit, rate in slabs:
        if units <= prev:
            break
        used = min(units, limit) - prev
        bill += used * rate
        prev = limit
    return bill


# --------------------------------------------------------------------------- #
# Full pipeline (used by both the CLI and the web app)
# --------------------------------------------------------------------------- #
def run_pipeline(df, horizon=7, folds=6, future_temps=None):
    """Backtest all models, pick the best, retrain on all data, forecast.
    Returns a dict with the backtest table, best model name, and forecast frame."""
    bt, residuals, factories = backtest(df, horizon, folds)
    best_name = bt.index[0]
    best_baseline = bt.loc[bt.index.str.contains("naive|average"), "MAE"].min()
    improvement = (1 - bt.loc[best_name, "MAE"] / best_baseline) * 100

    model = factories[best_name]()
    if hasattr(model, "fit"):
        model.fit(df)
    future_dates = pd.date_range(df["date"].max() + pd.Timedelta(days=1),
                                 periods=horizon, freq="D")
    future = pd.DataFrame({"date": future_dates})
    if "temp" in df.columns:
        if future_temps is not None:
            future["temp"] = np.asarray(future_temps)[:horizon]
        else:
            # Without a weather forecast, assume same-calendar-day temp from last year
            temps = df.set_index("date")["temp"]
            fallback = df["temp"].iloc[-7:].mean()
            future["temp"] = [temps.get(d - pd.DateOffset(years=1), fallback)
                              for d in future_dates]
    pred = model.forecast(df, future)
    r_lo, r_hi = np.percentile(residuals[best_name], [10, 90])
    fc = pd.DataFrame({
        "date": future_dates,
        "forecast_kwh": pred.round(2),
        "low_80": np.clip(pred + r_lo, 0, None).round(2),
        "high_80": (pred + r_hi).round(2),
    })
    if "temp" in future.columns:
        fc["assumed_temp"] = np.round(future["temp"].values, 1)
    return {"backtest": bt, "best_model": best_name,
            "improvement_pct": improvement, "forecast": fc}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Electricity usage forecaster")
    ap.add_argument("--csv", help="CSV with date,kwh[,temp]. Omit to use synthetic demo data.")
    ap.add_argument("--horizon", type=int, default=7, help="Days to forecast (default 7)")
    ap.add_argument("--folds", type=int, default=6, help="Backtest folds (default 6)")
    ap.add_argument("--out", default="output", help="Output folder")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_data(args.csv) if args.csv else generate_synthetic()
    print(f"Loaded {len(df)} days: {df['date'].min().date()} → {df['date'].max().date()}")
    if len(df) < 90:
        print("Warning: under 90 days of data — forecasts will be rough.")

    # 1–2. Backtest, pick best model, forecast
    print(f"\nBacktest ({args.folds} folds × {args.horizon}-day horizon):")
    res = run_pipeline(df, args.horizon, args.folds)
    bt, best_name, fc = res["backtest"], res["best_model"], res["forecast"]
    print(bt.to_string())
    bt.to_csv(os.path.join(args.out, "backtest_results.csv"))
    print(f"\nBest model: {best_name} — MAE {bt.loc[best_name, 'MAE']:.2f} kWh/day, "
          f"{res['improvement_pct']:.1f}% better than the best baseline.")
    fc.to_csv(os.path.join(args.out, "forecast.csv"), index=False)
    print(f"\nForecast for next {args.horizon} days:")
    print(fc.assign(date=fc["date"].dt.date).to_string(index=False))
    future_dates, pred = fc["date"], fc["forecast_kwh"].values
    lo, hi = fc["low_80"].values, fc["high_80"].values

    # 3. Monthly bill estimate at the forecast pace
    month_units = pred.mean() * 30
    print(f"\nAt the forecast pace, 30-day usage ≈ {month_units:.0f} units")
    print(f"Estimated monthly bill: Rs {slab_bill(month_units):,.0f}")
    print("  (illustrative slab tariff — edit DEFAULT_SLABS for your DISCOM)")

    # 4. Plot
    hist_tail = df.tail(90)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hist_tail["date"], hist_tail["kwh"], label="Actual", color="#2b6cb0", lw=1.5)
    ax.plot(future_dates, pred, label="Forecast", color="#dd6b20", lw=2, marker="o", ms=4)
    ax.fill_between(future_dates, lo, hi, color="#dd6b20", alpha=0.2, label="80% interval")
    ax.set_title("Daily electricity usage: last 90 days + forecast")
    ax.set_ylabel("kWh per day")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(args.out, "forecast_plot.png")
    fig.savefig(path, dpi=130)
    print(f"\nSaved: {args.out}/forecast.csv, backtest_results.csv, forecast_plot.png")


if __name__ == "__main__":
    main()
