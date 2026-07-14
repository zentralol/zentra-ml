"""
inference.py
------------
Loads all trained models once when the server starts.
Has one public function: run(lat, lon, when, task)

task = "crowd"   → GradientBoosting + LightGBM + RandomForest blend
task = "score"   → same blend, result converted to 0-100 score
task = "future"  → GradientBoosting + LightGBM only (no lag history needed)
"""

import json
import math
from datetime import date, datetime
from pathlib import Path

import h3
import holidays
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from astral import LocationInfo
from astral.sun import sun as astral_sun
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from tzutil import now_in_manhattan, to_manhattan_time

# Where are the model files? 
MODELS = Path("../models")
DATA   = Path("../data/master")
PROC   = Path("../data/processed")

# Manhattan location (used for weather + sunrise/sunset) 
MANHATTAN_LAT = 40.7831
MANHATTAN_LON = -73.9712

# What hour ranges map to which "period" label? 
PERIOD_MAP = {
    "EARLY": range(0, 7),
    "AM":    range(7, 10),
    "MD":    range(10, 14),
    "PM":    range(14, 18),
    "EVE":   range(18, 22),
    "NIGHT": range(22, 24),
}

# Crowd score labels 
SCORE_LABELS = [
    (20,  "Quiet"),
    (40,  "Light"),
    (60,  "Moderate"),
    (80,  "Busy"),
    (101, "Very Busy"),
]

# LOAD EVERYTHING ONCE AT STARTUP
print("Loading models...")

# Feature metadata (columns, medians for imputation, etc.)
bundle       = joblib.load(MODELS / "inference_bundle.joblib")
FEATURE_COLS = bundle["feature_cols"]
MEDIANS      = pd.Series(bundle["train_medians"])
MISSING_COLS = bundle["missing_flag_cols"]   # columns that get a _missing flag
DUMMY_GROUPS = bundle["dummy_groups"]         # categorical one-hot groups

# The 3 trained models
lgb_model  = lgb.Booster(model_file=str(MODELS / "lgbm_tuned.txt"))
gb_model   = joblib.load(MODELS / "gb_tuned.pkl")
rf_model   = joblib.load(MODELS / "rf_tuned.pkl")
rf_imputer = joblib.load(MODELS / "rf_imputer.pkl")   # median imputer for RF

# Ensemble blend weights (saved by the notebook)
weights_path = MODELS / "ensemble_weights.json"
if weights_path.exists():
    WEIGHTS = json.loads(weights_path.read_text())
else:
    # Fallback: equal weights
    WEIGHTS = {
        "crowd":  {"lgb": 0.33, "gb": 0.34, "rf": 0.33},
        "future": {"gb": 0.6,   "lgb": 0.4},
        "hist_p95_log": 8.0,
    }

HIST_P95 = float(WEIGHTS["hist_p95_log"])   # 95th pct of log_pedestrians in training

# Grid and POI lookup tables
grid_df        = pd.read_csv(DATA / "manhattan_grid_h3.csv")
poi_hex_counts = pd.read_csv(DATA / "poi_hex_counts.csv")

GRID_LOOKUP = grid_df.set_index("h3_cell")
POI_LOOKUP  = poi_hex_counts.set_index("h3_cell")

# Corridor-level lag baselines — used for future predictions instead of 0
# Loaded from the training master: median ped_lag1 per (h3_cell, period)
LAG_BASELINE     = {}   # {(h3_cell, period): {lag_col: median_value}}
TRAINED_CELLS    = set()  # h3 cells that actually appear in the training data
PROXY_SCORE      = {}   # {h3_cell: crowd_score_proxy} — POI+mobility score, all 524 cells
MEAN_TRAIN_PROXY = 50.0  # fallback; overwritten below

_master_path = DATA / "zentra_training_master.csv"
if _master_path.exists():
    _m = pd.read_csv(_master_path)
    _lag_cols = [c for c in _m.columns if c.startswith("ped_lag") or
                 c.startswith("ped_roll") or c.startswith("corridor_") or c == "ped_yoy_ratio"]
    if _lag_cols and "h3_cell" in _m.columns and "period" in _m.columns:
        _grp = _m.groupby(["h3_cell", "period"])[_lag_cols].median()
        LAG_BASELINE  = _grp.to_dict(orient="index")
        TRAINED_CELLS = set(_m["h3_cell"].unique())
        print(f"  Lag baselines loaded for {len(LAG_BASELINE)} (cell, period) pairs")
        print(f"  Training corridors: {len(TRAINED_CELLS)} unique cells")

# Proxy crowd scores for ALL 524 Manhattan cells (POI + mobility signals)
# Same formula as data processing notebook Section 11:
#   crowd_score_proxy = 0.28*tlc + 0.28*mta + 0.14*citibike + 0.20*poi + 0.10*events
_live_path = DATA / "zentra_live_grid_snapshot.csv"
if _live_path.exists():
    _live = pd.read_csv(_live_path)
    if "h3_cell" in _live.columns:
        PROXY_WEIGHTS = {
            "tlc_load_score":      0.28,
            "mta_load_score":      0.28,
            "citibike_load_score": 0.14,
            "poi_density_score":   0.20,
            "event_intensity_score": 0.10,
        }
        _live["crowd_score_proxy"] = sum(
            _live[col].fillna(0) * w
            for col, w in PROXY_WEIGHTS.items()
            if col in _live.columns
        ).clip(0, 100)
        PROXY_SCORE = _live.groupby("h3_cell")["crowd_score_proxy"].mean().to_dict()
        train_proxies = [PROXY_SCORE[c] for c in TRAINED_CELLS if c in PROXY_SCORE]
        MEAN_TRAIN_PROXY = float(np.mean(train_proxies)) if train_proxies else 50.0
        print(f"  Proxy scores computed for {len(PROXY_SCORE)} cells "
              f"(mean training proxy={MEAN_TRAIN_PROXY:.1f})")

# Reference mobility signals: typical TLC / MTA / Citi Bike per cell+period
def _load_ref(filename, cols):
    """Load a reference CSV and average per (h3_cell, period)."""
    for path in [PROC / filename.replace("_reference", "_all_dates"), PROC / filename]:
        if path.exists():
            df  = pd.read_csv(path, usecols=["h3_cell", "period"] + cols)
            avg = df.groupby(["h3_cell", "period"])[cols].mean()
            print(f"  Loaded {path.name} ({len(avg)} rows)")
            return avg
    print(f"  WARNING: {filename} not found")
    return None

TLC_REF = _load_ref("tlc_h3_reference.csv",
                    ["tlc_yellow_trips", "tlc_hvfhv_trips", "tlc_trip_count", "tlc_load_score"])
MTA_REF = _load_ref("mta_h3_reference.csv",
                    ["mta_ridership_total", "mta_ridership_avg", "mta_load_score"])
CB_REF  = _load_ref("citibike_h3_reference.csv",
                    ["citibike_trip_count", "citibike_load_score"])

# Seasonal weather averages (used when live forecast is unavailable)
WEATHER_SEASONAL = None
weather_hist = PROC / "weather_all_dates.csv"
if weather_hist.exists():
    w = pd.read_csv(weather_hist, parse_dates=["survey_date"])
    w["month"] = w["survey_date"].dt.month
    WEATHER_SEASONAL = w.groupby(["month", "period"])[
        ["avg_temperature_f", "precip_mm", "avg_wind_kmh"]
    ].mean()

US_HOLIDAYS = holidays.UnitedStates(years=range(2007, now_in_manhattan().year + 3))

print(f"Ready — {len(FEATURE_COLS)} features, {len(grid_df)} grid cells.")


# SMALL HELPERS
def get_period(hour: int) -> str:
    """Turn an hour (0-23) into a period name like AM, PM, EVE."""
    for name, hours in PERIOD_MAP.items():
        if hour in hours:
            return name
    return "NIGHT"


def get_score_label(score: float) -> str:
    """Turn a 0-100 crowd score into a human label."""
    for upper, label in SCORE_LABELS:
        if score < upper:
            return label
    return "Very Busy"


def snap_to_grid(lat: float, lon: float) -> str:
    """Find the nearest Manhattan H3 cell for a lat/lon point."""
    cell = h3.latlng_to_cell(lat, lon, 9)   # resolution 9
    if cell in GRID_LOOKUP.index:
        return cell
    # GPS drift — search nearby rings until we find a valid cell
    for ring in range(1, 4):
        nearby = [c for c in h3.grid_disk(cell, ring) if c in GRID_LOOKUP.index]
        if nearby:
            return nearby[0]
    return cell   # last resort fallback


def get_sunlight(day: date) -> dict:
    """Sunrise hour, sunset hour, day length for Manhattan."""
    loc = LocationInfo("Manhattan", "USA", "America/New_York", MANHATTAN_LAT, MANHATTAN_LON)
    s   = astral_sun(loc.observer, date=day, tzinfo="America/New_York")
    return {
        "sunrise_hour":  s["sunrise"].hour + s["sunrise"].minute / 60,
        "sunset_hour":   s["sunset"].hour  + s["sunset"].minute  / 60,
        "day_length_hr": round((s["sunset"] - s["sunrise"]).total_seconds() / 3600, 2),
    }


def is_school_break(d) -> int:
    """1 if NYC schools are on break, 0 otherwise."""
    m, day = d.month, d.day
    if m in (7, 8):             return 1
    if m == 12 and day >= 22:   return 1
    if m == 1  and day <= 2:    return 1
    return 0


def get_calendar_features(d) -> dict:
    """Day of week, weekend flag, holiday flag, school break, etc."""
    d = pd.Timestamp(d)
    return {
        "day_of_week":     d.dayofweek,
        "is_weekend":      int(d.dayofweek >= 5),
        "is_holiday":      int(d.date() in US_HOLIDAYS),
        "is_school_break": is_school_break(d),
        "week_of_year":    int(d.isocalendar().week),
        "day_of_year":     d.dayofyear,
    }


def get_weather(when: pd.Timestamp, period: str) -> dict:
    """
    Try 3 sources in order:
    1. Cached forecast CSV (written by data notebook)
    2. Live Open-Meteo API (works up to 16 days ahead)
    3. Seasonal historical average (fallback)
    """
    # 1. Cached forecast file
    fc_path = PROC / "weather_forecast.csv"
    if fc_path.exists():
        fc    = pd.read_csv(fc_path, parse_dates=["survey_date"])
        match = fc[(fc["survey_date"].dt.date == when.date()) & (fc["period"] == period)]
        if not match.empty:
            r = match.iloc[0]
            return {
                "avg_temperature_f": r.get("avg_temperature_f", np.nan),
                "precip_mm":         r.get("precip_mm", np.nan),
                "avg_wind_kmh":      r.get("avg_wind_kmh", np.nan),
            }

    # 2. Live forecast API (only for dates within 16 days)
    today_ny = now_in_manhattan().date()
    days_out = (when.date() - today_ny).days
    if 0 <= days_out <= 16:
        try:
            resp = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": MANHATTAN_LAT, "longitude": MANHATTAN_LON,
                    "forecast_days": min(days_out + 1, 16),
                    "hourly": "temperature_2m,precipitation,wind_speed_10m",
                    "temperature_unit": "fahrenheit",
                    "timezone": "America/New_York",
                },
                timeout=10,
            )
            resp.raise_for_status()
            wdf = pd.DataFrame(resp.json()["hourly"])
            wdf["time"]   = pd.to_datetime(wdf["time"])
            wdf["period"] = wdf["time"].dt.hour.apply(get_period)
            wdf["date"]   = wdf["time"].dt.date
            match = wdf[(wdf["date"] == when.date()) & (wdf["period"] == period)]
            if not match.empty:
                return {
                    "avg_temperature_f": match["temperature_2m"].mean(),
                    "precip_mm":         match["precipitation"].sum(),
                    "avg_wind_kmh":      match["wind_speed_10m"].mean(),
                }
        except Exception:
            pass   # silently fall through to seasonal average

    # 3. Seasonal average fallback
    if WEATHER_SEASONAL is not None:
        key = (when.month, period)
        if key in WEATHER_SEASONAL.index:
            r = WEATHER_SEASONAL.loc[key]
            return {
                "avg_temperature_f": r.get("avg_temperature_f", np.nan),
                "precip_mm":         r.get("precip_mm", np.nan),
                "avg_wind_kmh":      r.get("avg_wind_kmh", np.nan),
            }

    return {}

# FEATURE BUILDER
def build_features(lat: float, lon: float, when: datetime, zero_lags: bool = False) -> tuple:
    """
    Build one feature row for the given location + time.

    Returns:
        row    - dict of all raw features
        cell   - the H3 cell the point snapped to
        period - time period (AM / PM / EVE etc.)
    """
    when = to_manhattan_time(when)
    when = pd.Timestamp(when)
    cell = snap_to_grid(lat, lon)
    period = get_period(when.hour)

    row = {"h3_cell": cell, "lat": lat, "lon": lon, "period": period}

    # POI counts (how many restaurants, subway stops, etc. nearby)
    if cell in POI_LOOKUP.index:
        for col in POI_LOOKUP.columns:
            row[col] = POI_LOOKUP.loc[cell, col]
    else:
        for col in poi_hex_counts.columns:
            if col != "h3_cell":
                row[col] = 0

    # Typical mobility signals for this cell + time of day
    for ref, cols in [
        (TLC_REF, ["tlc_yellow_trips", "tlc_hvfhv_trips", "tlc_trip_count", "tlc_load_score"]),
        (MTA_REF, ["mta_ridership_total", "mta_ridership_avg", "mta_load_score"]),
        (CB_REF,  ["citibike_trip_count", "citibike_load_score"]),
    ]:
        if ref is not None and (cell, period) in ref.index:
            for col in cols:
                if col in ref.columns:
                    row[col] = ref.loc[(cell, period), col]
        else:
            for col in cols:
                row[col] = 0

    # Weather, sunlight, calendar
    row.update(get_weather(when, period))
    row.update(get_sunlight(when.date()))
    row.update(get_calendar_features(when))

    # Event score — not available at inference time
    row["event_intensity_score"] = 0.0

    # Cyclic time encodings
    period_num = {"EARLY": 0, "AM": 1, "MD": 2, "PM": 3, "EVE": 4, "NIGHT": 5}.get(period, 0)
    row["month_sin"]  = np.sin(2 * np.pi * when.month / 12)
    row["month_cos"]  = np.cos(2 * np.pi * when.month / 12)
    row["period_sin"] = np.sin(2 * np.pi * period_num / 6)
    row["period_cos"] = np.cos(2 * np.pi * period_num / 6)

    # Lag features — use this corridor's own historical median if available
    lag_prefixes = ("ped_lag", "ped_roll", "ped_yoy", "corridor_")
    baseline = LAG_BASELINE.get((cell, period), {})
    for col in FEATURE_COLS:
        if any(col.startswith(p) for p in lag_prefixes):
            if col not in row:
                if baseline:
                    row[col] = baseline.get(col, 0.0)   # this cell's own history
                else:
                    row[col] = 0.0   # unknown cell — lags stay 0, proxy scaling handles it

    # Convert numpy types to plain Python so FastAPI can serialise them
    row = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
    return row, cell, period


def align_to_feature_cols(raw_row: dict) -> pd.DataFrame:
    """
    Turn a raw feature dict into a DataFrame with exactly the columns
    the model was trained on, filling missing ones with training medians.
    """
    out = {}
    for col in FEATURE_COLS:
        if col in raw_row:
            out[col] = raw_row[col]
        elif col in MISSING_COLS:
            out[col] = 1          # signal that this feature was absent
        else:
            out[col] = MEDIANS.get(col, 0.0)

    # If a whole categorical group is missing, set its _nan dummy to 1
    for cat, dummy_cols in DUMMY_GROUPS.items():
        if dummy_cols and not any(d in raw_row for d in dummy_cols):
            nan_col = f"{cat}_nan"
            if nan_col in out:
                for d in dummy_cols:
                    out[d] = 0
                out[nan_col] = 1

    df = pd.DataFrame([out], columns=FEATURE_COLS)
    df = df.fillna(MEDIANS.reindex(FEATURE_COLS)).fillna(0.0)
    return df


# PREDICTION LOGIC
def predict_crowd(df: pd.DataFrame) -> float:
    """
    3-model blend: HistGradientBoosting + LightGBM + RandomForest.
    Returns log_pedestrians (log scale).
    """
    w     = WEIGHTS["crowd"]
    lgb_p = float(lgb_model.predict(df)[0])
    gb_p  = float(gb_model.predict(df)[0])                           # handles NaN natively
    rf_p  = float(rf_model.predict(rf_imputer.transform(df))[0])     # needs imputed input
    return w["lgb"] * lgb_p + w["gb"] * gb_p + w["rf"] * rf_p


def predict_future(df: pd.DataFrame) -> float:
    """
    2-model blend: HistGradientBoosting + LightGBM.
    Used for future dates where lag history is unavailable.
    """
    w     = WEIGHTS["future"]
    lgb_p = float(lgb_model.predict(df)[0])
    gb_p  = float(gb_model.predict(df)[0])
    return w["lgb"] * lgb_p + w["gb"] * gb_p


def log_to_crowd_score(log_pred: float, cell: str = "") -> float:
    """
    Convert log_pedestrians to a 0-100 crowd score.
    For cells outside the training corridors, scale by the cell's proxy score
    relative to the average training corridor — prevents unknown cells from
    inheriting busy-corridor bias.
    """
    raw_score = log_pred / HIST_P95 * 100

    # Apply proxy scaling for cells with no training data
    if cell and cell not in TRAINED_CELLS and PROXY_SCORE:
        cell_proxy  = PROXY_SCORE.get(cell, MEAN_TRAIN_PROXY)
        scale       = cell_proxy / MEAN_TRAIN_PROXY if MEAN_TRAIN_PROXY > 0 else 1.0
        raw_score   = raw_score * scale

    return round(min(max(raw_score, 0.0), 100.0), 1)


# PUBLIC FUNCTION — called by all route handlers
def run(lat: float, lon: float, when: datetime, task: str = "crowd") -> dict:
    """
    Main entry point for route handlers.

    task options:
        "crowd"  → current/past crowd prediction (uses all 3 models + lag features)
        "score"  → same as crowd but returns score-only fields
        "future" → future crowd prediction (GB + LGB only, lags zeroed)
    """
    zero_lags = (task == "future")
    raw_row, cell, period = build_features(lat, lon, when, zero_lags=zero_lags)
    feature_df = align_to_feature_cols(raw_row)

    if task == "future":
        log_pred = predict_future(feature_df)
    else:
        log_pred = predict_crowd(feature_df)

    crowd_score = log_to_crowd_score(log_pred, cell=cell)

    return {
        "h3_cell":        cell,
        "period":         period,
        "pedestrians":    round(max(0.0, math.expm1(log_pred)), 1),
        "crowd_score":    crowd_score,
        "crowd_category": get_score_label(crowd_score),
    }
