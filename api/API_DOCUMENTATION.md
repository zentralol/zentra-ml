# Zentra Crowd Prediction API — Documentation

**Version:** 1.0  
**Base URL:** `http://localhost:8000` (development)  
**Interactive Docs:** `http://localhost:8000/docs`

---

## Overview

The Zentra Crowd Prediction API predicts pedestrian crowd levels at any location in Manhattan. It accepts a latitude/longitude and a timestamp, and returns a predicted pedestrian count, a 0–100 crowd score, and a crowd category label.

**Coverage:** Manhattan only (lat 40.679–40.882, lon -74.020 to -73.907)

**Three prediction modes:**
| Mode | Use case |
|------|----------|
| Current crowd | What is it like right now (or at a past time)? |
| Score only | Lightweight — just the score + label for a badge or map pin |
| Future crowd | What will it be like next weekend / next month? |

---

## How to Start the Server

```bash
cd api
uvicorn main:app --reload --port 8000
```

On startup the server prints what was loaded:
```
Loading models...
  Lag baselines loaded for 90 (cell, period) pairs
  Training corridors: 30 unique cells
  Proxy scores computed for 524 cells (mean training proxy=19.7)
  Loaded tlc_h3_reference.csv (...)
  Loaded mta_h3_reference.csv (...)
  Loaded citibike_h3_reference.csv (...)
Ready — 81 features, 524 grid cells.
```

---

## Endpoints

### 1. Health Check

```
GET /
```

Returns server status and a summary of what data was loaded at startup.

**Response:**
```json
{
  "message": "Zentra Crowd Prediction API is running!",
  "docs": "/docs",
  "loaded": {
    "trained_cells": 30,
    "lag_baselines": 90,
    "proxy_cells": 524,
    "mean_train_proxy": 19.7,
    "feature_cols": 81
  }
}
```

---

### 2. Current Crowd Prediction

```
POST /predict/crowd
```

Predicts the crowd level at a location right now, or at a specific past/present time. Uses a 3-model ensemble (LightGBM + HistGradientBoosting + RandomForest).

**Request Body:**
```json
{
  "lat":  40.7580,
  "lon": -73.9855,
  "when": "2026-06-30T15:00:00"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `lat` | float | Yes | Latitude (40.679 – 40.882) |
| `lon` | float | Yes | Longitude (-74.020 – -73.907) |
| `when` | datetime (ISO 8601) | No | Defaults to current time if omitted |

**Response:**
```json
{
  "lat": 40.758,
  "lon": -73.9855,
  "h3_cell": "892a100d647ffff",
  "period": "PM",
  "timestamp": "2026-06-30T15:00:00",
  "pedestrians": 8423.5,
  "crowd_score": 76.2,
  "crowd_category": "Busy"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `lat` | float | Echoed from request |
| `lon` | float | Echoed from request |
| `h3_cell` | string | H3 resolution-9 grid cell ID (Uber H3 format) |
| `period` | string | Time bucket (see Period Labels below) |
| `timestamp` | string | ISO datetime the prediction was made for |
| `pedestrians` | float | Predicted pedestrian count |
| `crowd_score` | float | 0–100 crowd intensity score |
| `crowd_category` | string | Human label (see Crowd Categories below) |

**Example cURL:**
```bash
curl -X POST "http://localhost:8000/predict/crowd" \
  -H "Content-Type: application/json" \
  -d '{"lat": 40.7580, "lon": -73.9855}'
```

---

### 3. Score Only (Lightweight)

```
GET /predict/crowd-score?lat=...&lon=...&when=...
```

Returns only the crowd score and category — no pedestrian count. Faster and lighter, good for map pins, badges, or dashboards.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `lat` | float | Yes | Latitude |
| `lon` | float | Yes | Longitude |
| `when` | datetime (ISO 8601) | No | Defaults to now |

**Response:**
```json
{
  "lat": 40.7580,
  "lon": -73.9855,
  "h3_cell": "892a100d647ffff",
  "period": "PM",
  "timestamp": "2026-06-30T15:00:00",
  "crowd_score": 76.2,
  "crowd_category": "Busy"
}
```

**Example cURL:**
```bash
curl "http://localhost:8000/predict/crowd-score?lat=40.7580&lon=-73.9855"
```

---

### 4. Future Crowd Prediction

```
POST /predict/future
```

Predicts crowd levels for a future date and time. Uses a 2-model ensemble (LightGBM + HistGradientBoosting). Weather data comes from the live Open-Meteo forecast API if within 16 days, or from seasonal historical averages for dates further out.

> **Note:** The `when` field **must be in the future**. For current/past times use `/predict/crowd`.

**Request Body:**
```json
{
  "lat":  40.7580,
  "lon": -73.9855,
  "when": "2026-07-04T20:00:00"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `lat` | float | Yes | Latitude |
| `lon` | float | Yes | Longitude |
| `when` | datetime (ISO 8601) | Yes | Must be in the future |

**Response:**
```json
{
  "lat": 40.758,
  "lon": -73.9855,
  "h3_cell": "892a100d647ffff",
  "period": "EVE",
  "timestamp": "2026-07-04T20:00:00",
  "pedestrians": 9102.1,
  "crowd_score": 82.4,
  "crowd_category": "Very Busy",
  "days_ahead": 4.21,
  "weather_source": "forecast"
}
```

Extra fields compared to `/predict/crowd`:

| Field | Type | Description |
|-------|------|-------------|
| `days_ahead` | float | How many days from now this prediction is |
| `weather_source` | string | `"forecast"` (live API, ≤16 days) or `"seasonal_average"` (historical fallback) |

**Example cURL:**
```bash
curl -X POST "http://localhost:8000/predict/future" \
  -H "Content-Type: application/json" \
  -d '{"lat": 40.7580, "lon": -73.9855, "when": "2026-07-04T20:00:00"}'
```

---

### 5. Debug (Developer Only)

```
GET /predict/debug?lat=...&lon=...
```

Shows per-model predictions and blend weights for a location. Useful for frontend developers to verify the model is working correctly.

> Remove or restrict this endpoint before going to production.

**Example cURL:**
```bash
curl "http://localhost:8000/predict/debug?lat=40.7580&lon=-73.9855"
```

**Response:**
```json
{
  "cell": "892a100d647ffff",
  "period": "PM",
  "model_predictions": {
    "lgb_log": 8.9312, "lgb_raw": 7564.0,
    "gb_log":  9.0241, "gb_raw":  8248.1,
    "rf_log":  8.8833, "rf_raw":  7192.3,
    "ensemble_log": 8.9421, "ensemble_raw": 7647.2
  },
  "crowd_score": 76.2,
  "blend_weights": {"lgb": 0.31, "gb": 0.42, "rf": 0.27},
  "feature_count": 81,
  "features_present": 74
}
```

---

## Reference Data

### Period Labels

The API divides the day into 6 named time buckets:

| Period | Hours |
|--------|-------|
| `EARLY` | 12am – 6am |
| `AM` | 7am – 9am |
| `MD` | 10am – 1pm |
| `PM` | 2pm – 5pm |
| `EVE` | 6pm – 9pm |
| `NIGHT` | 10pm – 11pm |

### Crowd Categories

| crowd_score | crowd_category |
|-------------|----------------|
| 0 – 19 | Quiet |
| 20 – 39 | Light |
| 40 – 59 | Moderate |
| 60 – 79 | Busy |
| 80 – 100 | Very Busy |

---

## Error Responses

All errors follow standard HTTP status codes with a JSON detail field.

| Status | Cause |
|--------|-------|
| `422 Unprocessable Entity` | Missing required fields, out-of-range lat/lon, or `when` is not in the future (for `/predict/future`) |
| `500 Internal Server Error` | Model inference failure (check `/predict/debug` for diagnostics) |

**Example 422:**
```json
{
  "detail": [
    {
      "loc": ["body", "when"],
      "msg": "'when' must be in the future. For current predictions use /predict/crowd.",
      "type": "value_error"
    }
  ]
}
```

---

## File Structure

```
api/
├── main.py        # FastAPI app setup + health check endpoint
├── inference.py   # Model loading + feature engineering + prediction logic
├── predict.py     # Route definitions (request/response models + endpoint handlers)
```

### main.py
Creates the FastAPI app, attaches the router from `predict.py`, and exposes the root health check at `GET /`. Also imports `inference` so the health check can report what was loaded.

### inference.py
Loaded once at server startup. Contains:
- Model loading (LightGBM, HistGradientBoosting, RandomForest + imputer)
- Feature builders (POI lookup, mobility signals, weather, calendar, lag features)
- `run(lat, lon, when, task)` — the single public function called by all route handlers

### predict.py
All FastAPI route logic. Contains Pydantic request/response models and 4 endpoint handlers that call `inference.run()`.

---

## Data Sources Used

| Source | What it provides |
|--------|-----------------|
| NYC DOT Pedestrian Counts | Ground truth pedestrian counts (30 Manhattan corridors, 2007–2025) |
| TLC Trip Records | Yellow cab + rideshare pickup/dropoff volumes per H3 cell |
| MTA Subway Ridership | Station-level ridership mapped to H3 grid |
| Citi Bike Trip Data | Bike trip counts per H3 cell |
| OpenStreetMap POI | Count of restaurants, subway stops, offices, retail, etc. per cell |
| Open-Meteo API | Live 16-day weather forecast (temperature, precipitation, wind) |
| US Holidays (Python) | Federal holiday flags |

---

## Known Limitations

Understanding these limitations is important before integrating or presenting results to end users.

### 1. Training Data Covers Only 30 High-Traffic Corridors

The model was trained exclusively on NYC Department of Transportation (DOT) pedestrian count surveys. The DOT only surveys locations that are already known to be high-footfall — busy intersections, commercial strips, and transit hubs across Manhattan.

**What this means in practice:**
- The 30 trained corridors include places like Times Square, Grand Central, the High Line, Wall Street, and Herald Square
- Residential side streets, parks in off-peak hours, and quiet neighbourhood blocks have **no ground-truth training data**
- The model has never seen a "quiet" location during training — every training sample is from a busy corridor

**How the API handles untrained locations:**
For any lat/lon that does not snap to one of the 30 trained H3 cells, the API applies a **proxy scaling** step. It computes a relative activity score for that cell using TLC trip counts, MTA ridership, Citi Bike trips, and POI density — then scales the model's raw output down proportionally. This prevents the model from assigning "Very Busy" scores to genuinely quiet areas, but it is an approximation, not a trained prediction.

**Confidence levels by location type:**

| Location type | Example | Confidence |
|---------------|---------|------------|
| Trained corridor | Times Square, Wall Street | High — model has 15+ years of ground truth |
| Near trained corridor | Midtown side streets | Medium — proxy scaling + nearby signal |
| Far from training data | East Harlem, Inwood | Low — proxy scaling only, no pedestrian ground truth |

---

### 2. Training Data is Weekday-Only

The DOT surveys are conducted on specific weekdays (typically mid-May and mid-September each year). **No weekend ground truth exists in the training data.**

**What this means:**
- When you predict for a Saturday or Sunday, the model uses the `is_weekend=1` flag and weekend calendar features, but the lag features (the biggest driver of accuracy, R²=0.93+) are populated with the corridor's **weekday historical medians**
- This can cause weekend predictions for trained corridors to be slightly inflated relative to actual weekend crowds, because weekday foot traffic at those corridors is higher than weekend
- Example: Wall Street on a Sunday PM is predicted at crowd_score ~85 (Very Busy) because the model's lag baseline is weekday commuter traffic — real Sunday crowds would likely be lower

**Workaround for the frontend:** consider showing a reduced-confidence indicator (e.g. a softer colour or a disclaimer) when `is_weekend=1` or `is_holiday=1`.

---

### 3. Survey Dates are Fixed (Mid-May and Mid-October)

The DOT only surveys in spring and fall. The model has **no summer or winter ground truth** for any corridor.

**What this means:**
- Summer (June–August) and winter (December–February) predictions rely on calendar features (month_sin, month_cos, school_break, etc.) to extrapolate
- Weather features compensate somewhat (a hot summer day vs. a cold winter day)
- But the model has not actually seen a Times Square reading in August or January — those predictions are extrapolations

---

### 4. Lag Features Require Historical Data — Future Predictions Are Weaker

The model's single biggest accuracy driver is corridor-level lag features: how many pedestrians were counted in the same corridor 1 survey ago, 2 surveys ago, and the rolling average. These features account for most of the R²=0.93 score.

For **future predictions** (`/predict/future`):
- Actual lag values are unknown (we don't know what will happen next week)
- For trained corridors: the API fills lags with the corridor's historical median (a reasonable estimate)
- For untrained cells: lags are set to 0 and proxy scaling is applied
- The future endpoint uses only 2 models (LGB + HistGB) instead of 3, as RandomForest is less reliable without lag data

**The further into the future, the less reliable the prediction.** Short-term (1–3 days) benefits from live weather forecasts; beyond 16 days, weather falls back to seasonal averages.

---

### 5. Geographic Coverage — Manhattan Only

The H3 grid covers Manhattan only (524 hexagonal cells at resolution 9, approximately 150m × 150m each). Requests for Brooklyn, Queens, the Bronx, or Staten Island will be **rejected with a 422 error**.

If a lat/lon is within Manhattan's bounding box but outside any valid H3 cell (e.g. on the Hudson River), the API snaps to the nearest valid cell within 3 rings (~450m radius).

---

### 6. No Real-Time Pedestrian Data

The API does not consume any live pedestrian sensor feed. All predictions are model-based estimates. There is no live camera feed, footfall sensor, or real-time crowd count ingested at prediction time.

**What "real-time" actually means in this API:**
- Live weather (temperature, rain, wind) from Open-Meteo API — refreshed per request
- Time of day and calendar (holiday, weekend, school break) — computed per request
- Everything else (mobility signals, POI counts, lag baselines) is from the last time the data pipeline was run

---

### 7. Events Are Not Modelled

The `event_intensity_score` feature exists in the training data but is set to `0.0` at inference time because there is no live event data feed integrated. This means:
- A major concert at Madison Square Garden on a Tuesday evening will not cause the model to predict higher crowds for that area
- Holiday parades, marathons, and street fairs are not captured in real time

The seasonal and holiday calendar features partially compensate (e.g. July 4th gets `is_holiday=1`), but individual event-level spikes are not predicted.

---

### Summary Table

| Limitation | Impact | Severity |
|-----------|--------|----------|
| Only 30 trained corridors | Untrained cells use proxy scaling | Medium — proxy gives directional accuracy but not calibrated counts |
| Weekday-only training data | Weekend predictions may be inflated | Medium — workaround: show lower confidence on weekends |
| No summer/winter surveys | Seasonal extrapolation for Jun–Aug, Dec–Feb | Low–Medium — calendar + weather features compensate partially |
| No real lag for future dates | Future predictions less accurate than current | Medium — acceptable for planning use cases |
| Manhattan only | 422 error for other boroughs | Hard limit |
| No live pedestrian sensors | All predictions are model estimates | Architectural — would require hardware investment to change |
| No event modelling | Spikes from concerts/parades not captured | Medium for specific venues (MSG, Barclays proximity) |

---

## Notes for Frontend Team

1. **Always send lat/lon in decimal degrees** (e.g. `40.7580`, not `40°45'28.8"N`)
2. **`when` must be ISO 8601** — `"2026-07-04T20:00:00"` works; date-only strings do not
3. **Use `/predict/crowd-score` for map overlays** — it's the lightest endpoint and returns only what you need for a colour-coded pin
4. **Use `/predict/future` for planning features** — pass any future datetime, weather is handled automatically
5. **`h3_cell` is an Uber H3 hex ID** — if you want to render the hex boundary on a map, use the [H3-js library](https://h3geo.org/) with `h3.cellToBoundary(h3_cell)`
6. **`crowd_score` is always 0–100** — map it directly to a colour gradient (green → yellow → red)

## Notes for Backend Team

1. **Models are loaded once at startup** — inference is fast (~20ms per request)
2. **The server requires these files at startup** (relative to `api/`):
   - `../models/lgbm_tuned.txt`
   - `../models/gb_tuned.pkl`
   - `../models/rf_tuned.pkl`
   - `../models/rf_imputer.pkl`
   - `../models/ensemble_weights.json`
   - `../models/inference_bundle.joblib`
   - `../data/master/manhattan_grid_h3.csv`
   - `../data/master/poi_hex_counts.csv`
   - `../data/master/zentra_training_master.csv`
   - `../data/master/zentra_live_grid_snapshot.csv`
3. **Weather calls go to Open-Meteo** (`api.open-meteo.com`) — ensure outbound HTTP is allowed on the server
4. **No authentication is implemented** — add API key middleware before exposing publicly
5. **The `/predict/debug` endpoint exposes model internals** — disable it in production
