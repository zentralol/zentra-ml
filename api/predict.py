"""
predict.py
----------
Three prediction endpoints:

  POST /predict/crowd        → current crowd prediction (count + score + label)
  GET  /predict/crowd-score  → lightweight score-only (good for badges/gauges)
  POST /predict/future       → future date prediction (must be in the future)
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

import inference
from tzutil import now_in_manhattan, to_manhattan_time

router = APIRouter(prefix="/predict", tags=["Predictions"])

# Manhattan bounding box — requests outside this box are rejected automatically
LAT_MIN, LAT_MAX =  40.679,  40.882
LON_MIN, LON_MAX = -74.020, -73.907

# REQUEST MODELS  (what the user sends us)
class CrowdRequest(BaseModel):
    lat:  float            = Field(..., ge=LAT_MIN, le=LAT_MAX,
                                   description="Latitude (Manhattan only)")
    lon:  float            = Field(..., ge=LON_MIN, le=LON_MAX,
                                   description="Longitude (Manhattan only)")
    when: Optional[datetime] = Field(None,
                                     description="Datetime to predict for. Defaults to right now.")


class FutureRequest(BaseModel):
    lat:  float    = Field(..., ge=LAT_MIN, le=LAT_MAX)
    lon:  float    = Field(..., ge=LON_MIN, le=LON_MAX)
    when: datetime = Field(..., description="Must be a future datetime.")

    @field_validator("when")
    @classmethod
    def when_must_be_future(cls, v):
        v = to_manhattan_time(v)
        if v <= now_in_manhattan():
            raise ValueError("'when' must be in the future. For current predictions use /predict/crowd.")
        return v


# RESPONSE MODELS  (what we send back)
class CrowdResponse(BaseModel):
    lat:             float
    lon:             float
    h3_cell:         str     # the H3 grid cell this point belongs to
    period:          str     # time bucket: EARLY / AM / MD / PM / EVE / NIGHT
    timestamp:       str     # the datetime we predicted for
    pedestrians:     float   # predicted pedestrian count
    crowd_score:     float   # 0 to 100
    crowd_category:  str     # Quiet / Light / Moderate / Busy / Very Busy


class ScoreOnlyResponse(BaseModel):
    lat:            float
    lon:            float
    h3_cell:        str
    period:         str
    timestamp:      str
    crowd_score:    float
    crowd_category: str


class FutureResponse(CrowdResponse):
    days_ahead:     float    # how far into the future this prediction is
    weather_source: str      # "forecast" (live API) or "seasonal_average" (fallback)


# ENDPOINTS
@router.post("/crowd", response_model=CrowdResponse)
def predict_crowd(req: CrowdRequest):
    """
    Predict pedestrian crowd level at a Manhattan location right now (or at a past time).

    Uses a 3-model ensemble: GradientBoosting + LightGBM + RandomForest.
    """
    when = to_manhattan_time(req.when) if req.when is not None else now_in_manhattan()
    result = inference.run(req.lat, req.lon, when, task="crowd")

    return CrowdResponse(
        lat=req.lat,
        lon=req.lon,
        timestamp=when.isoformat(),
        **result
    )


@router.get("/crowd-score", response_model=ScoreOnlyResponse)
def get_crowd_score(
    lat:  float            = Query(..., ge=LAT_MIN, le=LAT_MAX, description="Latitude"),
    lon:  float            = Query(..., ge=LON_MIN, le=LON_MAX, description="Longitude"),
    when: Optional[datetime] = Query(None, description="Defaults to now"),
):
    """
    Lightweight endpoint — returns just the 0-100 crowd score and category label.
    Good for dashboards and map badges where you only need the score.
    """
    when = to_manhattan_time(when) if when is not None else now_in_manhattan()
    result = inference.run(lat, lon, when, task="score")

    return ScoreOnlyResponse(
        lat=lat,
        lon=lon,
        timestamp=when.isoformat(),
        h3_cell=result["h3_cell"],
        period=result["period"],
        crowd_score=result["crowd_score"],
        crowd_category=result["crowd_category"],
    )


@router.post("/future", response_model=FutureResponse)
def predict_future_crowd(req: FutureRequest):
    """
    Predict crowd levels at a future date and time.

    Uses GB + LightGBM ensemble (lag features zeroed since we have no past data for
    that future date). Weather comes from the live forecast API if within 16 days,
    or seasonal averages if further out.
    """
    result    = inference.run(req.lat, req.lon, req.when, task="future")
    days_out  = (req.when - now_in_manhattan()).total_seconds() / 86400

    return FutureResponse(
        lat=req.lat,
        lon=req.lon,
        timestamp=req.when.isoformat(),
        days_ahead=round(days_out, 2),
        weather_source="forecast" if days_out <= 16 else "seasonal_average",
        **result
    )


@router.get("/debug")
def debug_features(lat: float = Query(...), lon: float = Query(...)):
    """
    Shows exactly what features the model sees for a given location.
    Useful for debugging — remove or restrict this in production.
    """
    import traceback
    from fastapi.responses import JSONResponse

    def to_python(obj):
        """Convert numpy types to plain Python so JSON doesn't break."""
        import numpy as np
        if isinstance(obj, dict):   return {k: to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):   return [to_python(v) for v in obj]
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return obj

    try:
        when = now_in_manhattan()
        raw_row, cell, period = inference.build_features(lat, lon, when)
        feature_df = inference.align_to_feature_cols(raw_row)

        lgb_p = float(inference.lgb_model.predict(feature_df)[0])
        gb_p  = float(inference.gb_model.predict(feature_df)[0])
        rf_p  = float(inference.rf_model.predict(feature_df)[0])

        w     = inference.WEIGHTS["crowd"]
        ens_p = w["lgb"]*lgb_p + w["gb"]*gb_p + w["rf"]*rf_p

        import math
        result = {
            "cell":   cell,
            "period": period,
            "model_predictions": {
                "lgb_log": round(lgb_p, 4),  "lgb_raw": round(math.expm1(lgb_p), 1),
                "gb_log":  round(gb_p, 4),   "gb_raw":  round(math.expm1(gb_p), 1),
                "rf_log":  round(rf_p, 4),   "rf_raw":  round(math.expm1(rf_p), 1),
                "ensemble_log": round(ens_p, 4), "ensemble_raw": round(math.expm1(ens_p), 1),
            },
            "crowd_score": round(min(max(ens_p / float(inference.HIST_P95) * 100, 0), 100), 1),
            "blend_weights": w,
            "feature_count": len(inference.FEATURE_COLS),
            "features_present": len([c for c in inference.FEATURE_COLS if c in raw_row]),
        }
        return JSONResponse(content=to_python(result))

    except Exception:
        return JSONResponse(content={"error": traceback.format_exc()}, status_code=500)
