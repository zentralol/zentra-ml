# api/routers/predict.py
# Prediction endpoints for LightGBM, Prophet, and the ensemble blend.

import json
import logging
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from lightgbm import Booster
from prophet.serialize import model_from_json

from api.config import LGBM_MODEL_PATH, MODELS_DIR
from api.dependencies import get_model
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    EnsembleRequest,
    EnsembleResponse,
    PredictionRequest,
    PredictionResponse,
    ProphetRequest,
    ProphetResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["predict"])

# Prophet model files live here: models/prophet/prophet_{location_id}_{period}.json
PROPHET_DIR = MODELS_DIR / "prophet"

# Ensemble weights — must sum to 1.0
LGBM_WEIGHT   = 0.70
PROPHET_WEIGHT = 0.30

# These are the extra regressors Prophet was trained 
PROPHET_REGRESSORS = [
    "avg_temperature_f",
    "total_precip_in",
    "is_bad_weather",
    "tlc_trip_count",
    "mta_ridership_total",
    "ped_lag_1survey",
    "loc_period_baseline",
]


# Helper functions 

def build_lgbm_dataframe(rows, feature_names):
    """
    Turn a list of feature dicts into a DataFrame with the exact columns
    LightGBM expects. Any feature the caller didn't send becomes NaN —
    LightGBM handles missing values natively so this is fine.
    """
    df = pd.DataFrame(rows)
    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan
    return df[feature_names]


def run_lgbm(model, df):
    """Run LightGBM and return a list of PredictionResponse objects."""
    log_preds = model.predict(df)
    raw_preds = np.expm1(log_preds)   # undo the log1p from training
    results = []
    for log_val, raw_val in zip(log_preds, raw_preds):
        results.append(PredictionResponse(
            log_pedestrians_pred=float(log_val),
            pedestrians_pred=float(raw_val),
            model_version=LGBM_MODEL_PATH.name,
        ))
    return results


@lru_cache(maxsize=512)
def load_prophet_model(location_id, period):
    """
    Load a Prophet model from its JSON file and cache it in memory.
    lru_cache means we only pay the load cost once per (location_id, period) pair.
    Raises FileNotFoundError if the notebook hasn't exported a model for this key.
    """
    filename = f"prophet_{location_id}_{period}.json"
    filepath = PROPHET_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"No Prophet model file at {filepath}")
    with open(filepath) as f:
        return model_from_json(json.load(f))


def build_prophet_dataframe(ds, regressors):
    """Build the one-row DataFrame that Prophet.predict() expects."""
    row = {"ds": pd.to_datetime(ds)}
    for reg in PROPHET_REGRESSORS:
        row[reg] = regressors.get(reg)   # None if the caller didn't send it
    return pd.DataFrame([row])


# POST /predict 

@router.post("", response_model=PredictionResponse)
def predict(request: PredictionRequest, model: Booster = Depends(get_model)):
    """Single-row LightGBM prediction."""
    df = build_lgbm_dataframe([request.features], model.feature_name())
    try:
        return run_lgbm(model, df)[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


# POST /predict/batch 

@router.post("/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest, model: Booster = Depends(get_model)):
    """Predict for multiple rows in one call."""
    if not request.rows:
        raise HTTPException(status_code=400, detail="rows list cannot be empty")
    df = build_lgbm_dataframe(request.rows, model.feature_name())
    try:
        return BatchPredictionResponse(predictions=run_lgbm(model, df))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {e}")


# POST /predict/prophet 

@router.post("/prophet", response_model=ProphetResponse)
def predict_prophet(request: ProphetRequest):
    """
    Prophet forecast for a single location + period + date.
    Needs models/prophet/prophet_{location_id}_{period}.json to exist
    (written by notebook Section 8).
    """
    # Load the model for this location/period
    try:
        model = load_prophet_model(request.location_id, request.period)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No Prophet model for location_id={request.location_id}, period={request.period}. "
                f"Run the notebook export step first."
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load Prophet model: {e}")

    # Build regressor values from the request
    regressors = {reg: getattr(request, reg, None) for reg in PROPHET_REGRESSORS}
    future_df = build_prophet_dataframe(request.ds, regressors)

    # Run Prophet
    try:
        forecast = model.predict(future_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prophet prediction failed: {e}")

    # Clip to a sane log range 
    yhat = float(np.clip(forecast["yhat"].iloc[0], 0, 15))
    yhat_low = float(np.clip(forecast["yhat_lower"].iloc[0], 0, 15)) if "yhat_lower" in forecast.columns else None
    yhat_hi = float(np.clip(forecast["yhat_upper"].iloc[0], 0, 15)) if "yhat_upper" in forecast.columns else None

    return ProphetResponse(
        location_id=request.location_id,
        period=request.period,
        ds=request.ds,
        log_pedestrians_pred=yhat,
        pedestrians_pred=float(np.expm1(yhat)),
        yhat_lower=yhat_low,
        yhat_upper=yhat_hi,
        model_file=f"prophet/prophet_{request.location_id}_{request.period}.json",
    )


# POST /predict/ensemble 

@router.post("/ensemble", response_model=EnsembleResponse)
def predict_ensemble(request: EnsembleRequest, model: Booster = Depends(get_model)):
    """
    Blend: 70% LightGBM + 30% Prophet.
    If prophet_regressors is not sent, or no Prophet file exists for this
    location/period, it falls back silently to 100% LightGBM.
    """
    # Step 1 — get LightGBM prediction
    lgbm_df = build_lgbm_dataframe([request.lgbm_features], model.feature_name())
    try:
        lgbm_log = float(model.predict(lgbm_df)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"LightGBM prediction failed: {e}")

    # Step 2 — try to get Prophet prediction
    prophet_log = None
    prophet_available = False

    if request.prophet_regressors is not None:
        try:
            prophet_model = load_prophet_model(request.location_id, request.period)
            future_df     = build_prophet_dataframe(request.ds, request.prophet_regressors)
            forecast      = prophet_model.predict(future_df)
            prophet_log   = float(np.clip(forecast["yhat"].iloc[0], 0, 15))
            prophet_available = True
        except FileNotFoundError:
            # No model file for this location — silently use LightGBM only
            logger.info("No Prophet model for loc=%s period=%s, using LightGBM only",
                        request.location_id, request.period)
        except Exception:
            # Prophet failed for some other reason — still fall back gracefully
            logger.warning("Prophet failed for loc=%s period=%s, using LightGBM only",
                           request.location_id, request.period, exc_info=True)

    # Step 3 — blend (or use LightGBM only)
    if prophet_available and prophet_log is not None:
        blended = LGBM_WEIGHT * lgbm_log + PROPHET_WEIGHT * prophet_log
        # Sanity check — if blend is somehow out of range, fall back
        if not (np.isfinite(blended) and 0 <= blended <= 15):
            logger.warning("Blend out of range (%.3f), falling back to LightGBM", blended)
            blended = lgbm_log
            prophet_available = False
    else:
        blended = lgbm_log

    # Work out individual contributions for the response
    lgbm_contribution    = LGBM_WEIGHT * lgbm_log if prophet_available else lgbm_log
    prophet_contribution = PROPHET_WEIGHT * prophet_log if prophet_available else None

    return EnsembleResponse(
        log_pedestrians_pred=float(blended),
        pedestrians_pred=float(np.expm1(blended)),
        lgbm_contribution=lgbm_contribution,
        prophet_contribution=prophet_contribution,
        prophet_available=prophet_available,
        model_version=LGBM_MODEL_PATH.name,
    )