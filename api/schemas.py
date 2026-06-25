# api/schemas.py
# All Pydantic request and response models used across the API.
# Pydantic validates incoming JSON automatically — if a required field is
# missing or the wrong type, FastAPI returns a 422 error before our code runs.

from typing import Optional
from pydantic import BaseModel, Field


# LightGBM 

class PredictionRequest(BaseModel):
    # Feature name -> value dict. Missing features are filled with NaN by the router.
    # See GET /model/info for the full list of feature names.
    features: dict[str, Optional[float]]


class BatchPredictionRequest(BaseModel):
    # Same as above but a list of rows for bulk predictions
    rows: list[dict[str, Optional[float]]]


class PredictionResponse(BaseModel):
    log_pedestrians_pred: float   # model output in log1p space
    pedestrians_pred: float       # back-transformed to raw pedestrian count
    model_version: str            # filename of the model that made this prediction


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


# Prophet 

class ProphetRequest(BaseModel):
    # Which location and time period to forecast
    location_id: int
    period: str               # "AM", "MD", or "PM"
    ds: str                   # date string e.g. "2025-10-15"

    # Regressors — same ones Prophet was trained with in the notebook
    # All are optional; missing ones are passed as None to the model
    avg_temperature_f: Optional[float] = None
    total_precip_in: Optional[float] = None
    is_bad_weather: Optional[float] = None
    tlc_trip_count: Optional[float] = None
    mta_ridership_total: Optional[float] = None
    ped_lag_1survey: Optional[float] = None
    loc_period_baseline: Optional[float] = None


class ProphetResponse(BaseModel):
    location_id: int
    period: str
    ds: str
    log_pedestrians_pred: float         # forecast in log1p space
    pedestrians_pred: float             # back-transformed to raw count
    yhat_lower: Optional[float] = None  # 95% lower bound (log space)
    yhat_upper: Optional[float] = None  # 95% upper bound (log space)
    model_file: str                     # which JSON file was used


# Ensemble 

class EnsembleRequest(BaseModel):
    location_id: int
    period: str
    ds: str
    lgbm_features: dict[str, Optional[float]]           # features for LightGBM
    prophet_regressors: Optional[dict[str, Optional[float]]] = None  # if None, falls back to LightGBM only


class EnsembleResponse(BaseModel):
    log_pedestrians_pred: float
    pedestrians_pred: float
    lgbm_contribution: float            # the weighted LightGBM portion
    prophet_contribution: Optional[float] = None  # None if Prophet wasn't used
    prophet_available: bool             # tells the frontend whether blend was used
    model_version: str


# Model info 

class ModelInfoResponse(BaseModel):
    model_path: str
    num_features: int
    feature_names: list[str]
    num_trees: int
    best_params: Optional[dict] = None


class MetricsResponse(BaseModel):
    metrics: dict


# Run history 

class ModelRun(BaseModel):
    run_id: Optional[str]   = None
    model_name: Optional[str]   = None
    batch_tag: Optional[str]   = None
    mae_log: Optional[float] = None
    rmse_log: Optional[float] = None
    r2: Optional[float] = None
    mape_log: Optional[float] = None
    mae_raw: Optional[float] = None
    rmse_raw: Optional[float] = None
    train_time_s: Optional[float] = None
    created_at: Optional[str]   = None


class RunsResponse(BaseModel):
    runs: list[ModelRun]
    source: str   # "supabase" or "unavailable"


# Health 

class HealthResponse(BaseModel):
    status: str        # "ok" or "degraded"
    model_loaded: bool
    model_path: str
