# api/routers/model_info.py
# Read-only endpoints that describe the loaded model and its metrics.
# No training or heavy computation happens here — it just reads files
# that the notebook wrote during the export step.

from fastapi import APIRouter, Depends, HTTPException
from api.config import LGBM_MODEL_PATH
from api.dependencies import get_best_params, get_metrics, get_model
from api.schemas import MetricsResponse, ModelInfoResponse

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/info", response_model=ModelInfoResponse)
def model_info(model=Depends(get_model)):
    # Returns feature names, tree count, and Optuna best params
    feature_names = model.feature_name()
    return ModelInfoResponse(
        model_path=str(LGBM_MODEL_PATH),
        num_features=len(feature_names),
        feature_names=feature_names,
        num_trees=model.num_trees(),
        best_params=get_best_params(),
    )


@router.get("/metrics", response_model=MetricsResponse)
def model_metrics():
    # Returns all-model test metrics from reports/metrics.json
    metrics = get_metrics()
    if metrics is None:
        raise HTTPException(
            status_code=404,
            detail="reports/metrics.json not found — run the notebook export step first."
        )
    return MetricsResponse(metrics=metrics)
