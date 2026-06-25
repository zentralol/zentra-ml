# api/routers/health.py
# Simple liveness check — the frontend and deployment platform use this
# to know whether the API is up and the model loaded successfully.

from fastapi import APIRouter
from api.config import LGBM_MODEL_PATH
from api.dependencies import model_is_loaded
from api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health():
    loaded = model_is_loaded()
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_path=str(LGBM_MODEL_PATH),
    )
