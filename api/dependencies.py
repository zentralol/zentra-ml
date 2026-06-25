# api/dependencies.py
# Shared helpers used by multiple routers.
# The main job here is loading the LightGBM model once and reusing it
# across all requests (lru_cache keeps it in memory after first load).

import json
import logging
from functools import lru_cache

import lightgbm as lgb
from fastapi import HTTPException

from api.config import BEST_PARAMS_PATH, LGBM_MODEL_PATH, METRICS_PATH

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_model():
    # Load the LightGBM booster from disk and keep it cached.
    # FastAPI calls this as a dependency on every prediction request,
    # but lru_cache means the file is only read once.
    if not LGBM_MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model file not found at {LGBM_MODEL_PATH}. Run the notebook export step first."
        )
    logger.info("Loading LightGBM model from %s", LGBM_MODEL_PATH)
    return lgb.Booster(model_file=str(LGBM_MODEL_PATH))


def model_is_loaded():
    # Returns True/False without raising — used by the health endpoint.
    try:
        get_model()
        return True
    except Exception:
        return False


def load_json_file(path):
    # Read a JSON file from disk. Returns None if the file doesn't exist.
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.warning("Could not read %s: %s", path, e)
        return None


def get_metrics():
    return load_json_file(METRICS_PATH)


def get_best_params():
    return load_json_file(BEST_PARAMS_PATH)
