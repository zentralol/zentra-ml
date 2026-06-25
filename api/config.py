# api/config.py
# Central settings — paths, CORS origins, and API metadata.
# All paths are relative to the project root (zentra-ml/) so the API works
# whether you run it locally or inside Docker.

import os
from pathlib import Path

# Project root is one level above this file (api/ -> zentra-ml/)
ROOT = Path(__file__).resolve().parent.parent

# Folder paths
MODELS_DIR  = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MASTER_DIR  = ROOT / "data" / "master"

# Model and report file paths
# These can be overridden with environment variables if needed
LGBM_MODEL_PATH = Path(os.environ.get("ZENTRA_MODEL_PATH", MODELS_DIR  / "lgbm_tuned.txt"))
METRICS_PATH = Path(os.environ.get("ZENTRA_METRICS_PATH", REPORTS_DIR / "metrics.json"))
BEST_PARAMS_PATH = Path(os.environ.get("ZENTRA_BEST_PARAMS_PATH", REPORTS_DIR / "best_params.json"))
FEATURE_MANIFEST_PATH = Path(os.environ.get("ZENTRA_FEATURE_MANIFEST_PATH", MASTER_DIR / "feature_manifest_h3.csv"))

# CORS — who is allowed to call the API from a browser
# Defaults to "*" (allow all) for local dev. Set ZENTRA_API_CORS_ORIGINS to
# a comma-separated list of URLs in production, e.g. "https://zentra.lol"
_raw_origins = os.environ.get("ZENTRA_API_CORS_ORIGINS", "*")
CORS_ORIGINS = [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else ["*"]

# API metadata shown in the /docs page
API_TITLE       = "Zentra ML API"
API_VERSION     = "1.0.0"
API_DESCRIPTION = "Pedestrian crowd prediction API for Zentra Manhattan."
