# api/main.py
# FastAPI app entry point. Run with:
#   uvicorn api.main:app --reload --port 8000

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import API_DESCRIPTION, API_TITLE, API_VERSION, CORS_ORIGINS
from api.routers import health, model_info, predict, runs

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

app = FastAPI(title=API_TITLE, version=API_VERSION, description=API_DESCRIPTION)

# Allow the frontend to call this API from a browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routers
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(model_info.router)
app.include_router(runs.router)


@app.get("/", tags=["health"])
def root():
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
    }
