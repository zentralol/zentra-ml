from fastapi import FastAPI
import predict
import inference

# Create the app
app = FastAPI(
    title="Zentra Crowd Prediction API",
    description="Predict pedestrian crowd levels anywhere in Manhattan",
    version="2.0"
)

# Connect all the routes from predict.py
app.include_router(predict.router)

# Root endpoint — health check + shows what data was loaded at startup
@app.get("/")
def home():
    return {
        "message": "Zentra Crowd Prediction API is running!",
        "docs": "/docs",
        "loaded": {
            "trained_cells":   len(inference.TRAINED_CELLS),
            "lag_baselines":   len(inference.LAG_BASELINE),
            "proxy_cells":     len(inference.PROXY_SCORE),
            "mean_train_proxy": round(inference.MEAN_TRAIN_PROXY, 1),
            "feature_cols":    len(inference.FEATURE_COLS),
        }
    }

# Run with:  uvicorn main:app --reload --port 8000
# API docs:  http://localhost:8000/docs
