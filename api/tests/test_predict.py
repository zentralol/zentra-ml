"""Route-level tests for prediction endpoints.

These tests mock the `inference` module before importing `predict` so that
heavy model files are not loaded during test collection.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Create a lightweight mock for the inference module before predict imports it.
_mock_inference = MagicMock()
_mock_inference.run.return_value = {
    "h3_cell": "892a100d647ffff",
    "period": "PM",
    "pedestrians": 8423.5,
    "crowd_score": 76.2,
    "crowd_category": "Busy",
}
sys.modules["inference"] = _mock_inference

import predict  # noqa: E402


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(predict.router)
    return TestClient(app)


class TestPredictCrowd:
    def test_offset_aware_iso_timestamp_returns_200(self, client):
        response = client.post(
            "/predict/crowd",
            json={
                "lat": 40.758,
                "lon": -73.9855,
                "when": "2026-07-14T15:00:00Z",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["crowd_score"] == 76.2
        assert data["timestamp"].endswith("-04:00") or data["timestamp"].endswith("-05:00")

    def test_naive_iso_timestamp_returns_200(self, client):
        response = client.post(
            "/predict/crowd",
            json={
                "lat": 40.758,
                "lon": -73.9855,
                "when": "2026-07-14T15:00:00",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["timestamp"].endswith("-04:00") or data["timestamp"].endswith("-05:00")

    def test_default_now_returns_manhattan_timestamp(self, client):
        response = client.post(
            "/predict/crowd",
            json={"lat": 40.758, "lon": -73.9855},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["timestamp"].endswith("-04:00") or data["timestamp"].endswith("-05:00")


class TestPredictCrowdScore:
    def test_offset_aware_query_timestamp_returns_200(self, client):
        response = client.get(
            "/predict/crowd-score",
            params={
                "lat": 40.758,
                "lon": -73.9855,
                "when": "2026-07-14T15:00:00Z",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["crowd_score"] == 76.2


class TestPredictFuture:
    def test_offset_aware_future_timestamp_returns_200(self, client):
        response = client.post(
            "/predict/future",
            json={
                "lat": 40.758,
                "lon": -73.9855,
                "when": "2027-01-01T15:00:00Z",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["crowd_score"] == 76.2
        assert "days_ahead" in data

    def test_offset_aware_past_timestamp_returns_422(self, client):
        response = client.post(
            "/predict/future",
            json={
                "lat": 40.758,
                "lon": -73.9855,
                "when": "2020-01-01T15:00:00Z",
            },
        )

        assert response.status_code == 422

    def test_naive_future_timestamp_returns_200(self, client):
        response = client.post(
            "/predict/future",
            json={
                "lat": 40.758,
                "lon": -73.9855,
                "when": "2027-01-01T15:00:00",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["crowd_score"] == 76.2
