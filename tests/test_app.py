import numpy as np
from fastapi.testclient import TestClient

import app as app_module
import middlewares.internal_token as internal_token
from models.leaf_pnn import PNNPrediction


class FakeClassifier:
    def predict(self, features: np.ndarray) -> PNNPrediction:
        assert features.shape == (12,)
        return PNNPrediction(
            label=1,
            species="Phyllostachys edulis (pubescent bamboo)",
            confidence=0.8,
        )


def test_predict_endpoint_requires_token_and_returns_species(
    monkeypatch,
) -> None:
    monkeypatch.setattr(app_module, "load_classifier", FakeClassifier)
    monkeypatch.setattr(
        app_module,
        "extract_leaf_features",
        lambda image_bytes: np.zeros(12, dtype=np.float64),
    )
    monkeypatch.setattr(internal_token, "INTERNAL_TOKEN", "test-token")

    with TestClient(app_module.create_app()) as client:
        assert client.get("/health_check").status_code == 200

        files = {"image": ("leaf.png", b"image bytes", "image/png")}
        unauthorized = client.post("/predict", files=files)
        assert unauthorized.status_code == 401

        response = client.post(
            "/predict",
            files=files,
            headers={"INTERNAL-TOKEN": "test-token"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "label": 1,
        "species": "Phyllostachys edulis (pubescent bamboo)",
        "confidence": 0.8,
    }
