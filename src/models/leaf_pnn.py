from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FEATURE_COUNT = 12
PCA_COMPONENT_COUNT = 5
SMOOTHING_FACTOR = 1 / 32


@dataclass(frozen=True)
class PNNPrediction:
    label: int
    species: str
    confidence: float


class LeafPNN:
    """PNN do notebook, com padronização e PCA ajustados nas features do Flavia."""

    def __init__(
        self,
        raw_features: NDArray[np.float64],
        labels: NDArray[np.int64],
        plant_types: dict[int, str],
    ) -> None:
        if raw_features.ndim != 2 or raw_features.shape[1] != FEATURE_COUNT:
            raise ValueError(
                f"A matriz de treino deve ter {FEATURE_COUNT} features por amostra."
            )
        if raw_features.shape[0] != labels.shape[0] or labels.ndim != 1:
            raise ValueError("A quantidade de rótulos não corresponde às features.")
        if raw_features.shape[0] < 2 or not np.isfinite(raw_features).all():
            raise ValueError("A matriz de treino está vazia ou contém valores inválidos.")

        self.plant_types = plant_types
        self.species = np.unique(labels)

        self.feature_mean = raw_features.mean(axis=0)
        self.feature_std = raw_features.std(axis=0)
        self.feature_std[self.feature_std == 0] = 1
        scaled_features = (raw_features - self.feature_mean) / self.feature_std

        covariance = np.cov(scaled_features, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        component_order = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, component_order[:PCA_COMPONENT_COUNT]]
        self.training_features = scaled_features @ self.components
        self.training_labels = labels
        self.bias = np.sqrt(-np.log(0.5)) / SMOOTHING_FACTOR

    def predict(self, raw_features: NDArray[np.float64]) -> PNNPrediction:
        sample = np.asarray(raw_features, dtype=np.float64)
        if sample.shape != (FEATURE_COUNT,) or not np.isfinite(sample).all():
            raise ValueError(
                f"A imagem não produziu um vetor válido de {FEATURE_COUNT} features."
            )

        scaled_sample = (sample - self.feature_mean) / self.feature_std
        projected_sample = scaled_sample @ self.components
        distances = np.linalg.norm(
            self.training_features - projected_sample,
            axis=1,
        )
        activations = np.exp(-np.square(distances * self.bias))

        class_scores = np.array(
            [activations[self.training_labels == label].sum() for label in self.species]
        )
        total_score = float(class_scores.sum())
        if not np.isfinite(total_score) or total_score == 0:
            raise ValueError(
                "As features da imagem estão fora do alcance do classificador."
            )

        predicted_index = int(np.argmax(class_scores))
        label = int(self.species[predicted_index])
        return PNNPrediction(
            label=label,
            species=self.plant_types[label],
            confidence=float(class_scores[predicted_index] / total_score),
        )
