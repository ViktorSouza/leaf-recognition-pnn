import numpy as np
import pytest

import repositories.leaf_model_repository as repository


def test_load_classifier_generates_and_reuses_missing_training_features(
    tmp_path,
    monkeypatch,
) -> None:
    images_path = tmp_path / "Leaves"
    images_path.mkdir()
    feature_path = tmp_path / "data" / "leaf_features.npy"
    for index in repository.EXPECTED_IMAGE_INDICES:
        (images_path / f"{index}.jpg").touch()

    extracted = 0

    def extract_features(image_bytes: bytes) -> np.ndarray:
        nonlocal extracted
        extracted += 1
        assert image_bytes == b""
        return np.full(12, extracted, dtype=np.float64)

    monkeypatch.setattr(repository, "extract_leaf_features", extract_features)
    monkeypatch.setenv("LEAF_FEATURES_PATH", str(feature_path))
    monkeypatch.setenv("LEAF_IMAGES_PATH", str(images_path))

    classifier = repository.load_classifier()

    assert extracted == len(repository.EXPECTED_IMAGE_INDICES)
    assert classifier.training_labels.shape == (1907,)
    assert np.load(feature_path).shape == (1907, 12)

    def unexpected_extraction(image_bytes: bytes) -> np.ndarray:
        raise AssertionError("Existing features should be reused.")

    monkeypatch.setattr(repository, "extract_leaf_features", unexpected_extraction)
    reloaded_classifier = repository.load_classifier()

    assert reloaded_classifier.training_labels.shape == (1907,)


def test_missing_dataset_reports_setup_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="LEAF_IMAGES_PATH"):
        repository.ensure_training_features(
            tmp_path / "features.npy",
            tmp_path / "missing-dataset",
        )
