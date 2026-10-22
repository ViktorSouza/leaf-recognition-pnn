import numpy as np

from models.leaf_pnn import LeafPNN


def test_predicts_species_for_training_sample() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(20, 12))
    labels = np.array([1] * 10 + [2] * 10)
    features[10:, 0] += 5
    model = LeafPNN(features, labels, {1: "species one", 2: "species two"})

    prediction = model.predict(features[0])

    assert prediction.label == 1
    assert prediction.species == "species one"
    assert 0 <= prediction.confidence <= 1


def test_rejects_invalid_feature_vector() -> None:
    features = np.arange(120, dtype=np.float64).reshape(10, 12)
    labels = np.array([1] * 5 + [2] * 5)
    model = LeafPNN(features, labels, {1: "species one", 2: "species two"})

    try:
        model.predict(np.zeros(11))
    except ValueError as error:
        assert "12 features" in str(error)
    else:
        raise AssertionError("An invalid feature vector should be rejected.")
