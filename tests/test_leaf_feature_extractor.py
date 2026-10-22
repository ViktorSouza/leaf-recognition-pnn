from io import BytesIO

import numpy as np
import pytest
from PIL import Image, ImageDraw

from sources.leaf_feature_extractor import (
    InvalidLeafImage,
    extract_leaf_features,
)


def test_extracts_twelve_finite_features_from_leaf_image() -> None:
    image = Image.new("RGB", (128, 128), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((40, 8, 88, 120), fill=(25, 80, 30))
    draw.line((64, 16, 64, 112), fill=(230, 230, 230), width=2)
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")

    features = extract_leaf_features(image_bytes.getvalue())

    assert features.shape == (12,)
    assert np.isfinite(features).all()


def test_rejects_image_without_a_segmentable_leaf() -> None:
    image_bytes = BytesIO()
    Image.new("RGB", (128, 128), "white").save(image_bytes, format="PNG")

    with pytest.raises(InvalidLeafImage, match="segmentar uma folha"):
        extract_leaf_features(image_bytes.getvalue())


def test_rejects_non_image_bytes() -> None:
    with pytest.raises(InvalidLeafImage, match="imagem válida"):
        extract_leaf_features(b"not an image")
