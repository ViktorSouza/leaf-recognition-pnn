import os
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from models.leaf_pnn import LeafPNN
from sources.leaf_feature_extractor import extract_leaf_features
from utils.logger import get_logger

PLANT_TYPES = {
    1: "Phyllostachys edulis (pubescent bamboo)",
    2: "Aesculus chinensis (Chinese horse chestnut)",
    3: "Berberis anhweiensis (Anhui Barberry)",
    4: "Cercis chinensis (Chinese redbud)",
    5: "Indigofera tinctoria (true indigo)",
    6: "Acer palmatum (Japanese maple)",
    7: "Phoebe nanmu (Nanmu)",
    8: "Kalopanax septemlobus (castor aralia)",
    9: "Cinnamomum japonicum (Chinese cinnamon)",
    10: "Koelreuteria paniculata (goldenrain tree)",
    11: "Ilex macrocarpa (Big-fruited Holly)",
    12: "Pittosporum tobira (Japanese cheesewood)",
    14: "Chimonanthus praecox (wintersweet)",
    15: "Cinnamomum camphora (camphortree)",
    16: "Viburnum awabuki (Japan Arrowwood)",
    17: "Osmanthus fragrans (sweet osmanthus)",
    18: "Cedrus deodara (deodar)",
    19: "Ginkgo biloba (ginkgo)",
    20: "Lagerstroemia indica (crape myrtle)",
    21: "Nerium oleander (oleander)",
    22: "Podocarpus macrophyllus (yew plum pine)",
    23: "Prunus serrulata (Japanese Flowering Cherry)",
    24: "Ligustrum lucidum (Glossy Privet)",
    25: "Toona sinensis (Chinese Toon)",
    26: "Prunus persica (peach)",
    27: "Manglietia fordiana (Ford Woodlotus)",
    28: "Acer buergerianum (trident maple)",
    29: "Mahonia bealei (Beale's barberry)",
    30: "Magnolia grandiflora (southern magnolia)",
    31: "Populus ×canadensis (Canadian poplar)",
    32: "Liriodendron chinense (Chinese tulip tree)",
    33: "Citrus reticulata (tangerine)",
}

PLANT_RANGES = (
    (1001, 1059, 1),
    (1060, 1122, 2),
    (1123, 1194, 4),
    (1195, 1267, 5),
    (1268, 1323, 6),
    (1324, 1385, 7),
    (1386, 1437, 8),
    (1438, 1496, 10),
    (1497, 1551, 9),
    (1552, 1616, 3),
    (2001, 2050, 11),
    (2051, 2113, 12),
    (2114, 2165, 14),
    (2166, 2230, 15),
    (2231, 2290, 16),
    (2291, 2346, 17),
    (2347, 2423, 18),
    (2424, 2485, 19),
    (2486, 2546, 20),
    (2547, 2612, 21),
    (2616, 2675, 22),
    (3001, 3055, 23),
    (3056, 3110, 24),
    (3111, 3175, 25),
    (3176, 3229, 26),
    (3230, 3281, 27),
    (3282, 3334, 28),
    (3335, 3389, 29),
    (3390, 3446, 30),
    (3447, 3510, 31),
    (3511, 3563, 32),
    (3566, 3621, 33),
)
FEATURE_COUNT = 12
LOGGER = get_logger("LeafModelRepository")
EXPECTED_IMAGE_INDICES = tuple(
    index
    for start, end, _ in PLANT_RANGES
    for index in range(start, end + 1)
)


def _training_labels() -> NDArray[np.int64]:
    return np.array(
        [
            label
            for start, end, label in PLANT_RANGES
            for _ in range(start, end + 1)
        ],
        dtype=np.int64,
    )


def ensure_training_features(
    feature_path: Path,
    images_path: Path,
) -> Path:
    if feature_path.is_file():
        return feature_path

    if not images_path.is_dir():
        raise FileNotFoundError(
            f"Features de treino não existem em '{feature_path}' e o diretório "
            f"do dataset Flavia não foi encontrado em '{images_path}'. "
            "Disponibilize as imagens JPEG nesse diretório ou configure "
            "LEAF_IMAGES_PATH."
        )

    LOGGER.info(
        "Features de treino ausentes; extraindo %s imagens em %s.",
        len(EXPECTED_IMAGE_INDICES),
        images_path,
    )
    image_files = sorted(
        (
            path
            for path in images_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".jpg"
        ),
        key=lambda path: path.name,
    )
    try:
        image_indices = tuple(int(path.stem) for path in image_files)
    except ValueError as error:
        raise ValueError(
            f"O dataset em '{images_path}' contém JPEGs cujo nome não é um "
            "índice numérico Flavia."
        ) from error

    if image_indices != EXPECTED_IMAGE_INDICES:
        raise ValueError(
            f"O diretório '{images_path}' precisa conter as {len(EXPECTED_IMAGE_INDICES)} "
            "imagens JPEG do Flavia, nomeadas pelos índices esperados. "
            f"Foram encontradas {len(image_files)}."
        )

    features = np.vstack(
        [extract_leaf_features(image_file.read_bytes()) for image_file in image_files]
    )
    expected_shape = (len(EXPECTED_IMAGE_INDICES), FEATURE_COUNT)
    if features.shape != expected_shape or not np.isfinite(features).all():
        raise ValueError(
            "A extração do dataset Flavia produziu uma matriz inválida: "
            f"esperado {expected_shape}, recebido {features.shape}."
        )

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=feature_path.parent,
            prefix=f".{feature_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.save(temporary_file, features, allow_pickle=False)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, feature_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    LOGGER.info("Features de treino salvas em %s.", feature_path)
    return feature_path


def load_classifier() -> LeafPNN:
    repository_root = Path(__file__).resolve().parents[2]
    default_path = repository_root / "data" / "leaf_features.npy"
    feature_path = Path(os.environ.get("LEAF_FEATURES_PATH", default_path))
    images_path = Path(
        os.environ.get("LEAF_IMAGES_PATH", repository_root / "Leaves")
    )
    ensure_training_features(feature_path, images_path)

    raw_features = np.load(feature_path, allow_pickle=False)
    labels = _training_labels()
    if raw_features.ndim != 2:
        raise ValueError(
            f"O artefato em '{feature_path}' deve ser uma matriz bidimensional."
        )
    if raw_features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"O artefato contém {raw_features.shape[0]} amostras, "
            f"mas o mapeamento do Flavia espera {labels.shape[0]}."
        )

    return LeafPNN(raw_features, labels, PLANT_TYPES)
