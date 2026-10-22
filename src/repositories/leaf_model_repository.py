import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from models.leaf_pnn import LeafPNN

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


def _training_labels() -> NDArray[np.int64]:
    return np.array(
        [
            label
            for start, end, label in PLANT_RANGES
            for _ in range(start, end + 1)
        ],
        dtype=np.int64,
    )


def load_classifier() -> LeafPNN:
    default_path = (
        Path(__file__).resolve().parents[2] / "notebooks" / "leaf_features.npy"
    )
    feature_path = Path(os.environ.get("LEAF_FEATURES_PATH", default_path))
    if not feature_path.is_file():
        raise FileNotFoundError(
            f"Features de treino não encontradas em '{feature_path}'. "
            "Configure LEAF_FEATURES_PATH para leaf_features.npy."
        )

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
