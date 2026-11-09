from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from database import SessionLocal
from models.leaf import Leaf
from models.leaf_feature import LeafFeature
from models.specie import Specie
from repositories.leaf_model_repository import (
    EXPECTED_IMAGE_INDICES,
    PLANT_RANGES,
    PLANT_TYPES,
)
from sources.leaf_feature_extractor import InvalidLeafImage, extract_leaf_features

FEATURE_NAMES = (
    "smooth_factor",
    "length_width_ratio",
    "circularity",
    "rectangularity",
    "diameter_length_ratio",
    "perimeter_diameter_ratio",
    "perimeter_length_width_ratio",
    "vein_area_radius_1_ratio",
    "vein_area_radius_2_ratio",
    "vein_area_radius_3_ratio",
    "vein_area_radius_4_ratio",
    "vein_area_radius_4_to_1_ratio",
)


def _image_label_map() -> dict[int, int]:
    return {
        image_index: label
        for start, end, label in PLANT_RANGES
        for image_index in range(start, end + 1)
    }


def _get_image_files(images_path: Path) -> list[Path]:
    if not images_path.is_dir():
        raise FileNotFoundError(f"Diretório de imagens não encontrado: {images_path}")

    image_files = [
        path
        for path in images_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    ]
    try:
        image_files.sort(key=lambda path: int(path.stem))
    except ValueError as error:
        raise ValueError(
            f"Todas as imagens devem ter o índice Flavia como nome: {images_path}"
        ) from error

    image_indices = tuple(int(path.stem) for path in image_files)
    if len(set(image_indices)) != len(image_indices):
        duplicates = sorted(
            image_index
            for image_index in set(image_indices)
            if image_indices.count(image_index) > 1
        )
        raise ValueError(
            f"Há mais de uma imagem para estes índices: {duplicates[:10]}."
        )

    if image_indices != EXPECTED_IMAGE_INDICES:
        missing = sorted(set(EXPECTED_IMAGE_INDICES) - set(image_indices))
        unexpected = sorted(set(image_indices) - set(EXPECTED_IMAGE_INDICES))
        raise ValueError(
            "O diretório deve conter exatamente os índices de imagem do dataset "
            f"Flavia. Ausentes: {missing[:10]}; inesperados: {unexpected[:10]}."
        )
    return image_files


def _extract_dataset(
    image_files: list[Path],
) -> list[tuple[Path, int, tuple[float, ...]]]:
    image_labels = _image_label_map()
    dataset: list[tuple[Path, int, tuple[float, ...]]] = []

    for index, image_file in enumerate(image_files, start=1):
        print(f"Extraindo para imagem {image_file}")
        try:
            features = extract_leaf_features(image_file.read_bytes())
        except InvalidLeafImage as error:
            raise ValueError(
                f"Não foi possível extrair features de '{image_file}': {error}"
            ) from error

        if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
            raise ValueError(f"Features inválidas para a imagem '{image_file}'.")

        dataset.append(
            (
                image_file,
                image_labels[int(image_file.stem)],
                tuple(float(value) for value in features),
            )
        )
        if index % 100 == 0 or index == len(image_files):
            print(f"Features extraídas: {index}/{len(image_files)}")

    return dataset


def _persist_dataset(
    dataset: list[tuple[Path, int, tuple[float, ...]]],
) -> tuple[int, int]:
    created_species = 0
    created_leaves = 0

    with SessionLocal.begin() as session:
        species_by_name = {
            specie.name: specie for specie in session.scalars(select(Specie)).all()
        }
        leaves_by_path = {
            leaf.picture_url: leaf
            for leaf in session.scalars(
                select(Leaf).options(selectinload(Leaf.features))
            ).all()
        }

        for image_file, label, values in dataset:
            specie_name = PLANT_TYPES[label]
            picture_path = image_file.as_posix()
            specie = species_by_name.get(specie_name)
            if specie is None:
                specie = Specie(name=specie_name, picture_url=picture_path)
                species_by_name[specie_name] = specie
                session.add(specie)
                created_species += 1

            leaf = leaves_by_path.get(picture_path)
            if leaf is None:
                leaf = Leaf(picture_url=picture_path, specie=specie)
                leaves_by_path[picture_path] = leaf
                session.add(leaf)
                created_leaves += 1
            else:
                leaf.specie = specie

            existing_features = {feature.name: feature for feature in leaf.features}
            for feature_name, value in zip(FEATURE_NAMES, values, strict=True):
                feature = existing_features.pop(feature_name, None)
                if feature is None:
                    leaf.features.append(LeafFeature(name=feature_name, value=value))
                else:
                    feature.value = value
            for obsolete_feature in existing_features.values():
                leaf.features.remove(obsolete_feature)

    return created_species, created_leaves


def main() -> None:
    default_images_path = Path(__file__).resolve().parents[1] / "Leaves"
    images_path = Path(os.getenv("LEAF_IMAGES_PATH", default_images_path))
    image_files = _get_image_files(images_path)

    print(f"Extraindo features de {len(image_files)} imagens em {images_path}...")
    dataset = _extract_dataset(image_files)
    created_species, created_leaves = _persist_dataset(dataset)
    print(
        "Importação concluída: "
        f"{len(image_files)} imagens processadas, "
        f"{created_species} espécies e {created_leaves} folhas novas."
    )


if __name__ == "__main__":
    main()
