from pathlib import Path

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import scripts.import_leaves as import_leaves
from database import Base
from models.leaf import Leaf
from models.specie import Specie


def test_imports_each_image_before_processing_the_next(monkeypatch) -> None:
    image_files = [Path("/dataset/1001.jpg"), Path("/dataset/1060.jpg")]
    events: list[tuple[str, str]] = []

    def extract_features(image_bytes: bytes) -> np.ndarray:
        image_name = image_bytes.decode()
        events.append(("extract", image_name))
        return np.zeros(len(import_leaves.FEATURE_NAMES))

    def persist_leaf(
        image_file: Path,
        label: int,
        values: tuple[float, ...],
    ) -> tuple[bool, bool]:
        events.append(("persist", image_file.name))
        return False, True

    monkeypatch.setattr(
        import_leaves.Path,
        "read_bytes",
        lambda path: path.name.encode(),
    )
    monkeypatch.setattr(import_leaves, "extract_leaf_features", extract_features)
    monkeypatch.setattr(import_leaves, "_persist_leaf", persist_leaf)
    monkeypatch.setattr(import_leaves, "_get_existing_picture_paths", lambda _: set())

    counts = import_leaves._import_images(image_files)

    assert events == [
        ("extract", "1001.jpg"),
        ("persist", "1001.jpg"),
        ("extract", "1060.jpg"),
        ("persist", "1060.jpg"),
    ]
    assert counts == (0, 2)


def test_skips_existing_images_before_reading_or_extracting_them(monkeypatch) -> None:
    image_files = [Path("/dataset/1001.jpg"), Path("/dataset/1060.jpg")]
    events: list[tuple[str, str]] = []

    def extract_features(image_bytes: bytes) -> np.ndarray:
        image_name = image_bytes.decode()
        events.append(("extract", image_name))
        return np.zeros(len(import_leaves.FEATURE_NAMES))

    def persist_leaf(
        image_file: Path,
        label: int,
        values: tuple[float, ...],
    ) -> tuple[bool, bool]:
        events.append(("persist", image_file.name))
        return False, True

    monkeypatch.setattr(
        import_leaves.Path,
        "read_bytes",
        lambda path: path.name.encode(),
    )
    monkeypatch.setattr(import_leaves, "extract_leaf_features", extract_features)
    monkeypatch.setattr(import_leaves, "_persist_leaf", persist_leaf)
    monkeypatch.setattr(
        import_leaves,
        "_get_existing_picture_paths",
        lambda _: {image_files[0].as_posix()},
    )

    counts = import_leaves._import_images(image_files)

    assert events == [
        ("extract", "1060.jpg"),
        ("persist", "1060.jpg"),
    ]
    assert counts == (0, 1)


def test_finds_only_image_paths_already_stored_in_database(tmp_path, monkeypatch) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'leaves.db'}")
    Base.metadata.create_all(engine)
    test_session_local = sessionmaker(bind=engine)

    with test_session_local.begin() as session:
        specie = Specie(name="test species", picture_url="/dataset/1001.jpg")
        session.add(Leaf(picture_url="/dataset/1001.jpg", specie=specie))

    monkeypatch.setattr(import_leaves, "SessionLocal", test_session_local)

    try:
        existing_paths = import_leaves._get_existing_picture_paths(
            [Path("/dataset/1001.jpg"), Path("/dataset/1060.jpg")]
        )
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()

    assert existing_paths == {"/dataset/1001.jpg"}
