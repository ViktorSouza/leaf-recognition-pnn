from __future__ import annotations

from collections.abc import Callable
from io import BytesIO

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

MIN_LEAF_AREA = 101
MAX_IMAGE_PIXELS = 20_000_000


class InvalidLeafImage(ValueError):
    """A imagem não contém uma folha que possa ser processada pelo pipeline."""


def _grayscale(image_bytes: bytes) -> NDArray[np.float64]:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise InvalidLeafImage("A imagem excede o limite de 20 megapixels.")
            rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
    ) as error:
        raise InvalidLeafImage("O arquivo enviado não é uma imagem válida.") from error

    return rgb @ np.array([0.2989, 0.5870, 0.1140])


def _morphology(
    image: NDArray[np.float64],
    window: NDArray[np.uint8],
    operation: Callable[..., NDArray[np.float64]],
    initial_value: float,
    padding_value: float,
) -> NDArray[np.float64]:
    height, width = image.shape
    window_height, window_width = window.shape
    pad_height, pad_width = window_height // 2, window_width // 2
    padded = np.pad(
        image,
        ((pad_height, pad_height), (pad_width, pad_width)),
        mode="constant",
        constant_values=padding_value,
    )
    result = np.full_like(image, initial_value)

    for row in range(window_height):
        for column in range(window_width):
            if window[row, column]:
                operation(
                    result,
                    padded[row : row + height, column : column + width],
                    out=result,
                )
    return result


def _erosion(
    image: NDArray[np.float64],
    kernel: NDArray[np.uint8],
) -> NDArray[np.float64]:
    return _morphology(image, kernel, np.minimum, 1, 1)


def _opening(
    image: NDArray[np.float64],
    kernel: NDArray[np.uint8],
) -> NDArray[np.float64]:
    eroded = _erosion(image, kernel)
    return _morphology(eroded, kernel, np.maximum, 0, 0)


def _area_opening(
    image: NDArray[np.float64],
    minimum_area: int,
) -> NDArray[np.float64]:
    height, width = image.shape
    padded = np.zeros((height, width + 2), dtype=image.dtype)
    padded[:, 1:-1] = image
    differences = np.diff(padded, axis=1)

    row_runs: list[list[tuple[int, int, int]]] = []
    parent = [0]

    def find(label: int) -> int:
        root = label
        while parent[root] != root:
            root = parent[root]
        while parent[label] != root:
            parent[label], label = root, parent[label]
        return root

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_first] = root_second

    next_label = 1
    previous_runs: list[tuple[int, int, int]] = []
    for row in range(height):
        starts = np.flatnonzero(differences[row] == 1)
        ends = np.flatnonzero(differences[row] == -1)
        current_runs = []
        for start, end in zip(starts, ends):
            label = next_label
            next_label += 1
            parent.append(label)
            current_runs.append((int(start), int(end), label))

        previous_index = 0
        for start, end, label in current_runs:
            while (
                previous_index < len(previous_runs)
                and previous_runs[previous_index][1] <= start
            ):
                previous_index += 1
            overlap_index = previous_index
            while (
                overlap_index < len(previous_runs)
                and previous_runs[overlap_index][0] < end
            ):
                union(label, previous_runs[overlap_index][2])
                overlap_index += 1

        row_runs.append(current_runs)
        previous_runs = current_runs

    areas: dict[int, int] = {}
    for runs in row_runs:
        for start, end, label in runs:
            root = find(label)
            areas[root] = areas.get(root, 0) + end - start
    retained = {root for root, area in areas.items() if area >= minimum_area}

    result = np.zeros((height, width), dtype=image.dtype)
    for row, runs in enumerate(row_runs):
        for start, end, label in runs:
            if find(label) in retained:
                result[row, start:end] = 1
    return result


def _leaf_diameter(points: NDArray[np.float64]) -> float:
    maximum_squared_distance = 0.0
    chunk_size = 512
    for start in range(0, len(points), chunk_size):
        differences = points[start : start + chunk_size, None, :] - points[None, :, :]
        squared_distances = np.sum(np.square(differences), axis=2)
        maximum_squared_distance = max(
            maximum_squared_distance,
            float(squared_distances.max(initial=0)),
        )
    return float(np.sqrt(maximum_squared_distance))


def _extract_features(gray: NDArray[np.float64]) -> NDArray[np.float64]:
    mask = _area_opening((gray < 0.93).astype(np.float64), MIN_LEAF_AREA)
    kernel_5 = np.ones((5, 5), dtype=np.uint8)
    border = mask - _erosion(mask, kernel_5)

    y_points, x_points = np.nonzero(mask)
    points = np.column_stack((x_points, y_points)).astype(np.float64)
    y_border, x_border = np.nonzero(border)
    border_points = np.column_stack((x_border, y_border)).astype(np.float64)
    if len(points) < 2 or len(border_points) < 2:
        raise InvalidLeafImage(
            "Não foi possível segmentar uma folha. Use uma imagem com folha "
            "visível sobre fundo claro."
        )

    center = border_points.mean(axis=0)
    centered_border = border_points - center
    covariance = centered_border.T @ centered_border / (len(border_points) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]

    primary_projection = (points - center) @ eigenvectors[:, 0]
    p1 = points[np.argmin(primary_projection)]
    p2 = points[np.argmax(primary_projection)]
    secondary_axis = eigenvectors[:, 1]
    secondary_axis = secondary_axis / np.linalg.norm(secondary_axis)
    secondary_projection = (points - center) @ secondary_axis
    q1 = center + secondary_projection.min() * secondary_axis
    q2 = center + secondary_projection.max() * secondary_axis

    area = float(np.count_nonzero(mask))
    perimeter = float(np.count_nonzero(border))
    length = float(np.linalg.norm(p1 - p2))
    width = float(np.linalg.norm(q1 - q2))
    diameter = _leaf_diameter(border_points)
    if min(area, perimeter, length, width, diameter) <= 0:
        raise InvalidLeafImage("A folha segmentada não possui dimensões válidas.")

    eroded_5 = _erosion(mask, kernel_5)
    eroded_3 = _erosion(mask, np.ones((3, 3), dtype=np.uint8))
    area_5 = float(np.count_nonzero(eroded_5))
    area_3 = float(np.count_nonzero(eroded_3))
    if area_3 == 0:
        raise InvalidLeafImage("A folha segmentada é pequena demais.")
    smooth_factor = area_5 / area_3

    vein_areas = []
    for radius in (1, 2, 3, 4):
        y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        disk = ((x * x + y * y) <= radius * radius).astype(np.uint8)
        opened = _opening(gray, disk)
        vein_areas.append(float(np.count_nonzero((gray - opened) > 0)))
    if vein_areas[0] == 0:
        raise InvalidLeafImage(
            "Não foi possível extrair as features de textura da folha."
        )

    return np.array(
        [
            smooth_factor,
            length / width,
            4 * np.pi * area / np.square(perimeter),
            (length * width) / area,
            diameter / length,
            perimeter / diameter,
            perimeter / (length + width),
            vein_areas[0] / area,
            vein_areas[1] / area,
            vein_areas[2] / area,
            vein_areas[3] / area,
            vein_areas[3] / vein_areas[0],
        ],
        dtype=np.float64,
    )


def extract_leaf_features(image_bytes: bytes) -> NDArray[np.float64]:
    if not image_bytes:
        raise InvalidLeafImage("O arquivo enviado está vazio.")
    return _extract_features(_grayscale(image_bytes))
