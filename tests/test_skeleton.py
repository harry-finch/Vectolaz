"""Tests de continuité, d'épaisseur et de topologie sur six dessins connus."""

import cv2
import numpy as np
from numpy.typing import NDArray
import pytest
from skimage.measure import euler_number

from raster2vector.skeleton import skeletonize_mask


def synthetic_mask(name: str) -> NDArray[np.bool_]:
    """Dessine des formes de 7 pixels d'épaisseur, loin des bords."""
    image = np.zeros((81, 101), dtype=np.uint8)
    if name == "horizontal":
        cv2.line(image, (15, 40), (85, 40), 1, 7)
    elif name == "diagonal":
        cv2.line(image, (20, 15), (80, 65), 1, 7)
    elif name == "L":
        cv2.line(image, (25, 15), (25, 60), 1, 7)
        cv2.line(image, (25, 60), (80, 60), 1, 7)
    elif name == "T":
        cv2.line(image, (15, 20), (85, 20), 1, 7)
        cv2.line(image, (50, 20), (50, 65), 1, 7)
    elif name == "cross":
        cv2.line(image, (15, 40), (85, 40), 1, 7)
        cv2.line(image, (50, 15), (50, 65), 1, 7)
    elif name == "circle":
        cv2.circle(image, (50, 40), 25, 1, 7)
    else:
        raise ValueError(f"Cas inconnu : {name}")
    return image.astype(np.bool_)


@pytest.mark.parametrize("name, endpoints, holes", [
    ("horizontal", 2, 0), ("diagonal", 2, 0), ("L", 2, 0),
    ("T", 3, 0), ("cross", 4, 0), ("circle", 0, 1),
])
def test_synthetic_topology(name: str, endpoints: int, holes: int) -> None:
    mask = synthetic_mask(name)
    before = mask.copy()
    skeleton = skeletonize_mask(mask)
    assert skeleton.dtype == np.bool_
    assert skeleton.shape == mask.shape
    assert np.array_equal(mask, before)
    assert np.all(~skeleton | mask)  # Aucun pixel ajouté hors du trait.
    assert 0 < skeleton.sum() < mask.sum()

    # Une seule composante de trait ; le nombre de trous ne change pas.
    assert cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)[0] - 1 == 1
    assert euler_number(mask, connectivity=2) == 1 - holes
    assert euler_number(skeleton, connectivity=2) == 1 - holes

    # Vérifie les extrémités attendues sans construire de graphe.
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbors = cv2.filter2D(skeleton.astype(np.uint8), cv2.CV_16U, kernel,
                             borderType=cv2.BORDER_CONSTANT)
    assert np.count_nonzero(skeleton & (neighbors == 1)) == endpoints
    assert not np.any(skeleton[:-1, :-1] & skeleton[1:, :-1]
                      & skeleton[:-1, 1:] & skeleton[1:, 1:])


def test_horizontal_centerline() -> None:
    skeleton = skeletonize_mask(synthetic_mask("horizontal"))
    for col in range(20, 81):
        rows = np.flatnonzero(skeleton[:, col])
        assert rows.size == 1
        assert abs(int(rows[0]) - 40) <= 1


@pytest.mark.parametrize("thickness", [4, 5, 6])
def test_centerline_for_different_thicknesses(thickness: int) -> None:
    image = np.zeros((41, 81), dtype=np.uint8)
    cv2.line(image, (10, 20), (70, 20), 1, thickness)
    skeleton = skeletonize_mask(image.astype(np.bool_))
    for col in range(15, 66):
        rows = np.flatnonzero(skeleton[:, col])
        assert rows.size == 1
        assert abs(int(rows[0]) - 20) <= 1


def test_diagonal_centerline() -> None:
    rows, cols = np.nonzero(skeletonize_mask(synthetic_mask("diagonal")))
    # Distance perpendiculaire à la droite passant par (20,15) et (80,65).
    distance = np.abs(50 * (cols - 20) - 60 * (rows - 15)) / np.hypot(50, 60)
    assert np.max(distance) <= 1.5


@pytest.mark.parametrize("name, row, col", [("T", 20, 50), ("cross", 40, 50)])
def test_intersection_remains_near_center(name: str, row: int, col: int) -> None:
    skeleton = skeletonize_mask(synthetic_mask(name))
    assert skeleton[row - 1:row + 2, col - 1:col + 2].any()


def test_circle_centerline() -> None:
    rows, cols = np.nonzero(skeletonize_mask(synthetic_mask("circle")))
    assert np.max(np.abs(np.hypot(cols - 50, rows - 40) - 25)) <= 1.5


def test_empty_foreground_single_pixel_and_multiple_components() -> None:
    mask = np.zeros((20, 30), dtype=np.bool_)
    assert not skeletonize_mask(mask).any()
    mask[3, 4] = True
    assert np.array_equal(skeletonize_mask(mask), mask)
    mask[10:15, 20:25] = True
    skeleton = skeletonize_mask(mask)
    assert cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)[0] - 1 == 2


def test_noncontiguous_input_and_idempotence() -> None:
    mask = synthetic_mask("T")[:, ::-1]
    skeleton = skeletonize_mask(mask)
    assert np.array_equal(skeletonize_mask(skeleton), skeleton)


@pytest.mark.parametrize("mask", [
    np.zeros((5, 5), dtype=np.uint8), np.zeros((5, 5, 1), dtype=np.bool_),
    np.zeros((0, 5), dtype=np.bool_),
])
def test_invalid_mask(mask: NDArray) -> None:
    with pytest.raises(ValueError, match="masque booléen 2D non vide"):
        skeletonize_mask(mask)
