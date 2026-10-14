"""Vérifications reproductibles sur des dessins synthétiques."""

import cv2
import numpy as np
from numpy.typing import NDArray
import pytest

from raster2vector.preprocessing import binarize, remove_small_objects, to_grayscale


def drawing() -> NDArray[np.uint8]:
    """Un trait épais sur fond blanc, aux dimensions non carrées."""
    image = np.full((32, 60), 255, dtype=np.uint8)
    cv2.line(image, (8, 16), (50, 16), 0, 5)
    return image


def test_grayscale_preserves_dimensions_and_input() -> None:
    image = drawing()
    before = image.copy()
    result = to_grayscale(image)
    assert result.shape == (32, 60)
    assert result.dtype == np.uint8
    assert np.array_equal(result, before)
    result[0, 0] = 0
    assert np.array_equal(image, before)


def test_color_uses_bgr_order() -> None:
    image = np.array([[[0, 0, 255], [255, 0, 0]]], dtype=np.uint8)
    assert to_grayscale(image).tolist() == [[76, 29]]


@pytest.mark.parametrize("dtype, maximum", [(np.uint8, 255), (np.uint16, 65535)])
def test_transparency_is_composited_on_white(dtype: type, maximum: int) -> None:
    image = np.zeros((3, 4, 4), dtype=dtype)
    image[1, 1, 3] = maximum  # Noir opaque au centre, noir transparent ailleurs.
    grayscale = to_grayscale(image)
    expected = np.full((3, 4), 255, dtype=np.uint8)
    expected[1, 1] = 0
    assert np.array_equal(grayscale, expected)
    assert np.count_nonzero(binarize(grayscale)) == 1


def test_partial_alpha() -> None:
    image = np.array([[[0, 0, 0, 128]]], dtype=np.uint8)
    assert to_grayscale(image).item() == 127


def test_16_bit_grayscale_normalization() -> None:
    image = np.array([[0, 128 * 257, 65535]], dtype=np.uint16)
    assert to_grayscale(image).tolist() == [[0, 128, 255]]


def test_threshold_is_inclusive() -> None:
    grayscale = np.array([[0, 127, 128, 129, 255]], dtype=np.uint8)
    result = binarize(grayscale, threshold=128, invert="yes")
    assert result.dtype == np.bool_
    assert result.tolist() == [[True, True, True, False, False]]
    assert binarize(grayscale, 0, "yes").tolist() == [[True, False, False, False, False]]
    assert binarize(grayscale, 255, "yes").all()


def test_auto_inversion_for_both_polarities() -> None:
    image = drawing()
    expected = image == 0
    assert np.array_equal(binarize(image), expected)
    assert np.array_equal(binarize(255 - image), expected)
    assert np.array_equal(binarize(255 - image, invert="no"), expected)


def test_explicit_inversion_when_border_is_misleading() -> None:
    image = np.full((20, 20), 255, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (19, 19), 0, 2)
    assert np.array_equal(binarize(image, invert="yes"), image == 0)
    assert np.array_equal(binarize(image), image == 255)


def test_small_noise_removed_without_changing_thick_line() -> None:
    image = drawing()
    original_line = image == 0
    image[2, 2] = 0
    image[3, 55:57] = 0
    binary = binarize(image)
    before = binary.copy()
    cleaned = remove_small_objects(binary, min_component_size=3)
    assert np.array_equal(cleaned, original_line)
    assert np.array_equal(binary, before)
    assert np.array_equal(remove_small_objects(binary), binary)


def test_diagonal_connectivity_and_minimum_boundary() -> None:
    mask = np.eye(5, dtype=np.bool_)
    assert np.array_equal(remove_small_objects(mask, 5), mask)
    assert not remove_small_objects(mask, 6).any()


def test_empty_foreground_and_holes() -> None:
    assert not remove_small_objects(np.zeros((20, 20), dtype=np.bool_), 3).any()
    image = np.full((40, 40), 255, dtype=np.uint8)
    cv2.circle(image, (20, 20), 10, 0, 3)
    mask = binarize(image)
    cleaned = remove_small_objects(mask, 10)
    assert np.array_equal(cleaned, mask)
    assert not cleaned[20, 20]  # Le nettoyage ne remplit pas le cercle.


@pytest.mark.parametrize("threshold", [-1, 256])
def test_invalid_threshold(threshold: int) -> None:
    with pytest.raises(ValueError, match="seuil"):
        binarize(drawing(), threshold=threshold)


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="inversion"):
        binarize(drawing(), invert="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="taille minimale"):
        remove_small_objects(drawing() == 0, -1)
    with pytest.raises(ValueError, match="booléen"):
        remove_small_objects(drawing())
    with pytest.raises(ValueError, match="uint8"):
        binarize(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="canaux"):
        to_grayscale(np.zeros((2, 2, 2), dtype=np.uint8))

