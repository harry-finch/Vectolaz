"""Seuillage global automatique, polarités et diagnostic du seuil réellement appliqué."""

from pathlib import Path
import numpy as np
import pytest

from raster2vector import automatic_threshold, binarize
from raster2vector.cli import main
from raster2vector.image_io import save_png, load_image


@pytest.mark.parametrize("background,stroke,expected", [(220, 190, 190), (180, 60, 60), (20, 200, 20)])
def test_known_two_intensity_images(background: int, stroke: int, expected: int) -> None:
    image = np.full((30, 50), background, dtype=np.uint8)
    image[15, 5:45] = stroke
    before = image.copy()
    threshold = automatic_threshold(image)
    assert threshold == expected
    expected_mask = np.zeros_like(image, dtype=bool)
    expected_mask[15, 5:45] = True
    assert np.array_equal(binarize(image, threshold), expected_mask)
    assert np.array_equal(image, before)


@pytest.mark.parametrize("value", [0, 80, 255])
def test_uniform_image_is_empty_with_automatic_polarity(value: int) -> None:
    image = np.full((10, 12), value, dtype=np.uint8)
    assert not binarize(image, automatic_threshold(image)).any()


@pytest.mark.parametrize("image", [np.zeros((0, 5), dtype=np.uint8), np.zeros((5, 5), dtype=np.uint16), np.zeros((5, 5, 3), dtype=np.uint8)])
def test_invalid_input(image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="gris uint8"):
        automatic_threshold(image)


def test_cli_auto_finds_faint_stroke_and_reports_threshold(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    image = np.full((30, 60), 220, dtype=np.uint8)
    image[15, 5:55] = 190
    source = tmp_path / "faint.png"
    save_png(source, image)
    debug = tmp_path / "debug"
    assert main([str(source), "--threshold", "auto", "--output", str(tmp_path / "output.svg"), "--debug-dir", str(debug)]) == 0
    assert "Seuil : 190 (auto, Otsu)" in capsys.readouterr().out
    assert np.count_nonzero(load_image(debug / "binary.png") == 0) == 50
    assert (tmp_path / "output.svg").read_text(encoding="utf-8").count("<path") == 1


@pytest.mark.parametrize("value", ["AUTO", "128.5", "invalid"])
def test_invalid_threshold_argument(value: str) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["input.png", "--threshold", value])
    assert exc.value.code == 2
