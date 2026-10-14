"""Tests synthétiques du chargement et de la CLI."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from raster2vector.cli import main
from raster2vector.image_io import load_image


@pytest.fixture
def png_path(tmp_path: Path) -> Path:
    image = np.full((24, 40, 3), 255, dtype=np.uint8)
    image[10:14, 5:35] = 0
    success, encoded = cv2.imencode(".png", image)
    assert success
    path = tmp_path / "dessin été.png"
    path.write_bytes(encoded.tobytes())
    return path


def test_load_image_preserves_pixels(png_path: Path) -> None:
    image = load_image(png_path)
    assert image.shape == (24, 40, 3)
    assert image.dtype == np.uint8
    assert np.all(image[10:14, 5:35] == 0)
    assert np.all(image[0] == 255)


def test_cli_dimensions_and_debug_dir(png_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    debug_dir = tmp_path / "debug" / "nested"
    assert main([str(png_path), "--debug-dir", str(debug_dir)]) == 0
    assert "40 × 24 pixels" in capsys.readouterr().out
    assert debug_dir.is_dir()
    assert {p.name for p in debug_dir.iterdir()} == {"grayscale.png", "binary.png", "cleaned.png", "skeleton.png", "graph.png", "geometry.png", "geometry_before.png", "junctions.png"}
    assert load_image(debug_dir / "grayscale.png").shape == (24, 40)
    assert np.array_equal(load_image(debug_dir / "binary.png"), load_image(debug_dir / "cleaned.png"))
    skeleton = load_image(debug_dir / "skeleton.png")
    assert skeleton.shape == (24, 40)
    assert set(np.unique(skeleton)) == {0, 255}
    assert 0 < np.count_nonzero(skeleton == 0) < 120
    assert load_image(debug_dir / "graph.png").shape == (24, 40, 3)
    assert load_image(debug_dir / "geometry.png").shape == (24, 40, 3)


def test_cli_missing_image(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path / "missing.png")]) == 1
    assert "Erreur :" in capsys.readouterr().err


@pytest.mark.parametrize("content", [b"", b"not a PNG"])
def test_cli_invalid_image(tmp_path: Path, capsys: pytest.CaptureFixture[str], content: bytes) -> None:
    path = tmp_path / "invalid.png"
    path.write_bytes(content)
    assert main([str(path)]) == 1
    assert "Erreur :" in capsys.readouterr().err


def test_cli_debug_path_is_file(png_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    debug_file = tmp_path / "debug"
    debug_file.write_text("existing", encoding="utf-8")
    assert main([str(png_path), "--debug-dir", str(debug_file)]) == 1
    assert "Erreur :" in capsys.readouterr().err
    assert debug_file.read_text(encoding="utf-8") == "existing"


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--debug-dir" in capsys.readouterr().out


@pytest.mark.parametrize("option, value", [
    ("--threshold", "-1"), ("--threshold", "256"), ("--min-component-size", "-1"),
    ("--geometry-tolerance", "-1"), ("--geometry-tolerance", "nan"),
    ("--geometry-tolerance", "inf"),
    ("--polyline-tolerance", "-1"), ("--polyline-tolerance", "nan"),
    ("--polyline-tolerance", "inf"),
    ("--stroke-width", "0"), ("--stroke-width", "-1"),
    ("--stroke-width", "nan"), ("--stroke-width", "inf"),
    ("--gap-tolerance", "-1"), ("--gap-tolerance", "nan"), ("--gap-tolerance", "inf"),
])
def test_cli_rejects_invalid_options(
    png_path: Path, option: str, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        main([str(png_path), option, value])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_cli_threshold_and_cleanup(tmp_path: Path) -> None:
    image = np.full((30, 50), 255, dtype=np.uint8)
    image[12:16, 5:45] = 150
    image[2, 2] = 0
    success, encoded = cv2.imencode(".png", image)
    assert success
    source = tmp_path / "input.png"
    source.write_bytes(encoded.tobytes())
    before = source.read_bytes()
    debug_dir = tmp_path / "débogage"
    assert main([
        str(source), "--threshold", "160", "--invert", "yes",
        "--min-component-size", "3", "--debug-dir", str(debug_dir),
    ]) == 0
    binary = load_image(debug_dir / "binary.png")
    cleaned = load_image(debug_dir / "cleaned.png")
    expected = np.full_like(image, 255)
    expected[12:16, 5:45] = 0
    assert binary[2, 2] == 0
    assert np.array_equal(cleaned, expected)
    assert set(np.unique(binary)) == {0, 255}
    assert source.read_bytes() == before


@pytest.mark.parametrize("name", ["binary.png", "skeleton.png", "graph.png", "geometry.png", "geometry_before.png", "junctions.png"])
def test_debug_must_not_overwrite_source(tmp_path: Path, capsys: pytest.CaptureFixture[str], name: str) -> None:
    source = tmp_path / name
    success, encoded = cv2.imencode(".png", np.full((8, 8), 255, dtype=np.uint8))
    assert success
    source.write_bytes(encoded.tobytes())
    before = source.read_bytes()
    assert main([str(source), "--debug-dir", str(tmp_path)]) == 1
    assert "écraserait" in capsys.readouterr().err
    assert source.read_bytes() == before


@pytest.mark.parametrize("tolerance, counts", [("0.5", "segments=0"), ("2", "segments=1, polylignes=0, branches brutes=0")])
def test_cli_geometry_tolerance(png_path: Path, capsys: pytest.CaptureFixture[str], tolerance: str, counts: str) -> None:
    assert main([str(png_path), "--geometry-tolerance", tolerance]) == 0
    output = capsys.readouterr().out
    assert f"tolérance={tolerance} px," in output
    assert f"Géométrie : {counts}" in output
