"""Configuration TOML stricte, précédence et diagnostics CLI."""

from pathlib import Path
import numpy as np
import pytest

from raster2vector import VectorizationConfig, load_config
from raster2vector.cli import main
from raster2vector.image_io import save_png


def test_defaults_and_overrides() -> None:
    settings = VectorizationConfig()
    assert settings.threshold == 128 and settings.gap_tolerance == 0 and settings.junction_cleanup
    updated = settings.with_overrides(threshold="auto", geometry_tolerance=3, gap_tolerance=None)
    assert settings.threshold == 128 and updated.threshold == "auto"
    assert updated.geometry_tolerance == 3 and updated.gap_tolerance == 0


def test_partial_config_and_unicode_path(tmp_path: Path) -> None:
    path = tmp_path / "réglages été.toml"
    path.write_text('[vectorization]\nthreshold = "auto"\ngap_tolerance = 3\njunction_cleanup = false\n', encoding="utf-8")
    settings = load_config(path)
    assert settings.threshold == "auto" and settings.gap_tolerance == 3
    assert not settings.junction_cleanup and settings.stroke_width == 1


@pytest.mark.parametrize("content", [
    "", "[other]\nthreshold=128", "vectorization=3", "[vectorization]\nunknown=1",
    "[vectorization]\nthreshold=true", "[vectorization]\nthreshold=128.0",
    "[vectorization]\nthreshold=256", '[vectorization]\nthreshold="bad"',
    "[vectorization]\ngeometry_tolerance=nan", "[vectorization]\ngap_tolerance=-1",
    '[vectorization]\nstroke_width="1"', "[vectorization]\nstroke_width=0",
    "[vectorization]\nmin_component_size=2.5", "[vectorization]\nmin_component_size=true",
    "[vectorization]\njunction_cleanup=1", '[vectorization]\ninvert="maybe"',
    "[vectorization]\nthreshold=128\nthreshold=160", "[vectorization",
    "[vectorization]\n[extra]\nkey=1", "[vectorization]\ngeometry_tolerance=true",
])
def test_invalid_config_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)
    with pytest.raises(SystemExit) as exc:
        main([str(tmp_path / "input.png"), "--config", str(path)])
    assert exc.value.code == 2


def test_huge_numeric_value_is_reported_as_validation_error() -> None:
    with pytest.raises(ValueError, match="geometry-tolerance"):
        VectorizationConfig(geometry_tolerance=10**400)


def test_cli_explicit_defaults_override_file_and_booleans(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    image = np.full((30, 60), 255, dtype=np.uint8)
    image[15, 5:55] = 150
    source = tmp_path / "input.png"
    save_png(source, image)
    config = tmp_path / "settings.toml"
    config.write_text("[vectorization]\nthreshold=160\ngeometry_tolerance=3\ngap_tolerance=3\njunction_cleanup=false\n", encoding="utf-8")
    assert main([str(source), "--config", str(config)]) == 0
    output = capsys.readouterr().out
    assert "Seuil : 160" in output and "tolérance=3 px" in output and "Jonctions :" not in output
    assert main([str(source), "--config", str(config), "--threshold", "128", "--geometry-tolerance", "2",
                 "--gap-tolerance", "0", "--junction-cleanup"]) == 0
    output = capsys.readouterr().out
    assert "Seuil : 128" in output and "tolérance=2 px" in output
    assert "Raccords :" not in output and "Jonctions :" in output
    assert main([str(source), "--config", str(config), "--no-junction-cleanup"]) == 0
    assert "Jonctions :" not in capsys.readouterr().out


def test_config_source_must_not_be_overwritten(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "input.png"
    save_png(source, np.full((10, 10), 255, dtype=np.uint8))
    config = tmp_path / "settings.toml"
    config.write_text("[vectorization]\nthreshold=128\n", encoding="utf-8")
    before = config.read_bytes()
    assert main([str(source), "--config", str(config), "--output", str(config)]) == 1
    assert config.read_bytes() == before
    assert "configuration source" in capsys.readouterr().err


def test_help_version_missing_config_and_flag_conflict(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for flag in ("--help", "--version"):
        with pytest.raises(SystemExit) as exc:
            main([flag, "--config", str(tmp_path / "missing.toml")])
        assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "auto|0..255" in text and "Priorité" in text and "raster2vector 0.1.0" in text
    with pytest.raises(SystemExit) as exc:
        main(["input.png", "--config", str(tmp_path / "missing.toml")])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        main(["input.png", "--junction-cleanup", "--no-junction-cleanup"])
    assert exc.value.code == 2
