"""Tests unitaires pour le pipeline autonome et la configuration des primitives globales."""

import math
from pathlib import Path
import cv2
import numpy as np
import pytest

from raster2vector.coordinates import Point
from raster2vector.config import VectorizationConfig, load_config
from raster2vector.geometry import (
    Circle,
    CircularArc,
    LineSegment,
    Polyline,
    RawPolyline,
    primitive_source_edge_ids,
)
from raster2vector.bezier import BezierPath
from raster2vector.global_circles import GlobalCircle, GlobalCircleResult
from raster2vector.global_arcs import GlobalArc, GlobalArcResult
from raster2vector.pipeline import PipelineResult, run_pipeline
from raster2vector.cli import main
from raster2vector.image_io import save_png


def test_primitive_source_edge_ids() -> None:
    """Vérifie l'accès unifié aux identifiants sources pour toutes les primitives."""
    p1 = Point(10.0, 20.0)
    p2 = Point(30.0, 40.0)

    seg = LineSegment(p1, p2, 0.1, 0.05, 42, (p1, p2))
    assert primitive_source_edge_ids(seg) == (42,)

    poly = Polyline((p1, p2), 0.1, 0.05, 7, (p1, p2), (0, 1))
    assert primitive_source_edge_ids(poly) == (7,)

    raw = RawPolyline((p1, p2), 15, is_loop=False)
    assert primitive_source_edge_ids(raw) == (15,)

    circ = Circle(p1, 25.0, 0.2, 0.1, 9, (p1, p2))
    assert primitive_source_edge_ids(circ) == (9,)

    arc = CircularArc(p1, 25.0, 0.0, 1.5, 0.2, 0.1, 13, (p1, p2))
    assert primitive_source_edge_ids(arc) == (13,)

    bez = BezierPath((), 99, (p1, p2), 0.1, 0.05, 0.2)
    assert primitive_source_edge_ids(bez) == (99,)

    # Primitives globales multi-branches
    g_circ = GlobalCircle(p1, 25.0, 0.2, 0.1, 3, (p1, p2), (3, 4, 5))
    assert primitive_source_edge_ids(g_circ) == (3, 4, 5)

    g_arc = GlobalArc(p1, 25.0, 0.0, 1.5, 0.2, 0.1, 2, (p1, p2), (2, 8), (2, 8))
    assert primitive_source_edge_ids(g_arc) == (2, 8)


def test_config_defaults_and_dependencies() -> None:
    """Vérifie les valeurs par défaut et l'implication global_arcs => global_circles."""
    config = VectorizationConfig()
    assert config.global_circles is False
    assert config.global_arcs is False

    # Activer global_arcs active automatiquement global_circles
    config_arcs = VectorizationConfig(global_arcs=True)
    assert config_arcs.global_arcs is True
    assert config_arcs.global_circles is True

    # Activer global_circles seul ne touche pas à global_arcs
    config_circles = VectorizationConfig(global_circles=True)
    assert config_circles.global_circles is True
    assert config_circles.global_arcs is False


def test_toml_loading_with_and_without_globals(tmp_path: Path) -> None:
    """Vérifie le chargement TOML et la surcharge CLI via with_overrides."""
    # Fichier TOML classique sans options globales (rétrocompatibilité)
    p_classic = tmp_path / "classic.toml"
    p_classic.write_text("[vectorization]\nthreshold = 140\ngeometry_tolerance = 2.5\n", encoding="utf-8")
    cfg_classic = load_config(p_classic)
    assert cfg_classic.threshold == 140
    assert cfg_classic.geometry_tolerance == 2.5
    assert cfg_classic.global_circles is False
    assert cfg_classic.global_arcs is False

    # Fichier TOML avec global_circles = true
    p_circles = tmp_path / "circles.toml"
    p_circles.write_text("[vectorization]\nglobal_circles = true\n", encoding="utf-8")
    cfg_circles = load_config(p_circles)
    assert cfg_circles.global_circles is True
    assert cfg_circles.global_arcs is False

    # Fichier TOML avec global_arcs = true (entraîne global_circles = true)
    p_arcs = tmp_path / "arcs.toml"
    p_arcs.write_text("[vectorization]\nglobal_arcs = true\n", encoding="utf-8")
    cfg_arcs = load_config(p_arcs)
    assert cfg_arcs.global_arcs is True
    assert cfg_arcs.global_circles is True

    # Surcharges CLI via with_overrides
    overridden = cfg_classic.with_overrides(global_circles=True, global_arcs=None)
    assert overridden.global_circles is True
    assert overridden.global_arcs is False

    overridden_arcs = cfg_classic.with_overrides(global_arcs=True)
    assert overridden_arcs.global_arcs is True
    assert overridden_arcs.global_circles is True


@pytest.mark.parametrize("invalid_val", [1, 0, "true", "false", 1.0, "yes", None])
def test_reject_non_boolean_for_globals(invalid_val: object) -> None:
    """Rejette strictement toute valeur non-booléenne (entier, chaîne, etc.)."""
    with pytest.raises(ValueError, match="global_circles"):
        VectorizationConfig(global_circles=invalid_val)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="global_arcs"):
        VectorizationConfig(global_arcs=invalid_val)  # type: ignore[arg-type]


@pytest.mark.parametrize("toml_line", [
    "global_circles = 1",
    'global_circles = "true"',
    "global_arcs = 1",
    'global_arcs = "false"',
])
def test_reject_invalid_toml_types_for_globals(tmp_path: Path, toml_line: str) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(f"[vectorization]\n{toml_line}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_run_pipeline_synthetic_circle_and_cross() -> None:
    """Exécute run_pipeline en mémoire sur une image synthétique (cercle + croix)."""
    image = np.full((120, 120), 255, dtype=np.uint8)
    cv2.circle(image, (60, 60), 35, 0, 2)
    cv2.line(image, (20, 60), (100, 60), 0, 2)
    cv2.line(image, (60, 20), (60, 100), 0, 2)

    # 1. Pipeline nominal sans primitives globales
    cfg_default = VectorizationConfig(threshold="auto", geometry_tolerance=2.0)
    result_default = run_pipeline(image, cfg_default)

    assert isinstance(result_default, PipelineResult)
    assert result_default.grayscale.shape == (120, 120)
    assert result_default.binary.shape == (120, 120)
    assert result_default.skeleton.shape == (120, 120)
    assert len(result_default.graph.nodes) > 0
    assert len(result_default.graph.edges) > 0
    assert result_default.global_circles is None
    assert result_default.global_arcs is None
    assert len(result_default.geometry.primitives) > 0

    # 2. Pipeline avec global_circles = True
    cfg_circles = VectorizationConfig(threshold="auto", geometry_tolerance=2.0, global_circles=True)
    result_circles = run_pipeline(image, cfg_circles)
    assert result_circles.global_circles is not None
    assert isinstance(result_circles.global_circles, GlobalCircleResult)
    assert len(result_circles.global_circles.circles) == 1
    circle = result_circles.global_circles.circles[0]
    assert math.isclose(circle.radius, 35.0, abs_tol=2.0)
    assert math.isclose(circle.center.x, 60.0, abs_tol=2.0)
    assert math.isclose(circle.center.y, 60.0, abs_tol=2.0)
    assert len(circle.source_edge_ids) == 4  # 4 quadrants du cercle séparés par la croix


def test_cli_precedence_for_global_flags(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Vérifie la priorité CLI > TOML pour --global-circles et --global-arcs."""
    image = np.full((120, 120), 255, dtype=np.uint8)
    cv2.circle(image, (60, 60), 35, 0, 2)
    cv2.line(image, (20, 60), (100, 60), 0, 2)
    cv2.line(image, (60, 20), (60, 100), 0, 2)
    img_path = tmp_path / "test.png"
    save_png(img_path, image)

    # 1. TOML active global_circles, CLI par défaut -> cercles globaux détectés
    cfg_file = tmp_path / "circles.toml"
    cfg_file.write_text("[vectorization]\nglobal_circles = true\n", encoding="utf-8")
    assert main([str(img_path), "--config", str(cfg_file)]) == 0
    out = capsys.readouterr().out
    assert "Cercles globaux : 1" in out

    # 2. TOML sans options, CLI active --global-circles
    cfg_empty = tmp_path / "empty.toml"
    cfg_empty.write_text("[vectorization]\nthreshold = 128\n", encoding="utf-8")
    assert main([str(img_path), "--config", str(cfg_empty), "--global-circles"]) == 0
    out = capsys.readouterr().out
    assert "Cercles globaux : 1" in out

    # 3. CLI sans flag et TOML sans flag -> pas de cercles globaux
    assert main([str(img_path), "--config", str(cfg_empty)]) == 0
    out = capsys.readouterr().out
    assert "Cercles globaux" not in out
