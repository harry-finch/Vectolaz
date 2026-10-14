"""Fitting, borne continue indépendante, priorités et export des courbes libres."""

from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pytest

from raster2vector import (
    Point, BezierPath, GeometryResult, GraphEdge, SkeletonGraph,
    Polyline, fit_bezier, evaluate_cubic, reconstruct_geometry,
    skeletonize_mask, skeleton_to_graph, svg_document,
)
from raster2vector.cli import main


def source_curve(cycles: int = 1) -> tuple[Point, ...]:
    return tuple(Point(float(x), float(110+40*np.sin((x-20)*cycles*2*np.pi/240)))
                 for x in np.linspace(20, 260, 481))


@pytest.mark.parametrize("cycles,tolerance", [(1, 2), (2, 0.5), (1, 0.05)])
def test_continuous_error_and_metrics(cycles: int, tolerance: float) -> None:
    source = source_curve(cycles)
    fit = fit_bezier(source, tolerance, source_edge_id=7)
    assert fit is not None
    assert fit.source_points == source and fit.source_edge_id == 7
    assert 0 <= fit.fit_error_rms <= fit.fit_error_max <= fit.error_bound <= tolerance
    assert fit.segments[0].start == source[0] and fit.segments[-1].end == source[-1]
    assert all(a.end == b.start and a.source_end_index == b.source_start_index
               for a, b in zip(fit.segments, fit.segments[1:]))
    errors = []
    for segment in fit.segments:
        data = np.array([(p.x, p.y) for p in source[segment.source_start_index:segment.source_end_index+1]])
        parameters = np.array(segment.source_parameters)
        point_errors = np.linalg.norm(evaluate_cubic(segment, parameters)-data, axis=1)
        errors.extend(point_errors[:-1])
        assert point_errors.max() == pytest.approx(segment.fit_error_max)
        # Calcul indépendant : interpoler chaque segment source et vérifier
        # la courbe à 11 points intermédiaires, pas seulement aux pixels.
        for index in range(len(data)-1):
            fractions = np.linspace(0, 1, 11)
            t = parameters[index]+fractions*(parameters[index+1]-parameters[index])
            raster = data[index]+fractions[:, None]*(data[index+1]-data[index])
            deviations = np.linalg.norm(evaluate_cubic(segment, t)-raster, axis=1)
            assert deviations.max() <= segment.error_bound+1e-10
    errors.append(0)
    assert fit.fit_error_rms == pytest.approx(np.sqrt(np.mean(np.array(errors)**2)))
    if cycles == 2:
        assert len(fit.segments) > 1
    assert fit == fit_bezier(source, tolerance, source_edge_id=7)


def test_known_cubic_and_reverse_traversal() -> None:
    t = np.linspace(0, 1, 301)[:, None]
    controls = np.array([(20, 100), (80, 200), (180, 0), (260, 100)])
    data = (1-t)**3*controls[0]+3*(1-t)**2*t*controls[1]+3*(1-t)*t**2*controls[2]+t**3*controls[3]
    points = tuple(Point(float(x), float(y)) for x, y in data)
    fit, reverse = fit_bezier(points, 1), fit_bezier(points[::-1], 1)
    assert fit is not None and reverse is not None
    assert len(fit.segments) == len(reverse.segments) == 1
    forward_values = evaluate_cubic(fit.segments[0], np.linspace(0, 1, 51))
    reverse_values = evaluate_cubic(reverse.segments[0], np.linspace(1, 0, 51))
    assert np.allclose(forward_values, reverse_values, atol=1e-7)


def test_sparse_interpolation_does_not_hide_overshoot_between_points() -> None:
    points = (Point(0, 0), Point(10, 20), Point(20, -20), Point(30, 0))
    loose = fit_bezier(points, 100)
    tight = fit_bezier(points, 0.5)
    assert loose is not None and tight is not None
    assert len(loose.segments) == 1 and loose.fit_error_max < 1e-10
    assert loose.error_bound > 1
    assert len(tight.segments) > 1 and tight.error_bound <= 0.5


def test_noisy_curve_keeps_all_source_points() -> None:
    source = tuple(Point(p.x, p.y+0.3*np.sin(index*2.1)) for index, p in enumerate(source_curve()))
    fit = fit_bezier(source, 2)
    assert fit is not None and fit.source_points == source
    assert fit.error_bound <= 2 and len(fit.segments) <= 3


def test_selection_reduces_nodes_and_preserves_graph() -> None:
    points = source_curve()
    graph = SkeletonGraph(281, 221, (), (GraphEdge(7, 0, 1, points),))
    before = graph
    primitive = reconstruct_geometry(graph).primitives[0]
    baseline = reconstruct_geometry(graph, detect_beziers=False).primitives[0]
    assert isinstance(primitive, BezierPath) and isinstance(baseline, Polyline)
    assert 3*len(primitive.segments)+1 < len(baseline.points)
    assert graph == before and primitive.error_bound <= 2
    root = ET.fromstring(svg_document(GeometryResult(281, 221, (primitive,))))
    path = root.find("{http://www.w3.org/2000/svg}g/{http://www.w3.org/2000/svg}path")
    assert path is not None and path.get("id") == "edge-7"
    assert path.get("d", "").split().count("C") == len(primitive.segments)
    assert "Z" not in path.get("d", "")


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_invalid_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolérance Bézier"):
        fit_bezier(source_curve(), tolerance)


def test_degenerate_and_nonfinite_inputs() -> None:
    assert fit_bezier(()) is None
    assert fit_bezier((Point(1, 1),)*4) is None
    assert fit_bezier((Point(0, 0), Point(2, 3), Point(0, 0))) is None
    fit = fit_bezier((Point(0, 0), Point(0, 0), Point(3, 1), Point(5, 2)), 1)
    assert fit is not None and fit.error_bound <= 1
    with pytest.raises(ValueError, match="coordonnées"):
        fit_bezier((Point(0, 0), Point(float("nan"), 1)))
    with pytest.raises(ValueError, match="paramètres"):
        evaluate_cubic(fit.segments[0], [1.5])
    with pytest.raises(ValueError, match="Bézier SVG"):
        svg_document(GeometryResult(20, 20, (BezierPath((), 0, (), 0, 0, 0),)))


def test_rasterized_s_curve_cli_and_overlay(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    image = np.full((221, 281), 255, dtype=np.uint8)
    points = np.rint([(p.x, p.y) for p in source_curve()]).astype(np.int32)
    cv2.polylines(image, [points], False, 0, 5)
    graph = skeleton_to_graph(skeletonize_mask(image == 0))
    assert isinstance(reconstruct_geometry(graph).primitives[0], BezierPath)
    success, encoded = cv2.imencode(".png", image)
    assert success
    source = tmp_path / "curve.png"
    source.write_bytes(encoded.tobytes())
    output = tmp_path / "curve.svg"
    assert main([str(source), "--output", str(output), "--debug-dir", str(tmp_path / "debug"), "--debug-svg"]) == 0
    assert " C " in output.read_text(encoding="utf-8")
    assert "chemins Bézier=1" in capsys.readouterr().out
    assert (tmp_path / "debug" / "geometry.png").is_file()
    assert " C " in (tmp_path / "debug" / "overlay.svg").read_text(encoding="utf-8")
