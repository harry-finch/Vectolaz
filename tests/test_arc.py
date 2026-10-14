"""Ajustement d'arcs finis et conventions de parcours SVG dans le repère image."""

from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

from raster2vector import (
    Point, CircularArc, LineSegment, Polyline, RawPolyline, BezierPath, GeometryResult,
    GraphNode, GraphEdge, SkeletonGraph, fit_arc, reconstruct_geometry,
    skeleton_to_graph, skeletonize_mask, svg_document,
)
from raster2vector.cli import main


def arc_points(start: float, sweep: float) -> tuple[Point, ...]:
    angles = np.linspace(start, start + sweep, 101)
    return tuple(Point(float(100 + 60 * np.cos(a)), float(90 + 60 * np.sin(a))) for a in angles)


def graph_for(points: tuple[Point, ...]) -> SkeletonGraph:
    nodes = tuple(GraphNode(index, "endpoint", point, ((round(point.y), round(point.x)),))
                  for index, point in enumerate((points[0], points[-1])))
    return SkeletonGraph(201, 181, nodes, (GraphEdge(0, 0, 1, points),))


@pytest.mark.parametrize("start, sweep, large, direction", [
    (0, np.pi / 2, 0, 1), (0, -np.pi / 2, 0, 0),
    (0, np.pi, 0, 1), (0, -np.pi, 0, 0),
    (2.9, 0.4, 0, 1), (-2.9, -0.4, 0, 0),
    (0.2, 1.5 * np.pi, 1, 1), (0.2, -1.5 * np.pi, 1, 0),
])
def test_arcs_flags_and_svg(start: float, sweep: float, large: int, direction: int) -> None:
    points = arc_points(start, sweep)
    fit = fit_arc(points, 1e-7)
    assert fit is not None
    assert (fit.center.x, fit.center.y) == pytest.approx((100, 90), abs=1e-6)
    assert fit.radius == pytest.approx(60, abs=1e-6)
    assert fit.sweep_angle_rad == pytest.approx(sweep, abs=1e-8)
    assert fit.large_arc_flag == large and fit.sweep_flag == direction
    assert fit.fit_error_max < 1e-7
    assert (fit.start.x, fit.start.y) == pytest.approx((points[0].x, points[0].y), abs=1e-7)
    assert (fit.end.x, fit.end.y) == pytest.approx((points[-1].x, points[-1].y), abs=1e-7)
    arc = reconstruct_geometry(graph_for(points), 1e-7).primitives[0]
    assert isinstance(arc, CircularArc)
    assert arc.source_points == points and arc.source_edge_id == 0
    root = ET.fromstring(svg_document(GeometryResult(201, 181, (arc,))))
    path = root.find("{http://www.w3.org/2000/svg}g/{http://www.w3.org/2000/svg}path")
    assert path is not None
    tokens = path.get("d", "").split()
    assert tokens[0] == "M" and tokens[3] == "A"
    assert int(tokens[7]) == large and int(tokens[8]) == direction
    assert float(tokens[4]) == pytest.approx(60, abs=1e-6)
    assert "Z" not in tokens


def test_reverse_traversal_preserves_arc_and_changes_sweep() -> None:
    points = arc_points(2.8, 2)
    forward, reverse = fit_arc(points, 0.1), fit_arc(points[::-1], 0.1)
    assert forward is not None and reverse is not None
    assert forward.radius == pytest.approx(reverse.radius)
    assert forward.sweep_angle_rad == pytest.approx(-reverse.sweep_angle_rad)
    assert (forward.start.x, forward.start.y) == pytest.approx((reverse.end.x, reverse.end.y))


def test_noisy_arc_accepted_and_non_circular_curve_refused() -> None:
    angles = np.linspace(0, np.pi, 180)
    radii = 60 + 0.3 * np.sin(7 * angles)
    points = tuple(Point(float(100 + r * np.cos(a)), float(90 + r * np.sin(a))) for r, a in zip(radii, angles))
    fit = fit_arc(points, 0.5)
    assert fit is not None and 0 < fit.fit_error_rms <= fit.fit_error_max <= 0.5
    assert fit_arc(points, 0.01) is None
    ellipse = tuple(Point(float(100 + 75 * np.cos(a)), float(90 + 25 * np.sin(a))) for a in angles)
    assert fit_arc(ellipse, 2) is None
    primitive = reconstruct_geometry(graph_for(ellipse), 2).primitives[0]
    assert isinstance(primitive, (Polyline, RawPolyline, BezierPath))


def test_backtracking_and_closed_loop_refused() -> None:
    points = arc_points(0, 1.5)
    source = (*points, *points[-2:30:-1])
    assert fit_arc(source, 0.1) is None
    closed = (*points, *points[-2::-1])
    assert fit_arc(closed, 2) is None


def test_straight_line_has_priority_and_arc_can_be_disabled() -> None:
    short = arc_points(0, 0.03)
    assert isinstance(reconstruct_geometry(graph_for(short), 2).primitives[0], LineSegment)
    points = arc_points(0, np.pi / 2)
    assert isinstance(reconstruct_geometry(graph_for(points), 2).primitives[0], CircularArc)
    assert isinstance(reconstruct_geometry(graph_for(points), 2, detect_arcs=False, detect_beziers=False).primitives[0], Polyline)


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_invalid_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolérance d'arc"):
        fit_arc(arc_points(0, 1), tolerance)


def test_degenerate_points() -> None:
    assert fit_arc(()) is None
    assert fit_arc((Point(1, 1),) * 3) is None
    assert fit_arc((Point(1, 1), Point(2, 2), Point(3, 3))) is None


@pytest.mark.parametrize("radius, end", [(0, 1), (float("nan"), 1), (10, 0), (10, 7)])
def test_invalid_svg_arc(radius: float, end: float) -> None:
    arc = CircularArc(Point(20, 20), radius, 0, end, 0, 0, 0, ())
    with pytest.raises(ValueError, match="arc SVG"):
        svg_document(GeometryResult(40, 40, (arc,)))


def test_rasterized_quarter_arc_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    image = np.full((181, 201), 255, dtype=np.uint8)
    cv2.ellipse(image, (100, 90), (65, 65), 0, 0, 90, 0, 5)
    graph = skeleton_to_graph(skeletonize_mask(image == 0))
    result = reconstruct_geometry(graph, 2)
    assert len(result.primitives) == 1 and isinstance(result.primitives[0], CircularArc)
    source = tmp_path / "quarter.png"
    success, encoded = cv2.imencode(".png", image)
    assert success
    source.write_bytes(encoded.tobytes())
    output = tmp_path / "quarter.svg"
    assert main([str(source), "--output", str(output), "--debug-dir", str(tmp_path / "debug"), "--debug-svg"]) == 0
    assert " A " in output.read_text(encoding="utf-8")
    assert "arcs=1" in capsys.readouterr().out
