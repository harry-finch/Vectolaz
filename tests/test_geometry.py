"""Tests d'ajustement sur tous les points et de reconstruction sous tolérance."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from raster2vector.coordinates import Point
from raster2vector.debug import save_geometry_debug
from raster2vector.geometry import (
    GeometryResult, LineSegment, RawPolyline, Circle, fit_line, reconstruct_geometry,
)
from raster2vector.graph import GraphEdge, GraphNode, SkeletonGraph, skeleton_to_graph
from raster2vector.image_io import load_image
from raster2vector.skeleton import skeletonize_mask


def graph_for(points: tuple[Point, ...], loop: bool = False) -> SkeletonGraph:
    nodes = tuple(GraphNode(index, "loop" if loop else "endpoint", point,
                            ((round(point.y), round(point.x)),))
                  for index, point in enumerate(points[:1] if loop else (points[0], points[-1])))
    return SkeletonGraph(128, 96, nodes, (GraphEdge(0, 0, 0 if loop else 1, points),))


@pytest.mark.parametrize("points", [
    tuple(Point(float(x), 12.0) for x in range(5, 50)),
    tuple(Point(17.0, float(y)) for y in range(5, 50)),
    tuple(Point(float(x), 0.7 * x + 10) for x in range(5, 50)),
])
def test_exact_lines(points: tuple[Point, ...]) -> None:
    fit = fit_line(points)
    assert fit is not None
    assert fit.fit_error_max < 1e-12
    assert fit.fit_error_rms < 1e-12
    assert np.hypot(fit.direction.x, fit.direction.y) == pytest.approx(1)
    assert (fit.start.x, fit.start.y) == pytest.approx((points[0].x, points[0].y))
    assert (fit.end.x, fit.end.y) == pytest.approx((points[-1].x, points[-1].y))
    primitive = reconstruct_geometry(graph_for(points), 1e-10).primitives[0]
    assert isinstance(primitive, LineSegment)
    assert primitive.source_edge_id == 0
    assert primitive.source_points == points


def test_known_errors_and_inclusive_tolerance() -> None:
    # Covariance xy nulle ; droite optimale y=20, erreur exacte de 1 pixel.
    points = (Point(7, 19), Point(9, 21), Point(11, 21), Point(13, 19))
    fit = fit_line(points)
    assert fit is not None
    assert fit.origin == Point(10, 20)
    assert fit.fit_error_max == pytest.approx(1)
    assert fit.fit_error_rms == pytest.approx(1)
    assert fit.start == Point(7, 20)
    assert fit.end == Point(13, 20)
    graph = graph_for(points)
    assert isinstance(reconstruct_geometry(graph, 1).primitives[0], LineSegment)
    raw = reconstruct_geometry(graph, np.nextafter(1.0, 0.0), polyline_tolerance_px=None, detect_arcs=False).primitives[0]
    assert isinstance(raw, RawPolyline)
    assert raw.points == points
    assert raw.fit_error_max == pytest.approx(1)


def test_noisy_line_reconstructed_with_correct_orientation() -> None:
    points = tuple(Point(float(x), 0.5 * x + 15 + (0.4 if x % 2 else -0.4)) for x in range(10, 90))
    primitive = reconstruct_geometry(graph_for(points), 0.5).primitives[0]
    assert isinstance(primitive, LineSegment)
    slope = (primitive.end.y - primitive.start.y) / (primitive.end.x - primitive.start.x)
    assert slope == pytest.approx(0.5, abs=0.001)
    assert 0 < primitive.fit_error_rms <= primitive.fit_error_max <= 0.5


def test_curve_with_aligned_endpoints_remains_raw() -> None:
    points = tuple(Point(float(x), float(20 + 12 * np.sin(np.pi * x / 40))) for x in range(41))
    # Les deux extrémités sont sur y=20, mais l'intérieur s'en écarte largement.
    raw = reconstruct_geometry(graph_for(points), 2, polyline_tolerance_px=None, detect_arcs=False).primitives[0]
    assert isinstance(raw, RawPolyline)
    assert raw.points == points
    assert raw.fit_error_max is not None and raw.fit_error_max > 2


def test_single_outlier_controls_maximum_error() -> None:
    points = tuple(Point(float(x), 20.0 if x != 20 else 24.0) for x in range(41))
    fit = fit_line(points)
    assert fit is not None
    assert fit.fit_error_rms < 1 < fit.fit_error_max
    assert isinstance(reconstruct_geometry(graph_for(points), 1, polyline_tolerance_px=None, detect_arcs=False).primitives[0], RawPolyline)


def test_closed_loop_keeps_closed_primitive_with_large_tolerance() -> None:
    points = (Point(10, 10), Point(20, 10), Point(20, 20), Point(10, 20), Point(10, 10))
    raw = reconstruct_geometry(graph_for(points, loop=True), 1000).primitives[0]
    assert isinstance(raw, Circle)
    assert raw.source_points == points


def test_loop_with_distinct_contacts_is_preserved() -> None:
    points = (Point(10, 10), Point(30, 10), Point(30, 30), Point(10, 11))
    raw = reconstruct_geometry(graph_for(points, loop=True), 1000).primitives[0]
    assert isinstance(raw, RawPolyline)
    assert raw.is_loop and raw.points == points


def test_segment_covers_all_projected_points() -> None:
    points = (Point(10, 20), Point(50, 20), Point(12, 20))
    primitive = reconstruct_geometry(graph_for(points), 0).primitives[0]
    assert isinstance(primitive, LineSegment)
    assert primitive.start == Point(10, 20)
    assert primitive.end == Point(50, 20)


def test_reverse_traversal_and_translation() -> None:
    points = tuple(Point(1e9 + x, 1e9 + 2 * x) for x in range(10))
    forward = fit_line(points)
    reverse = fit_line(points[::-1])
    assert forward is not None and reverse is not None
    assert forward.fit_error_max < 1e-12
    assert reverse.start == forward.end
    assert reverse.end == forward.start


def test_degenerate_data_and_empty_graph() -> None:
    assert fit_line(()) is None
    assert fit_line((Point(2, 3),)) is None
    points = (Point(2, 3), Point(2, 3))
    assert fit_line(points) is None
    assert isinstance(reconstruct_geometry(graph_for(points), 2).primitives[0], RawPolyline)
    result = reconstruct_geometry(SkeletonGraph(20, 10, (), ()))
    assert result == GeometryResult(20, 10, ())


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf"), -float("inf")])
def test_invalid_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolérance"):
        reconstruct_geometry(SkeletonGraph(20, 10, (), ()), tolerance)


def test_nonfinite_coordinates_rejected() -> None:
    with pytest.raises(ValueError, match="coordonnées"):
        fit_line((Point(0, 0), Point(float("nan"), 1)))


@pytest.mark.parametrize("start, end", [((10, 40), (90, 40)), ((50, 10), (50, 70)), ((15, 10), (85, 70))])
def test_raster_to_segment(start: tuple[int, int], end: tuple[int, int]) -> None:
    image = np.zeros((81, 101), dtype=np.uint8)
    cv2.line(image, start, end, 1, 5)
    graph = skeleton_to_graph(skeletonize_mask(image.astype(np.bool_)))
    before = graph
    result = reconstruct_geometry(graph, 1)
    assert graph == before
    assert len(result.primitives) == 1
    assert isinstance(result.primitives[0], LineSegment)


def test_geometry_overlay_preserves_dimensions_and_source(tmp_path: Path) -> None:
    image = np.full((40, 60, 4), 255, dtype=np.uint8)
    image[20, 10:51, :3] = 0
    before = image.copy()
    points = tuple(Point(float(x), 20) for x in range(10, 51))
    graph = graph_for(points)
    graph = SkeletonGraph(60, 40, graph.nodes, graph.edges)
    geometry = reconstruct_geometry(graph, 0)
    save_geometry_debug(tmp_path, image, geometry)
    rendered = load_image(tmp_path / "geometry.png")
    assert rendered.shape == (40, 60, 3)
    assert np.array_equal(image, before)
    assert rendered[20, 30, 2] > rendered[20, 30, 0]  # Segment rouge au bon endroit.
