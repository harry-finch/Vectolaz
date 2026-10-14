"""Simplification ouverte, coins conservés et distances contrôlées."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from raster2vector import (
    Point, GraphNode, GraphEdge, SkeletonGraph, LineSegment, Polyline, RawPolyline,
    reconstruct_geometry, simplify_polyline, skeletonize_mask, skeleton_to_graph,
)
from raster2vector.debug import save_geometry_debug
from raster2vector.image_io import load_image


def densify(vertices: tuple[Point, ...]) -> tuple[Point, ...]:
    points = [vertices[0]]
    for start, end in zip(vertices, vertices[1:]):
        count = int(max(abs(end.x - start.x), abs(end.y - start.y)))
        for fraction in np.linspace(0, 1, count + 1)[1:]:
            points.append(Point(float(start.x + fraction * (end.x - start.x)),
                                float(start.y + fraction * (end.y - start.y))))
    return tuple(points)


def graph_for(points: tuple[Point, ...]) -> SkeletonGraph:
    nodes = tuple(GraphNode(index, "endpoint", point, ((round(point.y), round(point.x)),))
                  for index, point in enumerate((points[0], points[-1])))
    return SkeletonGraph(128, 96, nodes, (GraphEdge(0, 0, 1, points),))


@pytest.mark.parametrize("vertices", [
    (Point(10, 10), Point(10, 60), Point(80, 60)),
    (Point(10, 20), Point(40, 20), Point(40, 60), Point(80, 60), Point(80, 20)),
])
def test_L_and_zigzag_keep_only_corners(vertices: tuple[Point, ...]) -> None:
    source = densify(vertices)
    assert simplify_polyline(source, 2) == vertices
    graph = graph_for(source)
    polyline = reconstruct_geometry(graph, 2, 2, detect_arcs=False).primitives[0]
    assert isinstance(polyline, Polyline)
    assert polyline.points == vertices
    assert polyline.segment_count == len(vertices) - 1
    assert polyline.source_points == source
    assert polyline.source_edge_id == 0
    assert polyline.fit_error_max < 1e-12
    assert polyline.source_indices[0] == 0
    assert polyline.source_indices[-1] == len(source) - 1


def test_almost_straight_branch_keeps_one_fitted_segment() -> None:
    source = tuple(Point(float(x), 20 + 0.2 * np.sin(x)) for x in range(10, 100))
    result = reconstruct_geometry(graph_for(source), 0.5, 0.01)
    assert isinstance(result.primitives[0], LineSegment)


def test_smooth_curve_has_few_segments_and_bounded_errors() -> None:
    source = tuple(Point(float(x), float(20 + 25 * np.sin(np.pi * (x - 10) / 100)))
                   for x in np.linspace(10, 110, 501))
    counts = []
    for tolerance in (0.5, 2.0, 4.0):
        polyline = reconstruct_geometry(graph_for(source), 0.1, tolerance, detect_arcs=False, detect_beziers=False).primitives[0]
        assert isinstance(polyline, Polyline)
        assert polyline.points[0] == source[0] and polyline.points[-1] == source[-1]
        assert polyline.source_points == source
        assert 0 < polyline.fit_error_rms <= polyline.fit_error_max <= tolerance
        counts.append(polyline.segment_count)
        # Calcul indépendant des distances aux segments affectés à chaque
        # sous-chaîne ; vérifie la tolérance et les métriques exposées.
        errors = np.zeros(len(source))
        for a, b in zip(polyline.source_indices, polyline.source_indices[1:]):
            start = np.array((source[a].x, source[a].y))
            end = np.array((source[b].x, source[b].y))
            direction = end - start
            for index in range(a, b + 1):
                point = np.array((source[index].x, source[index].y))
                fraction = np.clip(np.dot(point - start, direction) / np.dot(direction, direction), 0, 1)
                errors[index] = np.linalg.norm(point - (start + fraction * direction))
        assert polyline.fit_error_max == pytest.approx(errors.max())
        assert polyline.fit_error_rms == pytest.approx(np.sqrt(np.mean(errors ** 2)))
    assert counts[0] >= counts[1] >= counts[2]
    assert counts[1] <= 10  # Pas de fragmentation en dizaines de petits segments.


def test_distances_are_to_finite_segments() -> None:
    source = (Point(0, 0), Point(20, 0), Point(1, 0))
    assert simplify_polyline(source, 1) == source


def test_redundant_short_segment_at_rounded_corner_is_removed() -> None:
    source = (Point(21, 14), Point(20, 63), Point(23, 65), Point(141, 64))
    assert simplify_polyline(source, 2) == (source[0], source[1], source[-1])


def test_tolerance_is_inclusive_and_zero_removes_collinear_points() -> None:
    source = (Point(0, 0), Point(10, 2), Point(20, 0))
    assert simplify_polyline(source, 2) == (source[0], source[-1])
    assert simplify_polyline(source, np.nextafter(2.0, 0.0)) == source
    assert simplify_polyline((Point(0, 0), Point(10, 0), Point(20, 0)), 0) == (Point(0, 0), Point(20, 0))


def test_separate_tolerances_and_optional_disable() -> None:
    source = tuple(Point(float(x), float(20 + 12 * np.sin(np.pi * x / 40))) for x in range(41))
    graph = graph_for(source)
    assert isinstance(reconstruct_geometry(graph, 1, 2, detect_arcs=False, detect_beziers=False).primitives[0], Polyline)
    assert isinstance(reconstruct_geometry(graph, 1, None, detect_arcs=False).primitives[0], RawPolyline)
    assert isinstance(reconstruct_geometry(graph, 20, 0.1).primitives[0], LineSegment)


def test_no_reduction_keeps_raw_points() -> None:
    source = (Point(0, 0), Point(20, 0), Point(20, 20))
    primitive = reconstruct_geometry(graph_for(source), 0, 0, detect_arcs=False).primitives[0]
    assert isinstance(primitive, RawPolyline)
    assert primitive.points == source


def test_short_duplicate_and_closed_inputs() -> None:
    assert simplify_polyline(()) == ()
    assert simplify_polyline((Point(1, 1),)) == (Point(1, 1),)
    assert simplify_polyline((Point(1, 1), Point(1, 1))) == (Point(1, 1), Point(1, 1))
    with pytest.raises(ValueError, match="ouverte"):
        simplify_polyline((Point(1, 1), Point(3, 3), Point(1, 1)))


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_invalid_polyline_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolérance de polyligne"):
        simplify_polyline((), tolerance)
    with pytest.raises(ValueError, match="tolérance de polyligne"):
        reconstruct_geometry(SkeletonGraph(10, 10, (), ()), polyline_tolerance_px=tolerance)


def test_nonfinite_coordinates() -> None:
    with pytest.raises(ValueError, match="coordonnées"):
        simplify_polyline((Point(0, 0), Point(float("inf"), 1)))


def test_raster_L_and_green_overlay(tmp_path: Path) -> None:
    image = np.full((96, 128), 255, dtype=np.uint8)
    cv2.polylines(image, [np.array([(20, 15), (20, 70), (100, 70)], dtype=np.int32)], False, 0, 5)
    graph = skeleton_to_graph(skeletonize_mask(image == 0))
    geometry = reconstruct_geometry(graph, 2, 2)
    assert len(geometry.primitives) == 1
    assert isinstance(geometry.primitives[0], Polyline)
    assert geometry.primitives[0].segment_count == 2
    save_geometry_debug(tmp_path, image, geometry)
    result = load_image(tmp_path / "geometry.png")
    assert result.shape == (96, 128, 3)
    assert result[70, 70, 1] > result[70, 70, 2]
