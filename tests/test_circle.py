"""Ajustement radial de cercles, refus des ellipses et export de vraies primitives."""

from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

from raster2vector import (
    Point, Circle, RawPolyline, GraphEdge, GraphNode, SkeletonGraph, GeometryResult,
    fit_circle, reconstruct_geometry, skeleton_to_graph, skeletonize_mask, svg_document,
)
from raster2vector.cli import main


def circle_points(noise: float = 0) -> tuple[Point, ...]:
    angles = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    radii = 30 + noise * np.sin(7 * angles)
    points = tuple(Point(float(60 + r * np.cos(a)), float(50 + r * np.sin(a)))
                   for r, a in zip(radii, angles))
    return (*points, points[0])


def graph_for(points: tuple[Point, ...], closed: bool = True) -> SkeletonGraph:
    node = GraphNode(0, "loop" if closed else "endpoint", points[0], ((round(points[0].y), round(points[0].x)),))
    nodes = (node,) if closed else (node, GraphNode(1, "endpoint", points[-1], ((round(points[-1].y), round(points[-1].x)),)))
    return SkeletonGraph(121, 101, nodes, (GraphEdge(0, 0, 0 if closed else 1, points),))


def test_perfect_circle_and_source_preservation() -> None:
    points = circle_points()
    fit = fit_circle(points)
    assert fit is not None
    assert (fit.center.x, fit.center.y) == pytest.approx((60, 50), abs=1e-10)
    assert fit.radius == pytest.approx(30, abs=1e-10)
    assert fit.fit_error_max < 1e-10
    assert fit.fit_error_rms < 1e-10
    assert fit_circle(points[:-1]) == fit  # L'ancre répétée n'est pas pondérée deux fois.
    graph = graph_for(points)
    circle = reconstruct_geometry(graph, 1e-9).primitives[0]
    assert isinstance(circle, Circle)
    assert circle.source_points == points and circle.source_edge_id == 0
    assert graph == graph_for(points)


def test_noisy_circle_and_inclusive_radial_tolerance() -> None:
    points = circle_points(0.4)
    fit = fit_circle(points)
    assert fit is not None
    assert (fit.center.x, fit.center.y) == pytest.approx((60, 50), abs=0.01)
    assert fit.radius == pytest.approx(30, abs=0.01)
    assert fit.fit_error_max == pytest.approx(0.4, abs=0.01)
    assert 0 < fit.fit_error_rms < fit.fit_error_max
    graph = graph_for(points)
    assert isinstance(reconstruct_geometry(graph, fit.fit_error_max).primitives[0], Circle)
    raw = reconstruct_geometry(graph, np.nextafter(fit.fit_error_max, 0), detect_beziers=False).primitives[0]
    assert isinstance(raw, RawPolyline)
    assert raw.points == points and raw.is_loop
    assert raw.fit_error_max == fit.fit_error_max


def test_ellipse_exceeds_tolerance() -> None:
    angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    points = tuple(Point(float(60 + 40 * np.cos(a)), float(50 + 20 * np.sin(a))) for a in angles)
    points = (*points, points[0])
    raw = reconstruct_geometry(graph_for(points), 2, detect_beziers=False).primitives[0]
    assert isinstance(raw, RawPolyline)
    assert raw.fit_error_max is not None and raw.fit_error_max > 2
    assert raw.points == points


def test_open_arc_and_retraced_arc_do_not_become_full_circles() -> None:
    arc = circle_points()[:50]
    assert not isinstance(reconstruct_geometry(graph_for(arc, closed=False), 1).primitives[0], Circle)
    retraced = (*arc, *arc[-2::-1])
    raw = reconstruct_geometry(graph_for(retraced), 1).primitives[0]
    assert isinstance(raw, RawPolyline)
    assert raw.points == retraced


def test_distinct_junction_contacts_remain_raw() -> None:
    points = circle_points()[:-1]
    raw = reconstruct_geometry(graph_for(points), 1).primitives[0]
    assert isinstance(raw, RawPolyline)
    assert raw.fit_error_max is None


def test_radial_metrics_match_direct_distances() -> None:
    points = circle_points(0.7)
    fit = fit_circle(points)
    assert fit is not None
    errors = np.array([abs(np.hypot(p.x - fit.center.x, p.y - fit.center.y) - fit.radius) for p in points[:-1]])
    assert fit.fit_error_max == pytest.approx(errors.max(), abs=1e-12)
    assert fit.fit_error_rms == pytest.approx(np.sqrt(np.mean(errors ** 2)), abs=1e-12)


def test_degenerate_and_nonfinite_points() -> None:
    assert fit_circle(()) is None
    assert fit_circle((Point(0, 0), Point(1, 1))) is None
    assert fit_circle((Point(0, 0), Point(1, 1), Point(2, 2))) is None
    assert fit_circle((Point(1, 1),) * 4) is None
    with pytest.raises(ValueError, match="coordonnées"):
        fit_circle((Point(0, 0), Point(1, 1), Point(float("inf"), 2)))


def test_large_coordinate_translation() -> None:
    points = tuple(Point(p.x + 1e9, p.y - 1e9) for p in circle_points())
    fit = fit_circle(points)
    assert fit is not None
    assert (fit.center.x, fit.center.y) == pytest.approx((1e9 + 60, -1e9 + 50), abs=1e-6)
    assert fit.radius == pytest.approx(30, abs=1e-6)
    assert fit.fit_error_max < 1e-6


@pytest.mark.parametrize("ellipse", [False, True])
def test_rasterized_circle_and_ellipse(ellipse: bool) -> None:
    image = np.zeros((101, 121), dtype=np.uint8)
    if ellipse:
        cv2.ellipse(image, (60, 50), (40, 20), 0, 0, 360, 1, 5)
    else:
        cv2.circle(image, (60, 50), 30, 1, 5)
    graph = skeleton_to_graph(skeletonize_mask(image.astype(np.bool_)))
    result = reconstruct_geometry(graph, 2, detect_beziers=False)
    assert len(result.primitives) == 1
    if ellipse:
        assert isinstance(result.primitives[0], RawPolyline)
    else:
        circle = result.primitives[0]
        assert isinstance(circle, Circle)
        assert (circle.center.x, circle.center.y) == pytest.approx((60, 50), abs=0.5)
        assert circle.radius == pytest.approx(30, abs=0.5)
        assert circle.fit_error_max <= 2


def test_svg_circle_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "circle.png"
    image = np.full((101, 121), 255, dtype=np.uint8)
    cv2.circle(image, (60, 50), 30, 0, 5)
    success, encoded = cv2.imencode(".png", image)
    assert success
    source.write_bytes(encoded.tobytes())
    output = tmp_path / "circle.svg"
    assert main([str(source), "--geometry-tolerance", "2", "--output", str(output)]) == 0
    ns = {"s": "http://www.w3.org/2000/svg"}
    root = ET.parse(output).getroot()
    elements = root.findall("s:g/s:circle", ns)
    assert len(elements) == 1 and root.findall("s:g/s:path", ns) == []
    assert elements[0].get("fill") == "none" and elements[0].get("stroke") == "black"
    assert float(elements[0].get("r", "0")) == pytest.approx(30, abs=0.5)
    assert root.find("s:g", ns).get("transform") == "translate(0.5 0.5)"
    assert "cercles=1" in capsys.readouterr().out


@pytest.mark.parametrize("radius", [0, -1, float("nan"), float("inf")])
def test_invalid_svg_circle_radius(radius: float) -> None:
    geometry = GeometryResult(40, 40, (Circle(Point(20, 20), radius, 0, 0, 0, ()),))
    with pytest.raises(ValueError, match="rayon"):
        svg_document(geometry)
