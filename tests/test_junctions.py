"""Jonctions communes, arcs contraints, continuités et conservation du graphe."""

from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pytest

from raster2vector import (
    Point, GraphNode, GraphEdge, SkeletonGraph, GeometryResult,
    LineSegment, CircularArc, JoinedArc, RawPolyline, Circle, BezierPath,
    fit_bezier, reconstruct_geometry, refine_junctions, svg_document,
    evaluate_cubic,
    skeleton_to_graph, skeletonize_mask,
)
from raster2vector.cli import main
from raster2vector.image_io import save_png


def line(start: tuple[float, float], end: tuple[float, float]) -> tuple[Point, ...]:
    return tuple(Point(float(x), float(y)) for x, y in np.linspace(start, end, 41))


def junction_graph(*branches: tuple[Point, ...]) -> SkeletonGraph:
    nodes = [GraphNode(0, "junction", Point(50, 50), ((50, 49), (50, 50), (50, 51), (49, 50)))]
    edges = []
    for index, points in enumerate(branches):
        p = points[-1]
        nodes.append(GraphNode(index+1, "endpoint", p, ((round(p.y), round(p.x)),)))
        edges.append(GraphEdge(index, 0, index+1, points))
    return SkeletonGraph(121, 121, tuple(nodes), tuple(edges))


def t_graph() -> SkeletonGraph:
    return junction_graph(line((49, 50), (10, 50)), line((51, 50), (100, 50)), line((50, 49), (50, 10)))


def test_T_common_contact_and_collinear_continuation() -> None:
    graph = t_graph()
    original = reconstruct_geometry(graph)
    result = refine_junctions(graph, original)
    assert len(result.adjustments) == 1 and not result.rejected_node_ids
    anchor = result.adjustments[0].position
    assert (anchor.x, anchor.y) == pytest.approx((50, 50))
    assert all(p.start == anchor for p in result.geometry.primitives)
    assert len(result.continuations) == 1 and result.continuations[0].edge_ids == (0, 1)
    assert all(p.fit_error_max <= 2 and p.source_points == graph.edges[p.source_edge_id].points
               for p in result.geometry.primitives)
    assert original.primitives[0].start == Point(49, 50)
    assert len(graph.nodes) == 4 and len(graph.edges) == 3
    assert result == refine_junctions(graph, original)


def test_cross_has_two_continuations_without_merging_edges() -> None:
    graph = junction_graph(line((49, 50), (10, 50)), line((51, 50), (100, 50)),
                           line((50, 49), (50, 10)), line((50, 51), (50, 100)))
    result = refine_junctions(graph, reconstruct_geometry(graph))
    assert len(result.continuations) == 2 and len(result.geometry.primitives) == 4
    assert all(p.start == result.adjustments[0].position for p in result.geometry.primitives)


def test_fork_is_not_recognized_as_opposite_collinear_segments() -> None:
    graph = junction_graph(line((50, 50), (10, 45)), line((50, 50), (10, 55)), line((50, 50), (50, 100)))
    assert not refine_junctions(graph, reconstruct_geometry(graph)).continuations


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("sweep", [np.pi/2-0.01, 1.5*np.pi])
def test_arc_refit_passes_exactly_through_common_contact(reverse: bool, sweep: float) -> None:
    angles = np.linspace(-np.pi/2+0.01, -np.pi/2+0.01+sweep, 201)
    curve = tuple(Point(float(50+30*np.cos(a)), float(80+30*np.sin(a))) for a in angles)
    graph = junction_graph(curve, line((49, 50), (10, 50)), line((50, 49), (50, 10)))
    if reverse:
        edge = graph.edges[0]
        graph = replace(graph, edges=(replace(edge, start_node_id=1, end_node_id=0, points=edge.points[::-1]), *graph.edges[1:]))
    original = reconstruct_geometry(graph)
    assert isinstance(original.primitives[0], CircularArc)
    result = refine_junctions(graph, original)
    arc = result.geometry.primitives[0]
    assert isinstance(arc, JoinedArc) and result.adjustments
    anchor = result.adjustments[0].position
    assert (arc.end if reverse else arc.start) == anchor
    assert arc.source_points == graph.edges[0].points and arc.fit_error_max <= 2
    for point in (arc.start, arc.end):
        assert np.hypot(point.x-arc.center.x, point.y-arc.center.y) == pytest.approx(arc.radius, abs=1e-8)
    assert arc.sweep_flag == original.primitives[0].sweep_flag
    assert arc.large_arc_flag == original.primitives[0].large_arc_flag
    root = ET.fromstring(svg_document(result.geometry))
    paths = root.findall("{http://www.w3.org/2000/svg}g/{http://www.w3.org/2000/svg}path")
    assert " A " in paths[0].get("d", "")


def test_bezier_keeps_sources_and_continuous_error_bound() -> None:
    t = np.linspace(0, 1, 121)[:, None]
    c = np.array([(50.3, 50.2), (65, 90), (90, 45), (110, 90)])
    data = (1-t)**3*c[0]+3*(1-t)**2*t*c[1]+3*(1-t)*t**2*c[2]+t**3*c[3]
    points = tuple(Point(float(x), float(y)) for x, y in data)
    graph = junction_graph(points, line((49, 50), (10, 50)), line((50, 49), (50, 10)))
    original = reconstruct_geometry(graph)
    bezier = fit_bezier(points, 0.5, source_edge_id=0)
    assert bezier is not None
    original = replace(original, primitives=(bezier, *original.primitives[1:]))
    result = refine_junctions(graph, original)
    path = result.geometry.primitives[0]
    assert isinstance(path, BezierPath) and result.adjustments
    assert path.segments[0].start == result.adjustments[0].position
    assert path.source_points == points and path.error_bound <= 2
    assert all(a.end == b.start for a, b in zip(path.segments, path.segments[1:]))
    for segment in path.segments:
        source = np.array([(p.x, p.y) for p in points[segment.source_start_index:segment.source_end_index+1]])
        for index, (a, b) in enumerate(zip(segment.source_parameters, segment.source_parameters[1:])):
            f = np.linspace(0, 1, 7)
            curve = evaluate_cubic(segment, a+f*(b-a))
            reference = source[index]+f[:, None]*(source[index+1]-source[index])
            assert np.linalg.norm(curve-reference, axis=1).max() <= segment.error_bound+1e-10


def test_raw_source_points_retained_after_endpoint_adjustment() -> None:
    graph = t_graph()
    geometry = reconstruct_geometry(graph)
    raw = RawPolyline(graph.edges[0].points, 0, False)
    geometry = replace(geometry, primitives=(raw, *geometry.primitives[1:]))
    result = refine_junctions(graph, geometry)
    adjusted = result.geometry.primitives[0]
    assert isinstance(adjusted, RawPolyline) and result.adjustments
    assert adjusted.source_points == raw.points
    assert adjusted.points[0] == result.adjustments[0].position
    assert adjusted.fit_error_max <= 2


def test_atomic_refusal_when_circle_cannot_share_contact() -> None:
    graph = t_graph()
    original = reconstruct_geometry(graph)
    circle = Circle(Point(20, 20), 10, 0, 0, 0, graph.edges[0].points)
    original = replace(original, primitives=(circle, *original.primitives[1:]))
    result = refine_junctions(graph, original)
    assert result.rejected_node_ids == (0,) and not result.adjustments
    assert result.geometry == original


def test_large_contact_zone_is_not_forced_into_tolerance() -> None:
    graph = junction_graph(line((40, 50), (10, 50)), line((60, 50), (100, 50)), line((50, 40), (50, 10)))
    original = reconstruct_geometry(graph)
    result = refine_junctions(graph, original)
    assert not result.adjustments and result.geometry == original


@pytest.mark.parametrize("tolerance", [-1, float("inf"), float("nan")])
def test_invalid_tolerance(tolerance: float) -> None:
    graph = t_graph()
    with pytest.raises(ValueError, match="tolérance de jonction"):
        refine_junctions(graph, reconstruct_geometry(graph), tolerance)


def test_zero_empty_and_mismatched_inputs() -> None:
    graph = t_graph()
    geometry = reconstruct_geometry(graph)
    assert refine_junctions(graph, geometry, 0).geometry is geometry
    empty = SkeletonGraph(20, 20, (), ())
    assert refine_junctions(empty, reconstruct_geometry(empty)).geometry.primitives == ()
    with pytest.raises(ValueError, match="dimensions"):
        refine_junctions(graph, replace(geometry, width=5))
    with pytest.raises(ValueError, match="une primitive"):
        refine_junctions(graph, replace(geometry, primitives=geometry.primitives[:-1]))


@pytest.mark.parametrize("case", ["T", "cross", "double_T"])
def test_raster_junctions_preserve_graph_and_cycles(case: str) -> None:
    image = np.full((121, 141), 255, dtype=np.uint8)
    cv2.line(image, (10, 40), (130, 40), 0, 5)
    cv2.line(image, (70, 40 if case != "cross" else 10), (70, 110), 0, 5)
    if case == "double_T":
        cv2.line(image, (10, 85), (130, 85), 0, 5)
    graph = skeleton_to_graph(skeletonize_mask(image == 0))
    before = graph
    result = refine_junctions(graph, reconstruct_geometry(graph))
    assert result.adjustments and graph == before
    assert len(result.geometry.primitives) == len(graph.edges)
    assert {p.source_edge_id for p in result.geometry.primitives} == {e.id for e in graph.edges}
    for adjustment in result.adjustments:
        for index in adjustment.edge_ids:
            edge = graph.edges[index]
            primitive = result.geometry.primitives[index]
            if isinstance(primitive, LineSegment):
                if edge.start_node_id == adjustment.node_id:
                    assert primitive.start == adjustment.position
                if edge.end_node_id == adjustment.node_id:
                    assert primitive.end == adjustment.position


def test_cli_comparison_and_disable(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    image = np.full((101, 121), 255, dtype=np.uint8)
    cv2.line(image, (10, 50), (110, 50), 0, 5)
    cv2.line(image, (60, 15), (60, 50), 0, 5)
    source = tmp_path / "T.png"
    save_png(source, image)
    debug = tmp_path / "debug"
    assert main([str(source), "--debug-dir", str(debug), "--debug-svg", "--output", str(tmp_path / "output.svg")]) == 0
    assert "Jonctions :" in capsys.readouterr().out
    assert (debug / "geometry_before.png").exists() and (debug / "junctions.png").exists()
    assert (debug / "overlay_before.svg").exists()
    baseline = tmp_path / "baseline"
    assert main([str(source), "--no-junction-cleanup", "--debug-dir", str(baseline)]) == 0
    assert "Jonctions :" not in capsys.readouterr().out
    assert not (baseline / "junctions.png").exists()


def test_real_loop_is_retained_with_sources_and_closure() -> None:
    image = np.full((151, 201), 255, dtype=np.uint8)
    cv2.ellipse(image, (100, 95), (55, 55), 0, 180, 270, 0, 5)
    cv2.line(image, (45, 95), (185, 95), 0, 5)
    cv2.line(image, (100, 40), (100, 125), 0, 5)
    graph = skeleton_to_graph(skeletonize_mask(image == 0))
    loop = next(e for e in graph.edges if e.is_loop)
    original = reconstruct_geometry(graph)
    result = refine_junctions(graph, original)
    primitive = result.geometry.primitives[loop.id]
    assert isinstance(primitive, (RawPolyline, BezierPath)) and primitive.is_loop
    assert primitive.source_points == loop.points
    if isinstance(primitive, BezierPath):
        assert primitive.segments[0].start == primitive.segments[-1].end
        assert primitive.error_bound <= 2
    else:
        assert primitive.points[0] == primitive.points[-1]
    assert len(result.geometry.primitives) == len(graph.edges)
    assert " Z" in svg_document(result.geometry)


def test_overlay_before_must_not_overwrite_source(tmp_path: Path) -> None:
    source = tmp_path / "overlay_before.svg"
    save_png(source, np.full((8, 8), 255, dtype=np.uint8))
    before = source.read_bytes()
    assert main([str(source), "--debug-dir", str(tmp_path), "--debug-svg"]) == 1
    assert source.read_bytes() == before
