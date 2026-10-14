"""Arcs multibranches, erreurs continues et provenance aux intersections."""

from dataclasses import replace
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pytest

from raster2vector import (
    Point, GraphNode, GraphEdge, SkeletonGraph, GlobalArc, LineSegment,
    reconstruct_geometry, reconstruct_global_arcs, svg_document,
)
from raster2vector.global_arcs import _finite_arc_bound
from raster2vector.cli import main
from raster2vector.image_io import save_png


def arc_graph(start: float = 0, sweep: float = math.pi, reverse: bool = False,
              disconnected: bool = False) -> SkeletonGraph:
    limits = np.linspace(start, start+sweep, 4)
    contacts = [Point(float(130+90*np.cos(t)), float(130+90*np.sin(t))) for t in limits]
    nodes = [GraphNode(i, "junction" if 0 < i < 3 else "endpoint", p, ((round(p.y), round(p.x)),))
             for i, p in enumerate(contacts)]
    edges = []
    for i in range(3):
        t = np.linspace(limits[i], limits[i+1], 101)
        points = [Point(float(130+90*np.cos(a)), float(130+90*np.sin(a))) for a in t]
        points[0], points[-1] = contacts[i], contacts[i+1]
        a, b = i, i+1
        if reverse:
            a, b, points = b, a, points[::-1]
        if disconnected:
            a = len(nodes)
            nodes.append(GraphNode(a, "endpoint", points[0], ((round(points[0].y), round(points[0].x)),)))
            b = len(nodes)
            nodes.append(GraphNode(b, "endpoint", points[-1], ((round(points[-1].y), round(points[-1].x)),)))
        edges.append(GraphEdge(i, a, b, tuple(points)))
    return SkeletonGraph(261, 261, tuple(nodes), tuple(edges))


@pytest.mark.parametrize("start,sweep", [(0, math.pi), (2.9, math.pi/2), (0.2, 1.4*math.pi)])
@pytest.mark.parametrize("reverse", [False, True])
def test_long_arc_through_intersections(start: float, sweep: float, reverse: bool) -> None:
    graph = arc_graph(start, sweep, reverse)
    original = reconstruct_geometry(graph, 1)
    result = reconstruct_global_arcs(graph, original, 1)
    assert len(result.arcs) == len(result.geometry.primitives) == 1
    arc = result.arcs[0]
    assert arc.source_edge_ids == arc.support_edge_ids == (0, 1, 2)
    assert arc.radius == pytest.approx(90, abs=1e-7)
    assert arc.sweep_angle_rad == pytest.approx(sweep, abs=0.02)
    assert arc.large_arc_flag == int(sweep > math.pi)
    assert arc.fit_error_max <= 1
    assert len(graph.edges) == 3 and len(original.primitives) == 3
    assert result == reconstruct_global_arcs(graph, original, 1)
    for edge in graph.edges:
        for a, b in zip(edge.points, edge.points[1:]):
            fractions = np.linspace(0, 1, 5)
            data = np.array((a.x, a.y))+fractions[:, None]*np.array((b.x-a.x, b.y-a.y))
            vectors = data-[arc.center.x, arc.center.y]
            progress = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0])-arc.start_angle_rad, 2*math.pi)
            errors = abs(np.linalg.norm(vectors, axis=1)-arc.radius)
            outside = progress > arc.sweep_angle_rad
            errors[outside] = np.minimum(np.linalg.norm(data[outside]-[arc.start.x, arc.start.y], axis=1),
                                         np.linalg.norm(data[outside]-[arc.end.x, arc.end.y], axis=1))
            assert errors.max() <= arc.fit_error_max+1e-9
    root = ET.fromstring(svg_document(result.geometry))
    path = root.find("{http://www.w3.org/2000/svg}g/{http://www.w3.org/2000/svg}path")
    assert path is not None and path.get("id") == "global-arc-0"
    assert path.get("data-support-edges") == "0 1 2" and path.get("data-source-edges") == "0 1 2"
    assert " A " in path.get("d", "")


def test_disconnected_fragments_remain_separate() -> None:
    graph = arc_graph(disconnected=True)
    geometry = reconstruct_geometry(graph, 1)
    result = reconstruct_global_arcs(graph, geometry, 1)
    assert not result.arcs and result.geometry == geometry


def test_non_circular_sources_not_forced_into_long_arc() -> None:
    graph = arc_graph()
    edges = tuple(replace(e, points=tuple(Point(p.x, 130+(p.y-130)*0.4) for p in e.points)) for e in graph.edges)
    graph = replace(graph, edges=edges)
    assert not reconstruct_global_arcs(graph, reconstruct_geometry(graph, 1), 1).arcs


def test_nearby_branch_outside_tolerance_is_retained() -> None:
    graph = arc_graph()
    points = (graph.nodes[1].position, Point(130, 130))
    node = GraphNode(4, "endpoint", points[-1], ((130, 130),))
    graph = replace(graph, nodes=(*graph.nodes, node), edges=(*graph.edges, GraphEdge(3, 1, 4, points)))
    geometry = reconstruct_geometry(graph, 1)
    result = reconstruct_global_arcs(graph, geometry, 1)
    assert len(result.arcs) == 1
    assert geometry.primitives[3] in result.geometry.primitives
    assert isinstance(geometry.primitives[3], LineSegment)


def test_finite_arc_bound_detects_points_on_missing_side_of_circle() -> None:
    arc = GlobalArc(Point(0, 0), 10, 0, math.pi/2, 0, 0, 0, (), (0,), (0,))
    points = (Point(-10, 0), Point(0, -10))
    assert _finite_arc_bound(points, arc) >= math.sqrt(200)-1e-10


def test_finite_arc_bound_checks_nearest_endpoint_switch() -> None:
    arc = GlobalArc(Point(0, 0), 10, 0, math.pi/2, 0, 0, 0, (), (0,), (0,))
    points = (Point(-8, 0), Point(0, -6))
    expected = math.hypot(10+24/7, 24/7)
    assert _finite_arc_bound(points, arc) == pytest.approx(expected)


def test_small_compatible_loop_absorbed_without_mutating_graph() -> None:
    graph = arc_graph()
    contact = graph.nodes[1].position
    points = (contact, Point(contact.x+0.2, contact.y+0.2),
              Point(contact.x+0.3, contact.y), contact)
    graph = replace(graph, edges=(*graph.edges, GraphEdge(3, 1, 1, points)))
    original = graph
    result = reconstruct_global_arcs(graph, reconstruct_geometry(graph, 1), 1)
    assert len(result.arcs) == len(result.geometry.primitives) == 1
    assert result.arcs[0].source_edge_ids == (0, 1, 2, 3)
    assert result.arcs[0].fit_error_max <= 1
    assert graph == original and graph.edges[3].is_loop


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_invalid_tolerance(tolerance: float) -> None:
    graph = arc_graph()
    with pytest.raises(ValueError, match="tolérance d'arc global"):
        reconstruct_global_arcs(graph, reconstruct_geometry(graph), tolerance)


def test_zero_mismatches_and_idempotence() -> None:
    graph = arc_graph()
    geometry = reconstruct_geometry(graph, 1)
    assert reconstruct_global_arcs(graph, geometry, 0).geometry is geometry
    with pytest.raises(ValueError, match="dimensions"):
        reconstruct_global_arcs(graph, replace(geometry, width=5))
    with pytest.raises(ValueError, match="toutes les branches"):
        reconstruct_global_arcs(graph, replace(geometry, primitives=geometry.primitives[:-1]))
    result = reconstruct_global_arcs(graph, geometry, 1)
    assert reconstruct_global_arcs(graph, result.geometry, 1).geometry == result.geometry


def test_raster_cli_and_global_circle_option_implied(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    image = np.full((221, 221), 255, dtype=np.uint8)
    cv2.ellipse(image, (110, 110), (80, 80), 0, 0, 180, 0, 3)
    cv2.line(image, (110, 190), (110, 130), 0, 3)
    source = tmp_path / "arc.png"
    save_png(source, image)
    output = tmp_path / "output.svg"
    debug = tmp_path / "debug"
    assert main([str(source), "--global-arcs", "--geometry-tolerance", "3", "--output", str(output),
                 "--debug-dir", str(debug), "--debug-svg"]) == 0
    text = capsys.readouterr().out
    assert "Arcs globaux : 1" in text and "Cercles globaux : 0" in text
    assert 'id="global-arc-' in output.read_text(encoding="utf-8")
    assert (debug / "overlay_local.svg").exists()
