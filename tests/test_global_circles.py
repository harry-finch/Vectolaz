"""Cercles multibranches : couverture, provenance et refus de fermetures inventées."""

from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pytest
from raster2vector import (
    Point, GraphNode, GraphEdge, SkeletonGraph, GlobalCircle, Circle,
    reconstruct_geometry, reconstruct_global_circles, svg_document,
)
from raster2vector.cli import main
from raster2vector.image_io import save_png


def circle_graph(closed: bool = True, ellipse: bool = False, spokes: bool = False) -> SkeletonGraph:
    count = 6
    angles = np.linspace(0, 2*np.pi, count+1)
    contacts = [Point(float(130+100*np.cos(a)), float(130+(70 if ellipse else 100)*np.sin(a))) for a in angles[:-1]]
    if not closed:
        contacts.append(Point(230, 130))
    nodes = [GraphNode(i, "junction", p, ((round(p.y), round(p.x)),)) for i, p in enumerate(contacts)]
    edges = []
    for i in range(count):
        a = np.linspace(angles[i], angles[i+1], 101)
        points = [Point(float(130+100*np.cos(t)), float(130+(70 if ellipse else 100)*np.sin(t))) for t in a]
        end = (i+1)%count if closed else i+1
        points[0], points[-1] = contacts[i], contacts[end]
        edges.append(GraphEdge(i, i, end, tuple(points)))
    if spokes:
        for i, p in enumerate(contacts[:6]):
            end = Point(130+(p.x-130)*0.5, 130+(p.y-130)*0.5)
            index = len(nodes)
            nodes.append(GraphNode(index, "endpoint", end, ((round(end.y), round(end.x)),)))
            points = tuple(Point(float(x), float(y)) for x, y in np.linspace((p.x, p.y), (end.x, end.y), 51))
            edges.append(GraphEdge(len(edges), i, index, points))
    return SkeletonGraph(261, 261, tuple(nodes), tuple(edges))


def test_full_circle_across_intersections_preserves_spokes_and_sources() -> None:
    graph = circle_graph(spokes=True)
    local = reconstruct_geometry(graph, 1)
    result = reconstruct_global_circles(graph, local, 1)
    assert len(result.circles) == 1
    circle = result.circles[0]
    assert isinstance(circle, GlobalCircle) and isinstance(circle, Circle)
    assert circle.source_edge_ids == tuple(range(6))
    assert circle.radius == pytest.approx(100, abs=1e-7)
    assert (circle.center.x, circle.center.y) == pytest.approx((130, 130), abs=1e-7)
    assert circle.fit_error_max <= 1
    assert circle.source_points[0] == circle.source_points[-1]
    assert len(result.geometry.primitives) == 7 and len(graph.edges) == 12
    assert all(p in result.geometry.primitives for p in local.primitives[6:])
    for edge in graph.edges[:6]:
        assert all(point in circle.source_points for point in edge.points)
    assert result == reconstruct_global_circles(graph, local, 1)
    root = ET.fromstring(svg_document(result.geometry))
    element = root.find("{http://www.w3.org/2000/svg}g/{http://www.w3.org/2000/svg}circle")
    assert element is not None and element.get("data-source-edges") == "0 1 2 3 4 5"


@pytest.mark.parametrize("closed,ellipse", [(False, False), (True, True)])
def test_no_circle_when_path_open_or_not_circular(closed: bool, ellipse: bool) -> None:
    graph = circle_graph(closed=closed, ellipse=ellipse)
    local = reconstruct_geometry(graph, 1)
    result = reconstruct_global_circles(graph, local, 1)
    assert not result.circles and result.geometry == local


def test_disconnected_arcs_are_not_closed_even_if_angles_cover_a_turn() -> None:
    graph = circle_graph()
    nodes = []
    edges = []
    for edge in graph.edges:
        for point in (edge.points[0], edge.points[-1]):
            nodes.append(GraphNode(len(nodes), "endpoint", point, ((round(point.y), round(point.x)),)))
        edges.append(replace(edge, start_node_id=2*edge.id, end_node_id=2*edge.id+1))
    graph = replace(graph, nodes=tuple(nodes), edges=tuple(edges))
    assert not reconstruct_global_circles(graph, reconstruct_geometry(graph, 1), 1).circles


def test_non_circular_junction_zone_blocks_shortcuts() -> None:
    graph = circle_graph()
    node = replace(graph.nodes[0], source_pixels=(*graph.nodes[0].source_pixels, (130, 130)))
    graph = replace(graph, nodes=(node, *graph.nodes[1:]))
    assert not reconstruct_global_circles(graph, reconstruct_geometry(graph, 1), 1).circles


def test_continuous_bound_includes_chord_sagitta() -> None:
    graph = circle_graph()
    local = reconstruct_geometry(graph, 1)
    result = reconstruct_global_circles(graph, local, 1)
    circle = result.circles[0]
    data = np.array([(p.x, p.y) for p in circle.source_points])
    midpoints = (data[:-1]+data[1:])/2
    error = abs(np.linalg.norm(midpoints-[circle.center.x, circle.center.y], axis=1)-circle.radius)
    assert error.max() <= circle.fit_error_max+1e-10
    assert circle.fit_error_max > 0.001


def test_two_distinct_circles_are_kept_separate() -> None:
    a = circle_graph()
    nodes = tuple(replace(n, id=n.id+6, position=Point(n.position.x+320, n.position.y),
                          source_pixels=tuple((row, col+320) for row, col in n.source_pixels)) for n in a.nodes)
    edges = tuple(replace(e, id=e.id+6, start_node_id=e.start_node_id+6, end_node_id=e.end_node_id+6,
                          points=tuple(Point(p.x+320, p.y) for p in e.points)) for e in a.edges)
    graph = SkeletonGraph(601, 261, (*a.nodes, *nodes), (*a.edges, *edges))
    result = reconstruct_global_circles(graph, reconstruct_geometry(graph, 1), 1)
    assert len(result.circles) == 2
    assert not set(result.circles[0].source_edge_ids) & set(result.circles[1].source_edge_ids)


def test_raster_cli_reconstructs_one_circle_and_keeps_comparison(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    image = np.full((261, 261), 255, dtype=np.uint8)
    cv2.circle(image, (130, 130), 100, 0, 5)
    for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
        cv2.line(image, (130, 130), (round(130+100*np.cos(angle)), round(130+100*np.sin(angle))), 0, 3)
    source = tmp_path / "circle.png"
    save_png(source, image)
    output, debug = tmp_path / "output.svg", tmp_path / "debug"
    assert main([str(source), "--global-circles", "--geometry-tolerance", "3", "--output", str(output),
                 "--debug-dir", str(debug), "--debug-svg"]) == 0
    assert "Cercles globaux : 1" in capsys.readouterr().out
    root = ET.fromstring(output.read_text(encoding="utf-8"))
    circles = root.findall("{http://www.w3.org/2000/svg}g/{http://www.w3.org/2000/svg}circle")
    assert len(circles) == 1 and circles[0].get("data-source-edges")
    assert (debug / "geometry_local.png").exists() and (debug / "overlay_local.svg").exists()


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_invalid_tolerance(tolerance: float) -> None:
    graph = circle_graph()
    with pytest.raises(ValueError, match="tolérance globale"):
        reconstruct_global_circles(graph, reconstruct_geometry(graph), tolerance)


def test_zero_and_mismatched_input() -> None:
    graph = circle_graph()
    geometry = reconstruct_geometry(graph)
    assert reconstruct_global_circles(graph, geometry, 0).geometry is geometry
    with pytest.raises(ValueError, match="dimensions"):
        reconstruct_global_circles(graph, replace(geometry, width=5))
    with pytest.raises(ValueError, match="primitive par branche"):
        reconstruct_global_circles(graph, replace(geometry, primitives=geometry.primitives[:-1]))


@pytest.mark.parametrize("filename", ["geometry_local.png", "overlay_local.svg"])
def test_cli_comparison_files_cannot_overwrite_source(tmp_path: Path, filename: str) -> None:
    source = tmp_path / filename
    save_png(source, np.full((8, 8), 255, dtype=np.uint8))
    before = source.read_bytes()
    assert main([str(source), "--global-circles", "--debug-dir", str(tmp_path), "--debug-svg"]) == 1
    assert source.read_bytes() == before
