"""Raccords plausibles, refus conservateurs, provenance et intégration CLI."""

from pathlib import Path
import time
import cv2
import numpy as np
import pytest

from raster2vector import (
    Point, GraphNode, GraphEdge, SkeletonGraph, LineSegment,
    connect_gaps, reconstruct_geometry, skeleton_to_graph,
)
from raster2vector.gaps import _segment_distance, _segments_to_segment_min_distance, GapConnection
from raster2vector.cli import main
from raster2vector.image_io import save_png


def graph_for(*chains: tuple[Point, ...]) -> SkeletonGraph:
    nodes = []
    edges = []
    for index, points in enumerate(chains):
        for offset, point in enumerate((points[0], points[-1])):
            nodes.append(GraphNode(2*index+offset, "endpoint", point, ((round(point.y), round(point.x)),)))
        edges.append(GraphEdge(index, 2*index, 2*index+1, points))
    return SkeletonGraph(120, 100, tuple(nodes), tuple(edges))


def line(a: tuple[float, float], b: tuple[float, float]) -> tuple[Point, ...]:
    count = max(2, int(np.hypot(b[0]-a[0], b[1]-a[1]))+1)
    return tuple(Point(float(x), float(y)) for x, y in np.linspace(a, b, count))


def broken_line() -> SkeletonGraph:
    return graph_for(line((10, 30), (40, 30)), line((46, 30), (90, 30)))


def test_distance_boundary_disable_merge_and_provenance() -> None:
    graph = broken_line()
    assert connect_gaps(graph).graph is graph
    assert not connect_gaps(graph, np.nextafter(6.0, 0)).connections
    result = connect_gaps(graph, 6)
    assert len(result.connections) == 1
    assert result.connections[0].distance_px == 6
    assert len(result.graph.nodes) == 2 and len(result.graph.edges) == 1
    assert result.source_edge_ids == ((0, 1),)
    assert result.graph.edges[0].points == (*graph.edges[0].points, *graph.edges[1].points)
    assert len(graph.edges) == 2 and len(graph.nodes) == 4
    geometry = reconstruct_geometry(result.graph)
    assert len(geometry.primitives) == 1 and isinstance(geometry.primitives[0], LineSegment)
    assert result == connect_gaps(graph, 6)


def test_multiple_cuts_and_reverse_branch_orientation() -> None:
    graph = graph_for(line((5, 20), (30, 20)), line((60, 20), (36, 20)), line((66, 20), (110, 20)))
    result = connect_gaps(graph, 6)
    assert len(result.connections) == 2 and len(result.graph.edges) == 1
    assert result.source_edge_ids == ((0, 1, 2),)
    assert result.graph.edges[0].points[0] == Point(5, 20)
    assert result.graph.edges[0].points[-1] == Point(110, 20)


@pytest.mark.parametrize("chains", [
    (line((10, 30), (40, 30)), line((46, 34), (46, 70))),  # perpendiculaire
    (line((10, 30), (40, 30)), line((10, 35), (40, 35))),  # parallèles voisines
    (line((10, 30), (40, 30)), line((42, 35), (90, 35))),  # décalage latéral
    (line((10, 30), (12, 30)), line((18, 30), (60, 30))),  # tangente trop courte
])
def test_incompatible_or_unreliable_endpoints_refused(chains: tuple[tuple[Point, ...], ...]) -> None:
    graph = graph_for(*chains)
    assert connect_gaps(graph, 10).graph is graph


def test_ambiguity_refused_instead_of_selecting_nearest() -> None:
    graph = graph_for(line((10, 40), (40, 40)),
                      line((48, 38), (80, 30)), line((48, 42), (80, 50)))
    assert not connect_gaps(graph, 10).connections


def test_obstacle_and_isolated_pixel_refused() -> None:
    graph = graph_for(*[e.points for e in broken_line().edges], line((43, 5), (43, 70)))
    assert not connect_gaps(graph, 6).connections


def test_incident_branch_returning_into_gap_is_an_obstacle() -> None:
    vertices = [(40, 30), (10, 30), (10, 60), (44, 60), (44, 26), (70, 26)]
    points = tuple(point for a, b in zip(vertices, vertices[1:]) for point in line(a, b)[:-1])
    points = (*points, Point(*vertices[-1]))
    graph = graph_for(points, line((46, 30), (90, 30)))
    assert not connect_gaps(graph, 6).connections
    graph = broken_line()
    isolated = GraphNode(4, "isolated", Point(43, 30), ((30, 43),))
    graph = SkeletonGraph(graph.width, graph.height, (*graph.nodes, isolated), graph.edges)
    assert not connect_gaps(graph, 6).connections


def test_crossing_candidate_bridges_are_both_refused() -> None:
    graph = graph_for(line((5, 50), (45, 50)), line((55, 50), (100, 50)),
                      line((50, 5), (50, 45)), line((50, 55), (50, 90)))
    assert not connect_gaps(graph, 10).connections


def test_closing_broken_loop_retains_all_points_and_anchor() -> None:
    angles = np.linspace(0.08, 2*np.pi-0.08, 301)
    points = tuple(Point(float(50+30*np.cos(a)), float(50+30*np.sin(a))) for a in angles)
    graph = graph_for(points)
    result = connect_gaps(graph, 6)
    assert len(result.connections) == 1 and len(result.graph.nodes) == len(result.graph.edges) == 1
    assert result.graph.nodes[0].kind == "loop" and result.graph.edges[0].is_loop
    assert result.graph.edges[0].points == (*points, points[0])
    assert result.source_edge_ids == ((0,),)


def test_existing_junctions_and_loops_untouched() -> None:
    mask = np.zeros((60, 80), dtype=bool)
    mask[30, 5:75] = True
    mask[5:55, 40] = True
    graph = skeleton_to_graph(mask)
    assert connect_gaps(graph, 10).graph is graph


@pytest.mark.parametrize("tolerance", [-1, float("inf"), float("nan")])
def test_invalid_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolérance de raccordement"):
        connect_gaps(broken_line(), tolerance)


def test_cli_raster_gap_debug_and_original_preserved(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    image = np.full((61, 121), 255, dtype=np.uint8)
    cv2.line(image, (10, 30), (48, 30), 0, 3)
    cv2.line(image, (56, 30), (110, 30), 0, 3)
    source = tmp_path / "cut.png"
    save_png(source, image)
    before = source.read_bytes()
    debug = tmp_path / "debug"
    output = tmp_path / "output.svg"
    assert main([str(source), "--gap-tolerance", "10", "--debug-dir", str(debug),
                 "--output", str(output), "--debug-svg"]) == 0
    assert "Raccords : 1" in capsys.readouterr().out
    assert output.read_text(encoding="utf-8").count("<path") == 1
    assert (debug / "gaps.png").exists() and (debug / "graph_connected.png").exists()
    assert source.read_bytes() == before


@pytest.mark.parametrize("name", ["gaps.png", "graph_connected.png"])
def test_extra_debug_must_not_overwrite_input(tmp_path: Path, name: str) -> None:
    source = tmp_path / name
    save_png(source, np.full((8, 8), 255, dtype=np.uint8))
    before = source.read_bytes()
    assert main([str(source), "--gap-tolerance", "5", "--debug-dir", str(tmp_path)]) == 1
    assert source.read_bytes() == before


def test_segment_distance_cross_touching_colinear() -> None:
    # 1. Croisement strict
    assert _segment_distance(Point(0, 5), Point(10, 5), Point(5, 0), Point(5, 10)) == 0.0

    # 2. Contact par sommet (T-junction ou contact d'extrémité)
    assert _segment_distance(Point(0, 0), Point(5, 0), Point(5, 0), Point(5, 5)) == 0.0
    assert _segment_distance(Point(0, 0), Point(10, 0), Point(5, 0), Point(5, 5)) == 0.0

    # 3. Colinéaires se chevauchant partiellement
    assert _segment_distance(Point(0, 0), Point(10, 0), Point(5, 0), Point(15, 0)) == 0.0

    # 4. Colinéaires avec inclusion totale
    assert _segment_distance(Point(0, 0), Point(10, 0), Point(2, 0), Point(8, 0)) == 0.0
    assert _segment_distance(Point(2, 0), Point(8, 0), Point(0, 0), Point(10, 0)) == 0.0

    # 5. Colinéaires disjoints
    assert _segment_distance(Point(0, 0), Point(4, 0), Point(6, 0), Point(10, 0)) == pytest.approx(2.0)

    # 6. Parallèles disjoints
    assert _segment_distance(Point(0, 0), Point(10, 0), Point(0, 3), Point(10, 3)) == pytest.approx(3.0)

    # Vérification vectorisée _segments_to_segment_min_distance
    p = np.array([0.0, 0.0])
    q = np.array([10.0, 0.0])
    c = np.array([[5.0, 0.0], [0.0, 3.0]])
    d = np.array([[15.0, 0.0], [10.0, 3.0]])
    # Le premier segment est colinéaire chevauchant (distance 0), le min doit valoir 0.0
    assert _segments_to_segment_min_distance(p, q, c, d) == 0.0


def test_empty_graph_and_isolated_nodes_edge_cases() -> None:
    # 1. Graphe sans aucun segment (M == 0)
    empty_graph = SkeletonGraph(100, 100, (), ())
    res_empty = connect_gaps(empty_graph, 10.0)
    assert res_empty.graph is empty_graph
    assert len(res_empty.connections) == 0

    # 2. Graphe composé uniquement de nœuds isolated
    isolated_node = GraphNode(0, "isolated", Point(20.0, 20.0), ((20, 20),))
    iso_graph = SkeletonGraph(100, 100, (isolated_node,), ())
    res_iso = connect_gaps(iso_graph, 10.0)
    assert res_iso.graph is iso_graph
    assert len(res_iso.connections) == 0


def test_colinear_overlapping_candidate_bridges_rejection() -> None:
    # 1. Deux segments candidats colinéaires qui se chevauchent partiellement :
    # la distance doit être exactement 0.0 et ils doivent être rejetés lors du filtrage de croisement mutuel
    c1 = GapConnection(0, 1, Point(10.0, 50.0), Point(30.0, 50.0), 20.0)
    c2 = GapConnection(2, 3, Point(20.0, 50.0), Point(40.0, 50.0), 20.0)
    assert _segment_distance(c1.start, c1.end, c2.start, c2.end) == 0.0

    unique_candidates = [c1, c2]
    accepted = tuple(
        c for c in unique_candidates
        if all(c == other or _segment_distance(c.start, c.end, other.start, other.end) > 0.75
               for other in unique_candidates)
    )
    assert len(accepted) == 0

    # 2. Rejet de ponts concurrents qui se croisent dans connect_gaps
    graph = graph_for(
        line((5, 50), (45, 50)), line((55, 50), (100, 50)),
        line((50, 5), (50, 45)), line((50, 55), (50, 90)),
    )
    result = connect_gaps(graph, 10.0)
    assert len(result.connections) == 0


def test_performance_large_synthetic_graph() -> None:
    # 50 arêtes réparties sur une grille, chacune avec 41 points (40 segments) -> 2 000 segments.
    # 50 arêtes * 2 extrémités = 100 endpoints.
    # 25 paires d'arêtes en vis-à-vis avec un gap de 5 px (doivent être raccordées).
    nodes = []
    edges = []
    node_id = 0
    edge_id = 0

    for r in range(25):
        y = float(20 + 15 * r)
        pts_a = tuple(Point(float(x), y) for x in np.linspace(10, 50, 41))
        nodes.append(GraphNode(node_id, "endpoint", pts_a[0], ((round(y), round(pts_a[0].x)),)))
        nodes.append(GraphNode(node_id + 1, "endpoint", pts_a[-1], ((round(y), round(pts_a[-1].x)),)))
        edges.append(GraphEdge(edge_id, node_id, node_id + 1, pts_a))
        node_id += 2
        edge_id += 1

        pts_b = tuple(Point(float(x), y) for x in np.linspace(55, 95, 41))
        nodes.append(GraphNode(node_id, "endpoint", pts_b[0], ((round(y), round(pts_b[0].x)),)))
        nodes.append(GraphNode(node_id + 1, "endpoint", pts_b[-1], ((round(y), round(pts_b[-1].x)),)))
        edges.append(GraphEdge(edge_id, node_id, node_id + 1, pts_b))
        node_id += 2
        edge_id += 1

    graph = SkeletonGraph(120, 450, tuple(nodes), tuple(edges))
    assert sum(len(e.points) - 1 for e in graph.edges) == 2000
    assert sum(1 for n in graph.nodes if n.kind == "endpoint") == 100

    # Mesure du temps d'exécution
    t0 = time.perf_counter()
    res = connect_gaps(graph, 10.0)
    elapsed = time.perf_counter() - t0

    # Critère : exécution en moins de 100 ms
    assert elapsed < 0.100, f"connect_gaps trop lent: {elapsed*1000:.2f} ms >= 100 ms"
    assert len(res.connections) == 25
    # Déterminisme bit-à-bit
    assert connect_gaps(graph, 10.0) == res

