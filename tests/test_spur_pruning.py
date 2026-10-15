"""Tests de l'élagage topologique des micro-barbes de squelette (spur_tolerance)."""

from pathlib import Path
import pytest
import numpy as np

from raster2vector.coordinates import Point
from raster2vector.graph import (
    GraphNode,
    GraphEdge,
    SkeletonGraph,
    skeleton_to_graph,
    prune_spurs,
)
from raster2vector.geometry import reconstruct_geometry, LineSegment
from raster2vector.config import VectorizationConfig, load_config
from raster2vector.cli import main
from raster2vector.image_io import save_png


def test_horizontal_line_with_spur_pruned_to_single_segment() -> None:
    """Ligne droite de 50 px avec barbe de 3 px au milieu : réduite à 1 arête et 1 LineSegment."""
    mask = np.zeros((30, 60), dtype=bool)
    mask[15, 5:56] = True  # Ligne horizontale de 50 px (x=5 à x=55)
    mask[16:19, 30] = True  # Barbe verticale de 3 px (y=16, 17, 18 à x=30)

    graph = skeleton_to_graph(mask)
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 3
    assert sum(1 for n in graph.nodes if n.kind == "junction") == 1
    assert sum(1 for n in graph.nodes if n.kind == "endpoint") == 3

    # Sans élagage, reconstruct_geometry produit 3 segments
    res_raw = reconstruct_geometry(graph, geometry_tolerance_px=2.0)
    assert len(res_raw.primitives) == 3

    # Avec élagage de tolérance 4.0 px : la barbe de 3 px est supprimée et la fausse jonction est dissoute
    pruned = prune_spurs(graph, max_spur_length_px=4.0)
    assert len(pruned.nodes) == 2
    assert len(pruned.edges) == 1
    assert all(n.kind == "endpoint" for n in pruned.nodes)

    edge = pruned.edges[0]
    assert len(edge.points) == 51
    assert (edge.points[0].x, edge.points[0].y) == (5.0, 15.0)
    assert (edge.points[-1].x, edge.points[-1].y) == (55.0, 15.0)

    # Reconstruct_geometry produit un unique LineSegment continu de 50 px
    res_pruned = reconstruct_geometry(pruned, geometry_tolerance_px=2.0)
    assert len(res_pruned.primitives) == 1
    prim = res_pruned.primitives[0]
    assert isinstance(prim, LineSegment)
    assert (prim.start.x, prim.start.y) == pytest.approx((5.0, 15.0))
    assert (prim.end.x, prim.end.y) == pytest.approx((55.0, 15.0))


def test_star_preserves_at_least_two_branches() -> None:
    """La règle de préservation conserve au moins 2 branches sur une étoile de petites branches."""
    # Jonction centrale à (15, 15), 4 branches de longueur 3 px
    mask = np.zeros((31, 31), dtype=bool)
    mask[15, 12:19] = True  # horizontal 6 px
    mask[12:19, 15] = True  # vertical 6 px

    graph = skeleton_to_graph(mask)
    assert any(n.kind == "junction" for n in graph.nodes)
    assert len(graph.edges) == 4

    # Tolérance supérieure à la longueur des branches (5.0 px > 3 px)
    pruned = prune_spurs(graph, max_spur_length_px=5.0)
    # Exactement 2 branches sont conservées, la jonction devenue de degré 2 est dissoute
    assert len(pruned.nodes) == 2
    assert len(pruned.edges) == 1
    assert all(n.kind == "endpoint" for n in pruned.nodes)

    # Même principe avec 3 branches : 1 seule est élaguée, 2 sont conservées
    mask3 = np.zeros((31, 31), dtype=bool)
    mask3[15, 15:20] = True
    mask3[15:20, 15] = True
    mask3[10:16, 15] = True
    graph3 = skeleton_to_graph(mask3)
    assert any(n.kind == "junction" for n in graph3.nodes)
    pruned3 = prune_spurs(graph3, max_spur_length_px=6.0)
    assert len(pruned3.nodes) == 2
    assert len(pruned3.edges) == 1


def test_cli_and_toml_spur_tolerance(tmp_path: Path) -> None:
    """--spur-tolerance en CLI et spur_tolerance en TOML fonctionnent de bout en bout."""
    img_path = tmp_path / "spur_line.png"
    img = np.full((30, 60), 255, dtype=np.uint8)
    img[15, 5:56] = 0
    img[16:19, 30] = 0
    save_png(img_path, img)

    # 1. Sans option (défaut = 0) -> 3 chemins
    svg_raw = tmp_path / "raw.svg"
    assert main([str(img_path), "--output", str(svg_raw)]) == 0
    assert svg_raw.read_text("utf-8").count('id="edge-') == 3

    # 2. Avec --spur-tolerance 4.0 -> 1 chemin unique
    svg_cli = tmp_path / "cli.svg"
    assert main([str(img_path), "--output", str(svg_cli), "--spur-tolerance", "4.0"]) == 0
    assert svg_cli.read_text("utf-8").count('id="edge-') == 1

    # 3. Avec fichier TOML
    config_file = tmp_path / "config.toml"
    config_file.write_text("[vectorization]\nspur_tolerance = 4.0\n", encoding="utf-8")
    cfg = load_config(config_file)
    assert cfg.spur_tolerance == 4.0

    svg_toml = tmp_path / "toml.svg"
    assert main([str(img_path), "--output", str(svg_toml), "--config", str(config_file)]) == 0
    assert svg_toml.read_text("utf-8").count('id="edge-') == 1


@pytest.mark.parametrize("val", [-1.0, float("nan"), float("inf")])
def test_invalid_spur_tolerance_raises(val: float) -> None:
    """Une tolérance d'élagage négative ou non finie lève ValueError."""
    g = SkeletonGraph(10, 10, (), ())
    with pytest.raises(ValueError, match="tolérance d'élagage"):
        prune_spurs(g, val)
    with pytest.raises(ValueError, match="spur-tolerance"):
        VectorizationConfig(spur_tolerance=val)


def test_zero_spur_tolerance_returns_original_graph() -> None:
    """max_spur_length_px == 0 retourne le graphe d'origine sans modification."""
    e0 = GraphEdge(0, 0, 1, (Point(0, 0), Point(5, 5)))
    n0 = GraphNode(0, "endpoint", Point(0, 0), ((0, 0),))
    n1 = GraphNode(1, "junction", Point(5, 5), ((5, 5),))
    g = SkeletonGraph(10, 10, (n0, n1), (e0,))
    assert prune_spurs(g, 0.0) is g


def test_multi_pixel_junction_bridging() -> None:
    """Une jonction dissoute avec plusieurs source_pixels est pontée de manière continue."""
    # e1 se termine au pixel (10, 24)
    e1 = GraphEdge(0, 0, 1, tuple(Point(float(x), 10.0) for x in range(0, 25)))
    # e2 commence au pixel (10, 26)
    e2 = GraphEdge(1, 1, 2, tuple(Point(float(x), 10.0) for x in range(26, 51)))
    # e3 est une barbe de 3 px attachée à la jonction
    e3 = GraphEdge(2, 1, 3, tuple(Point(25.0, float(y)) for y in range(10, 14)))

    n0 = GraphNode(0, "endpoint", Point(0, 10), ((10, 0),))
    n1 = GraphNode(1, "junction", Point(25, 10), ((10, 24), (10, 25), (10, 26)))
    n2 = GraphNode(2, "endpoint", Point(50, 10), ((10, 50),))
    n3 = GraphNode(3, "endpoint", Point(25, 13), ((13, 25),))

    g = SkeletonGraph(100, 100, (n0, n1, n2, n3), (e1, e2, e3))
    pruned = prune_spurs(g, 4.0)

    assert len(pruned.nodes) == 2
    assert len(pruned.edges) == 1
    edge = pruned.edges[0]
    assert len(edge.points) == 51
    # Vérifier la 8-connexité sans trou sur tous les points fusionnés
    for i in range(len(edge.points) - 1):
        p1 = edge.points[i]
        p2 = edge.points[i + 1]
        assert max(abs(p2.x - p1.x), abs(p2.y - p1.y)) <= 1.0


def test_cycle_of_dissolved_junctions_becomes_loop() -> None:
    """Un cycle fermé composé uniquement de jonctions dissoutes conserve un nœud d'ancre 'loop'."""
    # 3 jonctions formant un triangle, chacune avec une petite barbe de 1 px
    n0 = GraphNode(0, "junction", Point(0, 0), ((0, 0),))
    n1 = GraphNode(1, "junction", Point(10, 0), ((0, 10),))
    n2 = GraphNode(2, "junction", Point(5, 8), ((8, 5),))

    sp0 = GraphNode(3, "endpoint", Point(0, 1), ((1, 0),))
    sp1 = GraphNode(4, "endpoint", Point(10, 1), ((1, 10),))
    sp2 = GraphNode(5, "endpoint", Point(5, 9), ((9, 5),))

    e_sp0 = GraphEdge(0, 0, 3, (Point(0, 0), Point(0, 1)))
    e_sp1 = GraphEdge(1, 1, 4, (Point(10, 0), Point(10, 1)))
    e_sp2 = GraphEdge(2, 2, 5, (Point(5, 8), Point(5, 9)))

    e01 = GraphEdge(3, 0, 1, (Point(0, 0), Point(5, 0), Point(10, 0)))
    e12 = GraphEdge(4, 1, 2, (Point(10, 0), Point(7.5, 4), Point(5, 8)))
    e20 = GraphEdge(5, 2, 0, (Point(5, 8), Point(2.5, 4), Point(0, 0)))

    g = SkeletonGraph(50, 50, (n0, n1, n2, sp0, sp1, sp2), (e_sp0, e_sp1, e_sp2, e01, e12, e20))
    pruned = prune_spurs(g, max_spur_length_px=1.5)

    assert len(pruned.nodes) == 1
    assert pruned.nodes[0].kind == "loop"
    assert len(pruned.edges) == 1
    edge = pruned.edges[0]
    assert edge.is_loop
    assert edge.points[0] == edge.points[-1]
