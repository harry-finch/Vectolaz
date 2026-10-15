"""Tests pour la simplification des boucles fermées en polylignes ou Bézier."""

import math
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

from raster2vector.coordinates import Point
from raster2vector.bezier import BezierPath
from raster2vector.geometry import (
    GeometryResult, LineSegment, Polyline, RawPolyline, Circle,
    _closed_polyline_approximation, reconstruct_geometry, simplify_polyline,
)
from raster2vector.graph import GraphEdge, GraphNode, SkeletonGraph, skeleton_to_graph
from raster2vector.junctions import _adjust
from raster2vector.skeleton import skeletonize_mask
from raster2vector.svg_export import svg_document


def loop_graph_for(points: tuple[Point, ...]) -> SkeletonGraph:
    """Crée un SkeletonGraph contenant une seule arête boucle fermée."""
    node = GraphNode(0, "loop", points[0], ((round(points[0].y), round(points[0].x)),))
    return SkeletonGraph(128, 128, (node,), (GraphEdge(0, 0, 0, points),))


def test_closed_rectangle_80px_becomes_4_segment_polyline_and_svg_z() -> None:
    """Vérifie qu'un rectangle rasterisé de 80 pixels devient une Polyline à 4 segments exportée avec Z."""
    # Rectangle de coins (10, 10), (30, 10), (30, 30), (10, 30)
    # 20 px par côté = boucle de 80 pixels fermée
    points = []
    for x in range(10, 30):
        points.append(Point(float(x), 10.0))
    for y in range(10, 30):
        points.append(Point(30.0, float(y)))
    for x in range(30, 10, -1):
        points.append(Point(float(x), 30.0))
    for y in range(30, 10, -1):
        points.append(Point(10.0, float(y)))
    points.append(Point(10.0, 10.0))
    source = tuple(points)
    assert len(source) == 81
    assert source[0] == source[-1]

    graph = loop_graph_for(source)
    result = reconstruct_geometry(graph, geometry_tolerance_px=2.0, polyline_tolerance_px=2.0)
    assert len(result.primitives) == 1
    primitive = result.primitives[0]

    assert isinstance(primitive, Polyline)
    assert primitive.is_loop is True
    assert primitive.segment_count == 4
    assert len(primitive.points) == 5
    assert primitive.points[0] == primitive.points[-1]
    assert primitive.fit_error_max < 1e-10

    # Vérification de la correspondance des sous-chaînes sources
    for i in range(primitive.segment_count):
        a = primitive.source_indices[i]
        b = primitive.source_indices[i + 1]
        assert primitive.source_points[a] == primitive.points[i]
        assert primitive.source_points[b] == primitive.points[i + 1]

    # Export SVG : doit contenir Z et ne pas répéter le point de départ avant Z
    svg = svg_document(result)
    assert " Z" in svg
    root = ET.fromstring(svg)
    ns = {"s": "http://www.w3.org/2000/svg"}
    paths = root.findall("s:g/s:path", ns)
    assert len(paths) == 1
    d = paths[0].get("d", "")
    assert d.endswith(" Z")
    # Exactement 1 M, 3 L et 1 Z (pas de 4e L répétant le premier point)
    tokens = d.split()
    assert tokens.count("M") == 1
    assert tokens.count("L") == 3
    assert tokens.count("Z") == 1


def test_closed_triangle_becomes_3_segment_polyline() -> None:
    """Vérifie qu'un triangle fermé devient une Polyline à 3 segments."""
    # Sommets : (10, 10), (50, 10), (30, 45)
    def densify(vertices: tuple[Point, ...]) -> tuple[Point, ...]:
        pts = [vertices[0]]
        for start, end in zip(vertices, vertices[1:]):
            count = int(max(abs(end.x - start.x), abs(end.y - start.y)))
            for fraction in np.linspace(0, 1, count + 1)[1:]:
                pts.append(Point(float(start.x + fraction * (end.x - start.x)),
                                 float(start.y + fraction * (end.y - start.y))))
        return tuple(pts)

    corners = (Point(10, 10), Point(50, 10), Point(30, 45), Point(10, 10))
    source = densify(corners)
    assert source[0] == source[-1]

    graph = loop_graph_for(source)
    result = reconstruct_geometry(graph, geometry_tolerance_px=2.0, polyline_tolerance_px=2.0)
    assert len(result.primitives) == 1
    primitive = result.primitives[0]

    assert isinstance(primitive, Polyline)
    assert primitive.is_loop is True
    assert primitive.segment_count == 3
    assert len(primitive.points) == 4
    assert primitive.points[0] == primitive.points[-1]
    assert primitive.fit_error_max < 1e-10

    # Vérification SVG
    svg = svg_document(result)
    root = ET.fromstring(svg)
    ns = {"s": "http://www.w3.org/2000/svg"}
    d = root.find("s:g/s:path", ns).get("d", "")
    assert d.endswith(" Z")
    tokens = d.split()
    assert tokens.count("M") == 1
    assert tokens.count("L") == 2
    assert tokens.count("Z") == 1


def test_closed_d_shape_reconstructed_as_closed_bezier_via_corner_pivot() -> None:
    """Vérifie qu'une forme en D démarrant sur l'arc pivote sur les coins et devient un BezierPath fermé."""
    # Segment vertical de (10, 10) à (10, 50)
    # Demi-cercle centré en (10, 30) de rayon 20 reliant (10, 50) à (10, 10)
    # La chaîne commence au milieu de l'arc (angle 0, point (30, 30)) où le virage est nul.
    points = []
    # Arc de 0 à -pi/2 : (30, 30) -> (10, 10)
    for th in np.linspace(0, -np.pi / 2, 35):
        points.append(Point(float(round(10.0 + 20.0 * np.cos(th), 4)),
                            float(round(30.0 + 20.0 * np.sin(th), 4))))
    points.pop()
    # Ligne droite de (10, 10) à (10, 50)
    for y in np.linspace(10, 50, 41):
        points.append(Point(10.0, float(round(y, 4))))
    points.pop()
    # Arc de +pi/2 à 0 : (10, 50) -> (30, 30)
    for th in np.linspace(np.pi / 2, 0, 35):
        points.append(Point(float(round(10.0 + 20.0 * np.cos(th), 4)),
                            float(round(30.0 + 20.0 * np.sin(th), 4))))
    source = tuple(points)
    assert source[0] == source[-1]
    assert len(set(source[:-1])) == len(source) - 1

    graph = loop_graph_for(source)
    result = reconstruct_geometry(graph, geometry_tolerance_px=2.0, polyline_tolerance_px=2.0)
    assert len(result.primitives) == 1
    primitive = result.primitives[0]

    assert isinstance(primitive, BezierPath)
    assert primitive.is_loop is True
    assert len(primitive.segments) >= 2
    assert primitive.segments[0].start == primitive.segments[-1].end
    assert primitive.error_bound <= 2.0

    # L'un des coins (10, 10) ou (10, 50) doit être le sommet d'ouverture
    start_pt = primitive.segments[0].start
    assert (start_pt.x, start_pt.y) in ((10.0, 10.0), (10.0, 50.0))

    # SVG export
    svg = svg_document(result)
    assert " Z" in svg


def test_simplify_polyline_rejects_closed_loop() -> None:
    """Vérifie que l'API publique simplify_polyline continue de rejeter une polyligne fermée."""
    loop = (Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 0))
    with pytest.raises(ValueError, match="La simplification requiert une polyligne ouverte."):
        simplify_polyline(loop, tolerance_px=2.0)


def test_degenerate_loops_return_none_and_fallback_to_raw() -> None:
    """Vérifie que les boucles dégénérées (< 3 sommets, surface nulle, points identiques) restent RawPolyline."""
    # 1. Tous les points identiques
    identical = (Point(5, 5), Point(5, 5), Point(5, 5), Point(5, 5))
    assert _closed_polyline_approximation(identical, 0, 2.0) is None

    # 2. Aller-retour plat sur un segment sans surface
    flat_segment = (
        Point(0, 0), Point(5, 0), Point(10, 0), Point(15, 0), Point(20, 0),
        Point(15, 0), Point(10, 0), Point(5, 0), Point(0, 0),
    )
    assert _closed_polyline_approximation(flat_segment, 0, 2.0) is None

    # Dans reconstruct_geometry, le fallback doit être RawPolyline
    graph = loop_graph_for(flat_segment)
    result = reconstruct_geometry(graph, geometry_tolerance_px=2.0, polyline_tolerance_px=2.0)
    assert isinstance(result.primitives[0], RawPolyline)
    assert result.primitives[0].is_loop is True


def test_junction_adjust_refuses_different_endpoints_on_closed_polyline() -> None:
    """Vérifie que _adjust refuse tout déplacement où start != end pour une Polyline fermée."""
    pts = (Point(0, 0), Point(20, 0), Point(20, 20), Point(0, 0))
    poly = Polyline(
        points=pts,
        fit_error_max=0.0,
        fit_error_rms=0.0,
        source_edge_id=0,
        source_points=pts,
        source_indices=(0, 1, 2, 3),
        is_loop=True,
    )
    # start != end doit être refusé
    assert _adjust(poly, Point(1, 1), Point(2, 2), tolerance=5.0) is None

    # start == end dans la tolérance doit être accepté
    adjusted = _adjust(poly, Point(0.5, 0.5), Point(0.5, 0.5), tolerance=5.0)
    assert adjusted is not None
    assert isinstance(adjusted, Polyline)
    assert adjusted.is_loop is True
    assert adjusted.points[0] == Point(0.5, 0.5)
    assert adjusted.points[-1] == Point(0.5, 0.5)


def test_raster_image_closed_rectangle(tmp_path) -> None:
    """Vérifie la détection d'un polygone fermé sur une image binaire réelle."""
    image = np.full((100, 100), 255, dtype=np.uint8)
    cv2.rectangle(image, (25, 25), (75, 75), 0, 3)
    graph = skeleton_to_graph(skeletonize_mask(image == 0))
    assert len(graph.edges) == 1 and graph.edges[0].is_loop

    result = reconstruct_geometry(graph, geometry_tolerance_px=2.0, polyline_tolerance_px=2.0)
    assert len(result.primitives) == 1
    primitive = result.primitives[0]
    assert isinstance(primitive, Polyline)
    assert primitive.is_loop is True
    assert primitive.segment_count == 4
    assert len(primitive.points) == 5
