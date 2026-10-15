"""Tests du dé-biaisage des inflexions de squelettisation aux jonctions (Zhang-Suen)."""

import math
import numpy as np
import pytest

from raster2vector.coordinates import Point
from raster2vector.graph import GraphNode, GraphEdge, SkeletonGraph
from raster2vector.geometry import (
    LineSegment,
    CircularArc,
    Polyline,
    RawPolyline,
    fit_line,
    fit_arc,
    reconstruct_geometry,
    _trimmed_junction_slice,
    _debias_line_at_junctions,
    _debias_arc_at_junctions,
)


def test_line_debiasing_at_junction_vs_endpoint() -> None:
    """Une déviation de 2.2 px à une jonction devient un LineSegment, mais une Polyline à un endpoint."""
    tolerance = 1.5

    # 1. Branche de 25 points avec inflexion de 2.2 px côté jonction (fin)
    points_junc = tuple(
        [Point(float(x), 0.0) for x in range(23)]
        + [Point(23.0, -1.2), Point(24.0, 2.2)]
    )
    # L'ajustement global dépasse la tolérance de 1.5 px
    fit_global = fit_line(points_junc)
    assert fit_global is not None
    assert fit_global.fit_error_max > tolerance

    node_endpoint = GraphNode(0, "endpoint", points_junc[0], ())
    node_junction = GraphNode(1, "junction", points_junc[-1], ())
    graph_junc = SkeletonGraph(100, 100, (node_endpoint, node_junction), (GraphEdge(0, 0, 1, points_junc),))

    result_junc = reconstruct_geometry(graph_junc, geometry_tolerance_px=tolerance)
    prim_junc = result_junc.primitives[0]
    assert isinstance(prim_junc, LineSegment)
    assert prim_junc.source_edge_id == 0

    # 2. Même déviation placée côté endpoint (début)
    points_end = tuple(
        [Point(0.0, 2.2), Point(1.0, -1.2)]
        + [Point(float(x), 0.0) for x in range(2, 25)]
    )
    node_endpoint_start = GraphNode(0, "endpoint", points_end[0], ())
    node_junction_end = GraphNode(1, "junction", points_end[-1], ())
    graph_end = SkeletonGraph(100, 100, (node_endpoint_start, node_junction_end), (GraphEdge(0, 0, 1, points_end),))

    result_end = reconstruct_geometry(graph_end, geometry_tolerance_px=tolerance)
    prim_end = result_end.primitives[0]
    # L'extrémité libre (crochet) ne doit JAMAIS être rognée : refus de LineSegment et repli en Polyline
    assert not isinstance(prim_end, LineSegment)
    assert isinstance(prim_end, Polyline)


def test_arc_debiasing_at_junction() -> None:
    """Un arc avec une inflexion de 2 px arrivant sur une jonction est reconnu en CircularArc."""
    tolerance = 1.0
    theta = np.linspace(0, math.pi / 2, 25)
    coords = np.column_stack([50.0 + 50.0 * np.cos(theta), 50.0 + 50.0 * np.sin(theta)])
    # Inflexion radiale de ~2 px aux 2 derniers pixels touchant la jonction
    coords[-1] += [0.0, 2.0]
    coords[-2] += [0.0, 1.2]
    points = tuple(Point(float(x), float(y)) for x, y in coords)

    # L'ajustement direct de l'arc entier échoue à tolérance 1.0 px
    assert fit_arc(points, tolerance) is None

    node_endpoint = GraphNode(0, "endpoint", points[0], ())
    node_junction = GraphNode(1, "junction", points[-1], ())
    graph = SkeletonGraph(100, 100, (node_endpoint, node_junction), (GraphEdge(0, 0, 1, points),))

    result = reconstruct_geometry(graph, geometry_tolerance_px=tolerance)
    prim = result.primitives[0]
    assert isinstance(prim, CircularArc)
    assert prim.source_edge_id == 0


def test_zero_tolerance_disables_trimming() -> None:
    """geometry_tolerance_px = 0 désactive tout rognage."""
    points = tuple(
        [Point(float(x), 0.0) for x in range(23)]
        + [Point(23.0, -1.2), Point(24.0, 2.2)]
    )
    # _trimmed_junction_slice retourne None
    assert _trimmed_junction_slice(points, True, True, 0.0) is None
    assert _trimmed_junction_slice(points, False, True, 0.0) is None

    node_ep = GraphNode(0, "endpoint", points[0], ())
    node_junc = GraphNode(1, "junction", points[-1], ())
    graph = SkeletonGraph(100, 100, (node_ep, node_junc), (GraphEdge(0, 0, 1, points),))

    result = reconstruct_geometry(graph, geometry_tolerance_px=0.0, polyline_tolerance_px=None)
    prim = result.primitives[0]
    assert not isinstance(prim, LineSegment)
    assert isinstance(prim, RawPolyline)


def test_trimmed_junction_slice_rules() -> None:
    """Vérifie les règles de calcul de la tranche interne rognée."""
    points = tuple(Point(float(x), 0.0) for x in range(25))

    # Moins de 10 points -> None
    assert _trimmed_junction_slice(points[:9], True, True, 1.5) is None

    # Ni début ni fin en jonction -> None
    assert _trimmed_junction_slice(points, False, False, 1.5) is None

    # Tolérance <= 0 -> None
    assert _trimmed_junction_slice(points, True, False, -0.5) is None

    # Seulement fin en jonction : trim = min(max(1, ceil(1.5)), 25 // 5) = 2
    sl_end = _trimmed_junction_slice(points, False, True, 1.5)
    assert sl_end == slice(0, 23)

    # Seulement début en jonction
    sl_start = _trimmed_junction_slice(points, True, False, 1.5)
    assert sl_start == slice(2, 25)

    # Deux côtés en jonction : trim aux deux bouts
    sl_both = _trimmed_junction_slice(points, True, True, 1.5)
    assert sl_both == slice(2, 23)

    # Rognage qui laisserait moins de 4 points -> None
    short_points = tuple(Point(float(x), 0.0) for x in range(10))
    # trim = min(max(1, ceil(4.0)), 2) = 2 -> 10 - 4 = 6 >= 4
    assert _trimmed_junction_slice(short_points, True, True, 4.0) == slice(2, 8)


def test_excessive_deviation_rejected() -> None:
    """Une déviation supérieure à 2.0 * tolerance_px est rejetée du dé-biaisage."""
    tolerance = 1.0
    # Déviation de 2.5 px (> 2.0 * 1.0 = 2.0 px)
    points = tuple(
        [Point(float(x), 0.0) for x in range(23)]
        + [Point(23.0, 0.0), Point(24.0, 2.5)]
    )
    core_slice = slice(0, 23)
    assert _debias_line_at_junctions(points, 0, core_slice, tolerance) is None
