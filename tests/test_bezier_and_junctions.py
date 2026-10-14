"""Tests de validation pour la précision RMS des chemins Bézier et la robustesse des jonctions de boucles."""

import math
import numpy as np
import pytest

from raster2vector.coordinates import Point
from raster2vector.bezier import (
    BezierPath,
    CubicBezier,
    evaluate_cubic,
    fit_bezier,
    fit_piecewise_bezier,
    _path_rms,
)
from raster2vector.geometry import GeometryResult, LineSegment
from raster2vector.graph import GraphEdge, GraphNode, SkeletonGraph
from raster2vector.junctions import _adjust, refine_junctions


def test_bezier_path_rms_accuracy_multi_segment() -> None:
    """Vérifie l'exactitude de fit_error_rms sur un BezierPath à 3 segments.

    L'erreur RMS doit être strictement égale à la racine de la moyenne des carrés des
    distances point-courbe sur les N points sources uniques (sans doublon aux raccords).
    """
    # Construction d'une courbe source avec 3 régimes bien distincts pour forcer 3 segments
    t1 = np.linspace(0, 1, 41)
    pts1 = [Point(float(10 + 30 * v), float(20 + 25 * np.sin(v * np.pi))) for v in t1]

    t2 = np.linspace(0, 1, 41)[1:]
    pts2 = [Point(float(40 + 30 * v), float(20 - 25 * np.sin(v * np.pi))) for v in t2]

    t3 = np.linspace(0, 1, 41)[1:]
    pts3 = [Point(float(70 + 30 * v), float(20 + 25 * np.sin(v * np.pi))) for v in t3]

    source_points = tuple(pts1 + pts2 + pts3)
    cut1 = len(pts1) - 1
    cut2 = cut1 + len(pts2)

    path = fit_piecewise_bezier(source_points, tolerance_px=2.0, break_indices=[cut1, cut2])
    assert path is not None
    assert len(path.segments) == 3

    # Calcul direct via evaluate_cubic sur les points sources uniques
    evaluated_chunks = []
    for i, segment in enumerate(path.segments):
        evals = evaluate_cubic(segment, segment.source_parameters)
        if i > 0:
            evals = evals[1:]
        evaluated_chunks.append(evals)

    all_evaluated = np.vstack(evaluated_chunks)
    coordinates = np.array([(p.x, p.y) for p in source_points], dtype=np.float64)

    assert len(all_evaluated) == len(coordinates)
    point_squared_errors = np.sum((all_evaluated - coordinates) ** 2, axis=1)
    direct_rms = float(np.sqrt(np.mean(point_squared_errors)))

    assert math.isclose(path.fit_error_rms, direct_rms, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(_path_rms(path.segments, coordinates), direct_rms, rel_tol=1e-12, abs_tol=1e-12)


def test_fit_piecewise_bezier_closed_loops_various_sizes() -> None:
    """Vérifie le comportement de fit_piecewise_bezier sur des boucles fermées de 2, 3, 4 et 8 points."""
    # 1. Boucle fermée de 2 points : [P0, P0]
    p0 = Point(10.0, 10.0)
    loop_2 = (p0, p0)
    res_2 = fit_piecewise_bezier(loop_2, tolerance_px=1.0, break_indices=[], is_loop=True)
    assert res_2 is None

    # 2. Boucle fermée de 3 points : [P0, P1, P0]
    p1 = Point(20.0, 30.0)
    loop_3 = (p0, p1, p0)
    res_3 = fit_piecewise_bezier(loop_3, tolerance_px=1.0, break_indices=[], is_loop=True)
    assert res_3 is None

    # 3. Boucle fermée de 4 points : [P0, P1, P2, P0]
    p2 = Point(30.0, 10.0)
    loop_4 = (p0, p1, p2, p0)
    # Sans découpage explicite, moins de 2 intervalles valides subsistent -> doit retourner None sans erreur
    res_4 = fit_piecewise_bezier(loop_4, tolerance_px=1.0, break_indices=[], is_loop=True)
    assert res_4 is None

    # Boucle fermée de 4 points avec découpage valide (ex: cut à l'indice 2 -> 2 intervalles)
    res_4_cuts = fit_piecewise_bezier(loop_4, tolerance_px=5.0, break_indices=[2], is_loop=True)
    assert res_4_cuts is not None
    assert res_4_cuts.is_loop is True
    assert res_4_cuts.segments[0].start == res_4_cuts.segments[-1].end == p0

    # 4. Boucle fermée de 8 points
    theta = np.linspace(0, 2 * np.pi, 8)
    center_x, center_y, radius = 50.0, 50.0, 20.0
    pts_8 = [Point(float(center_x + radius * np.cos(t)), float(center_y + radius * np.sin(t))) for t in theta[:-1]]
    pts_8.append(pts_8[0])
    loop_8 = tuple(pts_8)
    assert len(loop_8) == 8
    assert loop_8[0] == loop_8[-1]

    res_8 = fit_piecewise_bezier(loop_8, tolerance_px=2.0, break_indices=[], is_loop=True)
    assert res_8 is not None
    assert res_8.is_loop is True
    assert len(res_8.segments) >= 2
    assert res_8.segments[0].start == res_8.segments[-1].end == loop_8[0]
    assert all(a.end == b.start for a, b in zip(res_8.segments, res_8.segments[1:]))


def test_refine_junctions_preserves_closed_bezier_loop() -> None:
    """Vérifie la préservation de la fermeture segments[0].start == segments[-1].end après ajustement."""
    # Création d'une boucle Bézier fermée valide
    theta = np.linspace(0, 2 * np.pi, 17)
    pts = [Point(float(50.0 + 20.0 * np.cos(t)), float(50.0 + 20.0 * np.sin(t))) for t in theta[:-1]]
    pts.append(pts[0])
    loop_points = tuple(pts)

    loop_bezier = fit_piecewise_bezier(loop_points, tolerance_px=1.0, break_indices=[], is_loop=True)
    assert loop_bezier is not None
    assert loop_bezier.is_loop is True
    assert loop_bezier.segments[0].start == loop_bezier.segments[-1].end

    original_start = loop_bezier.segments[0].start

    # Test unitaire de _adjust :
    # Cas A : start != end sur un BezierPath en boucle -> doit refuser l'ajustement
    shifted_start = Point(original_start.x + 0.2, original_start.y + 0.2)
    shifted_end = Point(original_start.x + 0.4, original_start.y + 0.1)
    rejected = _adjust(loop_bezier, shifted_start, shifted_end, tolerance=2.0)
    assert rejected is None

    # Cas B : start == end sur un BezierPath en boucle -> translation uniforme, fermeture préservée
    adjusted = _adjust(loop_bezier, shifted_start, shifted_start, tolerance=2.0)
    assert adjusted is not None
    assert isinstance(adjusted, BezierPath)
    assert adjusted.is_loop is True
    assert adjusted.segments[0].start == adjusted.segments[-1].end == shifted_start

    # Test global via refine_junctions dans un graphe avec un nœud jonction
    # Nœud 0 : jonction située à shifted_start
    # Nœud 1 : point d'extrémité pour une seconde branche
    # Arête 0 : la boucle fermée connectée à elle-même sur le nœud 0
    # Arête 1 : un segment linéaire connecté entre le nœud 0 et le nœud 1
    junction_pt = shifted_start
    branch_end_pt = Point(junction_pt.x + 15.0, junction_pt.y)

    # Coordonnées des arêtes du graphe
    edge_0_points = (junction_pt, *loop_points[1:-1], junction_pt)
    edge_1_points = (junction_pt, branch_end_pt)

    graph = SkeletonGraph(
        width=100,
        height=100,
        nodes=(
            GraphNode(0, "junction", junction_pt, ((round(junction_pt.y), round(junction_pt.x)),)),
            GraphNode(1, "endpoint", branch_end_pt, ((round(branch_end_pt.y), round(branch_end_pt.x)),)),
        ),
        edges=(
            GraphEdge(0, 0, 0, edge_0_points),
            GraphEdge(1, 0, 1, edge_1_points),
        ),
    )

    line_segment = LineSegment(
        start=junction_pt,
        end=branch_end_pt,
        fit_error_max=0.0,
        fit_error_rms=0.0,
        source_edge_id=1,
        source_points=edge_1_points,
    )

    # loop_bezier initialisé avec source_edge_id=0
    loop_primitive = fit_piecewise_bezier(
        edge_0_points, tolerance_px=1.0, break_indices=[], source_edge_id=0, is_loop=True
    )
    assert loop_primitive is not None

    geometry = GeometryResult(
        width=100,
        height=100,
        primitives=(loop_primitive, line_segment),
    )

    junction_result = refine_junctions(graph, geometry, tolerance_px=2.0)
    refined_loop = junction_result.geometry.primitives[0]

    assert isinstance(refined_loop, BezierPath)
    assert refined_loop.is_loop is True
    assert refined_loop.segments[0].start == refined_loop.segments[-1].end
