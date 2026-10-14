"""Tests de validation numérique et robustesse vectorielle pour les primitives globales.

Vérifie la conformité vectorielle de _finite_arc_bound, _radial_bound,
_radial_bound_array, reconstruct_global_circles et reconstruct_global_arcs,
notamment face à un échantillonnage fin (10 000 points/segment), des traversées
de discontinuités angulaires (±pi, 0/2pi), et des cas limites singuliers
(longueur nulle, parallélisme médiatrice/rayons).
"""

import math
import numpy as np
import pytest

from raster2vector.coordinates import Point
from raster2vector.geometry import CircleFit, reconstruct_geometry
from raster2vector.global_circles import (
    GlobalCircle,
    _radial_bound,
    _radial_bound_array,
    reconstruct_global_circles,
)
from raster2vector.global_arcs import (
    GlobalArc,
    _finite_arc_bound,
    reconstruct_global_arcs,
)
from raster2vector.graph import GraphEdge, GraphNode, SkeletonGraph


def _sample_arc_distance(coords: np.ndarray, arc: GlobalArc, num_samples_per_seg: int = 10000) -> float:
    """Évaluation de référence par échantillonnage fin sur chaque segment."""
    center = np.array([arc.center.x, arc.center.y], dtype=np.float64)
    start_pt = center + np.array(
        [arc.radius * math.cos(arc.start_angle_rad), arc.radius * math.sin(arc.start_angle_rad)],
        dtype=np.float64,
    )
    end_pt = center + np.array(
        [arc.radius * math.cos(arc.end_angle_rad), arc.radius * math.sin(arc.end_angle_rad)],
        dtype=np.float64,
    )

    if len(coords) < 2:
        if len(coords) == 0:
            return 0.0
        pts = coords
    else:
        seg_starts = coords[:-1]
        seg_ends = coords[1:]
        t = np.linspace(0.0, 1.0, num_samples_per_seg + 1, dtype=np.float64)[:, None, None]
        pts_grid = seg_starts[None, :, :] + t * (seg_ends - seg_starts)[None, :, :]
        pts = pts_grid.reshape(-1, 2)

    rel = pts - center
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    progress = np.mod(angles - arc.start_angle_rad, 2.0 * math.pi)
    radii = np.linalg.norm(rel, axis=1)
    errors = np.abs(radii - arc.radius)
    outside = progress > arc.sweep_angle_rad
    if np.any(outside):
        errors[outside] = np.minimum(
            np.linalg.norm(pts[outside] - start_pt, axis=1),
            np.linalg.norm(pts[outside] - end_pt, axis=1),
        )
    return float(np.max(errors))


@pytest.fixture
def base_center() -> Point:
    return Point(120.0, 150.0)


@pytest.mark.parametrize(
    "start_angle,sweep",
    [
        (0.2, 1.1),  # Quadrant 1
        (1.8, 1.0),  # Quadrant 2
        (3.4, 0.9),  # Quadrant 3
        (4.9, 1.2),  # Quadrant 4
        (2.8, 0.8),  # Traverse la discontinuité ±pi (de 2.8 à 3.6 rad)
        (5.8, 1.2),  # Traverse la coupure 0 / 2pi (de 5.8 à 0.7 rad)
        (0.5, 1.8 * math.pi),  # Arc très large (> pi)
    ],
)
def test_finite_arc_bound_vs_sampling_across_quadrants(
    base_center: Point, start_angle: float, sweep: float
) -> None:
    radius = 60.0
    arc = GlobalArc(
        center=base_center,
        radius=radius,
        start_angle_rad=start_angle,
        end_angle_rad=start_angle + sweep,
        fit_error_max=0.0,
        fit_error_rms=0.0,
        source_edge_id=0,
        source_points=(),
        source_edge_ids=(0,),
        support_edge_ids=(0,),
    )

    test_segments = [
        # Segment traversant l'intérieur du balayage de l'arc
        np.array([[70.0, 100.0], [170.0, 200.0]]),
        # Segment extérieur au balayage de l'arc
        np.array([[20.0, 20.0], [40.0, 50.0]]),
        # Segment passant très près du centre
        np.array([[119.0, 149.0], [160.0, 190.0]]),
        # Polyline multi-segments avec virages
        np.array([
            [80.0, 130.0],
            [120.0, 150.0],
            [160.0, 120.0],
            [190.0, 170.0],
        ]),
    ]

    for seg in test_segments:
        bound = _finite_arc_bound(seg, arc)
        sampled = _sample_arc_distance(seg, arc, num_samples_per_seg=10000)

        # La borne analytique doit être supérieure ou égale au maximum échantillonné
        assert bound >= sampled - 1e-9, f"Bound {bound} < Sampled {sampled}"
        # L'écart avec 10 000 points doit être inférieur à 5e-3 (précision d'échantillonnage)
        assert math.isclose(bound, sampled, abs_tol=5e-3), (
            f"Écart bound vs sampled trop élevé: {abs(bound - sampled):.6e}"
        )


def test_finite_arc_bound_singularities_and_degeneracies(base_center: Point) -> None:
    """Vérifie l'absence stricte de RuntimeWarning sur les cas singuliers."""
    radius = 50.0
    start_angle = 0.6
    sweep = 2.1
    end_angle = start_angle + sweep

    arc = GlobalArc(
        center=base_center,
        radius=radius,
        start_angle_rad=start_angle,
        end_angle_rad=end_angle,
        fit_error_max=0.0,
        fit_error_rms=0.0,
        source_edge_id=0,
        source_points=(),
        source_edge_ids=(0,),
        support_edge_ids=(0,),
    )

    cx, cy = base_center.x, base_center.y
    start_vec = np.array([radius * math.cos(start_angle), radius * math.sin(start_angle)])
    end_vec = np.array([radius * math.cos(end_angle), radius * math.sin(end_angle)])
    direction = end_vec - start_vec

    # 1. Vecteur orthogonal à direction -> parallèle à la médiatrice des extrémités
    perp_direction = np.array([-direction[1], direction[0]])
    seg_parallel_med = np.array([[cx + 10.0, cy + 10.0], [cx + 10.0 + perp_direction[0], cy + 10.0 + perp_direction[1]]])

    # 2. Vecteur colinéaire au rayon de départ (cross_delta == 0)
    ray_start = np.array([math.cos(start_angle), math.sin(start_angle)])
    seg_parallel_ray_start = np.array([[cx - 20.0 * ray_start[0], cy - 20.0 * ray_start[1]],
                                       [cx + 70.0 * ray_start[0], cy + 70.0 * ray_start[1]]])

    # 3. Vecteur colinéaire au rayon d'arrivée (cross_delta == 0)
    ray_end = np.array([math.cos(end_angle), math.sin(end_angle)])
    seg_parallel_ray_end = np.array([[cx - 30.0 * ray_end[0], cy - 30.0 * ray_end[1]],
                                     [cx + 80.0 * ray_end[0], cy + 80.0 * ray_end[1]]])

    # 4. Segments dégénérés (longueur 0, length2 == 0)
    seg_degenerate = np.array([[cx + 30.0, cy + 40.0], [cx + 30.0, cy + 40.0]])

    # 5. Polyline mixte combinant segments dégénérés, parallèles et normaux
    mixed_polyline = np.array([
        [cx + 30.0, cy + 40.0],
        [cx + 30.0, cy + 40.0],  # segment de longueur nulle
        [cx + 30.0 + perp_direction[0], cy + 40.0 + perp_direction[1]],  # parallèle médiatrice
        [cx + 30.0 + perp_direction[0], cy + 40.0 + perp_direction[1]],  # segment de longueur nulle
        [cx, cy],  # passe au centre
    ])

    test_cases = [
        ("parallel_med", seg_parallel_med),
        ("parallel_ray_start", seg_parallel_ray_start),
        ("parallel_ray_end", seg_parallel_ray_end),
        ("degenerate", seg_degenerate),
        ("mixed_polyline", mixed_polyline),
    ]

    for name, seg in test_cases:
        bound = _finite_arc_bound(seg, arc)
        sampled = _sample_arc_distance(seg, arc, num_samples_per_seg=10000)
        assert bound >= sampled - 1e-9, f"[{name}] Bound {bound} < Sampled {sampled}"
        assert math.isclose(bound, sampled, abs_tol=5e-3), (
            f"[{name}] Écart bound vs sampled trop élevé: {abs(bound - sampled):.6e}"
        )


def test_finite_arc_bound_input_formats(base_center: Point) -> None:
    radius = 50.0
    arc = GlobalArc(
        center=base_center,
        radius=radius,
        start_angle_rad=0.0,
        end_angle_rad=math.pi / 2,
        fit_error_max=0.0,
        fit_error_rms=0.0,
        source_edge_id=0,
        source_points=(),
        source_edge_ids=(0,),
        support_edge_ids=(0,),
    )

    pts_tuple = (Point(120.0, 150.0), Point(170.0, 150.0), Point(170.0, 200.0))
    pts_array = np.array([[p.x, p.y] for p in pts_tuple], dtype=np.float64)

    bound_tuple = _finite_arc_bound(pts_tuple, arc)
    bound_array = _finite_arc_bound(pts_array, arc)

    assert math.isclose(bound_tuple, bound_array, abs_tol=1e-12)

    # Cas limite : séquence vide
    assert _finite_arc_bound((), arc) == 0.0
    assert _finite_arc_bound(np.empty((0, 2), dtype=np.float64), arc) == 0.0

    # Cas limite : point unique
    single_pt = (Point(170.0, 150.0),)
    single_array = np.array([[170.0, 150.0]], dtype=np.float64)
    expected_dist = abs(math.hypot(170.0 - base_center.x, 150.0 - base_center.y) - radius)
    assert math.isclose(_finite_arc_bound(single_pt, arc), expected_dist, abs_tol=1e-12)
    assert math.isclose(_finite_arc_bound(single_array, arc), expected_dist, abs_tol=1e-12)


def test_radial_bound_array_and_compat(base_center: Point) -> None:
    fit = CircleFit(base_center, 40.0, 0.0, 0.0)
    pts_tuple = (Point(100.0, 150.0), Point(120.0, 190.0), Point(160.0, 150.0))
    pts_array = np.array([[p.x, p.y] for p in pts_tuple], dtype=np.float64)

    bound_tuple = _radial_bound(pts_tuple, fit)
    bound_array = _radial_bound(pts_array, fit)
    bound_direct = _radial_bound_array(pts_array, fit)

    assert math.isclose(bound_tuple, bound_array, abs_tol=1e-12)
    assert math.isclose(bound_array, bound_direct, abs_tol=1e-12)

    # Cas limite : coordonnées vides
    assert _radial_bound((), fit) == 0.0
    assert _radial_bound(np.empty((0, 2), dtype=np.float64), fit) == 0.0
    assert _radial_bound_array(np.empty((0, 2), dtype=np.float64), fit) == 0.0

    # Segments de longueur nulle
    deg_array = np.array([[120.0, 190.0], [120.0, 190.0]], dtype=np.float64)
    assert math.isclose(_radial_bound_array(deg_array, fit), 0.0, abs_tol=1e-12)


def test_reconstruct_global_circles_with_edge_coords() -> None:
    """Vérifie la détection globale de cercle et l'utilisation de edge_coords."""
    center_x, center_y, r = 130.0, 130.0, 80.0
    count = 4
    angles = np.linspace(0, 2 * np.pi, count + 1)
    contacts = [Point(center_x + r * math.cos(a), center_y + r * math.sin(a)) for a in angles[:-1]]
    nodes = [GraphNode(i, "junction", p, ((round(p.y), round(p.x)),)) for i, p in enumerate(contacts)]
    edges = []
    for i in range(count):
        a = np.linspace(angles[i], angles[i + 1], 60)
        pts = [Point(center_x + r * math.cos(t), center_y + r * math.sin(t)) for t in a]
        end = (i + 1) % count
        pts[0], pts[-1] = contacts[i], contacts[end]
        edges.append(GraphEdge(i, i, end, tuple(pts)))

    graph = SkeletonGraph(260, 260, tuple(nodes), tuple(edges))
    local = reconstruct_geometry(graph, 1.0)

    result = reconstruct_global_circles(graph, local, tolerance_px=1.0)
    assert len(result.circles) == 1
    circle = result.circles[0]
    assert isinstance(circle, GlobalCircle)
    assert set(circle.source_edge_ids) == {0, 1, 2, 3}
    assert math.isclose(circle.radius, r, abs_tol=1e-3)
    assert math.isclose(circle.center.x, center_x, abs_tol=1e-3)
    assert math.isclose(circle.center.y, center_y, abs_tol=1e-3)


def test_reconstruct_global_arcs_with_edge_coords() -> None:
    """Vérifie la détection globale d'arc et l'utilisation de edge_coords."""
    center_x, center_y, r = 130.0, 130.0, 90.0
    limits = np.linspace(0.2, 0.2 + math.pi, 4)
    contacts = [Point(center_x + r * math.cos(t), center_y + r * math.sin(t)) for t in limits]
    nodes = [
        GraphNode(i, "junction" if 0 < i < 3 else "endpoint", p, ((round(p.y), round(p.x)),))
        for i, p in enumerate(contacts)
    ]
    edges = []
    for i in range(3):
        t = np.linspace(limits[i], limits[i + 1], 60)
        pts = [Point(center_x + r * math.cos(a), center_y + r * math.sin(a)) for a in t]
        pts[0], pts[-1] = contacts[i], contacts[i + 1]
        edges.append(GraphEdge(i, i, i + 1, tuple(pts)))

    graph = SkeletonGraph(260, 260, tuple(nodes), tuple(edges))
    local = reconstruct_geometry(graph, 1.0)

    result = reconstruct_global_arcs(graph, local, tolerance_px=1.0)
    assert len(result.arcs) == 1
    arc = result.arcs[0]
    assert isinstance(arc, GlobalArc)
    assert set(arc.source_edge_ids) == {0, 1, 2}
    assert math.isclose(arc.radius, r, abs_tol=1e-2)
    assert math.isclose(arc.sweep_angle_rad, math.pi, abs_tol=0.02)
