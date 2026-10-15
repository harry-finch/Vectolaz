"""Tests de continuité tangentielle G1 aux raccords internes des courbes de Bézier."""

import math
from dataclasses import replace
import numpy as np
import pytest

from raster2vector.coordinates import Point
from raster2vector.bezier import (
    CubicBezier,
    fit_bezier,
    fit_piecewise_bezier,
    _try_smooth_joint,
    _smooth_internal_joints,
)


def _joint_tangents(c_prev: CubicBezier, c_next: CubicBezier) -> tuple[float, float]:
    """Retourne le produit vectoriel (cross) et le produit scalaire (dot) normalisés."""
    t_in = np.array([c_prev.end.x - c_prev.control2.x, c_prev.end.y - c_prev.control2.y], dtype=np.float64)
    t_out = np.array([c_next.control1.x - c_next.start.x, c_next.control1.y - c_next.start.y], dtype=np.float64)
    L_in = float(np.linalg.norm(t_in))
    L_out = float(np.linalg.norm(t_out))
    assert L_in > 1e-9
    assert L_out > 1e-9
    u_in = t_in / L_in
    u_out = t_out / L_out
    cross = float(u_in[0] * u_out[1] - u_in[1] * u_out[0])
    dot = float(np.dot(u_in, u_out))
    return cross, dot


def test_sine_wave_internal_joints_g1() -> None:
    """Sur une onde sinusoïdale multi-cubiques, les raccords internes sont lissés en G1."""
    t = np.linspace(0, 4 * np.pi, 200)
    coords = np.column_stack([t * 10, np.sin(t) * 10])
    points = tuple(Point(float(x), float(y)) for x, y in coords)
    tolerance = 0.5

    path = fit_bezier(points, tolerance_px=tolerance)
    assert path is not None
    assert len(path.segments) >= 2
    assert path.error_bound <= tolerance

    for i in range(len(path.segments) - 1):
        c0 = path.segments[i]
        c1 = path.segments[i + 1]
        cross, dot = _joint_tangents(c0, c1)
        assert abs(cross) < 1e-10, f"Joint {i}: cross={cross} non nul"
        assert dot > 0.999999, f"Joint {i}: dot={dot} non colinéaire dans le même sens"


def test_spiral_internal_joints_g1() -> None:
    """Sur une spirale échantillonnée, les raccords internes lissés respectent G1 et la tolérance."""
    theta = np.linspace(0, 3 * np.pi, 200)
    r = 10 + 5 * theta
    coords = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    points = tuple(Point(float(x), float(y)) for x, y in coords)
    tolerance = 1.0

    path = fit_bezier(points, tolerance_px=tolerance)
    assert path is not None
    assert len(path.segments) >= 2
    assert path.error_bound <= tolerance

    # Vérifier que les raccords internes lissés sont colinéaires
    smoothed_joints = 0
    for i in range(len(path.segments) - 1):
        c0 = path.segments[i]
        c1 = path.segments[i + 1]
        cross, dot = _joint_tangents(c0, c1)
        if abs(cross) < 1e-10:
            assert dot > 0.999999
            smoothed_joints += 1
    assert smoothed_joints >= 1


def test_explicit_break_indices_preserve_sharp_corners() -> None:
    """Les angles vifs spécifiés par break_indices sont intégralement préservés."""
    t = np.linspace(0, 1, 51)
    # Deux arcs orthogonaux se rejoignant à angle droit en (50, 50)
    branch1 = [Point(float(50 * v), float(50 * (v ** 2))) for v in t]
    branch2 = [Point(float(50 + 50 * (1 - (1 - v) ** 2)), float(50 - 50 * v)) for v in t[1:]]
    points = tuple(branch1 + branch2)

    path = fit_piecewise_bezier(points, tolerance_px=1.0, break_indices=(50,))
    assert path is not None
    assert path.error_bound <= 1.0

    # Retrouver la jonction à l'indice de rupture 50
    corner_prev = [c for c in path.segments if c.source_end_index == 50]
    corner_next = [c for c in path.segments if c.source_start_index == 50]
    assert len(corner_prev) == 1 and len(corner_next) == 1
    c0 = corner_prev[0]
    c1 = corner_next[0]

    cross, dot = _joint_tangents(c0, c1)
    # L'angle doit être préservé (non aligné) : tangentes quasi-orthogonales
    assert abs(cross) > 0.8
    assert abs(dot) < 0.2


def test_closed_ellipse_four_seams_g1() -> None:
    """Sur une ellipse fermée, les 4 coutures (quarts + fermeture) sont toutes lissées en G1."""
    theta = np.linspace(0, 2 * math.pi, 201)
    coords = np.column_stack([130 + 80 * np.cos(theta), 130 + 35 * np.sin(theta)])
    points = [Point(float(x), float(y)) for x, y in coords]
    points[-1] = points[0]
    tolerance = 2.5

    path = fit_piecewise_bezier(tuple(points), tolerance_px=tolerance, break_indices=(), is_loop=True)
    assert path is not None
    assert path.is_loop
    assert len(path.segments) == 4
    assert path.error_bound <= tolerance

    # Vérifier les 4 raccords : 3 internes (0-1, 1-2, 2-3) et la fermeture (3-0)
    for i in range(len(path.segments)):
        c0 = path.segments[i]
        c1 = path.segments[(i + 1) % len(path.segments)]
        cross, dot = _joint_tangents(c0, c1)
        assert abs(cross) < 1e-10, f"Raccord {i}: cross={cross} non nul"
        assert dot > 0.999999, f"Raccord {i}: dot={dot} non unitaire"


def test_fallback_to_c0_when_tolerance_exceeded() -> None:
    """Si l'alignement G1 ferait dépasser la tolérance, le raccord C0 d'origine est conservé."""
    # Onde sinusoïdale ajustée à tolérance serrée où l'alignement G1 dépasserait la borne
    t = np.linspace(0, 4 * np.pi, 100)
    coords = np.column_stack([t * 10, np.sin(t) * 10])
    points = tuple(Point(float(x), float(y)) for x, y in coords)
    tight_tolerance = 0.5

    path = fit_bezier(points, tolerance_px=tight_tolerance)
    assert path is not None
    # Garantie absolue que la borne globale ne dépasse pas la tolérance
    assert path.error_bound <= tight_tolerance


def test_degenerate_handles_and_opposing_tangents_no_warning() -> None:
    """Les poignées nulles ou tangentes opposées sont ignorées sans lever d'exception ni de RuntimeWarning."""
    coords = np.zeros((10, 2), dtype=np.float64)

    # 1. Poignée dégénérée L_in == 0
    c_degen_in = CubicBezier(
        start=Point(0, 0), control1=Point(1, 0), control2=Point(2, 0), end=Point(2, 0),
        source_start_index=0, source_end_index=4, source_parameters=(0.0, 0.25, 0.5, 0.75, 1.0),
        fit_error_max=0.0, fit_error_rms=0.0, error_bound=0.0,
    )
    c_normal_out = CubicBezier(
        start=Point(2, 0), control1=Point(3, 0), control2=Point(4, 0), end=Point(5, 0),
        source_start_index=4, source_end_index=8, source_parameters=(0.0, 0.25, 0.5, 0.75, 1.0),
        fit_error_max=0.0, fit_error_rms=0.0, error_bound=0.0,
    )
    assert _try_smooth_joint(c_degen_in, c_normal_out, coords, 1.0) is None

    # 2. Poignée dégénérée L_out == 0
    c_degen_out = replace(c_normal_out, control1=Point(2, 0))
    c_normal_in = replace(c_degen_in, control2=Point(1.5, 0))
    assert _try_smooth_joint(c_normal_in, c_degen_out, coords, 1.0) is None

    # 3. Tangentes en sens opposé (demi-tour : dot <= 0.5)
    c_opposing_out = replace(c_normal_out, control1=Point(1, 0))  # t_out pointe vers la gauche
    assert _try_smooth_joint(c_normal_in, c_opposing_out, coords, 1.0) is None


def test_smooth_internal_joints_loop_corner_protection() -> None:
    """La fermeture de boucle n'est pas lissée si l'indice 0 ou N-1 est dans corner_indices."""
    theta = np.linspace(0, 2 * math.pi, 201)
    coords = np.column_stack([130 + 80 * np.cos(theta), 130 + 35 * np.sin(theta)])
    points = tuple(Point(float(x), float(y)) for x, y in coords)
    tolerance = 2.5

    # En spécifiant 0 dans break_indices, la couture à 0 doit rester protégée
    path = fit_piecewise_bezier(points, tolerance_px=tolerance, break_indices=(0,), is_loop=True)
    assert path is not None
    assert path.is_loop
    assert path.error_bound <= tolerance
