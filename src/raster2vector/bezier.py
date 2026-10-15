"""Bézier cubiques ouvertes ou fermées, avec borne continue de Bernstein."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
import math

import numpy as np
from numpy.typing import NDArray

from raster2vector.coordinates import Point


@dataclass(frozen=True)
class CubicBezier:
    """Cubique et sous-chaîne source ; erreurs aux points et borne continue en pixels."""
    start: Point
    control1: Point
    control2: Point
    end: Point
    source_start_index: int
    source_end_index: int
    source_parameters: tuple[float, ...]
    fit_error_max: float
    fit_error_rms: float
    error_bound: float


@dataclass(frozen=True)
class BezierPath:
    """Chemin à contacts partagés, ouvert ou fermé, conservant ses sources."""
    segments: tuple[CubicBezier, ...]
    source_edge_id: int
    source_points: tuple[Point, ...]
    fit_error_max: float
    fit_error_rms: float
    error_bound: float
    is_loop: bool = False


def evaluate_cubic(curve: CubicBezier, parameters: Sequence[float]) -> NDArray[np.float64]:
    """Évalue les coordonnées (x, y) pour des paramètres dans [0, 1]."""
    t = np.asarray(parameters, dtype=np.float64).reshape(-1, 1)
    if not np.isfinite(t).all() or np.any((t < 0) | (t > 1)):
        raise ValueError("Les paramètres Bézier doivent être finis dans [0, 1].")
    controls = np.array([(p.x, p.y) for p in (curve.start, curve.control1, curve.control2, curve.end)])
    return (1-t)**3 * controls[0] + 3*(1-t)**2*t * controls[1] + 3*(1-t)*t**2 * controls[2] + t**3 * controls[3]


def _candidate(coordinates: NDArray[np.float64], start: int, end: int) -> tuple[CubicBezier, int]:
    data = coordinates[start:end+1]
    lengths = np.linalg.norm(np.diff(data, axis=0), axis=1)
    total = float(lengths.sum())
    t = np.concatenate(([0.0], np.cumsum(lengths))) / total if total else np.linspace(0, 1, len(data))
    # Raffinement déterministe du paramétrage par Newton sur la distance.
    # Chaque mise à jour garde l'ordre des points ; les extrémités restent fixes.
    parameter_refinement_iterations = 12
    for iteration in range(parameter_refinement_iterations+1):
        u = t[:, None]
        design = np.column_stack((3*(1-t)**2*t, 3*(1-t)*t**2))
        residual = data - (1-u)**3*data[0] - u**3*data[-1]
        controls, _, rank, _ = np.linalg.lstsq(design, residual, rcond=None)
        if rank < 2:
            controls = np.array([(2*data[0]+data[-1])/3, (data[0]+2*data[-1])/3])
        c = np.vstack((data[0], controls, data[-1]))
        if iteration == parameter_refinement_iterations or rank < 2:
            break
        values = (1-u)**3*c[0] + 3*(1-u)**2*u*c[1] + 3*(1-u)*u**2*c[2] + u**3*c[3]
        first = 3*(1-u)**2*(c[1]-c[0]) + 6*(1-u)*u*(c[2]-c[1]) + 3*u**2*(c[3]-c[2])
        second = 6*(1-u)*(c[2]-2*c[1]+c[0]) + 6*u*(c[3]-2*c[2]+c[1])
        numerator = np.sum((values-data)*first, axis=1)
        denominator = np.sum(first**2+(values-data)*second, axis=1)
        step = np.divide(numerator, denominator, out=np.zeros_like(t), where=denominator > 0)
        refined = np.clip(t-step, 0, 1)
        refined[0], refined[-1] = 0, 1
        if np.all(np.diff(refined) >= 0):
            t = refined
        else:
            break

    def value(v: NDArray[np.float64]) -> NDArray[np.float64]:
        return (1-v)**3*c[0] + 3*(1-v)**2*v*c[1] + 3*(1-v)*v**2*c[2] + v**3*c[3]

    values = value(u)
    derivative = 3*(1-u)**2*(c[1]-c[0]) + 6*(1-u)*u*(c[2]-c[1]) + 3*u**2*(c[3]-c[2])
    dt = np.diff(t)[:, None]
    # Contrôles de chaque restriction cubique, comparés à ceux du segment
    # raster correspondant. L'enveloppe convexe borne toute la différence,
    # y compris ENTRE les points sources, dans les deux sens.
    differences = np.stack((values[:-1]-data[:-1],
                            values[:-1]+dt*derivative[:-1]/3-(2*data[:-1]+data[1:])/3,
                            values[1:]-dt*derivative[1:]/3-(data[:-1]+2*data[1:])/3,
                            values[1:]-data[1:]), axis=1)
    bounds = np.linalg.norm(differences, axis=2).max(axis=1)
    errors = np.linalg.norm(values-data, axis=1)
    split = start + int(np.argmax(bounds)) + 1
    split = max(start+1, min(end-1, split))
    points = tuple(Point(float(x), float(y)) for x, y in c)
    curve = CubicBezier(*points, start, end, tuple(float(v) for v in t),
                        float(errors.max()), float(np.sqrt(np.mean(errors**2))), float(bounds.max()))
    return curve, split


def _evaluate_curve(curve: CubicBezier, coordinates: NDArray[np.float64]) -> CubicBezier:
    """Ré-évalue les erreurs et la borne continue de Bernstein d'une cubique."""
    data = coordinates[curve.source_start_index:curve.source_end_index + 1]
    t = np.asarray(curve.source_parameters, dtype=np.float64)
    u = t[:, None]
    c = np.array([(p.x, p.y) for p in (curve.start, curve.control1, curve.control2, curve.end)], dtype=np.float64)
    values = (1 - u)**3 * c[0] + 3 * (1 - u)**2 * u * c[1] + 3 * (1 - u) * u**2 * c[2] + u**3 * c[3]
    derivative = 3 * (1 - u)**2 * (c[1] - c[0]) + 6 * (1 - u) * u * (c[2] - c[1]) + 3 * u**2 * (c[3] - c[2])
    dt = np.diff(t)[:, None]
    differences = np.stack((values[:-1] - data[:-1],
                            values[:-1] + dt * derivative[:-1] / 3 - (2 * data[:-1] + data[1:]) / 3,
                            values[1:] - dt * derivative[1:] / 3 - (data[:-1] + 2 * data[1:]) / 3,
                            values[1:] - data[1:]), axis=1)
    bounds = np.linalg.norm(differences, axis=2).max(axis=1)
    errors = np.linalg.norm(values - data, axis=1)
    return replace(curve, fit_error_max=float(errors.max()),
                   fit_error_rms=float(np.sqrt(np.mean(errors**2))),
                   error_bound=float(bounds.max()))


def _try_smooth_joint(c_prev: CubicBezier, c_next: CubicBezier,
                      coordinates: NDArray[np.float64],
                      tolerance_px: float) -> tuple[CubicBezier, CubicBezier] | None:
    """Tente un alignement G1 des poignées à une jonction sans dépasser la tolérance."""
    t_in = np.array([c_prev.end.x - c_prev.control2.x, c_prev.end.y - c_prev.control2.y], dtype=np.float64)
    L_in = float(np.linalg.norm(t_in))
    t_out = np.array([c_next.control1.x - c_next.start.x, c_next.control1.y - c_next.start.y], dtype=np.float64)
    L_out = float(np.linalg.norm(t_out))
    if L_in <= 1e-9 or L_out <= 1e-9:
        return None
    u_in = t_in / L_in
    u_out = t_out / L_out
    dot_val = float(np.dot(u_in, u_out))
    if dot_val <= 0.5:
        return None
    u_sum = u_in + u_out
    norm_sum = float(np.linalg.norm(u_sum))
    if norm_sum <= 1e-9:
        return None
    u_tangent = u_sum / norm_sum
    joint_prev = np.array([c_prev.end.x, c_prev.end.y], dtype=np.float64)
    joint_next = np.array([c_next.start.x, c_next.start.y], dtype=np.float64)
    p2 = joint_prev - L_in * u_tangent
    p1 = joint_next + L_out * u_tangent
    new_prev = _evaluate_curve(replace(c_prev, control2=Point(float(p2[0]), float(p2[1]))), coordinates)
    new_next = _evaluate_curve(replace(c_next, control1=Point(float(p1[0]), float(p1[1]))), coordinates)
    if new_prev.error_bound <= tolerance_px and new_next.error_bound <= tolerance_px:
        return new_prev, new_next
    return None


def _smooth_internal_joints(curves: list[CubicBezier], coordinates: NDArray[np.float64],
                            tolerance_px: float,
                            corner_indices: set[int] | frozenset[int] = frozenset(),
                            is_loop: bool = False) -> list[CubicBezier]:
    """Lisse les cassures de tangente aux raccords internes sous réserve de respecter la borne."""
    if len(curves) < 2:
        return curves
    smoothed = list(curves)
    for i in range(len(smoothed) - 1):
        c_prev = smoothed[i]
        c_next = smoothed[i + 1]
        joint_idx = c_prev.source_end_index
        if joint_idx in corner_indices:
            continue
        pair = _try_smooth_joint(c_prev, c_next, coordinates, tolerance_px)
        if pair is not None:
            smoothed[i], smoothed[i + 1] = pair
    if is_loop and len(smoothed) >= 2:
        if 0 not in corner_indices and (len(coordinates) - 1) not in corner_indices:
            c_prev = smoothed[-1]
            c_next = smoothed[0]
            pair = _try_smooth_joint(c_prev, c_next, coordinates, tolerance_px)
            if pair is not None:
                smoothed[-1], smoothed[0] = pair
    return smoothed


def _path_rms(curves: Sequence[CubicBezier], coordinates: NDArray[np.float64]) -> float:
    """Calcule l'erreur quadratique moyenne exacte sur l'ensemble des points sources.

    Pour chaque courbe d'indice i > 0, son premier point est exclu du cumul
    des erreurs au carré car son erreur est déjà comptée à la fin de la courbe i - 1.
    Le total est ensuite divisé par len(coordinates).
    """
    if not curves or len(coordinates) == 0:
        return 0.0
    total_squared_error = 0.0
    for i, curve in enumerate(curves):
        evals = evaluate_cubic(curve, curve.source_parameters)
        target = coordinates[curve.source_start_index:curve.source_end_index + 1]
        errors_sq = np.sum((evals - target) ** 2, axis=1)
        if i > 0:
            errors_sq = errors_sq[1:]
        total_squared_error += float(np.sum(errors_sq))
    return float(math.sqrt(total_squared_error / len(coordinates)))


def fit_bezier(points: Sequence[Point], tolerance_px: float = 2.0, *, source_edge_id: int = 0) -> BezierPath | None:
    """Ajuste des cubiques par moindres carrés, puis découpe si la borne dépasse la tolérance.

    Extrémités imposées ; paramétrage initial par longueur de corde, raffiné
    par 12 itérations de Newton au plus, en conservant l'ordre. Les morceaux
    adjacents sont fusionnés quand une nouvelle cubique respecte la borne.
    Cette recherche gloutonne ne garantit pas le minimum global. Continuité
    C0 : les extrémités sont partagées, les tangentes ne sont pas imposées.
    Les boucles sont laissées aux autres primitives du moteur.
    """
    if not math.isfinite(tolerance_px) or tolerance_px < 0:
        raise ValueError("La tolérance Bézier doit être finie et positive ou nulle.")
    source = tuple(points)
    coordinates = np.array([(p.x, p.y) for p in source], dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(coordinates).all():
        raise ValueError("Les coordonnées doivent être finies.")
    if len(source) < 2 or source[0] == source[-1]:
        return None
    pending = [(0, len(source)-1)]
    curves: list[CubicBezier] = []
    while pending:
        start, end = pending.pop()
        curve, split = _candidate(coordinates, start, end)
        if curve.error_bound <= tolerance_px:
            curves.append(curve)
        elif end-start == 1:
            # À tolérance nulle, une simple ligne peut avoir des arrondis
            # flottants non nuls : conserver alors les points bruts.
            return None
        else:
            pending.extend(((split, end), (start, split)))
    index = 0
    while index < len(curves)-1:
        merged, _ = _candidate(coordinates, curves[index].source_start_index, curves[index+1].source_end_index)
        if merged.error_bound <= tolerance_px:
            curves[index:index+2] = [merged]
            index = max(0, index-1)
        else:
            index += 1
    curves = _smooth_internal_joints(curves, coordinates, tolerance_px)
    rms = _path_rms(curves, coordinates)
    return BezierPath(tuple(curves), source_edge_id, source, max(c.fit_error_max for c in curves),
                      rms, max(c.error_bound for c in curves))


def fit_piecewise_bezier(points: Sequence[Point], tolerance_px: float,
                         break_indices: Sequence[int], *, source_edge_id: int = 0,
                         is_loop: bool = False) -> BezierPath | None:
    """Ajuste entre contacts imposés, sans arrondir les angles qui les séparent.

    Pour une fermeture explicite, des coupures supplémentaires empêchent une
    cubique à extrémités confondues. Chaque indice reste relatif à la source
    entière pour le contrôle continu et l'affinage ultérieur des jonctions.
    """
    if not math.isfinite(tolerance_px) or tolerance_px < 0:
        raise ValueError("La tolérance Bézier doit être finie et positive ou nulle.")
    source = tuple(points)
    if len(source) < 2:
        return None
    if any(index < 0 or index >= len(source) for index in break_indices):
        raise ValueError("Les indices de découpage Bézier doivent appartenir à la source.")
    cuts = sorted({0, len(source)-1, *break_indices})
    if source[0] == source[-1]:
        # Même sans angle, quatre portions donnent une fermeture exploitable si la chaîne est assez longue.
        if len(source) >= 5:
            cuts = sorted(set(cuts) | {i*(len(source)-1)//4 for i in range(1, 4)})
    filtered_cuts: list[int] = [cuts[0]]
    for c in cuts[1:]:
        if c - filtered_cuts[-1] >= 1 and source[filtered_cuts[-1]] != source[c]:
            filtered_cuts.append(c)
        elif c == cuts[-1]:
            while len(filtered_cuts) > 1 and source[filtered_cuts[-1]] == source[c]:
                filtered_cuts.pop()
            if source[filtered_cuts[-1]] != source[c]:
                filtered_cuts.append(c)
    valid_intervals = list(zip(filtered_cuts, filtered_cuts[1:]))
    if source[0] == source[-1] and len(valid_intervals) < 2:
        return None
    if not valid_intervals or filtered_cuts[-1] != len(source) - 1:
        return None
    curves = []
    for start, end in valid_intervals:
        result = fit_bezier(source[start:end+1], tolerance_px, source_edge_id=source_edge_id)
        if result is None:
            return None
        curves.extend(replace(curve, source_start_index=curve.source_start_index+start,
                              source_end_index=curve.source_end_index+start) for curve in result.segments)
    if not curves:
        return None
    coordinates = np.array([(p.x, p.y) for p in source], dtype=np.float64).reshape(-1, 2)
    curves = _smooth_internal_joints(curves, coordinates, tolerance_px,
                                     corner_indices=set(break_indices),
                                     is_loop=is_loop)
    rms = _path_rms(curves, coordinates)
    return BezierPath(tuple(curves), source_edge_id, source,
                      max(c.fit_error_max for c in curves), rms,
                      max(c.error_bound for c in curves), is_loop)
