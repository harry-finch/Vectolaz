"""Jonctions géométriques communes, sans contraction du graphe ni perte de cycles."""

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.optimize import least_squares

from raster2vector.bezier import BezierPath, CubicBezier, evaluate_cubic
from raster2vector.coordinates import Point
from raster2vector.geometry import (
    CircularArc, Circle, GeometryResult, LineFit, LineSegment, Polyline, RawPolyline,
    fit_line, _segment_distances,
)
from raster2vector.graph import SkeletonGraph

COLLINEAR_ANGLE_RAD = math.radians(5)
MIN_SUPPORT_SPAN_PX = 6.0
SUPPORT_CONDITION_RATIO = 0.05
LOCAL_WINDOW_PX = 12.0
Primitive = LineSegment | Polyline | RawPolyline | CircularArc | Circle | BezierPath


@dataclass(frozen=True)
class JoinedArc(CircularArc):
    """Arc contraint passant par les contacts communs, exportés exactement."""
    start_contact: Point
    end_contact: Point

    @property
    def start(self) -> Point:
        return self.start_contact

    @property
    def end(self) -> Point:
        return self.end_contact


@dataclass(frozen=True)
class JunctionAdjustment:
    node_id: int
    original_position: Point
    position: Point
    edge_ids: tuple[int, ...]


@dataclass(frozen=True)
class Continuation:
    node_id: int
    edge_ids: tuple[int, int]


@dataclass(frozen=True)
class JunctionResult:
    geometry: GeometryResult
    adjustments: tuple[JunctionAdjustment, ...]
    continuations: tuple[Continuation, ...]
    rejected_node_ids: tuple[int, ...]


def _array(point: Point) -> np.ndarray:
    return np.array((point.x, point.y), dtype=float)


def _point(value: np.ndarray) -> Point:
    return Point(float(value[0]), float(value[1]))


def _ends(primitive: Primitive) -> tuple[Point, Point] | None:
    if isinstance(primitive, (LineSegment, CircularArc)):
        return primitive.start, primitive.end
    if isinstance(primitive, BezierPath) and primitive.segments:
        return primitive.segments[0].start, primitive.segments[-1].end
    if isinstance(primitive, (Polyline, RawPolyline)) and len(primitive.points) >= 2:
        return primitive.points[0], primitive.points[-1]
    return None


def _support(primitive: Primitive, at_start: bool) -> tuple[np.ndarray, np.ndarray] | None:
    ends = _ends(primitive)
    if ends is None:
        return None
    contact = ends[0 if at_start else 1]
    if isinstance(primitive, LineSegment):
        vector = _array(primitive.end)-_array(primitive.start)
        if np.linalg.norm(vector) < MIN_SUPPORT_SPAN_PX:
            return None
    elif isinstance(primitive, CircularArc):
        radial = _array(contact)-_array(primitive.center)
        vector = np.array((-radial[1], radial[0])) * np.sign(primitive.sweep_angle_rad)
    elif isinstance(primitive, BezierPath):
        segment = primitive.segments[0 if at_start else -1]
        vector = (_array(segment.control1)-_array(segment.start) if at_start
                  else _array(segment.end)-_array(segment.control2))
    else:
        points = primitive.points if at_start else primitive.points[::-1]
        # Estimation locale, au plus 12 pixels de longueur, comme à l'étape 10.
        local = [points[0]]
        length = 0.0
        for point in points[1:]:
            length += float(np.linalg.norm(_array(point)-_array(local[-1])))
            if length > LOCAL_WINDOW_PX:
                break
            local.append(point)
        fit = fit_line(local)
        if fit is None or np.linalg.norm(_array(local[-1])-_array(local[0])) < MIN_SUPPORT_SPAN_PX:
            return None
        return _array(fit.origin), _array(fit.direction)
    if np.linalg.norm(vector) == 0:
        return None
    return _array(contact), vector/np.linalg.norm(vector)


def _refit_arc(arc: CircularArc, start: Point, end: Point, tolerance: float) -> JoinedArc | None:
    a, b = _array(start), _array(end)
    chord = b-a
    length = float(np.linalg.norm(chord))
    if length == 0:
        return None
    midpoint = (a+b)/2
    normal = np.array((-chord[1], chord[0]))/length
    data = np.array([(p.x, p.y) for p in arc.source_points])
    offset = float(np.dot(_array(arc.center)-midpoint, normal))

    def residual(h: np.ndarray) -> np.ndarray:
        center = midpoint+h[0]*normal
        radius = np.linalg.norm(a-center)
        return np.linalg.norm(data-center, axis=1)-radius

    fit = least_squares(residual, [offset])
    if not fit.success or not np.isfinite(fit.x).all():
        return None
    center = midpoint+fit.x[0]*normal
    radius = float(np.linalg.norm(a-center))
    first = math.atan2(a[1]-center[1], a[0]-center[0])
    last = math.atan2(b[1]-center[1], b[0]-center[0])
    sign = math.copysign(1, arc.sweep_angle_rad)
    sweep = sign*((sign*(last-first)) % (2*math.pi))
    if not 0 < abs(sweep) < 2*math.pi or abs(sweep-arc.sweep_angle_rad)*radius > 2*tolerance:
        return None
    angles = np.unwrap(np.arctan2(data[:, 1]-center[1], data[:, 0]-center[0]))
    # Choisir la même branche de l'angle que le début imposé.
    angles += round((first-angles[0])/(2*math.pi))*2*math.pi
    progress = sign*(angles-first)
    if np.max(np.maximum.accumulate(progress)-progress)*radius > tolerance:
        return None
    errors = np.abs(np.linalg.norm(data-center, axis=1)-radius)
    outside = (progress < 0) | (progress > abs(sweep))
    errors[outside] = np.minimum(np.linalg.norm(data[outside]-a, axis=1), np.linalg.norm(data[outside]-b, axis=1))
    if errors.max() > tolerance:
        return None
    return JoinedArc(_point(center), radius, first, first+sweep, float(errors.max()),
                     float(np.sqrt(np.mean(errors**2))), arc.source_edge_id, arc.source_points, start, end)


def _measure_cubic(curve: CubicBezier, source: tuple[Point, ...]) -> CubicBezier:
    data = np.array([(p.x, p.y) for p in source[curve.source_start_index:curve.source_end_index+1]])
    t = np.array(curve.source_parameters)[:, None]
    c = np.array([(p.x, p.y) for p in (curve.start, curve.control1, curve.control2, curve.end)])
    values = evaluate_cubic(curve, curve.source_parameters)
    derivative = 3*(1-t)**2*(c[1]-c[0])+6*(1-t)*t*(c[2]-c[1])+3*t**2*(c[3]-c[2])
    dt = np.diff(t, axis=0)
    differences = np.stack((values[:-1]-data[:-1],
                            values[:-1]+dt*derivative[:-1]/3-(2*data[:-1]+data[1:])/3,
                            values[1:]-dt*derivative[1:]/3-(data[:-1]+2*data[1:])/3,
                            values[1:]-data[1:]), axis=1)
    errors = np.linalg.norm(values-data, axis=1)
    return replace(curve, fit_error_max=float(errors.max()), fit_error_rms=float(np.sqrt(np.mean(errors**2))),
                   error_bound=float(np.linalg.norm(differences, axis=2).max()))


def _adjust(primitive: Primitive, start: Point, end: Point, tolerance: float) -> Primitive | None:
    if isinstance(primitive, CircularArc):
        return _refit_arc(primitive, start, end, tolerance)
    if isinstance(primitive, LineSegment):
        if start == end:
            return None
        data = np.array([(p.x, p.y) for p in primitive.source_points])
        errors = _segment_distances(data, _array(start), _array(end))
        if errors.max() > tolerance:
            return None
        return replace(primitive, start=start, end=end, fit_error_max=float(errors.max()),
                       fit_error_rms=float(np.sqrt(np.mean(errors**2))))
    if isinstance(primitive, Polyline):
        points = (start, *primitive.points[1:-1], end)
        data = np.array([(p.x, p.y) for p in primitive.source_points])
        errors = np.zeros(len(data))
        for index, (a, b) in enumerate(zip(primitive.source_indices, primitive.source_indices[1:])):
            errors[a:b+1] = _segment_distances(data[a:b+1], _array(points[index]), _array(points[index+1]))
        if errors.max() > max(tolerance, primitive.fit_error_max):
            return None
        return replace(primitive, points=points, fit_error_max=float(errors.max()),
                       fit_error_rms=float(np.sqrt(np.mean(errors**2))))
    if isinstance(primitive, RawPolyline):
        # Correspondance linéaire entre les deux chaînes : seules les deux
        # extrémités changent, donc cette borne vaut aussi entre les points.
        source = primitive.source_points or primitive.points
        errors = np.array((np.linalg.norm(_array(start)-_array(source[0])),
                           np.linalg.norm(_array(end)-_array(source[-1]))))
        if errors.max() > tolerance:
            return None
        return replace(primitive, points=(start, *primitive.points[1:-1], end),
                       fit_error_max=float(errors.max()),
                       fit_error_rms=float(np.sqrt(np.sum(errors**2)/len(primitive.points))),
                       source_points=primitive.source_points or primitive.points)
    if isinstance(primitive, BezierPath):
        segments = list(primitive.segments)
        first, last = segments[0], segments[-1]
        delta = _array(start)-_array(first.start)
        segments[0] = replace(first, start=start, control1=_point(_array(first.control1)+delta))
        last = segments[-1]
        delta = _array(end)-_array(last.end)
        segments[-1] = replace(last, end=end, control2=_point(_array(last.control2)+delta))
        segments = [_measure_cubic(c, primitive.source_points) for c in segments]
        bound = max(c.error_bound for c in segments)
        if bound > max(tolerance, primitive.error_bound):
            return None
        # Compter les points partagés une seule fois, même si l'arrondi n'est pas nul.
        errors = []
        for c in segments:
            data = np.array([(p.x, p.y) for p in primitive.source_points[c.source_start_index:c.source_end_index+1]])
            errors.extend(np.linalg.norm(evaluate_cubic(c, c.source_parameters)-data, axis=1)[:-1])
        errors.append(float(np.linalg.norm(_array(end)-_array(primitive.source_points[-1]))))
        return replace(primitive, segments=tuple(segments), error_bound=bound,
                       fit_error_max=max(errors), fit_error_rms=float(np.sqrt(np.mean(np.array(errors)**2))))
    return None


def refine_junctions(graph: SkeletonGraph, geometry: GeometryResult, tolerance_px: float = 2.0) -> JunctionResult:
    """Ajuste toutes les branches d'un nœud ensemble, ou conserve ce nœud entier.

    Aucune branche, composante ou boucle n'est supprimée. La tolérance borne
    le déplacement des contacts et les erreurs des primitives ajustées.
    Les branches trop courtes ne guident pas l'estimation d'une intersection.
    """
    if not math.isfinite(tolerance_px) or tolerance_px < 0:
        raise ValueError("La tolérance de jonction doit être finie et positive ou nulle.")
    if (geometry.width, geometry.height) != (graph.width, graph.height):
        raise ValueError("Le graphe et la géométrie doivent avoir les mêmes dimensions.")
    by_edge = {p.source_edge_id: p for p in geometry.primitives}
    if len(by_edge) != len(geometry.primitives) or set(by_edge) != {e.id for e in graph.edges}:
        raise ValueError("La géométrie doit contenir une primitive par branche du graphe.")
    if tolerance_px == 0:
        return JunctionResult(geometry, (), (), ())
    incidence: dict[int, list[tuple[int, bool]]] = {n.id: [] for n in graph.nodes if n.kind == "junction"}
    for edge in graph.edges:
        for node, at_start in ((edge.start_node_id, True), (edge.end_node_id, False)):
            if node in incidence:
                incidence[node].append((edge.id, at_start))
    anchors: dict[int, Point] = {}
    pairs: list[Continuation] = []
    common: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    rejected: set[int] = set()
    for node_id, contacts in incidence.items():
        node = graph.nodes[node_id]
        sources = [graph.edges[index].points[0 if at_start else -1] for index, at_start in contacts]
        supports = [(index, _support(by_edge[index], at_start)) for index, at_start in contacts]
        useful = [(i, s) for i, s in supports if s is not None]
        node_pairs: list[tuple[int, int, LineFit]] = []
        for offset, (a, sa) in enumerate(useful):
            if not isinstance(by_edge[a], LineSegment):
                continue
            for b, sb in useful[offset+1:]:
                if a == b or not isinstance(by_edge[b], LineSegment):
                    continue
                edge_a, edge_b = graph.edges[a], graph.edges[b]
                va = _array(edge_a.points[-1 if edge_a.start_node_id == node_id else 0])-_array(node.position)
                vb = _array(edge_b.points[-1 if edge_b.start_node_id == node_id else 0])-_array(node.position)
                if np.dot(va, vb) >= 0 or abs(np.dot(sa[1], sb[1])) < math.cos(COLLINEAR_ANGLE_RAD):
                    continue
                fit = fit_line((*by_edge[a].source_points, *by_edge[b].source_points))
                if fit is not None and fit.fit_error_max <= tolerance_px:
                    node_pairs.append((a, b, fit))
        # Une continuation n'est imposée que si chaque branche a un seul partenaire.
        counts = {i: sum(i in (a, b) for a, b, _ in node_pairs) for i, _ in useful}
        local_common = [(a, b, fit) for a, b, fit in node_pairs if counts[a] == counts[b] == 1]
        constraints = [(_array(fit.origin), _array(fit.direction)) for _, _, fit in local_common]
        if len(constraints) >= 2:
            normals = np.array([(-v[1], v[0]) for _, v in constraints])
            rhs = np.array([np.dot(n, origin) for n, (origin, _) in zip(normals, constraints)])
            value, _, rank, _ = np.linalg.lstsq(normals, rhs, rcond=None)
            if rank < 2 or np.max(np.abs(normals@value-rhs)) > tolerance_px:
                rejected.add(node_id)
                continue
        else:
            normals = np.array([(-s[1][1], s[1][0]) for _, s in useful])
            rhs = np.array([np.dot(n, s[0]) for n, (_, s) in zip(normals, useful)])
            if len(useful) >= 2:
                value, _, rank, singular = np.linalg.lstsq(normals, rhs, rcond=None)
                if rank < 2 or singular[-1] < SUPPORT_CONDITION_RATIO*singular[0]:
                    value = _array(node.position)
            else:
                value = _array(node.position)
            if constraints:
                origin, direction = constraints[0]
                value = origin+np.dot(value-origin, direction)*direction
        if not np.isfinite(value).all() or any(np.linalg.norm(value-_array(p)) > tolerance_px for p in sources):
            rejected.add(node_id)
            continue
        anchors[node_id] = _point(value)
        for a, b, fit in local_common:
            pairs.append(Continuation(node_id, (a, b)))
            common[a] = common[b] = (_array(fit.origin), _array(fit.direction))
    # Validation atomique : si une branche échoue, retirer les jonctions
    # concernées puis recalculer toutes les autres avec ces contacts d'origine.
    fallback_tried: set[int] = set()
    while True:
        updated: dict[int, Primitive] = {}
        failed: set[int] = set()
        for edge in graph.edges:
            primitive = by_edge[edge.id]
            ends = _ends(primitive)
            selected = {i for i in (edge.start_node_id, edge.end_node_id) if i in anchors}
            if not selected:
                updated[edge.id] = primitive
                continue
            if ends is None:
                failed.update(selected)
                continue
            start = anchors.get(edge.start_node_id, ends[0])
            end = anchors.get(edge.end_node_id, ends[1])
            if edge.id in common:
                origin, direction = common[edge.id]
                if graph.nodes[edge.start_node_id].kind != "junction":
                    start = _point(origin+np.dot(_array(start)-origin, direction)*direction)
                if graph.nodes[edge.end_node_id].kind != "junction":
                    end = _point(origin+np.dot(_array(end)-origin, direction)*direction)
            result = _adjust(primitive, start, end, tolerance_px)
            if result is None:
                failed.update(selected)
            else:
                updated[edge.id] = result
        if not failed:
            break
        for node in failed:
            centroid = graph.nodes[node].position
            contacts = [graph.edges[index].points[0 if at_start else -1] for index, at_start in incidence[node]]
            if (node not in fallback_tried and anchors[node] != centroid
                    and all(np.linalg.norm(_array(centroid)-_array(p)) <= tolerance_px for p in contacts)):
                anchors[node] = centroid
                fallback_tried.add(node)
            else:
                anchors.pop(node)
                rejected.add(node)
        common = {edge_id: support for edge_id, support in common.items()
                  if any(pair.node_id in anchors and edge_id in pair.edge_ids for pair in pairs)}
    adjustments = tuple(JunctionAdjustment(i, graph.nodes[i].position, p,
                                           tuple(sorted({index for index, _ in incidence[i]}))) for i, p in sorted(anchors.items()))
    final_pairs = []
    for pair in pairs:
        if pair.node_id not in anchors:
            continue
        a, b = (updated[index] for index in pair.edge_ids)
        va, vb = _array(a.end)-_array(a.start), _array(b.end)-_array(b.start)
        norm = np.linalg.norm(va)*np.linalg.norm(vb)
        if norm > 0 and abs(np.dot(va, vb))/norm >= math.cos(COLLINEAR_ANGLE_RAD):
            final_pairs.append(pair)
    result = replace(geometry, primitives=tuple(updated[p.source_edge_id] for p in geometry.primitives))
    return JunctionResult(result, adjustments, tuple(final_pairs), tuple(sorted(rejected)))
