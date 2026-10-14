"""Cercles complets répartis sur plusieurs branches du graphe."""

from dataclasses import dataclass, replace
import math
import numpy as np

from numpy.typing import NDArray

from raster2vector.coordinates import Point
from raster2vector.geometry import Circle, CircleFit, GeometryResult, fit_circle, _makes_full_turn, primitive_source_edge_ids
from raster2vector.graph import GraphEdge, SkeletonGraph

MIN_SEED_SWEEP_RAD = math.pi/6
SUPPORT_SEARCH_FACTOR = 4.0


@dataclass(frozen=True)
class GlobalCircle(Circle):
    """Un cercle, plusieurs branches sources ; le graphe original reste intact."""
    source_edge_ids: tuple[int, ...]


@dataclass(frozen=True)
class GlobalCircleResult:
    geometry: GeometryResult
    circles: tuple[GlobalCircle, ...]


def _radial_bound_array(coords: NDArray[np.float64], fit: CircleFit) -> float:
    """Borne exacte de l'écart radial sur des coordonnées (N, 2)."""
    if len(coords) == 0:
        return 0.0
    center = np.array((fit.center.x, fit.center.y), dtype=np.float64)
    data = coords - center
    radii = np.linalg.norm(data, axis=1)
    bound = float(np.max(np.abs(radii - fit.radius)))
    if len(data) > 1:
        delta = np.diff(data, axis=0)
        length2 = np.sum(delta ** 2, axis=1)
        t = np.clip(
            np.divide(-np.sum(data[:-1] * delta, axis=1), length2,
                      out=np.zeros_like(length2), where=length2 > 0),
            0.0,
            1.0,
        )
        closest = np.linalg.norm(data[:-1] + t[:, None] * delta, axis=1)
        bound = max(bound, float(np.max(np.abs(closest - fit.radius))))
    return bound


def _radial_bound(points: tuple[Point, ...] | NDArray[np.float64], fit: CircleFit) -> float:
    """Borne exacte de l'écart radial sur tous les segments de la chaîne source."""
    if isinstance(points, np.ndarray):
        return _radial_bound_array(points, fit)
    coords = np.array([(p.x, p.y) for p in points], dtype=np.float64).reshape(-1, 2)
    return _radial_bound_array(coords, fit)


def _cycles(graph: SkeletonGraph, selected: set[int]) -> list[tuple[tuple[int, ...], tuple[Point, ...]]]:
    adjacency: dict[int, set[int]] = {}
    for index in selected:
        edge = graph.edges[index]
        for node in (edge.start_node_id, edge.end_node_id):
            adjacency.setdefault(node, set()).add(index)
    # Éliminer les branches pendantes, pas les sources du graphe lui-même.
    pending = [node for node, edges in adjacency.items() if len(edges) < 2]
    while pending:
        node = pending.pop()
        for index in tuple(adjacency[node]):
            edge = graph.edges[index]
            for end in (edge.start_node_id, edge.end_node_id):
                adjacency[end].discard(index)
                if len(adjacency[end]) == 1:
                    pending.append(end)
    remaining = {node for node, edges in adjacency.items() if edges}
    cycles = []
    while remaining:
        component: set[int] = set()
        pending = [min(remaining)]
        while pending:
            node = pending.pop()
            if node in component:
                continue
            component.add(node)
            for index in adjacency[node]:
                edge = graph.edges[index]
                pending.append(edge.end_node_id if edge.start_node_id == node else edge.start_node_id)
        remaining -= component
        # Une ramification interne au candidat rend le choix de parcours ambigu.
        if any(len(adjacency[node]) != 2 for node in component):
            continue
        start = min(component)
        node, index = start, min(adjacency[start])
        used: list[int] = []
        points: list[Point] = []
        while True:
            edge = graph.edges[index]
            oriented = edge.points if edge.start_node_id == node else edge.points[::-1]
            points.extend(oriented[1:] if points and points[-1] == oriented[0] else oriented)
            used.append(index)
            node = edge.end_node_id if edge.start_node_id == node else edge.start_node_id
            if node == start:
                break
            index = next(i for i in adjacency[node] if i != index)
        if points[-1] != points[0]:
            points.append(points[0])
        cycles.append((tuple(used), tuple(points)))
    return cycles


def reconstruct_global_circles(graph: SkeletonGraph, geometry: GeometryResult,
                               tolerance_px: float = 3.0) -> GlobalCircleResult:
    """Regroupe les supports proches, puis exige un cycle réel et un tour complet.

    Les arcs incomplets ou séparés ne deviennent jamais des cercles. Les
    petits connecteurs ne sont inclus que s'ils sont déjà présents dans le
    graphe et radialement compatibles. Recherche déterministe et conservatrice.
    """
    if not math.isfinite(tolerance_px) or tolerance_px < 0:
        raise ValueError("La tolérance globale doit être finie et positive ou nulle.")
    if (graph.width, graph.height) != (geometry.width, geometry.height):
        raise ValueError("Le graphe et la géométrie doivent avoir les mêmes dimensions.")
    ids = {edge_id for p in geometry.primitives for edge_id in primitive_source_edge_ids(p)}
    if ids != {e.id for e in graph.edges} or len(ids) != len(geometry.primitives):
        raise ValueError("La géométrie locale doit contenir une primitive par branche.")
    if tolerance_px == 0:
        return GlobalCircleResult(geometry, ())

    edge_coords: dict[int, NDArray[np.float64]] = {
        edge.id: np.array([(p.x, p.y) for p in edge.points], dtype=np.float64).reshape(-1, 2)
        for edge in graph.edges
    }
    edge_lengths: dict[int, float] = {
        edge.id: float(np.sum(np.hypot(np.diff(edge_coords[edge.id][:, 0]), np.diff(edge_coords[edge.id][:, 1]))))
        if len(edge.points) > 1 else 0.0
        for edge in graph.edges
    }

    seeds: list[tuple[GraphEdge, CircleFit]] = []
    for edge in graph.edges:
        if edge.is_loop or len(edge.points) < 4:
            continue
        fit = fit_circle(edge.points)
        if fit is None or _radial_bound_array(edge_coords[edge.id], fit) > tolerance_px:
            continue
        coords = edge_coords[edge.id]
        angles = np.unwrap(np.arctan2(coords[:, 1] - fit.center.y, coords[:, 0] - fit.center.x))
        if abs(float(angles[-1]-angles[0])) >= MIN_SEED_SWEEP_RAD:
            seeds.append((edge, fit))
    proposals: dict[tuple[int, ...], GlobalCircle] = {}
    for seed, initial in seeds:
        group = [seed]
        fit = initial
        for edge, support in seeds:
            if edge.id == seed.id:
                continue
            discrepancy = math.hypot(support.center.x-fit.center.x, support.center.y-fit.center.y)+abs(support.radius-fit.radius)
            if discrepancy > SUPPORT_SEARCH_FACTOR*tolerance_px:
                continue
            candidate = fit_circle(tuple(p for e in (*group, edge) for p in e.points))
            if candidate is not None and all(_radial_bound_array(edge_coords[e.id], candidate) <= tolerance_px for e in (*group, edge)):
                group.append(edge)
                fit = candidate
        if len(group) < 2:
            continue
        selected = {e.id for e in group}
        # Les connecteurs courts constituent des données, pas des trous à inventer.
        for edge in graph.edges:
            length = edge_lengths[edge.id]
            if not edge.is_loop and length <= SUPPORT_SEARCH_FACTOR*tolerance_px and _radial_bound_array(edge_coords[edge.id], fit) <= tolerance_px:
                selected.add(edge.id)
        for edges, points in _cycles(graph, selected):
            if len(set(edges) & {e.id for e in group}) < 2:
                continue
            candidate = fit_circle(points)
            if candidate is None or not _makes_full_turn(points, candidate.center):
                continue
            bound = _radial_bound(points, candidate)
            nodes = {node for index in edges for node in (graph.edges[index].start_node_id, graph.edges[index].end_node_id)}
            # Les contacts distincts d'une même jonction ne justifient un
            # raccord circulaire que si toute sa zone raster est compatible.
            zone_errors = [abs(math.hypot(col-candidate.center.x, row-candidate.center.y)-candidate.radius)
                           for node in nodes for row, col in graph.nodes[node].source_pixels]
            bound = max(bound, max(zone_errors, default=0.0))
            if bound > tolerance_px:
                continue
            angles = np.unwrap(np.arctan2([p.y-candidate.center.y for p in points], [p.x-candidate.center.x for p in points]))
            progress = np.sign(angles[-1]-angles[0])*(angles-angles[0])
            if float(np.max(np.maximum.accumulate(progress)-progress))*candidate.radius > tolerance_px:
                continue
            key = tuple(sorted(edges))
            proposals[key] = GlobalCircle(candidate.center, candidate.radius, bound, candidate.fit_error_rms,
                                           min(edges), points, key)
    accepted: list[GlobalCircle] = []
    used: set[int] = set()
    for circle in sorted(proposals.values(), key=lambda c: (-len(c.source_points), c.fit_error_max, c.source_edge_ids)):
        if not used & set(circle.source_edge_ids):
            accepted.append(circle)
            used.update(circle.source_edge_ids)
    primitives = tuple(p for p in geometry.primitives if not (set(primitive_source_edge_ids(p)) & used)) + tuple(accepted)
    return GlobalCircleResult(replace(geometry, primitives=primitives), tuple(accepted))
