"""Arcs circulaires continus à travers plusieurs branches et intersections."""

from dataclasses import dataclass, replace
import math
import heapq
import numpy as np

from numpy.typing import NDArray

from raster2vector.coordinates import Point
from raster2vector.geometry import CircleFit, CircularArc, GeometryResult, fit_circle, primitive_source_edge_ids
from raster2vector.global_circles import _radial_bound, MIN_SEED_SWEEP_RAD
from raster2vector.graph import SkeletonGraph

ARC_SUPPORT_SEARCH_FACTOR = 8.0
MAX_GLOBAL_SWEEP_RAD = 2*math.pi-MIN_SEED_SWEEP_RAD
CONNECTOR_SEARCH_FACTOR = 2.0


@dataclass(frozen=True)
class GlobalArc(CircularArc):
    """Un arc long, avec provenance de toutes les branches qu'il remplace.

    source_points est un ensemble de sources, pas un chemin ordonné global.
    L'ordre de chaque branche reste disponible dans le graphe inchangé.
    """
    source_edge_ids: tuple[int, ...]
    support_edge_ids: tuple[int, ...]


@dataclass(frozen=True)
class GlobalArcResult:
    geometry: GeometryResult
    arcs: tuple[GlobalArc, ...]


def _finite_arc_bound(points: tuple[Point, ...] | NDArray[np.float64], arc: GlobalArc) -> float:
    """Distance maximale à l'arc fini sur chaque segment source, sans échantillonnage."""
    center = np.array((arc.center.x, arc.center.y), dtype=np.float64)
    if isinstance(points, np.ndarray):
        raw = points
    else:
        raw = np.array([(p.x, p.y) for p in points], dtype=np.float64).reshape(-1, 2)
    if len(raw) == 0:
        return 0.0
    data = raw - center
    start = np.array((arc.start.x, arc.start.y), dtype=np.float64) - center
    end = np.array((arc.end.x, arc.end.y), dtype=np.float64) - center

    if len(data) < 2:
        vectors = data
    else:
        a = data[:-1]
        b = data[1:]
        delta = b - a
        length2 = np.sum(delta ** 2, axis=1)

        candidate_chunks = [data]

        # 1. Projections radiales sur chaque segment
        valid_len = length2 > 0
        if np.any(valid_len):
            dot_a_delta = np.sum(a * delta, axis=1)
            t_rad = np.clip(
                np.divide(-dot_a_delta, length2, out=np.zeros_like(length2), where=valid_len),
                0.0,
                1.0,
            )
            candidate_chunks.append(a[valid_len] + t_rad[valid_len, None] * delta[valid_len])

        # 2. Hors du balayage, le plus proche des deux contacts change sur leur médiatrice
        direction = end - start
        denominator = 2.0 * (delta[:, 0] * direction[0] + delta[:, 1] * direction[1])
        where_med = denominator != 0
        if np.any(where_med):
            numerator = float(np.dot(end, end) - np.dot(start, start)) - 2.0 * (a[:, 0] * direction[0] + a[:, 1] * direction[1])
            t_med = np.divide(numerator, denominator, out=np.full_like(numerator, -1.0), where=where_med)
            valid_med = where_med & (t_med >= 0.0) & (t_med <= 1.0)
            if np.any(valid_med):
                candidate_chunks.append(a[valid_med] + t_med[valid_med, None] * delta[valid_med])

        # 3. Intersections avec les deux demi-droites d'angle start_angle_rad et end_angle_rad
        for angle in (arc.start_angle_rad, arc.end_angle_rad):
            ray = np.array((math.cos(angle), math.sin(angle)), dtype=np.float64)
            cross_delta = ray[0] * delta[:, 1] - ray[1] * delta[:, 0]
            where_ray = cross_delta != 0
            if np.any(where_ray):
                cross_a = ray[0] * a[:, 1] - ray[1] * a[:, 0]
                t_ray = np.divide(-cross_a, cross_delta, out=np.full_like(cross_a, -1.0), where=where_ray)
                valid_t = where_ray & (t_ray >= 0.0) & (t_ray <= 1.0)
                if np.any(valid_t):
                    cand_ray = a[valid_t] + t_ray[valid_t, None] * delta[valid_t]
                    dot_ray = cand_ray[:, 0] * ray[0] + cand_ray[:, 1] * ray[1]
                    valid_half = dot_ray >= 0.0
                    if np.any(valid_half):
                        candidate_chunks.append(cand_ray[valid_half])

        vectors = np.vstack(candidate_chunks)

    progress = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]) - arc.start_angle_rad, 2 * math.pi)
    errors = np.abs(np.linalg.norm(vectors, axis=1) - arc.radius)
    outside = progress > arc.sweep_angle_rad
    if np.any(outside):
        errors[outside] = np.minimum(
            np.linalg.norm(vectors[outside] - start, axis=1),
            np.linalg.norm(vectors[outside] - end, axis=1),
        )
    return float(errors.max())


def _components(graph: SkeletonGraph, selected: set[int]) -> list[tuple[int, ...]]:
    adjacency: dict[int, set[int]] = {}
    for index in selected:
        edge = graph.edges[index]
        for node in (edge.start_node_id, edge.end_node_id):
            adjacency.setdefault(node, set()).add(index)
    remaining = set(selected)
    result = []
    while remaining:
        pending = [min(remaining)]
        component: set[int] = set()
        while pending:
            index = pending.pop()
            if index in component:
                continue
            component.add(index)
            edge = graph.edges[index]
            pending.extend(adjacency[edge.start_node_id] | adjacency[edge.end_node_id])
        remaining -= component
        result.append(tuple(sorted(component)))
    return result


def _connecting_sources(graph: SkeletonGraph, group: list[int], available: set[int],
                        fit: CircleFit, tolerance: float,
                        edge_coords: dict[int, NDArray[np.float64]] | None = None,
                        edge_lengths: dict[int, float] | None = None) -> set[int] | None:
    """Propose les chemins nécessaires, puis laisse l'ajustement commun les valider."""
    adjacency: dict[int, list[tuple[int, int, float]]] = {}
    for index in sorted(available):
        edge = graph.edges[index]
        coords = edge_coords[index] if edge_coords is not None else edge.points
        bound = _radial_bound(coords, fit)
        if bound > CONNECTOR_SEARCH_FACTOR * tolerance:
            continue
        if edge_lengths is not None:
            length = edge_lengths[index]
        else:
            length = sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(edge.points, edge.points[1:]))
        cost = max(length, 1e-9) * (1 + bound / tolerance) ** 2
        for a, b in ((edge.start_node_id, edge.end_node_id), (edge.end_node_id, edge.start_node_id)):
            adjacency.setdefault(a, []).append((b, index, cost))
    required = {group[0]}
    for index in group[1:]:
        origins = {n for i in required for n in (graph.edges[i].start_node_id, graph.edges[i].end_node_id)}
        target = {graph.edges[index].start_node_id, graph.edges[index].end_node_id}
        queue = [(0.0, node, ()) for node in sorted(origins)]
        heapq.heapify(queue)
        seen: set[int] = set()
        found = None
        while queue:
            cost, node, path = heapq.heappop(queue)
            if node in seen:
                continue
            seen.add(node)
            if node in target:
                found = path
                break
            for other, edge, weight in adjacency.get(node, ()):
                if other not in seen:
                    heapq.heappush(queue, (cost+weight, other, (*path, edge)))
        if found is None:
            return None
        required.update((*found, index))
    return required


def reconstruct_global_arcs(graph: SkeletonGraph, geometry: GeometryResult,
                            tolerance_px: float = 3.0) -> GlobalArcResult:
    """Privilégie les plus longues portions circulaires réellement connectées.

    Une composante source compatible fournit la couverture angulaire ; les
    zones de jonction doivent aussi être dans la tolérance. Les portions
    déconnectées restent séparées. Les petits cycles compatibles sont absorbés
    dans la géométrie idéale mais restent dans le graphe et la provenance.
    """
    if not math.isfinite(tolerance_px) or tolerance_px < 0:
        raise ValueError("La tolérance d'arc global doit être finie et positive ou nulle.")
    if (graph.width, graph.height) != (geometry.width, geometry.height):
        raise ValueError("Le graphe et la géométrie doivent avoir les mêmes dimensions.")
    available: set[int] = set()
    represented: set[int] = set()
    for primitive in geometry.primitives:
        ids = set(primitive_source_edge_ids(primitive))
        if represented & ids:
            raise ValueError("Les branches sources de la géométrie doivent être distinctes.")
        represented.update(ids)
        if not hasattr(primitive, "source_edge_ids"):
            available.update(ids)
    if represented != {e.id for e in graph.edges}:
        raise ValueError("La géométrie doit représenter toutes les branches sources.")
    if tolerance_px == 0:
        return GlobalArcResult(geometry, ())

    edge_coords: dict[int, NDArray[np.float64]] = {
        edge.id: np.array([(p.x, p.y) for p in edge.points], dtype=np.float64).reshape(-1, 2)
        for edge in graph.edges
    }
    edge_lengths: dict[int, float] = {
        edge.id: float(np.sum(np.hypot(np.diff(edge_coords[edge.id][:, 0]), np.diff(edge_coords[edge.id][:, 1]))))
        if len(edge.points) > 1 else 0.0
        for edge in graph.edges
    }

    seeds = []
    for index in sorted(available):
        edge = graph.edges[index]
        if edge.is_loop or len(edge.points) < 4:
            continue
        fit = fit_circle(edge.points)
        if fit is None or _radial_bound(edge_coords[index], fit) > tolerance_px:
            continue
        coords = edge_coords[index]
        angles = np.unwrap(np.arctan2(coords[:, 1] - fit.center.y, coords[:, 0] - fit.center.x))
        if MIN_SEED_SWEEP_RAD <= abs(float(angles[-1]-angles[0])) <= MAX_GLOBAL_SWEEP_RAD:
            seeds.append((index, fit))
    proposals: dict[tuple[int, ...], GlobalArc] = {}
    for seed, initial in seeds:
        group = [seed]
        fit = initial
        for index, support in seeds:
            if index == seed:
                continue
            discrepancy = math.hypot(support.center.x-fit.center.x, support.center.y-fit.center.y)+abs(support.radius-fit.radius)
            if discrepancy > ARC_SUPPORT_SEARCH_FACTOR*tolerance_px:
                continue
            points = tuple(p for i in (*group, index) for p in graph.edges[i].points)
            candidate = fit_circle(points)
            if candidate is not None and all(_radial_bound(edge_coords[i], candidate) <= tolerance_px for i in (*group, index)):
                group.append(index)
                fit = candidate
        if len(group) < 2:
            continue
        required = _connecting_sources(graph, group, available, fit, tolerance_px, edge_coords, edge_lengths)
        if required is None:
            continue
        joined = fit_circle(tuple(p for i in sorted(required) for p in graph.edges[i].points))
        if joined is None or any(_radial_bound(edge_coords[i], joined) > tolerance_px for i in required):
            continue
        fit = joined
        selected = {i for i in available if _radial_bound(edge_coords[i], fit) <= tolerance_px}
        for component in _components(graph, selected):
            if len(set(component) & set(group)) < 2:
                continue
            points = tuple(p for i in component for p in graph.edges[i].points)
            candidate = fit_circle(points)
            if candidate is None:
                continue
            if any(_radial_bound(edge_coords[i], candidate) > tolerance_px for i in component):
                candidate = fit
            bound = max(_radial_bound(edge_coords[i], candidate) for i in component)
            nodes = {node for i in component for node in (graph.edges[i].start_node_id, graph.edges[i].end_node_id)}
            zone_points = tuple(Point(float(col), float(row)) for node in sorted(nodes)
                                for row, col in graph.nodes[node].source_pixels)
            if zone_points:
                bound = max(bound, max(abs(math.hypot(p.x-candidate.center.x, p.y-candidate.center.y)-candidate.radius) for p in zone_points))
            if bound > tolerance_px:
                continue
            # Couverture minimale : le plus grand intervalle angulaire sans
            # source est l'extérieur de l'arc, jamais un trou à combler.
            all_points = (*points, *zone_points)
            angles = np.sort(np.mod(np.arctan2([p.y-candidate.center.y for p in all_points],
                                               [p.x-candidate.center.x for p in all_points]), 2*math.pi))
            gaps = np.diff(np.r_[angles, angles[0]+2*math.pi])
            outside = int(np.argmax(gaps))
            start = float(angles[(outside+1)%len(angles)])
            sweep = float(2*math.pi-gaps[outside])
            if not MIN_SEED_SWEEP_RAD <= sweep <= MAX_GLOBAL_SWEEP_RAD:
                continue
            # Chaque branche doit rester dans ce balayage ; des détours autour
            # de l'autre côté du cercle ne sont pas représentés par un arc court.
            invalid = False
            for i in component:
                coords = edge_coords[i]
                theta = np.unwrap(np.arctan2(coords[:, 1] - candidate.center.y, coords[:, 0] - candidate.center.x))
                if np.ptp(theta) > sweep+1e-9:
                    invalid = True
                    break
            if invalid:
                continue
            arc = GlobalArc(candidate.center, candidate.radius, start, start+sweep,
                            bound, candidate.fit_error_rms, min(component), points, component, component)
            bound = max(bound, max(_finite_arc_bound(edge_coords[i], arc) for i in component))
            if bound <= tolerance_px:
                proposals[component] = replace(arc, fit_error_max=bound)
    accepted = []
    used: set[int] = set()
    for arc in sorted(proposals.values(), key=lambda a: (-a.radius*a.sweep_angle_rad, a.fit_error_max, a.source_edge_ids)):
        # Les longs fragments ne peuvent appartenir à deux primitives. Un
        # petit connecteur proche d'une intersection peut en revanche étayer
        # deux courbes ; il n'est retiré de l'export local qu'une seule fois.
        major_fragments = set()
        for i in arc.source_edge_ids:
            coords = edge_coords[i]
            theta = np.unwrap(np.arctan2(coords[:, 1] - arc.center.y, coords[:, 0] - arc.center.x))
            if np.ptp(theta) >= MIN_SEED_SWEEP_RAD:
                major_fragments.add(i)
        if used & major_fragments:
            continue
        owned = tuple(i for i in arc.source_edge_ids if i not in used)
        if owned:
            accepted.append(replace(arc, source_edge_id=min(owned), source_edge_ids=owned))
            used.update(owned)
    primitives = tuple(p for p in geometry.primitives if not (set(primitive_source_edge_ids(p)) & used)) + tuple(accepted)
    return GlobalArcResult(replace(geometry, primitives=primitives), tuple(accepted))

