"""Arcs circulaires continus à travers plusieurs branches et intersections."""

from dataclasses import dataclass, replace
import math
import heapq
import numpy as np

from raster2vector.coordinates import Point
from raster2vector.geometry import CircleFit, CircularArc, GeometryResult, fit_circle
from raster2vector.global_circles import GlobalCircle, _radial_bound, MIN_SEED_SWEEP_RAD
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


def _finite_arc_bound(points: tuple[Point, ...], arc: GlobalArc) -> float:
    """Distance maximale à l'arc fini sur chaque segment source, sans échantillonnage."""
    center = np.array((arc.center.x, arc.center.y))
    data = np.array([(p.x, p.y) for p in points])-center
    start = np.array((arc.start.x, arc.start.y))-center
    end = np.array((arc.end.x, arc.end.y))-center
    values = list(data)
    for a, b in zip(data, data[1:]):
        delta = b-a
        length2 = float(np.dot(delta, delta))
        if length2:
            t = float(np.clip(-np.dot(a, delta)/length2, 0, 1))
            values.append(a+t*delta)
        # Hors du balayage, le plus proche des deux contacts change sur leur
        # médiatrice. Ce changement peut être un maximum intérieur au segment.
        direction = end-start
        denominator = float(2*np.dot(delta, direction))
        if denominator:
            t = float((np.dot(end, end)-np.dot(start, start)-2*np.dot(a, direction))/denominator)
            if 0 <= t <= 1:
                values.append(a+t*delta)
        for angle in (arc.start_angle_rad, arc.end_angle_rad):
            ray = np.array((math.cos(angle), math.sin(angle)))
            cross_delta = ray[0]*delta[1]-ray[1]*delta[0]
            if cross_delta != 0:
                t = -(ray[0]*a[1]-ray[1]*a[0])/cross_delta
                value = a+t*delta
                if 0 <= t <= 1 and np.dot(value, ray) >= 0:
                    values.append(value)
    vectors = np.array(values)
    progress = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0])-arc.start_angle_rad, 2*math.pi)
    errors = np.abs(np.linalg.norm(vectors, axis=1)-arc.radius)
    outside = progress > arc.sweep_angle_rad
    errors[outside] = np.minimum(np.linalg.norm(vectors[outside]-start, axis=1),
                                 np.linalg.norm(vectors[outside]-end, axis=1))
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
                        fit: CircleFit, tolerance: float) -> set[int] | None:
    """Propose les chemins nécessaires, puis laisse l'ajustement commun les valider."""
    adjacency: dict[int, list[tuple[int, int, float]]] = {}
    for index in sorted(available):
        edge = graph.edges[index]
        bound = _radial_bound(edge.points, fit)
        if bound > CONNECTOR_SEARCH_FACTOR*tolerance:
            continue
        length = sum(math.hypot(b.x-a.x, b.y-a.y) for a, b in zip(edge.points, edge.points[1:]))
        cost = max(length, 1e-9)*(1+bound/tolerance)**2
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
        ids = set(primitive.source_edge_ids) if isinstance(primitive, (GlobalCircle, GlobalArc)) else {primitive.source_edge_id}
        if represented & ids:
            raise ValueError("Les branches sources de la géométrie doivent être distinctes.")
        represented.update(ids)
        if not isinstance(primitive, (GlobalCircle, GlobalArc)):
            available.update(ids)
    if represented != {e.id for e in graph.edges}:
        raise ValueError("La géométrie doit représenter toutes les branches sources.")
    if tolerance_px == 0:
        return GlobalArcResult(geometry, ())
    seeds = []
    for index in sorted(available):
        edge = graph.edges[index]
        if edge.is_loop or len(edge.points) < 4:
            continue
        fit = fit_circle(edge.points)
        if fit is None or _radial_bound(edge.points, fit) > tolerance_px:
            continue
        angles = np.unwrap(np.arctan2([p.y-fit.center.y for p in edge.points], [p.x-fit.center.x for p in edge.points]))
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
            if candidate is not None and all(_radial_bound(graph.edges[i].points, candidate) <= tolerance_px for i in (*group, index)):
                group.append(index)
                fit = candidate
        if len(group) < 2:
            continue
        required = _connecting_sources(graph, group, available, fit, tolerance_px)
        if required is None:
            continue
        joined = fit_circle(tuple(p for i in sorted(required) for p in graph.edges[i].points))
        if joined is None or any(_radial_bound(graph.edges[i].points, joined) > tolerance_px for i in required):
            continue
        fit = joined
        selected = {i for i in available if _radial_bound(graph.edges[i].points, fit) <= tolerance_px}
        for component in _components(graph, selected):
            if len(set(component) & set(group)) < 2:
                continue
            points = tuple(p for i in component for p in graph.edges[i].points)
            candidate = fit_circle(points)
            if candidate is None:
                continue
            if any(_radial_bound(graph.edges[i].points, candidate) > tolerance_px for i in component):
                candidate = fit
            bound = max(_radial_bound(graph.edges[i].points, candidate) for i in component)
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
                source = graph.edges[i].points
                theta = np.unwrap(np.arctan2([p.y-candidate.center.y for p in source], [p.x-candidate.center.x for p in source]))
                if np.ptp(theta) > sweep+1e-9:
                    invalid = True
                    break
            if invalid:
                continue
            arc = GlobalArc(candidate.center, candidate.radius, start, start+sweep,
                            bound, candidate.fit_error_rms, min(component), points, component, component)
            bound = max(bound, max(_finite_arc_bound(graph.edges[i].points, arc) for i in component))
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
            theta = np.unwrap(np.arctan2([p.y-arc.center.y for p in graph.edges[i].points],
                                         [p.x-arc.center.x for p in graph.edges[i].points]))
            if np.ptp(theta) >= MIN_SEED_SWEEP_RAD:
                major_fragments.add(i)
        if used & major_fragments:
            continue
        owned = tuple(i for i in arc.source_edge_ids if i not in used)
        if owned:
            accepted.append(replace(arc, source_edge_id=min(owned), source_edge_ids=owned))
            used.update(owned)
    primitives = tuple(p for p in geometry.primitives if p.source_edge_id not in used)+tuple(accepted)
    return GlobalArcResult(replace(geometry, primitives=primitives), tuple(accepted))
