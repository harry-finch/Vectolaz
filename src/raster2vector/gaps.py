"""Raccords explicites entre extrémités : distance, tangentes et absence d'obstacle."""

from dataclasses import dataclass, replace
import math

from raster2vector.coordinates import Point
from raster2vector.geometry import fit_line
from raster2vector.graph import GraphEdge, SkeletonGraph

# Critères internes conservateurs, en pixels et radians ; voir README.
TANGENT_WINDOW_PX = 12.0
MIN_TANGENT_SPAN_PX = 3.0
TANGENT_ERROR_PX = 1.25
MAX_DIRECTION_ANGLE_RAD = math.radians(20)
OBSTACLE_CLEARANCE_PX = 0.75


@dataclass(frozen=True)
class GapConnection:
    start_node_id: int
    end_node_id: int
    start: Point
    end: Point
    distance_px: float


@dataclass(frozen=True)
class GapResult:
    """Graphe raccordé, ponts acceptés et provenance des nouvelles branches.

    source_edge_ids[i] liste les branches du graphe original intégrées à la
    branche i. Les ponts sont explicites ; aucun pixel source n'est déplacé.
    """
    graph: SkeletonGraph
    connections: tuple[GapConnection, ...]
    source_edge_ids: tuple[tuple[int, ...], ...]


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a.x-b.x, a.y-b.y)


def _segment_distance(a: Point, b: Point, c: Point, d: Point) -> float:
    def cross(p: Point, q: Point, r: Point) -> float:
        return (q.x-p.x)*(r.y-p.y)-(q.y-p.y)*(r.x-p.x)

    def point_distance(p: Point, x: Point, y: Point) -> float:
        dx, dy = y.x-x.x, y.y-x.y
        length2 = dx*dx+dy*dy
        t = max(0.0, min(1.0, ((p.x-x.x)*dx+(p.y-x.y)*dy)/length2)) if length2 else 0.0
        return math.hypot(p.x-x.x-t*dx, p.y-x.y-t*dy)

    if cross(a, b, c)*cross(a, b, d) < 0 and cross(c, d, a)*cross(c, d, b) < 0:
        return 0.0
    return min(point_distance(a, c, d), point_distance(b, c, d),
               point_distance(c, a, b), point_distance(d, a, b))


def _endpoint(edge: GraphEdge, node_id: int) -> tuple[Point, Point] | None:
    points = edge.points if edge.start_node_id == node_id else edge.points[::-1]
    if len(points) < 2:
        return None
    local = [points[0]]
    length = 0.0
    for point in points[1:]:
        length += _distance(local[-1], point)
        if length > TANGENT_WINDOW_PX:
            break
        local.append(point)
    if _distance(local[0], local[-1]) < MIN_TANGENT_SPAN_PX:
        return None
    fit = fit_line(local)
    if fit is None or fit.fit_error_max > TANGENT_ERROR_PX:
        return None
    return points[0], Point(-fit.direction.x, -fit.direction.y)


def _merge(graph: SkeletonGraph, connections: tuple[GapConnection, ...]) -> GapResult:
    edges = list(graph.edges)
    edges.extend(GraphEdge(len(graph.edges)+index, c.start_node_id, c.end_node_id, (c.start, c.end))
                 for index, c in enumerate(connections))
    adjacency: dict[int, list[int]] = {n.id: [] for n in graph.nodes}
    for edge in edges:
        adjacency[edge.start_node_id].append(edge.id)
        if edge.end_node_id != edge.start_node_id:
            adjacency[edge.end_node_id].append(edge.id)
    internal = {i for c in connections for i in (c.start_node_id, c.end_node_id)}
    retained = set(adjacency)-internal
    # Une chaîne entièrement raccordée peut devenir une boucle sans endpoint.
    # Conserver alors une ancre logique, comme lors de l'extraction du squelette.
    seen: set[int] = set()
    anchors: set[int] = set()
    for seed in sorted(adjacency):
        if seed in seen:
            continue
        pending = [seed]
        component: set[int] = set()
        while pending:
            node = pending.pop()
            if node in component:
                continue
            component.add(node)
            for index in adjacency[node]:
                edge = edges[index]
                pending.append(edge.end_node_id if edge.start_node_id == node else edge.start_node_id)
        seen.update(component)
        if not component & retained:
            anchor = min(component)
            anchors.add(anchor)
            retained.add(anchor)
    old_ids = sorted(retained)
    mapping = {old: new for new, old in enumerate(old_ids)}
    nodes = tuple(replace(graph.nodes[old], id=mapping[old],
                          kind="loop" if old in anchors else graph.nodes[old].kind) for old in old_ids)
    merged: list[GraphEdge] = []
    provenance: list[tuple[int, ...]] = []
    visited: set[int] = set()
    for start in old_ids:
        for first in adjacency[start]:
            if first in visited:
                continue
            current, index = start, first
            points: list[Point] = []
            sources: list[int] = []
            while True:
                edge = edges[index]
                visited.add(index)
                oriented = edge.points if edge.start_node_id == current else edge.points[::-1]
                points.extend(oriented[1:] if points and points[-1] == oriented[0] else oriented)
                if index < len(graph.edges):
                    sources.append(index)
                current = edge.end_node_id if edge.start_node_id == current else edge.start_node_id
                if current in retained:
                    break
                index = next(i for i in adjacency[current] if i not in visited)
            merged.append(GraphEdge(len(merged), mapping[start], mapping[current], tuple(points)))
            provenance.append(tuple(sources))
    return GapResult(SkeletonGraph(graph.width, graph.height, nodes, tuple(merged)), connections, tuple(provenance))


def connect_gaps(graph: SkeletonGraph, gap_tolerance_px: float = 0.0) -> GapResult:
    """Raccorde seulement des endpoints compatibles et sans choix concurrent.

    Chaque tangente sortante doit viser l'autre endpoint à <= 20°. Le pont
    doit rester à plus de 0.75 px des autres traits. Un endpoint ayant plusieurs
    candidats compatibles est laissé inchangé ; des ponts concurrents qui se
    croisent sont tous refusés. Zéro désactive entièrement le raccordement.
    """
    if not math.isfinite(gap_tolerance_px) or gap_tolerance_px < 0:
        raise ValueError("La tolérance de raccordement doit être finie et positive ou nulle.")
    unchanged = GapResult(graph, (), tuple((e.id,) for e in graph.edges))
    if gap_tolerance_px == 0:
        return unchanged
    endpoint_edges: dict[int, list[GraphEdge]] = {n.id: [] for n in graph.nodes if n.kind == "endpoint"}
    for edge in graph.edges:
        for node in (edge.start_node_id, edge.end_node_id):
            if node in endpoint_edges:
                endpoint_edges[node].append(edge)
    endpoints = {node: info for node, edges in endpoint_edges.items() if len(edges) == 1
                 and (info := _endpoint(edges[0], node)) is not None}
    candidates: list[GapConnection] = []
    ids = sorted(endpoints)
    cosine = math.cos(MAX_DIRECTION_ANGLE_RAD)
    for index, a in enumerate(ids):
        p, direction_a = endpoints[a]
        for b in ids[index+1:]:
            q, direction_b = endpoints[b]
            distance = _distance(p, q)
            if not 0 < distance <= gap_tolerance_px:
                continue
            dx, dy = (q.x-p.x)/distance, (q.y-p.y)/distance
            if direction_a.x*dx+direction_a.y*dy < cosine or -(direction_b.x*dx+direction_b.y*dy) < cosine:
                continue
            blocked = False
            for edge in graph.edges:
                contacts = tuple(endpoints[node] for node in (a, b) if edge in endpoint_edges[node])
                for c, d in zip(edge.points, edge.points[1:]):
                    # Ignorer seulement les voisins locaux derrière le contact.
                    # Un trait incident qui revient devant le contact reste un obstacle.
                    if any(max(_distance(contact, c), _distance(contact, d)) <= TANGENT_WINDOW_PX
                           and max((point.x-contact.x)*direction.x+(point.y-contact.y)*direction.y
                                   for point in (c, d)) <= OBSTACLE_CLEARANCE_PX
                           for contact, direction in contacts):
                        continue
                    if _segment_distance(p, q, c, d) <= OBSTACLE_CLEARANCE_PX:
                        blocked = True
                        break
                if blocked:
                    break
            if any(n.kind == "isolated" and _segment_distance(p, q, n.position, n.position) <= OBSTACLE_CLEARANCE_PX
                   for n in graph.nodes):
                blocked = True
            if not blocked:
                candidates.append(GapConnection(a, b, p, q, distance))
    counts = {node: sum(node in (c.start_node_id, c.end_node_id) for c in candidates) for node in ids}
    unique = [c for c in candidates if counts[c.start_node_id] == counts[c.end_node_id] == 1]
    accepted = tuple(c for c in unique if all(c == other or _segment_distance(c.start, c.end, other.start, other.end) > OBSTACLE_CLEARANCE_PX
                                             for other in unique))
    return _merge(graph, accepted) if accepted else unchanged
