"""Raccords explicites entre extrémités : distance, tangentes et absence d'obstacle."""

from dataclasses import dataclass, replace
import math

import numpy as np
from numpy.typing import NDArray

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


def _cross(p: Point, q: Point, r: Point) -> float:
    return (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x)


def _point_distance(p: Point, x: Point, y: Point) -> float:
    dx, dy = y.x - x.x, y.y - x.y
    length2 = dx * dx + dy * dy
    if length2 == 0.0:
        return math.hypot(p.x - x.x, p.y - x.y)
    t = max(0.0, min(1.0, ((p.x - x.x) * dx + (p.y - x.y) * dy) / length2))
    return math.hypot(p.x - x.x - t * dx, p.y - x.y - t * dy)


def _on_segment(p: Point, a: Point, b: Point, tol: float = 1e-9) -> bool:
    return (min(a.x, b.x) - tol <= p.x <= max(a.x, b.x) + tol and
            min(a.y, b.y) - tol <= p.y <= max(a.y, b.y) + tol)


def _segment_distance(a: Point, b: Point, c: Point, d: Point) -> float:
    c1 = _cross(a, b, c)
    c2 = _cross(a, b, d)
    c3 = _cross(c, d, a)
    c4 = _cross(c, d, b)

    if c1 * c2 < 0 and c3 * c4 < 0:
        return 0.0

    if abs(c1) <= 1e-12 and _on_segment(c, a, b):
        return 0.0
    if abs(c2) <= 1e-12 and _on_segment(d, a, b):
        return 0.0
    if abs(c3) <= 1e-12 and _on_segment(a, c, d):
        return 0.0
    if abs(c4) <= 1e-12 and _on_segment(b, c, d):
        return 0.0

    dist = min(_point_distance(a, c, d), _point_distance(b, c, d),
               _point_distance(c, a, b), _point_distance(d, a, b))
    return 0.0 if dist <= 1e-12 else dist


def _segments_to_segment_min_distance(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    c: NDArray[np.float64],
    d: NDArray[np.float64],
) -> float:
    """Distance minimale entre le segment (p, q) et un ensemble de segments (c[i], d[i])."""
    if len(c) == 0:
        return float("inf")

    u = q - p
    v = d - c

    cp1 = u[0] * (c[:, 1] - p[1]) - u[1] * (c[:, 0] - p[0])
    cp2 = u[0] * (d[:, 1] - p[1]) - u[1] * (d[:, 0] - p[0])

    cd1 = v[:, 0] * (p[1] - c[:, 1]) - v[:, 1] * (p[0] - c[:, 0])
    cd2 = v[:, 0] * (q[1] - c[:, 1]) - v[:, 1] * (q[0] - c[:, 0])

    strict_cross = (cp1 * cp2 < 0) & (cd1 * cd2 < 0)
    if np.any(strict_cross):
        return 0.0

    tol = 1e-9
    c_on_pq = (
        (np.abs(cp1) <= 1e-12) &
        (np.minimum(p[0], q[0]) - tol <= c[:, 0]) & (c[:, 0] <= np.maximum(p[0], q[0]) + tol) &
        (np.minimum(p[1], q[1]) - tol <= c[:, 1]) & (c[:, 1] <= np.maximum(p[1], q[1]) + tol)
    )
    d_on_pq = (
        (np.abs(cp2) <= 1e-12) &
        (np.minimum(p[0], q[0]) - tol <= d[:, 0]) & (d[:, 0] <= np.maximum(p[0], q[0]) + tol) &
        (np.minimum(p[1], q[1]) - tol <= d[:, 1]) & (d[:, 1] <= np.maximum(p[1], q[1]) + tol)
    )
    p_on_cd = (
        (np.abs(cd1) <= 1e-12) &
        (np.minimum(c[:, 0], d[:, 0]) - tol <= p[0]) & (p[0] <= np.maximum(c[:, 0], d[:, 0]) + tol) &
        (np.minimum(c[:, 1], d[:, 1]) - tol <= p[1]) & (p[1] <= np.maximum(c[:, 1], d[:, 1]) + tol)
    )
    q_on_cd = (
        (np.abs(cd2) <= 1e-12) &
        (np.minimum(c[:, 0], d[:, 0]) - tol <= q[0]) & (q[0] <= np.maximum(c[:, 0], d[:, 0]) + tol) &
        (np.minimum(c[:, 1], d[:, 1]) - tol <= q[1]) & (q[1] <= np.maximum(c[:, 1], d[:, 1]) + tol)
    )
    if np.any(c_on_pq | d_on_pq | p_on_cd | q_on_cd):
        return 0.0

    len2_v = v[:, 0] ** 2 + v[:, 1] ** 2
    diff_p = p - c
    dot_p = diff_p[:, 0] * v[:, 0] + diff_p[:, 1] * v[:, 1]
    raw_t_p = np.divide(dot_p, len2_v, out=np.zeros_like(dot_p), where=len2_v > 0)
    t_p = np.clip(raw_t_p, 0.0, 1.0)
    proj_p_x = c[:, 0] + t_p * v[:, 0]
    proj_p_y = c[:, 1] + t_p * v[:, 1]
    dist_p = np.hypot(p[0] - proj_p_x, p[1] - proj_p_y)

    diff_q = q - c
    dot_q = diff_q[:, 0] * v[:, 0] + diff_q[:, 1] * v[:, 1]
    raw_t_q = np.divide(dot_q, len2_v, out=np.zeros_like(dot_q), where=len2_v > 0)
    t_q = np.clip(raw_t_q, 0.0, 1.0)
    proj_q_x = c[:, 0] + t_q * v[:, 0]
    proj_q_y = c[:, 1] + t_q * v[:, 1]
    dist_q = np.hypot(q[0] - proj_q_x, q[1] - proj_q_y)

    len2_u = u[0] ** 2 + u[1] ** 2
    if len2_u > 0:
        diff_c = c - p
        dot_c = diff_c[:, 0] * u[0] + diff_c[:, 1] * u[1]
        t_c = np.clip(dot_c / len2_u, 0.0, 1.0)
        proj_c_x = p[0] + t_c * u[0]
        proj_c_y = p[1] + t_c * u[1]
        dist_c = np.hypot(c[:, 0] - proj_c_x, c[:, 1] - proj_c_y)

        diff_d = d - p
        dot_d = diff_d[:, 0] * u[0] + diff_d[:, 1] * u[1]
        t_d = np.clip(dot_d / len2_u, 0.0, 1.0)
        proj_d_x = p[0] + t_d * u[0]
        proj_d_y = p[1] + t_d * u[1]
        dist_d = np.hypot(d[:, 0] - proj_d_x, d[:, 1] - proj_d_y)
    else:
        dist_c = np.hypot(c[:, 0] - p[0], c[:, 1] - p[1])
        dist_d = np.hypot(d[:, 0] - p[0], d[:, 1] - p[1])

    min_dists = np.minimum(np.minimum(dist_p, dist_q), np.minimum(dist_c, dist_d))
    val = float(np.min(min_dists))
    return 0.0 if val <= 1e-12 else val


def _points_to_segment_min_distance(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    pts: NDArray[np.float64],
) -> float:
    """Distance minimale entre un segment (p, q) et un ensemble de points pts."""
    if len(pts) == 0:
        return float("inf")
    u = q - p
    len2_u = u[0] ** 2 + u[1] ** 2
    if len2_u > 0:
        diff = pts - p
        dot = diff[:, 0] * u[0] + diff[:, 1] * u[1]
        t = np.clip(dot / len2_u, 0.0, 1.0)
        proj_x = p[0] + t * u[0]
        proj_y = p[1] + t * u[1]
        dists = np.hypot(pts[:, 0] - proj_x, pts[:, 1] - proj_y)
    else:
        dists = np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1])
    val = float(np.min(dists))
    return 0.0 if val <= 1e-12 else val


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

    # Représentation vectorisée de tous les segments du graphe
    seg_starts_list: list[tuple[float, float]] = []
    seg_ends_list: list[tuple[float, float]] = []
    seg_edge_ids_list: list[int] = []
    for edge in graph.edges:
        for c, d in zip(edge.points, edge.points[1:]):
            seg_starts_list.append((c.x, c.y))
            seg_ends_list.append((d.x, d.y))
            seg_edge_ids_list.append(edge.id)

    num_segments = len(seg_starts_list)
    if num_segments > 0:
        seg_starts: NDArray[np.float64] = np.array(seg_starts_list, dtype=np.float64)
        seg_ends: NDArray[np.float64] = np.array(seg_ends_list, dtype=np.float64)
        seg_edge_ids: NDArray[np.int32] = np.array(seg_edge_ids_list, dtype=np.int32)
        seg_min: NDArray[np.float64] = np.minimum(seg_starts, seg_ends)
        seg_max: NDArray[np.float64] = np.maximum(seg_starts, seg_ends)
    else:
        seg_starts = np.empty((0, 2), dtype=np.float64)
        seg_ends = np.empty((0, 2), dtype=np.float64)
        seg_edge_ids = np.empty((0,), dtype=np.int32)
        seg_min = np.empty((0, 2), dtype=np.float64)
        seg_max = np.empty((0, 2), dtype=np.float64)

    isolated_list = [(n.position.x, n.position.y) for n in graph.nodes if n.kind == "isolated"]
    if isolated_list:
        isolated_positions: NDArray[np.float64] = np.array(isolated_list, dtype=np.float64)
    else:
        isolated_positions = np.empty((0, 2), dtype=np.float64)

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

            p_arr = np.array((p.x, p.y), dtype=np.float64)
            q_arr = np.array((q.x, q.y), dtype=np.float64)
            cand_min = np.minimum(p_arr, q_arr) - OBSTACLE_CLEARANCE_PX
            cand_max = np.maximum(p_arr, q_arr) + OBSTACLE_CLEARANCE_PX

            blocked = False

            # Filtrage spatial des nœuds isolated
            if len(isolated_positions) > 0:
                iso_mask = (
                    (isolated_positions[:, 0] <= cand_max[0]) & (isolated_positions[:, 0] >= cand_min[0]) &
                    (isolated_positions[:, 1] <= cand_max[1]) & (isolated_positions[:, 1] >= cand_min[1])
                )
                if np.any(iso_mask):
                    sub_iso = isolated_positions[iso_mask]
                    if _points_to_segment_min_distance(p_arr, q_arr, sub_iso) <= OBSTACLE_CLEARANCE_PX:
                        blocked = True

            # Filtrage spatial AABB des segments du graphe
            if not blocked and num_segments > 0:
                aabb_mask = (
                    (seg_min[:, 0] <= cand_max[0]) & (seg_max[:, 0] >= cand_min[0]) &
                    (seg_min[:, 1] <= cand_max[1]) & (seg_max[:, 1] >= cand_min[1])
                )
                if np.any(aabb_mask):
                    sub_indices = np.flatnonzero(aabb_mask)
                    sub_starts = seg_starts[sub_indices]
                    sub_ends = seg_ends[sub_indices]
                    sub_edges = seg_edge_ids[sub_indices]

                    incident_contacts = []
                    if a in endpoint_edges and len(endpoint_edges[a]) == 1:
                        incident_contacts.append((p, direction_a, endpoint_edges[a][0].id))
                    if b in endpoint_edges and len(endpoint_edges[b]) == 1:
                        incident_contacts.append((q, direction_b, endpoint_edges[b][0].id))

                    to_ignore = np.zeros(len(sub_indices), dtype=bool)
                    for contact_pt, dir_pt, edge_id in incident_contacts:
                        edge_mask = (sub_edges == edge_id)
                        if np.any(edge_mask):
                            starts_e = sub_starts[edge_mask]
                            ends_e = sub_ends[edge_mask]
                            c_x, c_y = contact_pt.x, contact_pt.y
                            d_x, d_y = dir_pt.x, dir_pt.y

                            d_start = np.hypot(starts_e[:, 0] - c_x, starts_e[:, 1] - c_y)
                            d_end = np.hypot(ends_e[:, 0] - c_x, ends_e[:, 1] - c_y)
                            within_window = (d_start <= TANGENT_WINDOW_PX) & (d_end <= TANGENT_WINDOW_PX)

                            proj_start = (starts_e[:, 0] - c_x) * d_x + (starts_e[:, 1] - c_y) * d_y
                            proj_end = (ends_e[:, 0] - c_x) * d_x + (ends_e[:, 1] - c_y) * d_y
                            behind = (proj_start <= OBSTACLE_CLEARANCE_PX) & (proj_end <= OBSTACLE_CLEARANCE_PX)

                            to_ignore[edge_mask] |= (within_window & behind)

                    remaining = ~to_ignore
                    if np.any(remaining):
                        test_starts = sub_starts[remaining]
                        test_ends = sub_ends[remaining]
                        if _segments_to_segment_min_distance(p_arr, q_arr, test_starts, test_ends) <= OBSTACLE_CLEARANCE_PX:
                            blocked = True

            if not blocked:
                candidates.append(GapConnection(a, b, p, q, distance))
    counts = {node: sum(node in (c.start_node_id, c.end_node_id) for c in candidates) for node in ids}
    unique = [c for c in candidates if counts[c.start_node_id] == counts[c.end_node_id] == 1]
    accepted = tuple(c for c in unique if all(c == other or _segment_distance(c.start, c.end, other.start, other.end) > OBSTACLE_CLEARANCE_PX
                                              for other in unique))
    return _merge(graph, accepted) if accepted else unchanged
