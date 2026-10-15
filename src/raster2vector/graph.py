"""Topologie du squelette : voisins utiles, nœuds logiques et branches.

Les pixels internes utilisent (row, col). La conversion vers Point(x, y)
est explicite ; aucun ajustement géométrique n'est effectué.
"""

from collections import deque
from dataclasses import dataclass, replace
import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from raster2vector.coordinates import Point

Pixel = tuple[int, int]  # (row, col)
PixelLink = tuple[Pixel, Pixel]
NodeKind = Literal["endpoint", "junction", "loop", "isolated"]


@dataclass(frozen=True)
class GraphNode:
    """Nœud logique ; source_pixels conserve toute sa zone raster (row, col).

    position est le barycentre des pixels, destiné au debug, pas un ajustement
    d'intersection. loop est une ancre artificielle d'une boucle sans extrémité.
    """

    id: int
    kind: NodeKind
    position: Point
    source_pixels: tuple[Pixel, ...]


@dataclass(frozen=True)
class GraphEdge:
    """Branche ordonnée entre deux nœuds, avec ses pixels convertis en (x, y).

    Les extrémités sont les pixels réels de contact avec les zones des nœuds,
    et ne sont pas déplacées vers leur barycentre. Plusieurs branches entre
    les mêmes nœuds sont autorisées, ainsi que les arêtes revenant au même nœud.
    """

    id: int
    start_node_id: int
    end_node_id: int
    points: tuple[Point, ...]

    @property
    def is_loop(self) -> bool:
        """Indique une boucle topologique, même avec deux contacts distincts."""
        return self.start_node_id == self.end_node_id


@dataclass(frozen=True)
class SkeletonGraph:
    """Graphe déterministe ; les identifiants indexent nodes et edges."""

    width: int
    height: int
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]


def _point(pixel: Pixel) -> Point:
    row, col = pixel
    return Point(x=float(col), y=float(row))


def _link(a: Pixel, b: Pixel) -> PixelLink:
    return (a, b) if a < b else (b, a)


def _adjacency(pixels: set[Pixel]) -> dict[Pixel, tuple[Pixel, ...]]:
    """Connexité 8 sans raccourcis diagonaux ni cycles des carrés pleins.

    Dans chaque carré plein 2x2, le côté vertical droit est redondant et retiré.
    Les côtés horizontaux restent disponibles ; une liaison verticale retirée
    se retrouve en passant à gauche par les carrés voisins, jusqu'au bord du
    bloc. Ceci conserve la connexité sans inventer un trou dans un bloc plein.
    """
    adjacency: dict[Pixel, tuple[Pixel, ...]] = {}
    for row, col in sorted(pixels):
        neighbors: list[Pixel] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                neighbor = (row + dr, col + dc)
                if (dr == dc == 0) or neighbor not in pixels:
                    continue
                if dr and dc and ((row + dr, col) in pixels or (row, col + dc) in pixels):
                    continue
                if dr and not dc and (row, col - 1) in pixels and (row + dr, col - 1) in pixels:
                    continue
                neighbors.append(neighbor)
        adjacency[(row, col)] = tuple(sorted(neighbors))
    return adjacency


def _junction_groups(
    adjacency: dict[Pixel, tuple[Pixel, ...]],
) -> list[tuple[Pixel, ...]]:
    remaining = {pixel for pixel, neighbors in adjacency.items() if len(neighbors) > 2}
    # Un bloc plein 2x2 est une zone de junction, même si certains de ses
    # pixels ont un degré réduit après suppression des liaisons redondantes.
    for row, col in adjacency:
        block = ((row, col), (row + 1, col), (row, col + 1), (row + 1, col + 1))
        if all(pixel in adjacency for pixel in block):
            remaining.update(block)
    groups: list[tuple[Pixel, ...]] = []
    for seed in sorted(remaining):
        if seed not in remaining:
            continue
        remaining.remove(seed)
        group = [seed]
        queue = deque([seed])
        while queue:
            for neighbor in adjacency[queue.popleft()]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
                    group.append(neighbor)
        groups.append(tuple(sorted(group)))
    return groups


def _internal_cycles(
    pixels: tuple[Pixel, ...], adjacency: dict[Pixel, tuple[Pixel, ...]]
) -> list[tuple[Pixel, ...]]:
    """Préserve les cycles internes d'une zone de junction, s'il y en a.

    Contracter son arbre couvrant suffit pour regrouper la zone. Chaque liaison
    supplémentaire devient une boucle, avec son chemin raster via cet arbre.
    Ainsi un petit cycle n'est pas effacé par le seul regroupement des nœuds.
    """
    members = set(pixels)
    tree: dict[Pixel, list[Pixel]] = {pixel: [] for pixel in pixels}
    seen = {pixels[0]}
    queue = deque([pixels[0]])
    tree_links: set[PixelLink] = set()
    while queue:
        pixel = queue.popleft()
        for neighbor in adjacency[pixel]:
            if neighbor in members and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
                tree[pixel].append(neighbor)
                tree[neighbor].append(pixel)
                tree_links.add(_link(pixel, neighbor))
    all_links = {_link(p, q) for p in pixels for q in adjacency[p] if q in members}
    cycles: list[tuple[Pixel, ...]] = []
    for start, end in sorted(all_links - tree_links):
        parents: dict[Pixel, Pixel | None] = {end: None}
        queue = deque([end])
        while start not in parents:
            current = queue.popleft()
            for neighbor in tree[current]:
                if neighbor not in parents:
                    parents[neighbor] = current
                    queue.append(neighbor)
        path = [start]
        current = start
        while (parent := parents[current]) is not None:
            current = parent
            path.append(current)
        cycles.append(tuple([start, *reversed(path)]))
    return cycles


def skeleton_to_graph(skeleton: NDArray[np.bool_]) -> SkeletonGraph:
    """Construit les nœuds et toutes les branches d'un squelette booléen 2D.

    Degré utile 0 : isolated ; 1 : endpoint ; 2 : passage ; >=3 : junction.
    Les blocs pleins 2x2 appartiennent également aux zones de junction.
    Seuls les pixels de junction directement voisins sont regroupés. Les
    passages entre deux junctions, même courts, restent des branches distinctes.
    Une boucle sans junction reçoit une ancre loop et une arête fermée, avec
    répétition du premier point en fin de chemin. Un masque vide de traits
    donne un graphe sans nœud ni arête. Le tableau d'entrée n'est pas modifié.
    """
    if skeleton.ndim != 2 or skeleton.dtype != np.bool_ or skeleton.size == 0:
        raise ValueError("Un squelette booléen 2D non vide est requis.")
    height, width = skeleton.shape
    pixels = {(int(row), int(col)) for row, col in zip(*np.nonzero(skeleton))}
    adjacency = _adjacency(pixels)
    groups: list[tuple[NodeKind, tuple[Pixel, ...]]] = [
        ("junction", group) for group in _junction_groups(adjacency)
    ]
    junction_pixels = {pixel for _, members in groups for pixel in members}
    for pixel, neighbors in adjacency.items():
        if len(neighbors) <= 1 and pixel not in junction_pixels:
            groups.append(("endpoint" if neighbors else "isolated", (pixel,)))
    groups.sort(key=lambda item: item[1][0])
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    owner: dict[Pixel, int] = {}
    visited: set[PixelLink] = set()

    def add_node(kind: NodeKind, members: tuple[Pixel, ...]) -> int:
        node_id = len(nodes)
        position = Point(
            x=sum(col for _, col in members) / len(members),
            y=sum(row for row, _ in members) / len(members),
        )
        nodes.append(GraphNode(node_id, kind, position, members))
        for pixel in members:
            owner[pixel] = node_id
        return node_id

    def add_edge(start: int, end: int, path: list[Pixel] | tuple[Pixel, ...]) -> None:
        edges.append(GraphEdge(len(edges), start, end, tuple(_point(p) for p in path)))

    def trace(start: Pixel, neighbor: Pixel) -> None:
        path = [start, neighbor]
        visited.add(_link(start, neighbor))
        previous, current = start, neighbor
        while current not in owner:
            next_pixel = next(p for p in adjacency[current] if p != previous)
            visited.add(_link(current, next_pixel))
            path.append(next_pixel)
            previous, current = current, next_pixel
        add_edge(owner[start], owner[current], path)

    for kind, members in groups:
        add_node(kind, members)
    for node in nodes:
        members = set(node.source_pixels)
        for pixel in node.source_pixels:
            for neighbor in adjacency[pixel]:
                if neighbor in members:
                    visited.add(_link(pixel, neighbor))
        if node.kind == "junction":
            for cycle in _internal_cycles(node.source_pixels, adjacency):
                add_edge(node.id, node.id, cycle)
        for pixel in node.source_pixels:
            for neighbor in adjacency[pixel]:
                if _link(pixel, neighbor) not in visited:
                    trace(pixel, neighbor)

    # Les liens restants appartiennent aux composantes cycliques de degré 2.
    for pixel in sorted(pixels):
        for neighbor in adjacency[pixel]:
            if _link(pixel, neighbor) not in visited:
                add_node("loop", (pixel,))
                trace(pixel, neighbor)
    return SkeletonGraph(width, height, tuple(nodes), tuple(edges))


def _curvilinear_length(points: tuple[Point, ...]) -> float:
    """Longueur curviligne cumulée de la polyligne de points."""
    if len(points) < 2:
        return 0.0
    coords = np.array([(p.x, p.y) for p in points], dtype=np.float64)
    return float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))


def _shortest_path_in_pixels(start_px: Pixel, end_px: Pixel, pixels: set[Pixel]) -> list[Pixel]:
    """Plus court chemin pixel à pixel à travers pixels (8-connexe)."""
    if start_px == end_px:
        return [start_px]
    adj = _adjacency(pixels)
    queue = deque([start_px])
    parents: dict[Pixel, Pixel | None] = {start_px: None}
    while queue:
        curr = queue.popleft()
        if curr == end_px:
            break
        for nbr in adj.get(curr, ()):
            if nbr not in parents:
                parents[nbr] = curr
                queue.append(nbr)
    if end_px not in parents:
        queue = deque([start_px])
        parents = {start_px: None}
        while queue:
            curr = queue.popleft()
            if curr == end_px:
                break
            r, c = curr
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nbr = (r + dr, c + dc)
                    if nbr in pixels and nbr not in parents:
                        parents[nbr] = curr
                        queue.append(nbr)
    path: list[Pixel] = []
    curr_px: Pixel | None = end_px
    while curr_px is not None:
        path.append(curr_px)
        curr_px = parents[curr_px]
    path.reverse()
    return path


def prune_spurs(graph: SkeletonGraph, max_spur_length_px: float = 0.0) -> SkeletonGraph:
    """Élagage déterministe des micro-barbes et dissolution des fausses jonctions."""
    if not math.isfinite(max_spur_length_px) or max_spur_length_px < 0:
        raise ValueError("La tolérance d'élagage doit être finie et positive ou nulle.")
    if max_spur_length_px == 0 or not graph.edges:
        return graph

    nodes_by_id = {node.id: node for node in graph.nodes}

    # 1. Degré total initial de chaque nœud junction
    junction_degrees: dict[int, int] = {}
    for e in graph.edges:
        if e.is_loop:
            if e.start_node_id in nodes_by_id and nodes_by_id[e.start_node_id].kind == "junction":
                junction_degrees[e.start_node_id] = junction_degrees.get(e.start_node_id, 0) + 2
        else:
            for nid in (e.start_node_id, e.end_node_id):
                if nid in nodes_by_id and nodes_by_id[nid].kind == "junction":
                    junction_degrees[nid] = junction_degrees.get(nid, 0) + 1

    # 2. Identification des barbes candidates
    spurs_by_junction: dict[int, list[tuple[float, int, GraphEdge, int]]] = {}
    for e in graph.edges:
        if e.is_loop:
            continue
        n_start = nodes_by_id.get(e.start_node_id)
        n_end = nodes_by_id.get(e.end_node_id)
        if n_start is None or n_end is None:
            continue
        length = _curvilinear_length(e.points)
        if length > max_spur_length_px:
            continue
        if n_start.kind == "endpoint" and n_end.kind == "junction":
            spurs_by_junction.setdefault(n_end.id, []).append((length, e.id, e, n_start.id))
        elif n_start.kind == "junction" and n_end.kind == "endpoint":
            spurs_by_junction.setdefault(n_start.id, []).append((length, e.id, e, n_end.id))

    # 3. Règle de préservation topologique
    pruned_edge_ids: set[int] = set()
    removed_endpoint_ids: set[int] = set()
    for junc_id, candidates in spurs_by_junction.items():
        deg = junction_degrees.get(junc_id, 0)
        candidates.sort(key=lambda item: (item[0], item[1]))
        max_to_prune = max(0, deg - 2)
        for length, edge_id, edge, ep_id in candidates[:max_to_prune]:
            pruned_edge_ids.add(edge_id)
            removed_endpoint_ids.add(ep_id)

    remaining_edges = [e for e in graph.edges if e.id not in pruned_edge_ids]
    remaining_nodes = {n.id: n for n in graph.nodes if n.id not in removed_endpoint_ids}

    # Incidence parmi les arêtes restantes
    incidence: dict[int, list[tuple[GraphEdge, bool]]] = {nid: [] for nid in remaining_nodes}
    for e in remaining_edges:
        if e.is_loop:
            incidence[e.start_node_id].append((e, True))
            incidence[e.start_node_id].append((e, False))
        else:
            incidence[e.start_node_id].append((e, True))
            incidence[e.end_node_id].append((e, False))

    # 4. Dissolution topologique des fausses jonctions
    dissolved_node_ids: set[int] = set()
    for node_id, node in remaining_nodes.items():
        if node.kind == "junction":
            inc = incidence.get(node_id, [])
            if len(inc) == 2 and inc[0][0].id != inc[1][0].id:
                dissolved_node_ids.add(node_id)

    if not pruned_edge_ids and not dissolved_node_ids:
        return graph

    kept_nodes: dict[int, GraphNode] = {
        nid: n for nid, n in remaining_nodes.items() if nid not in dissolved_node_ids
    }

    visited_edge_ids: set[int] = set()
    new_edges_list: list[tuple[int, int, tuple[Point, ...]]] = []

    # Cas 1 : Tracé depuis les nœuds conservés
    for start_node_id in sorted(kept_nodes):
        for edge, is_start in sorted(incidence.get(start_node_id, []), key=lambda item: item[0].id):
            if edge.id in visited_edge_ids:
                continue
            visited_edge_ids.add(edge.id)
            if edge.start_node_id == start_node_id:
                pts = list(edge.points)
                curr_node_id = edge.end_node_id
            else:
                pts = list(reversed(edge.points))
                curr_node_id = edge.start_node_id

            while curr_node_id in dissolved_node_ids:
                J = remaining_nodes[curr_node_id]
                inc_J = incidence[curr_node_id]
                other_edge, _ = inc_J[1] if inc_J[0][0].id == edge.id else inc_J[0]
                visited_edge_ids.add(other_edge.id)
                if other_edge.start_node_id == curr_node_id:
                    next_pts = list(other_edge.points)
                    next_node_id = other_edge.end_node_id
                else:
                    next_pts = list(reversed(other_edge.points))
                    next_node_id = other_edge.start_node_id

                p_end = (int(round(pts[-1].y)), int(round(pts[-1].x)))
                p_start = (int(round(next_pts[0].y)), int(round(next_pts[0].x)))
                if p_end == p_start:
                    pts = pts[:-1] + next_pts
                else:
                    bridge = _shortest_path_in_pixels(p_end, p_start, set(J.source_pixels))
                    bridge_pts = [_point(p) for p in bridge[1:-1]]
                    pts = pts + bridge_pts + next_pts

                curr_node_id = next_node_id
                edge = other_edge

            end_node_id = curr_node_id
            new_edges_list.append((start_node_id, end_node_id, tuple(pts)))

    # Cas 2 : Composantes cycliques fermées de jonctions dissoutes sans nœud restant
    while True:
        unvisited = [e for e in remaining_edges if e.id not in visited_edge_ids]
        if not unvisited:
            break
        seed_edge = min(unvisited, key=lambda e: e.id)
        cycle_nodes: list[GraphNode] = []
        curr_edge = seed_edge
        curr_node_id = seed_edge.start_node_id
        while curr_node_id in dissolved_node_ids:
            node = remaining_nodes[curr_node_id]
            if node in cycle_nodes:
                break
            cycle_nodes.append(node)
            inc_J = incidence[curr_node_id]
            other_edge, _ = inc_J[1] if inc_J[0][0].id == curr_edge.id else inc_J[0]
            curr_node_id = other_edge.end_node_id if other_edge.start_node_id == curr_node_id else other_edge.start_node_id
            curr_edge = other_edge

        anchor_node = min(cycle_nodes, key=lambda n: n.id)
        anchor_pixel = sorted(anchor_node.source_pixels)[0]
        loop_node = GraphNode(anchor_node.id, "loop", _point(anchor_pixel), (anchor_pixel,))
        kept_nodes[anchor_node.id] = loop_node
        dissolved_node_ids.remove(anchor_node.id)

        inc_anchor = sorted(incidence[anchor_node.id], key=lambda item: item[0].id)
        edge, is_start = inc_anchor[0]
        visited_edge_ids.add(edge.id)
        if edge.start_node_id == anchor_node.id:
            pts = list(edge.points)
            curr_node_id = edge.end_node_id
        else:
            pts = list(reversed(edge.points))
            curr_node_id = edge.start_node_id

        while curr_node_id != anchor_node.id:
            J = remaining_nodes[curr_node_id]
            inc_J = incidence[curr_node_id]
            other_edge, _ = inc_J[1] if inc_J[0][0].id == edge.id else inc_J[0]
            visited_edge_ids.add(other_edge.id)
            if other_edge.start_node_id == curr_node_id:
                next_pts = list(other_edge.points)
                next_node_id = other_edge.end_node_id
            else:
                next_pts = list(reversed(other_edge.points))
                next_node_id = other_edge.start_node_id

            p_end = (int(round(pts[-1].y)), int(round(pts[-1].x)))
            p_start = (int(round(next_pts[0].y)), int(round(next_pts[0].x)))
            if p_end == p_start:
                pts = pts[:-1] + next_pts
            else:
                bridge = _shortest_path_in_pixels(p_end, p_start, set(J.source_pixels))
                bridge_pts = [_point(p) for p in bridge[1:-1]]
                pts = pts + bridge_pts + next_pts

            curr_node_id = next_node_id
            edge = other_edge

        p_end = (int(round(pts[-1].y)), int(round(pts[-1].x)))
        p_start = (int(round(pts[0].y)), int(round(pts[0].x)))
        if p_end != p_start:
            bridge = _shortest_path_in_pixels(p_end, p_start, set(anchor_node.source_pixels))
            pts = pts + [_point(p) for p in bridge[1:]]
        if pts[0] != pts[-1]:
            pts.append(pts[0])

        new_edges_list.append((anchor_node.id, anchor_node.id, tuple(pts)))

    # 5. Ré-indexation déterministe
    sorted_nodes = sorted(kept_nodes.values(), key=lambda n: n.id)
    node_id_map = {n.id: idx for idx, n in enumerate(sorted_nodes)}
    final_nodes = tuple(replace(n, id=node_id_map[n.id]) for n in sorted_nodes)

    sorted_edges = sorted(
        new_edges_list,
        key=lambda e: (
            min(node_id_map[e[0]], node_id_map[e[1]]),
            max(node_id_map[e[0]], node_id_map[e[1]]),
            e[2][0].x,
            e[2][0].y,
        ),
    )
    final_edges = tuple(
        GraphEdge(idx, node_id_map[s], node_id_map[t], pts)
        for idx, (s, t, pts) in enumerate(sorted_edges)
    )

    return SkeletonGraph(graph.width, graph.height, final_nodes, final_edges)
