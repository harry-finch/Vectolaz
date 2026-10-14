"""Topologie du squelette : voisins utiles, nœuds logiques et branches.

Les pixels internes utilisent (row, col). La conversion vers Point(x, y)
est explicite ; aucun ajustement géométrique n'est effectué.
"""

from collections import deque
from dataclasses import dataclass
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
