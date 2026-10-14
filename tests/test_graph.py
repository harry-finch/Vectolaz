"""Tests de topologie sans dépendance à NetworkX."""

import cv2
import numpy as np
from numpy.typing import NDArray
import pytest
from skimage.measure import euler_number

from raster2vector.graph import skeleton_to_graph
from raster2vector.skeleton import skeletonize_mask


def mask_from_pixels(pixels: list[tuple[int, int]]) -> NDArray[np.bool_]:
    mask = np.zeros((20, 30), dtype=np.bool_)
    for row, col in pixels:
        mask[row, col] = True
    return mask


def test_straight_line_and_xy_coordinates() -> None:
    mask = mask_from_pixels([(4, col) for col in range(8, 19)])
    graph = skeleton_to_graph(mask)
    assert (graph.width, graph.height) == (30, 20)
    assert len(graph.nodes) == 2
    assert all(node.kind == "endpoint" for node in graph.nodes)
    assert len(graph.edges) == 1
    assert [(point.x, point.y) for point in graph.edges[0].points] == [(col, 4) for col in range(8, 19)]
    assert not graph.edges[0].is_loop


@pytest.mark.parametrize("shape, nodes, edges", [("L", 2, 1), ("T", 4, 3), ("cross", 5, 4), ("X", 5, 4)])
def test_angles_and_intersections(shape: str, nodes: int, edges: int) -> None:
    image = np.zeros((21, 31), dtype=np.uint8)
    if shape == "L":
        cv2.line(image, (5, 3), (5, 15), 1, 1)
        cv2.line(image, (5, 15), (24, 15), 1, 1)
    elif shape == "X":
        cv2.line(image, (7, 3), (23, 19), 1, 1)
        cv2.line(image, (7, 19), (23, 3), 1, 1)
    else:
        cv2.line(image, (3, 10), (27, 10), 1, 1)
        cv2.line(image, (15, 10 if shape == "T" else 2), (15, 18), 1, 1)
    mask = image.astype(np.bool_)
    graph = skeleton_to_graph(mask)
    assert len(graph.nodes) == nodes
    assert len(graph.edges) == edges
    assert sum(node.kind == "junction" for node in graph.nodes) == (0 if shape == "L" else 1)
    assert not any(edge.is_loop for edge in graph.edges)


def test_adjacent_junction_pixels_are_grouped() -> None:
    # Deux pixels de jonction adjacents, avec deux branches chacun.
    pixels = [(10, col) for col in range(4, 22)]
    pixels += [(row, 12) for row in range(3, 10)]
    pixels += [(row, 13) for row in range(11, 18)]
    graph = skeleton_to_graph(mask_from_pixels(pixels))
    junctions = [node for node in graph.nodes if node.kind == "junction"]
    assert len(junctions) == 1
    assert junctions[0].source_pixels == ((10, 12), (10, 13))
    assert len(graph.nodes) == 5
    assert len(graph.edges) == 4


def test_separate_junctions_keep_short_connecting_branch() -> None:
    pixels = [(10, col) for col in range(4, 24)]
    pixels += [(row, 10) for row in range(3, 10)]
    pixels += [(row, 12) for row in range(11, 18)]
    graph = skeleton_to_graph(mask_from_pixels(pixels))
    junction_ids = {node.id for node in graph.nodes if node.kind == "junction"}
    assert len(junction_ids) == 2
    assert len(graph.edges) == 5
    connecting = [edge for edge in graph.edges if edge.start_node_id in junction_ids and edge.end_node_id in junction_ids]
    assert len(connecting) == 1
    assert len(connecting[0].points) == 3


def test_closed_circle_without_endpoints() -> None:
    image = np.zeros((61, 81), dtype=np.uint8)
    cv2.circle(image, (40, 30), 20, 1, 5)
    mask = skeletonize_mask(image.astype(np.bool_))
    graph = skeleton_to_graph(mask)
    assert len(graph.nodes) == len(graph.edges) == 1
    assert graph.nodes[0].kind == "loop"
    edge = graph.edges[0]
    assert edge.is_loop
    assert edge.points[0] == edge.points[-1]
    assert len(edge.points) == mask.sum() + 1


def test_loop_attached_to_a_branch() -> None:
    image = np.zeros((21, 31), dtype=np.uint8)
    cv2.rectangle(image, (8, 5), (20, 15), 1, 1)
    cv2.line(image, (3, 10), (8, 10), 1, 1)
    graph = skeleton_to_graph(image.astype(np.bool_))
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 2
    assert sum(edge.is_loop for edge in graph.edges) == 1


def test_parallel_branches_are_not_merged() -> None:
    image = np.zeros((21, 31), dtype=np.uint8)
    cv2.rectangle(image, (8, 5), (20, 15), 1, 1)
    cv2.line(image, (3, 10), (8, 10), 1, 1)
    cv2.line(image, (20, 10), (25, 10), 1, 1)
    graph = skeleton_to_graph(image.astype(np.bool_))
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 4
    junctions = {node.id for node in graph.nodes if node.kind == "junction"}
    assert sum(edge.start_node_id in junctions and edge.end_node_id in junctions for edge in graph.edges) == 2


def test_filled_square_does_not_invent_a_cycle() -> None:
    # Un carré plein 2x2 avec quatre branches n'a aucun trou raster.
    pixels = [(8, 12), (8, 13), (9, 12), (9, 13),
              (7, 12), (8, 14), (9, 11), (10, 13)]
    mask = mask_from_pixels(pixels)
    graph = skeleton_to_graph(mask)
    assert sum(node.kind == "junction" for node in graph.nodes) == 1
    assert sum(edge.is_loop for edge in graph.edges) == 0
    assert len(graph.edges) - len(graph.nodes) + 1 == 0


def test_real_hole_inside_a_junction_cluster_is_preserved() -> None:
    # Zone de junction épaisse, avec un vrai trou d'un pixel au centre.
    mask = np.zeros((15, 15), dtype=np.bool_)
    mask[4:9, 4:9] = True
    mask[6, 6] = False
    graph = skeleton_to_graph(mask)
    assert len(graph.nodes) == len(graph.edges) == 1
    assert graph.nodes[0].kind == "junction"
    assert graph.edges[0].is_loop
    assert graph.edges[0].points[0] == graph.edges[0].points[-1]
    assert len(graph.nodes) - len(graph.edges) == euler_number(mask, connectivity=2)


def test_empty_isolated_and_disconnected_components() -> None:
    mask = np.zeros((20, 30), dtype=np.bool_)
    graph = skeleton_to_graph(mask)
    assert graph.nodes == graph.edges == ()
    mask[0, 0] = True
    mask[10, 5:8] = True
    graph = skeleton_to_graph(mask)
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 1
    assert sum(node.kind == "isolated" for node in graph.nodes) == 1


def test_determinism_path_coverage_and_input_unchanged() -> None:
    image = np.zeros((41, 61), dtype=np.uint8)
    cv2.line(image, (5, 20), (55, 20), 1, 5)
    cv2.line(image, (30, 5), (30, 35), 1, 5)
    mask = skeletonize_mask(image.astype(np.bool_))
    before = mask.copy()
    graph = skeleton_to_graph(mask)
    assert graph == skeleton_to_graph(mask)
    assert np.array_equal(mask, before)
    covered = {pixel for node in graph.nodes for pixel in node.source_pixels}
    for edge in graph.edges:
        pixels = [(int(point.y), int(point.x)) for point in edge.points]
        covered.update(pixels)
        assert pixels[0] in graph.nodes[edge.start_node_id].source_pixels
        assert pixels[-1] in graph.nodes[edge.end_node_id].source_pixels
        for a, b in zip(pixels, pixels[1:]):
            assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1
    assert covered == set(zip(*np.nonzero(mask)))
    assert len(graph.nodes) - len(graph.edges) == euler_number(mask, connectivity=2)


def test_all_3_by_3_patterns_preserve_components_and_holes() -> None:
    # Couvre exhaustivement les cas locaux, dont les carrés pleins et les
    # diagonales : ni faux trou ni perte de connexion ne doit apparaître.
    for bits in range(512):
        mask = np.array([(bits >> index) & 1 for index in range(9)], dtype=np.bool_).reshape(3, 3)
        graph = skeleton_to_graph(mask)
        remaining = {node.id for node in graph.nodes}
        neighbors = {node.id: set() for node in graph.nodes}
        for edge in graph.edges:
            neighbors[edge.start_node_id].add(edge.end_node_id)
            neighbors[edge.end_node_id].add(edge.start_node_id)
        components = 0
        while remaining:
            components += 1
            pending = [remaining.pop()]
            while pending:
                for neighbor in neighbors[pending.pop()]:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        pending.append(neighbor)
        assert components == cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1, bits
        assert len(graph.nodes) - len(graph.edges) == euler_number(mask, connectivity=2), bits


@pytest.mark.parametrize("mask", [np.zeros((3, 3), dtype=np.uint8), np.zeros((0, 3), dtype=np.bool_), np.zeros((3, 3, 1), dtype=np.bool_)])
def test_invalid_skeleton(mask: NDArray) -> None:
    with pytest.raises(ValueError, match="squelette booléen"):
        skeleton_to_graph(mask)
