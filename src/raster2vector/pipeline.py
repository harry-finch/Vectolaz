"""Pipeline de vectorisation autonome, pur et indépendant des entrées/sorties."""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from raster2vector.config import VectorizationConfig
from raster2vector.gaps import GapResult, connect_gaps
from raster2vector.geometry import GeometryResult, reconstruct_geometry
from raster2vector.global_arcs import GlobalArcResult, reconstruct_global_arcs
from raster2vector.global_circles import GlobalCircleResult, reconstruct_global_circles
from raster2vector.graph import SkeletonGraph, skeleton_to_graph
from raster2vector.junctions import JunctionResult, refine_junctions
from raster2vector.preprocessing import (
    automatic_threshold,
    binarize,
    remove_small_objects,
    to_grayscale,
)
from raster2vector.skeleton import skeletonize_mask


@dataclass(frozen=True)
class PipelineResult:
    """Tous les artefacts intermédiaires et finaux du traitement d'une image."""

    grayscale: NDArray[np.uint8]
    threshold: int
    binary: NDArray[np.bool_]
    cleaned: NDArray[np.bool_]
    skeleton: NDArray[np.bool_]
    graph: SkeletonGraph
    gaps: GapResult
    geometry_before: GeometryResult
    junctions: JunctionResult | None
    geometry_local: GeometryResult
    global_circles: GlobalCircleResult | None
    global_arcs: GlobalArcResult | None
    geometry: GeometryResult


def run_pipeline(
    image: NDArray[np.uint8] | NDArray[np.uint16],
    config: VectorizationConfig,
) -> PipelineResult:
    """Exécute la chaîne de traitement complète sans effet de bord ni I/O disque."""
    grayscale = to_grayscale(image)
    threshold = automatic_threshold(grayscale) if config.threshold == "auto" else config.threshold
    binary = binarize(grayscale, threshold, config.invert)
    cleaned = remove_small_objects(binary, config.min_component_size)
    skeleton = skeletonize_mask(cleaned)
    graph = skeleton_to_graph(skeleton)
    gaps = connect_gaps(graph, config.gap_tolerance)
    geometry_before = reconstruct_geometry(
        gaps.graph, config.geometry_tolerance, config.polyline_tolerance
    )
    junctions = (
        refine_junctions(gaps.graph, geometry_before, config.geometry_tolerance)
        if config.junction_cleanup
        else None
    )
    geometry_local = geometry_before if junctions is None else junctions.geometry
    geometry = geometry_local

    global_circles = (
        reconstruct_global_circles(gaps.graph, geometry_local, config.geometry_tolerance)
        if config.global_circles
        else None
    )
    if global_circles is not None:
        geometry = global_circles.geometry

    global_arcs = (
        reconstruct_global_arcs(gaps.graph, geometry, config.geometry_tolerance)
        if config.global_arcs
        else None
    )
    if global_arcs is not None:
        geometry = global_arcs.geometry

    return PipelineResult(
        grayscale=grayscale,
        threshold=threshold,
        binary=binary,
        cleaned=cleaned,
        skeleton=skeleton,
        graph=graph,
        gaps=gaps,
        geometry_before=geometry_before,
        junctions=junctions,
        geometry_local=geometry_local,
        global_circles=global_circles,
        global_arcs=global_arcs,
        geometry=geometry,
    )
