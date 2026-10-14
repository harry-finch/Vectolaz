"""Outil de reconstruction géométrique de dessins au trait (étape 12)."""

from raster2vector.image_io import load_image
from raster2vector.config import VectorizationConfig, load_config
from raster2vector.global_circles import GlobalCircle, GlobalCircleResult, reconstruct_global_circles
from raster2vector.global_arcs import GlobalArc, GlobalArcResult, reconstruct_global_arcs
from raster2vector.preprocessing import automatic_threshold
from raster2vector.bezier import CubicBezier, BezierPath, fit_bezier, evaluate_cubic
from raster2vector.preprocessing import binarize, remove_small_objects, to_grayscale
from raster2vector.skeleton import skeletonize_mask
from raster2vector.graph import GraphNode, GraphEdge, SkeletonGraph, Point, skeleton_to_graph
from raster2vector.geometry import (
    LineFit, LineSegment, Polyline, RawPolyline, GeometryResult, fit_line,
    simplify_polyline, reconstruct_geometry,
    CircleFit, Circle, fit_circle,
    ArcFit, CircularArc, fit_arc,
    Primitive, primitive_source_edge_ids,
)
from raster2vector.svg_export import svg_document, export_svg
from raster2vector.gaps import GapConnection, GapResult, connect_gaps
from raster2vector.junctions import JoinedArc, JunctionAdjustment, Continuation, JunctionResult, refine_junctions
from raster2vector.pipeline import PipelineResult, run_pipeline

__all__ = [
    "GlobalArc", "GlobalArcResult", "reconstruct_global_arcs",
    "GlobalCircle", "GlobalCircleResult", "reconstruct_global_circles",
    "VectorizationConfig", "load_config", "automatic_threshold",
    "JoinedArc", "JunctionAdjustment", "Continuation", "JunctionResult", "refine_junctions",
    "GapConnection", "GapResult", "connect_gaps",
    "CubicBezier", "BezierPath", "fit_bezier", "evaluate_cubic",
    "load_image", "to_grayscale", "binarize", "remove_small_objects", "skeletonize_mask",
    "Point", "GraphNode", "GraphEdge", "SkeletonGraph", "skeleton_to_graph",
    "LineFit", "LineSegment", "Polyline", "RawPolyline", "GeometryResult", "fit_line",
    "simplify_polyline", "reconstruct_geometry",
    "svg_document", "export_svg",
    "CircleFit", "Circle", "fit_circle",
    "ArcFit", "CircularArc", "fit_arc",
    "Primitive", "primitive_source_edge_ids",
    "PipelineResult", "run_pipeline",
]
