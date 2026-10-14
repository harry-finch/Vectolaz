"""Rendus inspectables des étapes de traitement, indépendants des algorithmes."""

from pathlib import Path
import math

from raster2vector.bezier import BezierPath, evaluate_cubic

import cv2
import numpy as np
from numpy.typing import NDArray

from raster2vector.image_io import save_png
from raster2vector.graph import SkeletonGraph
from raster2vector.gaps import GapResult
from raster2vector.junctions import JunctionResult
from raster2vector.geometry import GeometryResult, LineSegment, Polyline, Circle, CircularArc
from raster2vector.preprocessing import to_grayscale
from raster2vector.svg_export import export_svg


def save_preprocessing_debug(
    directory: Path,
    grayscale: NDArray[np.uint8],
    binary: NDArray[np.bool_],
    cleaned: NDArray[np.bool_],
) -> None:
    """Écrit les gris et les masques affichés en traits noirs sur fond blanc.

    Le codage PNG d'affichage (trait=0, fond=255) est volontairement distinct
    du masque utilisé par les algorithmes (trait=True, fond=False).
    Les fichiers de debug du même nom sont remplacés à chaque exécution.
    """
    directory.mkdir(parents=True, exist_ok=True)
    save_png(directory / "grayscale.png", grayscale)
    save_png(directory / "binary.png", np.where(binary, 0, 255).astype(np.uint8))
    save_png(directory / "cleaned.png", np.where(cleaned, 0, 255).astype(np.uint8))


def save_skeleton_debug(directory: Path, skeleton: NDArray[np.bool_]) -> None:
    """Écrit skeleton.png en traits noirs d'un pixel sur fond blanc."""
    directory.mkdir(parents=True, exist_ok=True)
    save_png(directory / "skeleton.png", np.where(skeleton, 0, 255).astype(np.uint8))


def save_graph_debug(
    directory: Path, skeleton: NDArray[np.bool_], graph: SkeletonGraph,
    *, filename: str = "graph.png",
) -> None:
    """Dessine graph.png : branches colorées E#, nœuds N# et squelette.

    Vert : endpoint ; rouge : junction ; violet : ancre de boucle ; noir :
    pixel isolé. Les marqueurs/étiquettes sont un rendu de debug uniquement.
    Le repère et les dimensions restent ceux du raster source.
    """
    if skeleton.shape != (graph.height, graph.width):
        raise ValueError("Le squelette et le graphe doivent avoir les mêmes dimensions.")
    canvas = cv2.cvtColor(np.where(skeleton, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    palette = [(200, 100, 0), (180, 0, 180), (0, 140, 220), (160, 120, 20), (40, 120, 160)]
    occupied: list[tuple[int, int, int, int]] = [
        (round(n.position.x) - 4, round(n.position.y) - 4,
         round(n.position.x) + 4, round(n.position.y) + 4) for n in graph.nodes
    ]

    def label(text: str, col: int, row: int) -> None:
        (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        # Placement glouton autour du point pour éviter les identifiants
        # superposés dans les zones denses, sans modifier le graphe.
        offsets = [(4, -4), (4, height + 5), (-width - 4, -4), (-width - 4, height + 5)]
        for radius in (16, 32, 48, 64, 80, 96, 128):
            offsets.extend((dx, dy) for dx, dy in (
                (radius, 0), (-radius - width, 0), (0, radius), (0, -radius),
                (radius, radius), (-radius - width, radius),
                (radius, -radius), (-radius - width, -radius),
            ))
        max_x = max(0, graph.width - 1)
        max_y = max(0, graph.height - 1)
        pref_x = min(col, graph.width - width - 1) if graph.width > width else 0
        pref_y = max(height, min(row, graph.height - baseline - 1)) if graph.height > height + baseline else max_y
        origin = (max(0, min(pref_x, max_x)), max(0, min(pref_y, max_y)))
        for dx, dy in offsets:
            x, y = col + dx, row + dy
            box = (x - 2, y - height - 2, x + width + 2, y + baseline + 2)
            if box[0] < 0 or box[1] < 0 or box[2] >= graph.width or box[3] >= graph.height:
                continue
            if all(box[2] < other[0] or box[0] > other[2]
                   or box[3] < other[1] or box[1] > other[3] for other in occupied):
                origin = (x, y)
                occupied.append(box)
                break
        if abs(origin[0] - col) > 10 or abs(origin[1] - row) > 15:
            cv2.line(canvas, (col, row), (origin[0] + width // 2, origin[1] - height // 2),
                     (190, 190, 190), 1, cv2.LINE_AA)
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

    for edge in graph.edges:
        points = np.array([(round(p.x), round(p.y)) for p in edge.points], dtype=np.int32)
        cv2.polylines(canvas, [points], False, palette[edge.id % len(palette)], 1)
    colors = {"endpoint": (0, 160, 0), "junction": (0, 0, 220),
              "loop": (180, 0, 180), "isolated": (0, 0, 0)}
    for node in graph.nodes:
        center = (round(node.position.x), round(node.position.y))
        cv2.circle(canvas, center, 3, colors[node.kind], -1)
    for node in graph.nodes:
        label(f"N{node.id}", round(node.position.x), round(node.position.y))
    for edge in graph.edges:
        middle = edge.points[len(edge.points) // 2]
        label(f"E{edge.id}", round(middle.x), round(middle.y))
    directory.mkdir(parents=True, exist_ok=True)
    save_png(directory / filename, canvas)


def save_gap_debug(directory: Path, skeleton: NDArray[np.bool_], result: GapResult) -> None:
    """Ponts acceptés en magenta sur le squelette original inchangé."""
    if skeleton.shape != (result.graph.height, result.graph.width):
        raise ValueError("Le squelette et les raccords doivent avoir les mêmes dimensions.")
    canvas = cv2.cvtColor(np.where(skeleton, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for connection in result.connections:
        start, end = ((round(p.x), round(p.y)) for p in (connection.start, connection.end))
        cv2.line(canvas, start, end, (200, 0, 180), 1, cv2.LINE_AA)
        for point in (start, end):
            cv2.circle(canvas, point, 2, (200, 0, 180), 1, cv2.LINE_AA)
    directory.mkdir(parents=True, exist_ok=True)
    save_png(directory / "gaps.png", canvas)


def save_geometry_debug(
    directory: Path,
    original: NDArray[np.uint8] | NDArray[np.uint16],
    geometry: GeometryResult,
    *, filename: str = "geometry.png",
) -> None:
    """Superpose segments rouges, arcs orange, cercles violets et polylignes vertes.

    Le raster en niveaux de gris est éclairci pour rendre les lignes lisibles.
    Les coordonnées fractionnaires utilisent le dessin OpenCV à précision
    subpixel (8 bits fractionnaires), sans arrondir les primitives elles-mêmes.
    """
    grayscale = to_grayscale(original)
    if grayscale.shape != (geometry.height, geometry.width):
        raise ValueError("L'image et la géométrie doivent avoir les mêmes dimensions.")
    background = np.rint(0.5 * grayscale + 127.5).astype(np.uint8)
    canvas = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    shift = 8
    scale = 1 << shift
    # Les branches brutes sont dessinées avant les segments pour que ceux-ci
    # restent visibles près des intersections.
    for primitive in sorted(geometry.primitives, key=lambda p: isinstance(p, LineSegment)):
        if isinstance(primitive, BezierPath):
            for curve in primitive.segments:
                controls = (curve.start, curve.control1, curve.control2, curve.end)
                length_bound = sum(math.hypot(a.x-b.x, a.y-b.y) for a, b in zip(controls, controls[1:]))
                points = np.rint(evaluate_cubic(curve, np.linspace(0, 1, max(2, math.ceil(length_bound)+1)))*scale).astype(np.int32)
                cv2.polylines(canvas, [points], False, (180, 120, 0), 1, cv2.LINE_AA, shift=shift)
                for point in (curve.start, curve.end):
                    cv2.circle(canvas, (round(point.x*scale), round(point.y*scale)), 2*scale,
                               (180, 120, 0), -1, cv2.LINE_AA, shift=shift)
            continue
        if isinstance(primitive, CircularArc):
            # Le debug échantillonne à au plus un pixel de longueur d'arc ;
            # la primitive et l'export SVG restent des arcs exacts.
            count = max(2, math.ceil(primitive.radius * abs(primitive.sweep_angle_rad)) + 1)
            angles = np.linspace(primitive.start_angle_rad, primitive.end_angle_rad, count)
            points = np.rint(np.column_stack((primitive.center.x + primitive.radius * np.cos(angles),
                                              primitive.center.y + primitive.radius * np.sin(angles))) * scale).astype(np.int32)
            cv2.polylines(canvas, [points], False, (0, 130, 230), 1, cv2.LINE_AA, shift=shift)
            continue
        if isinstance(primitive, Circle):
            center = (round(primitive.center.x * scale), round(primitive.center.y * scale))
            cv2.circle(canvas, center, round(primitive.radius * scale), (180, 0, 180),
                       1, cv2.LINE_AA, shift=shift)
            continue
        if isinstance(primitive, LineSegment):
            points = (primitive.start, primitive.end)
            color = (0, 0, 230)
        else:
            points = primitive.points
            color = (0, 150, 0) if isinstance(primitive, Polyline) else (220, 100, 0)
        if len(points) < 2:
            continue
        raster_points = np.rint([(p.x * scale, p.y * scale) for p in points]).astype(np.int32)
        cv2.polylines(canvas, [raster_points], False, color, 1, cv2.LINE_AA, shift=shift)
        if isinstance(primitive, Polyline):
            for point in raster_points:
                cv2.circle(canvas, tuple(int(value) for value in point), scale * 2,
                           color, -1, cv2.LINE_AA, shift=shift)
    directory.mkdir(parents=True, exist_ok=True)
    save_png(directory / filename, canvas)


def save_junction_debug(directory: Path, original: NDArray[np.uint8] | NDArray[np.uint16],
                        graph: SkeletonGraph, result: JunctionResult) -> None:
    """Jonctions acceptées vertes, refusées rouges, déplacements magenta."""
    grayscale = to_grayscale(original)
    if grayscale.shape != (graph.height, graph.width):
        raise ValueError("L'image et les jonctions doivent avoir les mêmes dimensions.")
    canvas = cv2.cvtColor(np.rint(0.5*grayscale+127.5).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for adjustment in result.adjustments:
        old = (round(adjustment.original_position.x), round(adjustment.original_position.y))
        new = (round(adjustment.position.x), round(adjustment.position.y))
        cv2.line(canvas, old, new, (180, 0, 180), 1, cv2.LINE_AA)
        cv2.circle(canvas, new, 3, (0, 150, 0), 1, cv2.LINE_AA)
    for node_id in result.rejected_node_ids:
        p = graph.nodes[node_id].position
        cv2.circle(canvas, (round(p.x), round(p.y)), 3, (0, 0, 230), 1, cv2.LINE_AA)
    directory.mkdir(parents=True, exist_ok=True)
    save_png(directory / "junctions.png", canvas)


def save_svg_overlay(
    directory: Path,
    original: NDArray[np.uint8] | NDArray[np.uint16],
    geometry: GeometryResult,
    stroke_width: float = 1.0,
    *, filename: str = "overlay.svg",
) -> None:
    """Écrit overlay.svg, avec le raster original embarqué et les strokes noirs."""
    if original.shape[:2] != (geometry.height, geometry.width):
        raise ValueError("L'image et la géométrie doivent avoir les mêmes dimensions.")
    try:
        success, encoded = cv2.imencode(".png", original)
    except cv2.error as exc:
        raise ValueError("Impossible d'encoder le raster pour le SVG de comparaison.") from exc
    if not success:
        raise ValueError("Impossible d'encoder le raster pour le SVG de comparaison.")
    export_svg(geometry, directory / filename, stroke_width, background_png=encoded.tobytes())
