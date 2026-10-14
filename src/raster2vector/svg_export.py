"""Export SVG par XML standard, sans modification des primitives géométriques."""

import base64
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

from raster2vector.coordinates import Point
from raster2vector.bezier import BezierPath
from raster2vector.global_circles import GlobalCircle
from raster2vector.global_arcs import GlobalArc
from raster2vector.geometry import GeometryResult, LineSegment, Polyline, RawPolyline, Circle, CircularArc

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def _number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("Les coordonnées SVG doivent être finies.")
    return "0" if value == 0 else format(value, ".15g")


def _coordinates(point: Point) -> str:
    return f"{_number(point.x)} {_number(point.y)}"


def svg_document(
    geometry: GeometryResult,
    stroke_width: float = 1.0,
    *,
    background_png: bytes | None = None,
) -> str:
    """Retourne un SVG autonome, en unités pixels, de la taille du raster.

    Les chemins gardent les coordonnées sources (x, y) dans un groupe translaté
    de (0.5, 0.5) pour aligner leurs axes sur les centres des pixels de l'image.
    Le fond est transparent, sauf si un PNG de mêmes dimensions est fourni.
    Les boucles explicitement fermées sont exportées avec Z. Une boucle
    topologique dont les contacts raster diffèrent conserve son chemin ouvert,
    sans ajouter un raccord géométrique arbitraire dans la zone d'intersection.
    """
    if geometry.width <= 0 or geometry.height <= 0:
        raise ValueError("Les dimensions SVG doivent être strictement positives.")
    if not math.isfinite(stroke_width) or stroke_width <= 0:
        raise ValueError("L'épaisseur SVG doit être finie et strictement positive.")
    root = ET.Element(f"{{{SVG_NS}}}svg", {
        "version": "1.1", "width": f"{geometry.width}px", "height": f"{geometry.height}px",
        "viewBox": f"0 0 {geometry.width} {geometry.height}",
    })
    if background_png is not None:
        if (len(background_png) < 24 or background_png[:8] != b"\x89PNG\r\n\x1a\n"
                or background_png[12:16] != b"IHDR"):
            raise ValueError("Le fond SVG doit être un PNG.")
        dimensions = struct.unpack(">II", background_png[16:24])
        if dimensions != (geometry.width, geometry.height):
            raise ValueError("Le fond PNG et la géométrie doivent avoir les mêmes dimensions.")
        uri = "data:image/png;base64," + base64.b64encode(background_png).decode("ascii")
        ET.SubElement(root, f"{{{SVG_NS}}}image", {
            "id": "raster-background", "x": "0", "y": "0",
            "width": str(geometry.width), "height": str(geometry.height),
            "preserveAspectRatio": "none", "opacity": "0.35",
            "href": uri, f"{{{XLINK_NS}}}href": uri,
        })
    group = ET.SubElement(root, f"{{{SVG_NS}}}g", {
        "id": "vectors", "fill": "none", "stroke": "black",
        "stroke-width": _number(stroke_width),
        "stroke-linecap": "round", "stroke-linejoin": "round",
        "transform": "translate(0.5 0.5)",
    })
    for primitive in geometry.primitives:
        if isinstance(primitive, BezierPath):
            if not primitive.segments or any(a.end != b.start for a, b in zip(primitive.segments, primitive.segments[1:])):
                raise ValueError("Le chemin Bézier SVG doit être non vide et continu.")
            commands = [f"M {_coordinates(primitive.segments[0].start)}"]
            commands.extend(f"C {_coordinates(c.control1)} {_coordinates(c.control2)} {_coordinates(c.end)}"
                            for c in primitive.segments)
            if primitive.is_loop and primitive.segments[-1].end == primitive.segments[0].start:
                commands.append("Z")
            ET.SubElement(group, f"{{{SVG_NS}}}path", {
                "id": f"edge-{primitive.source_edge_id}", "d": " ".join(commands),
                "fill": "none", "stroke": "black",
            })
            continue
        if isinstance(primitive, CircularArc):
            if (not math.isfinite(primitive.radius) or primitive.radius <= 0
                    or not math.isfinite(primitive.start_angle_rad)
                    or not math.isfinite(primitive.end_angle_rad)
                    or not 0 < abs(primitive.sweep_angle_rad) < 2 * math.pi):
                raise ValueError("Les paramètres de l'arc SVG sont invalides.")
            large_arc = primitive.large_arc_flag
            sweep = primitive.sweep_flag
            radius = _number(primitive.radius)
            commands = (f"M {_coordinates(primitive.start)} A {radius} {radius} 0 "
                        f"{large_arc} {sweep} {_coordinates(primitive.end)}")
            attributes = {
                "id": f"global-arc-{primitive.source_edge_id}" if isinstance(primitive, GlobalArc) else f"edge-{primitive.source_edge_id}", "d": commands,
                "fill": "none", "stroke": "black",
            }
            if isinstance(primitive, GlobalArc):
                attributes["data-source-edges"] = " ".join(str(i) for i in primitive.source_edge_ids)
                attributes["data-support-edges"] = " ".join(str(i) for i in primitive.support_edge_ids)
            ET.SubElement(group, f"{{{SVG_NS}}}path", attributes)
            continue
        if isinstance(primitive, Circle):
            if not math.isfinite(primitive.radius) or primitive.radius <= 0:
                raise ValueError("Le rayon SVG doit être fini et strictement positif.")
            attributes = {
                "id": f"global-circle-{primitive.source_edge_id}" if isinstance(primitive, GlobalCircle) else f"edge-{primitive.source_edge_id}",
                "cx": _number(primitive.center.x), "cy": _number(primitive.center.y),
                "r": _number(primitive.radius), "fill": "none", "stroke": "black",
            }
            if isinstance(primitive, GlobalCircle):
                attributes["data-source-edges"] = " ".join(str(i) for i in primitive.source_edge_ids)
            ET.SubElement(group, f"{{{SVG_NS}}}circle", attributes)
            continue
        if isinstance(primitive, LineSegment):
            points = (primitive.start, primitive.end)
            closed = False
        elif isinstance(primitive, (Polyline, RawPolyline)):
            points = primitive.points
            closed = (isinstance(primitive, RawPolyline) and primitive.is_loop
                      and len(points) > 2 and points[0] == points[-1])
        else:
            raise TypeError(f"Primitive géométrique non supportée : {type(primitive)!r}")
        if len(points) < 2:
            continue
        path_points = points[:-1] if closed else points
        commands = [f"M {_coordinates(path_points[0])}"]
        commands.extend(f"L {_coordinates(point)}" for point in path_points[1:])
        if closed:
            commands.append("Z")
        ET.SubElement(group, f"{{{SVG_NS}}}path", {
            "id": f"edge-{primitive.source_edge_id}",
            "d": " ".join(commands), "fill": "none", "stroke": "black",
        })
    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def export_svg(
    geometry: GeometryResult,
    path: str | Path,
    stroke_width: float = 1.0,
    *,
    background_png: bytes | None = None,
) -> None:
    """Écrit le document UTF-8 et crée son dossier parent si nécessaire.

    Valide les données avant l'écriture. Un fichier de sortie existant est
    remplacé ; l'appelant doit protéger les chemins des fichiers sources.
    """
    document = svg_document(geometry, stroke_width, background_png=background_png)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
