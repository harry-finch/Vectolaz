"""CLI consolidée : réglages TOML, seuil manuel/automatique et diagnostics."""

import argparse
from importlib.metadata import version
from pathlib import Path
import sys
from collections.abc import Sequence
from typing import Literal

from raster2vector.image_io import load_image
from raster2vector.config import VectorizationConfig, load_config
from raster2vector.debug import (
    save_preprocessing_debug, save_skeleton_debug, save_graph_debug, save_geometry_debug,
    save_svg_overlay,
    save_gap_debug,
    save_junction_debug,
)
from raster2vector.geometry import LineSegment, Polyline, RawPolyline, Circle, CircularArc, reconstruct_geometry
from raster2vector.bezier import BezierPath
from raster2vector.graph import skeleton_to_graph
from raster2vector.gaps import connect_gaps
from raster2vector.junctions import refine_junctions
from raster2vector.global_circles import reconstruct_global_circles
from raster2vector.global_arcs import reconstruct_global_arcs
from raster2vector.preprocessing import automatic_threshold, binarize, remove_small_objects, to_grayscale
from raster2vector.skeleton import skeletonize_mask
from raster2vector.svg_export import export_svg


def _threshold(value: str) -> int | Literal["auto"]:
    if value == "auto":
        return "auto"
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Le seuil doit être auto ou un entier entre 0 et 255.") from exc
    if not 0 <= number <= 255:
        raise argparse.ArgumentTypeError("Le seuil doit être auto ou un entier entre 0 et 255.")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Construit une aide autonome, sans ouvrir de fichier de configuration."""
    parser = argparse.ArgumentParser(
        description="Vectorise les axes de dessins au trait en segments, cercles, arcs et Bézier SVG.",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Exemples :
  raster2vector dessin.png --threshold auto --output dessin.svg
  raster2vector dessin.png --config reglages.toml --geometry-tolerance 3 --output dessin.svg
  raster2vector dessin.png --gap-tolerance 3 --debug-dir debug --debug-svg

Priorité des réglages : options explicites > fichier TOML > valeurs par défaut.
Les distances sont en pixels du raster source ; 0 désactive les raccordements.
Sans --output ni --debug-dir, le traitement reste en mémoire.
Codes de sortie : 0 succès ; 1 erreur de fichier/traitement ; 2 arguments ou configuration invalides.""",
    )
    parser.add_argument("-h", "--help", action="help", help="Affiche cette aide et quitte.")
    parser.add_argument("--version", action="version", version=f"raster2vector {version('raster2vector')}",
                        help="Affiche la version installée et quitte.")
    parser.add_argument("--config", type=Path, metavar="FICHIER.toml",
                        help="Charge les réglages de la table TOML [vectorization].")
    parser.add_argument("input", type=Path, help="Chemin de l'image PNG à ouvrir.")
    parser.add_argument("--output", type=Path, help="Chemin du SVG final (optionnel).")
    parser.add_argument("--global-circles", action="store_true",
                        help="Recherche des cercles complets répartis sur plusieurs branches (optionnel).")
    parser.add_argument("--global-arcs", action="store_true",
                        help="Privilégie les longs arcs à travers les intersections ; active aussi --global-circles.")
    cleanup = parser.add_mutually_exclusive_group()
    cleanup.add_argument("--no-junction-cleanup", action="store_true", default=None,
                        help="Conserve les contacts de l'étape 10 pour comparaison.")
    cleanup.add_argument("--junction-cleanup", dest="no_junction_cleanup", action="store_false", default=None,
                        help="Active l'affinage des jonctions (défaut : activé).")
    parser.add_argument(
        "--stroke-width", type=float, default=None, metavar="PIXELS",
        help="Épaisseur des traits SVG, strictement positive (défaut : 1 px).",
    )
    parser.add_argument(
        "--debug-svg", action="store_true",
        help="Ajoute overlay.svg avec le raster embarqué dans --debug-dir.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        help="Enregistre les masques, squelette, graphe, géométrie et contrôles des jonctions.",
    )
    parser.add_argument(
        "--geometry-tolerance", type=float, default=None, metavar="PIXELS",
        help="Erreur maximale pour accepter un segment, cercle, arc ou Bézier (défaut : 2.0 px).",
    )
    parser.add_argument(
        "--polyline-tolerance", type=float, default=None, metavar="PIXELS",
        help="Écart maximal pour simplifier les branches non droites (défaut : 2.0 px).",
    )
    parser.add_argument(
        "--gap-tolerance", type=float, default=None, metavar="PIXELS",
        help="Distance maximale entre extrémités compatibles (défaut : 0, désactivé).",
    )
    parser.add_argument(
        "--threshold", type=_threshold, default=None, metavar="auto|0..255",
        help="Seuil inclusif manuel, ou auto pour Otsu (défaut : 128).",
    )
    parser.add_argument(
        "--invert", choices=("auto", "yes", "no"), default=None,
        help="auto : fond majoritaire au bord ; yes : traits sombres ; no : traits clairs.",
    )
    parser.add_argument(
        "--min-component-size", type=int, default=None, metavar="PIXELS",
        help="Supprime les objets plus petits que cette surface (défaut : 0, désactivé).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Exécute la CLI ; renvoie 0 en cas de succès et 1 en cas d'erreur I/O."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.global_circles = args.global_circles or args.global_arcs
    try:
        settings = load_config(args.config) if args.config is not None else VectorizationConfig()
        settings = settings.with_overrides(
            threshold=args.threshold, invert=args.invert, min_component_size=args.min_component_size,
            geometry_tolerance=args.geometry_tolerance, polyline_tolerance=args.polyline_tolerance,
            gap_tolerance=args.gap_tolerance, stroke_width=args.stroke_width,
            junction_cleanup=None if args.no_junction_cleanup is None else not args.no_junction_cleanup,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    for name in ("threshold", "invert", "min_component_size", "geometry_tolerance", "polyline_tolerance", "gap_tolerance", "stroke_width"):
        setattr(args, name, getattr(settings, name))
    args.no_junction_cleanup = not settings.junction_cleanup
    if args.debug_svg and args.debug_dir is None:
        parser.error("--debug-svg nécessite --debug-dir.")
    try:
        image = load_image(args.input)
        destinations = [] if args.output is None else [args.output]
        if args.debug_dir is not None:
            names = ["grayscale.png", "binary.png", "cleaned.png", "skeleton.png", "graph.png", "geometry.png"]
            if args.global_circles:
                names.append("geometry_local.png")
            if not args.no_junction_cleanup:
                names.extend(("geometry_before.png", "junctions.png"))
            if args.gap_tolerance > 0:
                names.extend(("gaps.png", "graph_connected.png"))
            if args.debug_svg:
                names.append("overlay.svg")
                if args.global_circles:
                    names.append("overlay_local.svg")
                if not args.no_junction_cleanup:
                    names.append("overlay_before.svg")
            destinations.extend(args.debug_dir / name for name in names)
        resolved = [path.resolve() for path in destinations]
        if len(set(resolved)) != len(resolved):
            raise ValueError("Les fichiers de sortie et de debug doivent avoir des chemins distincts.")
        for destination in destinations:
            if (destination.resolve() == args.input.resolve()
                    or (destination.exists() and destination.samefile(args.input))):
                raise ValueError("Un fichier de sortie écraserait l'image source.")
            if args.config is not None and (destination.resolve() == args.config.resolve()
                    or (destination.exists() and destination.samefile(args.config))):
                raise ValueError("Un fichier de sortie écraserait la configuration source.")
        grayscale = to_grayscale(image)
        threshold = automatic_threshold(grayscale) if args.threshold == "auto" else args.threshold
        binary = binarize(grayscale, threshold, args.invert)
        cleaned = remove_small_objects(binary, args.min_component_size)
        skeleton = skeletonize_mask(cleaned)
        graph = skeleton_to_graph(skeleton)
        gaps = connect_gaps(graph, args.gap_tolerance)
        geometry_before = reconstruct_geometry(gaps.graph, args.geometry_tolerance, args.polyline_tolerance)
        junctions = None if args.no_junction_cleanup else refine_junctions(gaps.graph, geometry_before, args.geometry_tolerance)
        geometry = geometry_before if junctions is None else junctions.geometry
        geometry_local = geometry
        global_circles = reconstruct_global_circles(gaps.graph, geometry_local, args.geometry_tolerance) if args.global_circles else None
        if global_circles is not None:
            geometry = global_circles.geometry
        global_arcs = reconstruct_global_arcs(gaps.graph, geometry, args.geometry_tolerance) if args.global_arcs else None
        if global_arcs is not None:
            geometry = global_arcs.geometry
        if args.debug_dir is not None:
            save_preprocessing_debug(args.debug_dir, grayscale, binary, cleaned)
            save_skeleton_debug(args.debug_dir, skeleton)
            save_graph_debug(args.debug_dir, skeleton, graph)
            if args.gap_tolerance > 0:
                save_gap_debug(args.debug_dir, skeleton, gaps)
                save_graph_debug(args.debug_dir, skeleton, gaps.graph, filename="graph_connected.png")
            save_geometry_debug(args.debug_dir, image, geometry)
            if global_circles is not None:
                save_geometry_debug(args.debug_dir, image, geometry_local, filename="geometry_local.png")
            if junctions is not None:
                save_geometry_debug(args.debug_dir, image, geometry_before, filename="geometry_before.png")
                save_junction_debug(args.debug_dir, image, gaps.graph, junctions)
            if args.debug_svg:
                save_svg_overlay(args.debug_dir, image, geometry, args.stroke_width)
                if global_circles is not None:
                    save_svg_overlay(args.debug_dir, image, geometry_local, args.stroke_width, filename="overlay_local.svg")
                if junctions is not None:
                    save_svg_overlay(args.debug_dir, image, geometry_before, args.stroke_width, filename="overlay_before.svg")
        if args.output is not None:
            export_svg(geometry, args.output, args.stroke_width)
    except (OSError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    height, width = image.shape[:2]
    print(f"Image : {args.input}")
    print(f"Dimensions : {width} × {height} pixels (largeur × hauteur)")
    print(f"Seuil : {threshold}" + (" (auto, Otsu)" if args.threshold == "auto" else " (manuel)"))
    if args.config is not None:
        print(f"Configuration : {args.config}")
    print(f"Pixels de trait : {int(binary.sum())} avant nettoyage, {int(cleaned.sum())} après")
    print(f"Pixels du squelette : {int(skeleton.sum())}")
    print(f"Graphe : nœuds={len(graph.nodes)}, branches={len(graph.edges)}, "
          f"boucles sur un même nœud={sum(edge.is_loop for edge in graph.edges)}")
    if args.gap_tolerance > 0:
        print(f"Raccords : {len(gaps.connections)} (distance maximale={args.gap_tolerance:g} px) ; "
              f"graphe raccordé : nœuds={len(gaps.graph.nodes)}, branches={len(gaps.graph.edges)}")
    if junctions is not None:
        print(f"Jonctions : {len(junctions.adjustments)} acceptées, {len(junctions.rejected_node_ids)} conservées ; "
              f"continuités colinéaires={len(junctions.continuations)}")
    segments = sum(isinstance(primitive, LineSegment) for primitive in geometry.primitives)
    if global_circles is not None:
        print(f"Cercles globaux : {len(global_circles.circles)} ; "
              f"branches regroupées={sum(len(c.source_edge_ids) for c in global_circles.circles)}")
    if global_arcs is not None:
        print(f"Arcs globaux : {len(global_arcs.arcs)} ; branches regroupées={sum(len(a.source_edge_ids) for a in global_arcs.arcs)}")
    polylines = sum(isinstance(primitive, Polyline) for primitive in geometry.primitives)
    raw = sum(isinstance(primitive, RawPolyline) for primitive in geometry.primitives)
    circles = sum(isinstance(primitive, Circle) for primitive in geometry.primitives)
    arcs = sum(isinstance(primitive, CircularArc) for primitive in geometry.primitives)
    beziers = [p for p in geometry.primitives if isinstance(p, BezierPath)]
    print(f"Géométrie : segments={segments}, polylignes={polylines}, branches brutes={raw}"
          f", cercles={circles}, arcs={arcs}, chemins Bézier={len(beziers)}, cubiques={sum(len(p.segments) for p in beziers)}"
          f" (tolérance={args.geometry_tolerance:g} px, polyligne={args.polyline_tolerance:g} px)")
    if args.debug_dir is not None:
        print(f"Images de debug : {args.debug_dir} (grayscale.png, binary.png, cleaned.png, skeleton.png, graph.png, geometry.png)")
        if args.gap_tolerance > 0:
            print("Contrôle des raccords : gaps.png, graph_connected.png")
        if junctions is not None:
            print("Contrôle des jonctions : geometry_before.png, junctions.png")
        if global_circles is not None:
            print("Contrôle global : geometry_local.png" + (", overlay_local.svg" if args.debug_svg else ""))
    if args.output is not None:
        print(f"SVG : {args.output}")
    if args.debug_svg:
        print(f"SVG de comparaison : {args.debug_dir / 'overlay.svg'}")
    return 0
