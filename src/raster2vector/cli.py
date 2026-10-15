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
from raster2vector.geometry import LineSegment, Polyline, RawPolyline, Circle, CircularArc
from raster2vector.bezier import BezierPath
from raster2vector.pipeline import run_pipeline
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
    parser.add_argument("--global-circles", action="store_true", default=None,
                        help="Recherche des cercles complets répartis sur plusieurs branches (optionnel).")
    parser.add_argument("--global-arcs", action="store_true", default=None,
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
        "--spur-tolerance", type=float, default=None, metavar="PIXELS",
        help="Supprime les micro-branches pendantes plus courtes que cette longueur (défaut : 0, désactivé).",
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
    try:
        settings = load_config(args.config) if args.config is not None else VectorizationConfig()
        settings = settings.with_overrides(
            threshold=args.threshold,
            invert=args.invert,
            min_component_size=args.min_component_size,
            geometry_tolerance=args.geometry_tolerance,
            polyline_tolerance=args.polyline_tolerance,
            gap_tolerance=args.gap_tolerance,
            stroke_width=args.stroke_width,
            spur_tolerance=args.spur_tolerance,
            junction_cleanup=None if args.no_junction_cleanup is None else not args.no_junction_cleanup,
            global_circles=args.global_circles,
            global_arcs=args.global_arcs,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.debug_svg and args.debug_dir is None:
        parser.error("--debug-svg nécessite --debug-dir.")
    try:
        destinations = [] if args.output is None else [args.output]
        if args.debug_dir is not None:
            names = ["grayscale.png", "binary.png", "cleaned.png", "skeleton.png", "graph.png", "geometry.png"]
            if settings.global_circles:
                names.append("geometry_local.png")
            if settings.junction_cleanup:
                names.extend(("geometry_before.png", "junctions.png"))
            if settings.gap_tolerance > 0:
                names.extend(("gaps.png", "graph_connected.png"))
            if args.debug_svg:
                names.append("overlay.svg")
                if settings.global_circles:
                    names.append("overlay_local.svg")
                if settings.junction_cleanup:
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

        image = load_image(args.input)
        pipeline_result = run_pipeline(image, settings)

        if args.debug_dir is not None:
            save_preprocessing_debug(args.debug_dir, pipeline_result.grayscale, pipeline_result.binary, pipeline_result.cleaned)
            save_skeleton_debug(args.debug_dir, pipeline_result.skeleton)
            save_graph_debug(args.debug_dir, pipeline_result.skeleton, pipeline_result.graph)
            if settings.gap_tolerance > 0:
                save_gap_debug(args.debug_dir, pipeline_result.skeleton, pipeline_result.gaps)
                save_graph_debug(args.debug_dir, pipeline_result.skeleton, pipeline_result.gaps.graph, filename="graph_connected.png")
            save_geometry_debug(args.debug_dir, image, pipeline_result.geometry)
            if pipeline_result.global_circles is not None:
                save_geometry_debug(args.debug_dir, image, pipeline_result.geometry_local, filename="geometry_local.png")
            if pipeline_result.junctions is not None:
                save_geometry_debug(args.debug_dir, image, pipeline_result.geometry_before, filename="geometry_before.png")
                save_junction_debug(args.debug_dir, image, pipeline_result.gaps.graph, pipeline_result.junctions)
            if args.debug_svg:
                save_svg_overlay(args.debug_dir, image, pipeline_result.geometry, settings.stroke_width)
                if pipeline_result.global_circles is not None:
                    save_svg_overlay(args.debug_dir, image, pipeline_result.geometry_local, settings.stroke_width, filename="overlay_local.svg")
                if pipeline_result.junctions is not None:
                    save_svg_overlay(args.debug_dir, image, pipeline_result.geometry_before, settings.stroke_width, filename="overlay_before.svg")
        if args.output is not None:
            export_svg(pipeline_result.geometry, args.output, settings.stroke_width)
    except (OSError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    height, width = image.shape[:2]
    print(f"Image : {args.input}")
    print(f"Dimensions : {width} × {height} pixels (largeur × hauteur)")
    print(f"Seuil : {pipeline_result.threshold}" + (" (auto, Otsu)" if settings.threshold == "auto" else " (manuel)"))
    if args.config is not None:
        print(f"Configuration : {args.config}")
    print(f"Pixels de trait : {int(pipeline_result.binary.sum())} avant nettoyage, {int(pipeline_result.cleaned.sum())} après")
    print(f"Pixels du squelette : {int(pipeline_result.skeleton.sum())}")
    print(f"Graphe : nœuds={len(pipeline_result.graph.nodes)}, branches={len(pipeline_result.graph.edges)}, "
          f"boucles sur un même nœud={sum(edge.is_loop for edge in pipeline_result.graph.edges)}")
    if settings.gap_tolerance > 0:
        print(f"Raccords : {len(pipeline_result.gaps.connections)} (distance maximale={settings.gap_tolerance:g} px) ; "
              f"graphe raccordé : nœuds={len(pipeline_result.gaps.graph.nodes)}, branches={len(pipeline_result.gaps.graph.edges)}")
    if pipeline_result.junctions is not None:
        print(f"Jonctions : {len(pipeline_result.junctions.adjustments)} acceptées, {len(pipeline_result.junctions.rejected_node_ids)} conservées ; "
              f"continuités colinéaires={len(pipeline_result.junctions.continuations)}")
    segments = sum(isinstance(primitive, LineSegment) for primitive in pipeline_result.geometry.primitives)
    if pipeline_result.global_circles is not None:
        print(f"Cercles globaux : {len(pipeline_result.global_circles.circles)} ; "
              f"branches regroupées={sum(len(c.source_edge_ids) for c in pipeline_result.global_circles.circles)}")
    if pipeline_result.global_arcs is not None:
        print(f"Arcs globaux : {len(pipeline_result.global_arcs.arcs)} ; branches regroupées={sum(len(a.source_edge_ids) for a in pipeline_result.global_arcs.arcs)}")
    polylines = sum(isinstance(primitive, Polyline) for primitive in pipeline_result.geometry.primitives)
    raw = sum(isinstance(primitive, RawPolyline) for primitive in pipeline_result.geometry.primitives)
    circles = sum(isinstance(primitive, Circle) for primitive in pipeline_result.geometry.primitives)
    arcs = sum(isinstance(primitive, CircularArc) for primitive in pipeline_result.geometry.primitives)
    beziers = [p for p in pipeline_result.geometry.primitives if isinstance(p, BezierPath)]
    print(f"Géométrie : segments={segments}, polylignes={polylines}, branches brutes={raw}"
          f", cercles={circles}, arcs={arcs}, chemins Bézier={len(beziers)}, cubiques={sum(len(p.segments) for p in beziers)}"
          f" (tolérance={settings.geometry_tolerance:g} px, polyligne={settings.polyline_tolerance:g} px)")
    if args.debug_dir is not None:
        print(f"Images de debug : {args.debug_dir} (grayscale.png, binary.png, cleaned.png, skeleton.png, graph.png, geometry.png)")
        if settings.gap_tolerance > 0:
            print("Contrôle des raccords : gaps.png, graph_connected.png")
        if pipeline_result.junctions is not None:
            print("Contrôle des jonctions : geometry_before.png, junctions.png")
        if pipeline_result.global_circles is not None:
            print("Contrôle global : geometry_local.png" + (", overlay_local.svg" if args.debug_svg else ""))
    if args.output is not None:
        print(f"SVG : {args.output}")
    if args.debug_svg:
        print(f"SVG de comparaison : {args.debug_dir / 'overlay.svg'}")
    return 0
