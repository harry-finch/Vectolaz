#!/usr/bin/env python3
"""Adaptateur Inkscape : PNG sélectionné → moteur externe → primitives SVG."""

from __future__ import annotations

import copy
import io
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace
import xml.etree.ElementTree as ET

try:
    import inkex
except ImportError:
    inkex = None

from image_source import read_png
from placement import image_mapping

ENGINE_TIMEOUT_SECONDS = 120
SVG_NS = "http://www.w3.org/2000/svg"


class ReadOnlyDict(dict):
    """Dictionnaire immuable en lecture seule."""

    def __setitem__(self, key, value):
        raise TypeError("PRESETS est un dictionnaire immuable.")

    def __delitem__(self, key):
        raise TypeError("PRESETS est un dictionnaire immuable.")

    def clear(self):
        raise TypeError("PRESETS est un dictionnaire immuable.")

    def update(self, *args, **kwargs):
        raise TypeError("PRESETS est un dictionnaire immuable.")

    def pop(self, *args, **kwargs):
        raise TypeError("PRESETS est un dictionnaire immuable.")

    def popitem(self):
        raise TypeError("PRESETS est un dictionnaire immuable.")

    def setdefault(self, *args, **kwargs):
        raise TypeError("PRESETS est un dictionnaire immuable.")


PRESETS: dict[str, dict[str, object]] = ReadOnlyDict({
    "technical": ReadOnlyDict({
        "auto_threshold": True,
        "geometry_tolerance": 2.5,
        "spur_tolerance": 3.0,
        "gap_tolerance": 0.0,
        "min_component_size": 5,
        "global_arcs": False,
    }),
    "rosace": ReadOnlyDict({
        "auto_threshold": False,
        "threshold": 160,
        "geometry_tolerance": 5.0,
        "spur_tolerance": 3.0,
        "gap_tolerance": 0.0,
        "min_component_size": 10,
        "global_arcs": True,
    }),
    "organic": ReadOnlyDict({
        "auto_threshold": True,
        "geometry_tolerance": 2.0,
        "spur_tolerance": 2.5,
        "gap_tolerance": 0.0,
        "min_component_size": 5,
        "global_arcs": False,
    }),
    "detailed": ReadOnlyDict({
        "auto_threshold": False,
        "threshold": 165,
        "geometry_tolerance": 1.0,
        "spur_tolerance": 1.0,
        "gap_tolerance": 0.0,
        "min_component_size": 2,
        "global_arcs": False,
    }),
    "scanned": ReadOnlyDict({
        "auto_threshold": False,
        "threshold": 175,
        "geometry_tolerance": 3.0,
        "spur_tolerance": 4.5,
        "gap_tolerance": 5.0,
        "min_component_size": 25,
        "global_arcs": False,
    }),
})


def resolve_options(options: object) -> SimpleNamespace:
    """Résout les options effectives selon le profil (preset) choisi."""
    preset = getattr(options, "preset", "custom")
    threshold_auto = getattr(options, "threshold_auto", getattr(options, "auto_threshold", True))
    threshold = getattr(options, "threshold", 128)
    geometry_tolerance = getattr(options, "geometry_tolerance", 2.0)
    gap_tolerance = getattr(options, "gap_tolerance", 0.0)
    spur_tolerance = getattr(options, "spur_tolerance", 0.0)
    min_component_size = getattr(options, "min_component_size", 0)
    stroke_width = getattr(options, "stroke_width", 1.0)
    global_arcs = getattr(options, "global_arcs", False)

    resolved = SimpleNamespace(
        preset=preset,
        threshold_auto=threshold_auto,
        auto_threshold=threshold_auto,
        threshold=threshold,
        geometry_tolerance=geometry_tolerance,
        gap_tolerance=gap_tolerance,
        spur_tolerance=spur_tolerance,
        min_component_size=min_component_size,
        stroke_width=stroke_width,
        global_arcs=global_arcs,
    )

    if preset and preset != "custom":
        if preset not in PRESETS:
            raise ValueError(f"Profil de préréglage inconnu : {preset}")
        cfg = PRESETS[preset]
        if "auto_threshold" in cfg:
            val = bool(cfg["auto_threshold"])
            resolved.threshold_auto = val
            resolved.auto_threshold = val
        if "threshold" in cfg:
            resolved.threshold = int(cfg["threshold"])
        if "geometry_tolerance" in cfg:
            resolved.geometry_tolerance = float(cfg["geometry_tolerance"])
        if "spur_tolerance" in cfg:
            resolved.spur_tolerance = float(cfg["spur_tolerance"])
        if "gap_tolerance" in cfg:
            resolved.gap_tolerance = float(cfg["gap_tolerance"])
        if "min_component_size" in cfg:
            resolved.min_component_size = int(cfg["min_component_size"])
        if "global_arcs" in cfg:
            resolved.global_arcs = bool(cfg["global_arcs"])
        # stroke_width est conservé tel quel depuis options

    return resolved


def run_engine(python: str, source: Path, output: Path, options: object) -> bytes:
    """La CLI est exécutée sans shell, dans son propre environnement Python."""
    executable = Path(python).expanduser()
    if not python or not executable.is_absolute() or not executable.is_file():
        raise ValueError("Indiquez le chemin absolu du Python du moteur dans l'onglet Installation.")
    opts = resolve_options(options)
    for name in ("geometry_tolerance", "gap_tolerance", "spur_tolerance", "stroke_width"):
        value = getattr(opts, name)
        if not math.isfinite(value) or value < 0 or (name == "stroke_width" and value == 0):
            raise ValueError("Les tolérances doivent être finies et positives ou nulles, et l'épaisseur positive.")
    if opts.min_component_size < 0 or not 0 <= opts.threshold <= 255:
        raise ValueError("Le seuil doit être entre 0 et 255 et la taille minimale positive ou nulle.")
    command = [
        str(executable), "-m", "raster2vector", str(source), "--output", str(output),
        "--threshold", "auto" if opts.threshold_auto else str(opts.threshold),
        "--geometry-tolerance", str(opts.geometry_tolerance),
        "--gap-tolerance", str(opts.gap_tolerance),
        "--spur-tolerance", str(opts.spur_tolerance),
        "--min-component-size", str(opts.min_component_size),
        "--stroke-width", str(opts.stroke_width),
    ]
    if opts.global_arcs:
        command.append("--global-arcs")
    # Le Python embarqué d'Inkscape peut exporter ces variables : ne pas les
    # transmettre au venv, qui doit charger ses propres modules et bibliothèques.
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE",
                "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        environment.pop(key, None)
    environment["PYTHONNOUSERSITE"] = "1"
    try:
        result = subprocess.run(command, cwd=source.parent, env=environment,
                                capture_output=True, timeout=ENGINE_TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired as error:
        raise ValueError(f"Le moteur a dépassé le délai de {ENGINE_TIMEOUT_SECONDS} secondes.") from error
    except OSError as error:
        raise ValueError(f"Impossible de lancer le Python du moteur : {error}") from error
    if result.returncode:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace").strip()
        raise ValueError(f"Le moteur a échoué (code {result.returncode}).\n{detail[-3000:]}")
    if not output.is_file():
        raise ValueError("Le moteur n'a pas produit le SVG attendu.")
    return output.read_bytes()


def validated_vectors(data: bytes, width: int, height: int) -> inkex.Group:
    """N'importe que le groupe de primitives du format SVG du moteur."""
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("Le SVG du moteur contient une déclaration inattendue.")
    try:
        root = ET.fromstring(data)
        if root.tag != f"{{{SVG_NS}}}svg" or [float(v) for v in root.get("viewBox", "").split()] != [0, 0, width, height]:
            raise ValueError("Les dimensions du SVG du moteur ne correspondent pas au PNG.")
        groups = [e for e in root if e.tag == f"{{{SVG_NS}}}g" and e.get("id") == "vectors"]
        if len(groups) != 1 or len(root) != 1:
            raise ValueError("Le SVG du moteur ne contient pas le groupe vectoriel attendu.")
        group = groups[0]
        if group.get("transform") != "translate(0.5 0.5)":
            raise ValueError("Le décalage des centres de pixels du SVG est invalide.")
        if not len(group):
            raise ValueError("Aucun trait vectoriel n'a été reconnu dans l'image.")
        allowed = {"id", "d", "cx", "cy", "r", "fill", "stroke", "stroke-width",
                   "stroke-linecap", "stroke-linejoin", "transform", "data-source-edges", "data-support-edges"}
        for element in group.iter():
            if element is not group and (element.tag not in (f"{{{SVG_NS}}}path", f"{{{SVG_NS}}}circle") or len(element)):
                raise ValueError("Le SVG contient une primitive inattendue.")
            if set(element.attrib)-allowed or (element is not group and "transform" in element.attrib):
                raise ValueError("Le SVG contient des attributs inattendus.")
            if element.tag == f"{{{SVG_NS}}}circle":
                values = [float(element.get(name, "nan")) for name in ("cx", "cy", "r")]
                if not all(math.isfinite(value) for value in values) or values[2] <= 0:
                    raise ValueError("Le SVG contient un cercle invalide.")
            elif element.tag == f"{{{SVG_NS}}}path":
                path = inkex.Path(element.get("d", ""))
                if len(path) < 2 or not all(math.isfinite(float(value)) for command in path for value in command.args):
                    raise ValueError("Le SVG contient un chemin invalide.")
        return copy.deepcopy(inkex.load_svg(io.BytesIO(data)).getroot()[0])
    except (ET.ParseError, TypeError, OverflowError) as error:
        raise ValueError("Le SVG produit par le moteur est illisible.") from error


def image_transform(image: inkex.Image, width: int, height: int) -> inkex.Transform:
    """Transform de l'image × placement des pixels ; les parents restent hérités."""
    root = image.root
    for node in (image, *image.ancestors()):
        if node is not root and node.tag == inkex.addNS("svg", "svg"):
            raise ValueError("Les images dans un viewport SVG imbriqué ne sont pas encore prises en charge.")
        style = node.cascaded_style()
        for name in ("clip-path", "mask", "filter"):
            if str(style.get(name, node.get(name, "none"))) != "none":
                raise ValueError("Les images avec clip, masque ou filtre ne sont pas encore prises en charge.")
        if any(key in style for key in ("transform", "transform-origin", "transform-box")):
            raise ValueError("Les transformations CSS ne sont pas encore prises en charge ; utilisez l'attribut SVG transform.")
        if node is image and any(key in style for key in ("x", "y", "width", "height")):
            raise ValueError("Les dimensions CSS de l'image ne sont pas encore prises en charge.")
        matrix = node.transform.matrix
        a, c, _ = matrix[0]
        b, d, _ = matrix[1]
        if not all(math.isfinite(value) for row in matrix for value in row) or a*d-b*c == 0:
            raise ValueError("Une transformation de l'image ou de ses parents est invalide ou singulière.")
    lengths = []
    for name in ("x", "y", "width", "height"):
        value = image.get(name, "0")
        if "%" in value:
            raise ValueError("Les dimensions en pourcentage doivent être converties en longueurs explicites.")
        if not re.fullmatch(r"\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:px|mm|cm|in|pt|pc|Q)?\s*", value):
            raise ValueError(f"La longueur SVG '{name}' est invalide ou utilise une unité relative non prise en charge.")
        # Sans unité, valeur déjà dans le repère local SVG. Les unités absolues
        # suivent les conversions CSS définies par SVG, sans facteur DPI ajouté.
        lengths.append(float(image.to_dimensionless(value)))
    sx, sy, dx, dy = image_mapping(*lengths, width, height, image.get("preserveAspectRatio"))
    return image.transform @ inkex.Transform((sx, 0, 0, sy, dx, dy))


if inkex is not None:
    _BaseExtension = inkex.EffectExtension
    _Boolean = inkex.Boolean
else:
    class _BaseExtension:  # type: ignore[no-redef]
        def __init__(self):
            import argparse
            self.arg_parser = argparse.ArgumentParser()
            self.add_arguments(self.arg_parser)
        def add_arguments(self, pars):
            pass
        def parse_arguments(self, args):
            self.options = self.arg_parser.parse_args(args)
    def _Boolean(val):  # type: ignore[no-redef]
        if isinstance(val, bool):
            return val
        return str(val).lower() in ("true", "1", "yes")


class VectolazExtension(_BaseExtension):
    """Reconstruit les axes d'un PNG sélectionné, en conservant son image."""

    def add_arguments(self, pars) -> None:
        pars.add_argument("--tab", default="vectorization")
        pars.add_argument("--preset", default="custom", choices=["custom", *PRESETS.keys()])
        pars.add_argument("--engine-python", default="")
        pars.add_argument("--threshold-auto", type=_Boolean, default=True)
        pars.add_argument("--threshold", type=int, default=128)
        pars.add_argument("--geometry-tolerance", type=float, default=2.0)
        pars.add_argument("--gap-tolerance", type=float, default=0.0)
        pars.add_argument("--spur-tolerance", "--spur_tolerance", dest="spur_tolerance", type=float, default=0.0)
        pars.add_argument("--min-component-size", type=int, default=0)
        pars.add_argument("--stroke-width", type=float, default=1.0)
        pars.add_argument("--global-arcs", type=_Boolean, default=False)

    def effect(self) -> None:
        selected = list(self.svg.selection.values())
        if len(selected) != 1 or not isinstance(selected[0], inkex.Image):
            raise inkex.AbortExtension("Sélectionnez exactement une image PNG (pas un groupe ni un clone).")
        image = selected[0]
        try:
            data, width, height = read_png(image.get("href") or image.get("xlink:href", ""), self.document_path())
            placement = image_transform(image, width, height)
            with tempfile.TemporaryDirectory(prefix="vectolaz-") as folder:
                source, output = Path(folder)/"input.png", Path(folder)/"output.svg"
                source.write_bytes(data)
                vectors = validated_vectors(run_engine(self.options.engine_python, source, output, self.options), width, height)
            opts = resolve_options(self.options)
            preset_labels = {
                "custom": "Personnalisé",
                "technical": "Dessin technique",
                "rosace": "Rosace",
                "organic": "Tracé libre",
                "detailed": "Motif fin",
                "scanned": "Croquis scanné",
            }
            p_label = preset_labels.get(opts.preset, opts.preset)
            wrapper = inkex.Group()
            wrapper.set("inkscape:label", f"Vectolaz — {p_label} (tol={opts.geometry_tolerance}px, barbes={opts.spur_tolerance}px)")
            wrapper.set("data-preset", opts.preset)
            wrapper.set("data-source-image", image.get("id", ""))
            wrapper.transform = placement
            wrapper.append(vectors)
            reserved = []
            for element in wrapper.iter():
                identifier = self.svg.get_unique_id("vectolaz-", blacklist=reserved)
                reserved.append(identifier)
                element.set("id", identifier)
            parent = image.getparent()
            parent.insert(parent.index(image)+1, wrapper)
        except (ValueError, OSError) as error:
            raise inkex.AbortExtension(str(error)) from error


if __name__ == "__main__":
    VectolazExtension().run()
