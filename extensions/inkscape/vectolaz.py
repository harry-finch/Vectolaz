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
import xml.etree.ElementTree as ET

import inkex

from image_source import read_png
from placement import image_mapping

ENGINE_TIMEOUT_SECONDS = 120
SVG_NS = "http://www.w3.org/2000/svg"


def run_engine(python: str, source: Path, output: Path, options: object) -> bytes:
    """La CLI est exécutée sans shell, dans son propre environnement Python."""
    executable = Path(python).expanduser()
    if not python or not executable.is_absolute() or not executable.is_file():
        raise ValueError("Indiquez le chemin absolu du Python du moteur dans l'onglet Installation.")
    for name in ("geometry_tolerance", "gap_tolerance", "stroke_width"):
        value = getattr(options, name)
        if not math.isfinite(value) or value < 0 or (name == "stroke_width" and value == 0):
            raise ValueError("Les tolérances doivent être finies et positives ou nulles, et l'épaisseur positive.")
    if options.min_component_size < 0 or not 0 <= options.threshold <= 255:
        raise ValueError("Le seuil doit être entre 0 et 255 et la taille minimale positive ou nulle.")
    command = [str(executable), "-m", "raster2vector", str(source), "--output", str(output),
               "--threshold", "auto" if options.threshold_auto else str(options.threshold),
               "--geometry-tolerance", str(options.geometry_tolerance),
               "--gap-tolerance", str(options.gap_tolerance),
               "--min-component-size", str(options.min_component_size),
               "--stroke-width", str(options.stroke_width)]
    if options.global_arcs:
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


class VectolazExtension(inkex.EffectExtension):
    """Reconstruit les axes d'un PNG sélectionné, en conservant son image."""

    def add_arguments(self, pars) -> None:
        pars.add_argument("--tab", default="vectorization")
        pars.add_argument("--engine-python", default="")
        pars.add_argument("--threshold-auto", type=inkex.Boolean, default=True)
        pars.add_argument("--threshold", type=int, default=128)
        pars.add_argument("--geometry-tolerance", type=float, default=2.0)
        pars.add_argument("--gap-tolerance", type=float, default=0.0)
        pars.add_argument("--min-component-size", type=int, default=0)
        pars.add_argument("--stroke-width", type=float, default=1.0)
        pars.add_argument("--global-arcs", type=inkex.Boolean, default=False)

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
            wrapper = inkex.Group()
            wrapper.set("inkscape:label", "Vectolaz — axes reconstruits")
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
