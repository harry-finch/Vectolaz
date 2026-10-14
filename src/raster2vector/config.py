"""Paramètres de traitement validés, chargement TOML sans dépendance supplémentaire."""

from dataclasses import dataclass, fields, replace
import math
from pathlib import Path
import tomllib
from typing import Literal


@dataclass(frozen=True)
class VectorizationConfig:
    """Réglages partagés ; les chemins d'entrée/sortie restent explicites dans la CLI."""
    threshold: int | Literal["auto"] = 128
    invert: Literal["auto", "yes", "no"] = "auto"
    min_component_size: int = 0
    geometry_tolerance: float = 2.0
    polyline_tolerance: float = 2.0
    gap_tolerance: float = 0.0
    stroke_width: float = 1.0
    junction_cleanup: bool = True
    global_circles: bool = False
    global_arcs: bool = False

    def __post_init__(self) -> None:
        if self.threshold != "auto" and (type(self.threshold) is not int or not 0 <= self.threshold <= 255):
            raise ValueError("--threshold doit être auto ou un entier entre 0 et 255.")
        if self.invert not in ("auto", "yes", "no"):
            raise ValueError("--invert doit être auto, yes ou no.")
        if type(self.min_component_size) is not int or self.min_component_size < 0:
            raise ValueError("--min-component-size doit être un entier positif ou nul.")
        for name in ("geometry_tolerance", "polyline_tolerance", "gap_tolerance", "stroke_width"):
            value = getattr(self, name)
            try:
                finite = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite or value < 0 or (name == "stroke_width" and value == 0):
                qualifier = "strictement positif" if name == "stroke_width" else "positif ou nul"
                raise ValueError(f"--{name.replace('_', '-')} doit être fini et {qualifier}.")
        for name in ("junction_cleanup", "global_circles", "global_arcs"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} doit être un booléen true ou false.")
        if self.global_arcs:
            object.__setattr__(self, "global_circles", True)

    def with_overrides(self, **overrides: object) -> "VectorizationConfig":
        """Valide le résultat ; les valeurs None signifient absence d'option CLI."""
        return replace(self, **{name: value for name, value in overrides.items() if value is not None})


def load_config(path: Path) -> VectorizationConfig:
    """Lit une seule table [vectorization] ; rejette clés inconnues et types invalides."""
    try:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Configuration TOML invalide : {exc}") from exc
    if set(document) != {"vectorization"} or not isinstance(document["vectorization"], dict):
        raise ValueError("La configuration doit contenir uniquement une table [vectorization].")
    values = document["vectorization"]
    unknown = set(values)-{field.name for field in fields(VectorizationConfig)}
    if unknown:
        raise ValueError("Paramètres de configuration inconnus : " + ", ".join(sorted(unknown)))
    return VectorizationConfig(**values)
