"""Placement du raster natif dans le rectangle SVG local de l'image."""

from __future__ import annotations

import math
import re


def image_mapping(x: float, y: float, width: float, height: float,
                  raster_width: int, raster_height: int,
                  aspect_ratio: str | None = None) -> tuple[float, float, float, float]:
    """Retourne sx, sy, dx, dy ; le transform de l'image est appliqué ensuite."""
    if not all(math.isfinite(v) for v in (x, y, width, height)) or min(width, height, raster_width, raster_height) <= 0:
        raise ValueError("Les dimensions de l'image doivent être finies et strictement positives.")
    tokens = (aspect_ratio or "xMidYMid meet").split()
    if tokens and tokens[0] == "defer":
        tokens.pop(0)
    if not tokens or len(tokens) > 2:
        raise ValueError("Le preserveAspectRatio de l'image est invalide.")
    align = tokens[0]
    mode = tokens[1] if len(tokens) == 2 else "meet"
    if mode not in ("meet", "slice"):
        raise ValueError("Le preserveAspectRatio de l'image est invalide.")
    sx, sy = width/raster_width, height/raster_height
    if align == "none":
        return sx, sy, x, y
    match = re.fullmatch(r"x(Min|Mid|Max)Y(Min|Mid|Max)", align)
    if match is None:
        raise ValueError("L'alignement preserveAspectRatio n'est pas pris en charge.")
    if mode == "slice":
        raise ValueError("Le recadrage preserveAspectRatio='slice' n'est pas encore pris en charge.")
    scale = min(sx, sy)
    factors = {"Min": 0.0, "Mid": 0.5, "Max": 1.0}
    return (scale, scale, x+(width-raster_width*scale)*factors[match[1]],
            y+(height-raster_height*scale)*factors[match[2]])
