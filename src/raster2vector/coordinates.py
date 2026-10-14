"""Coordonnées communes à la topologie et à la géométrie."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    """Coordonnées dans le repère du raster : x=col, y=row, en pixels."""

    x: float
    y: float
