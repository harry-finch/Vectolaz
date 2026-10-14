"""Reconstruction de segments, polylignes, cercles, arcs et Bézier sous tolérance."""

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from raster2vector.coordinates import Point
from raster2vector.bezier import BezierPath, fit_piecewise_bezier
from raster2vector.graph import SkeletonGraph


@dataclass(frozen=True)
class LineFit:
    """Droite optimale au sens des moindres carrés orthogonaux (PCA/SVD).

    origin est le barycentre et direction un vecteur unitaire. start et end
    bornent les projections de tous les points sur la droite, afin que le
    segment couvre toute la branche. Les erreurs sont en pixels.
    """

    origin: Point
    direction: Point
    start: Point
    end: Point
    fit_error_max: float
    fit_error_rms: float


@dataclass(frozen=True)
class LineSegment:
    """Segment idéal avec mesures d'erreur et points raster d'origine."""

    start: Point
    end: Point
    fit_error_max: float
    fit_error_rms: float
    source_edge_id: int
    source_points: tuple[Point, ...]


@dataclass(frozen=True)
class RawPolyline:
    """Chaîne brute d'une branche qui ne devient pas un segment.

    is_loop conserve la fermeture topologique, même si les contacts raster
    sont distincts. Les erreurs sont None si aucun ajustement n'a été tenté
    (contacts distincts d'une boucle) ou si la branche est dégénérée.
    source_points est None tant que points contient la source inchangée ;
    après affinage d'une jonction, il conserve la chaîne d'origine.
    """

    points: tuple[Point, ...]
    source_edge_id: int
    is_loop: bool
    fit_error_max: float | None = None
    fit_error_rms: float | None = None
    source_points: tuple[Point, ...] | None = None


@dataclass(frozen=True)
class Polyline:
    """Segments consécutifs partageant leurs sommets, issus de Douglas-Peucker.

    Les sommets sont des points sources inchangés. source_indices identifie
    les sous-chaînes approximées. Les erreurs mesurent la distance de chaque
    point à son segment correspondant, sans rechercher un segment voisin.
    """

    points: tuple[Point, ...]
    fit_error_max: float
    fit_error_rms: float
    source_edge_id: int
    source_points: tuple[Point, ...]
    source_indices: tuple[int, ...]

    @property
    def segment_count(self) -> int:
        """Nombre de segments dans la polyligne ouverte."""
        return max(0, len(self.points) - 1)


@dataclass(frozen=True)
class CircleFit:
    """Cercle ajusté, avec erreurs radiales maximale et RMS en pixels."""

    center: Point
    radius: float
    fit_error_max: float
    fit_error_rms: float


@dataclass(frozen=True)
class Circle:
    """Cercle idéal conservant sa branche et tous ses points raster sources."""

    center: Point
    radius: float
    fit_error_max: float
    fit_error_rms: float
    source_edge_id: int
    source_points: tuple[Point, ...]


@dataclass(frozen=True)
class ArcFit:
    """Arc ajusté dans le repère image, angles en radians, y vers le bas.

    end_angle_rad est déroulé par rapport à start_angle_rad. Un balayage
    positif est horaire dans ce repère et correspond à sweep-flag=1 en SVG.
    """

    center: Point
    radius: float
    start_angle_rad: float
    end_angle_rad: float
    fit_error_max: float
    fit_error_rms: float

    @property
    def sweep_angle_rad(self) -> float:
        return self.end_angle_rad - self.start_angle_rad

    @property
    def clockwise(self) -> bool:
        return self.sweep_angle_rad > 0

    @property
    def large_arc_flag(self) -> int:
        half_turn_roundoff_rad = 1e-12
        return int(abs(self.sweep_angle_rad) > math.pi and not math.isclose(
            abs(self.sweep_angle_rad), math.pi, rel_tol=0, abs_tol=half_turn_roundoff_rad))

    @property
    def sweep_flag(self) -> int:
        return int(self.clockwise)

    @property
    def start(self) -> Point:
        return Point(self.center.x + self.radius * math.cos(self.start_angle_rad),
                     self.center.y + self.radius * math.sin(self.start_angle_rad))

    @property
    def end(self) -> Point:
        return Point(self.center.x + self.radius * math.cos(self.end_angle_rad),
                     self.center.y + self.radius * math.sin(self.end_angle_rad))


@dataclass(frozen=True)
class CircularArc(ArcFit):
    """Arc circulaire conservant sa branche et tous ses points raster sources."""

    source_edge_id: int
    source_points: tuple[Point, ...]


@dataclass(frozen=True)
class GeometryResult:
    """Géométrie indépendante ; localement une primitive par branche.

    La reconstruction globale peut ensuite regrouper plusieurs branches.
    """

    width: int
    height: int
    primitives: tuple[LineSegment | Polyline | RawPolyline | Circle | CircularArc | BezierPath, ...]


def fit_circle(points: Sequence[Point]) -> CircleFit | None:
    """Ajuste un cercle à tous les points, sans Hough ni rejet d'aberrants.

    Une répétition finale du premier point est retirée pour ne pas le pondérer
    deux fois. Les données sont centrées et normalisées. Une estimation
    algébrique initialise la minimisation des distances radiales par SciPy.
    Retourne None pour moins de trois points, une configuration colinéaire,
    des points identiques ou un ajustement qui ne converge pas.
    """
    source = tuple(points)
    if len(source) > 1 and source[0] == source[-1]:
        source = source[:-1]
    coordinates = np.array([(p.x, p.y) for p in source], dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(coordinates).all():
        raise ValueError("Les coordonnées doivent être finies.")
    if len(source) < 3:
        return None
    origin = coordinates.mean(axis=0)
    centered = coordinates - origin
    scale = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
    if scale == 0:
        return None
    normalized = centered / scale
    design = np.column_stack((2 * normalized, np.ones(len(source))))
    solution, _, rank, _ = np.linalg.lstsq(design, np.sum(normalized ** 2, axis=1), rcond=None)
    if rank < 3:
        return None

    def residuals(center: NDArray[np.float64]) -> NDArray[np.float64]:
        radii = np.linalg.norm(normalized - center, axis=1)
        return radii - radii.mean()

    result = least_squares(residuals, solution[:2])
    if not result.success or not np.isfinite(result.x).all():
        return None
    radii = np.linalg.norm(normalized - result.x, axis=1)
    radius = float(radii.mean() * scale)
    center = origin + result.x * scale
    errors = np.abs((radii - radii.mean()) * scale)
    if not math.isfinite(radius) or radius <= 0 or not np.isfinite(center).all():
        return None
    return CircleFit(Point(float(center[0]), float(center[1])), radius,
                     float(errors.max()), float(np.sqrt(np.mean(errors ** 2))))


def _makes_full_turn(points: tuple[Point, ...], center: Point) -> bool:
    """Exige un tour autour du centre ; exclut un arc refermé par un retour."""
    vectors = np.array([(p.x - center.x, p.y - center.y) for p in points], dtype=np.float64)
    cross = vectors[:-1, 0] * vectors[1:, 1] - vectors[:-1, 1] * vectors[1:, 0]
    dot = np.sum(vectors[:-1] * vectors[1:], axis=1)
    turn = float(np.arctan2(cross, dot).sum())
    # Cette marge ne concerne que les arrondis numériques sur les angles.
    angular_roundoff_rad = 1e-8
    return math.isclose(abs(turn), 2 * math.pi, rel_tol=0, abs_tol=angular_roundoff_rad)


def fit_arc(points: Sequence[Point], tolerance_px: float = 2.0) -> ArcFit | None:
    """Ajuste un arc ouvert, avec erreur contrôlée à l'arc fini et sens ordonné.

    Le cercle est ajusté sur tous les points. Les angles sont déroulés pour
    traverser correctement +/-pi. Les retours angulaires plus grands que la
    tolérance exprimée en longueur d'arc sont refusés. Les points situés hors
    du balayage sont comparés aux extrémités projetées, pas au cercle infini.
    Une boucle fermée, un balayage nul ou >= un tour est refusé.
    """
    if not math.isfinite(tolerance_px) or tolerance_px < 0:
        raise ValueError("La tolérance d'arc doit être finie et positive ou nulle.")
    source = tuple(points)
    if len(source) < 3 or source[0] == source[-1]:
        return None
    circle = fit_circle(source)
    if circle is None:
        return None
    coordinates = np.array([(p.x, p.y) for p in source], dtype=np.float64)
    vectors = coordinates - np.array((circle.center.x, circle.center.y))
    angles = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
    sweep = float(angles[-1] - angles[0])
    if sweep == 0 or abs(sweep) >= 2 * math.pi:
        return None
    progress = np.sign(sweep) * (angles - angles[0])
    backtrack_px = (np.maximum.accumulate(progress) - progress) * circle.radius
    if float(backtrack_px.max()) > tolerance_px:
        return None
    start = np.array((circle.center.x + circle.radius * np.cos(angles[0]),
                      circle.center.y + circle.radius * np.sin(angles[0])))
    end = np.array((circle.center.x + circle.radius * np.cos(angles[-1]),
                    circle.center.y + circle.radius * np.sin(angles[-1])))
    errors = np.abs(np.linalg.norm(vectors, axis=1) - circle.radius)
    outside = (progress < 0) | (progress > abs(sweep))
    endpoint_errors = np.minimum(np.linalg.norm(coordinates - start, axis=1),
                                 np.linalg.norm(coordinates - end, axis=1))
    errors[outside] = endpoint_errors[outside]
    if float(errors.max()) > tolerance_px:
        return None
    return ArcFit(circle.center, circle.radius, float(angles[0]), float(angles[-1]),
                  float(errors.max()), float(np.sqrt(np.mean(errors ** 2))))


def _segment_distances(
    coordinates: NDArray[np.float64], start: NDArray[np.float64], end: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Distances à un segment fini, y compris si ses deux bornes coïncident."""
    direction = end - start
    length_squared = float(direction @ direction)
    if length_squared == 0:
        return np.linalg.norm(coordinates - start, axis=1)
    projections = np.clip(((coordinates - start) @ direction) / length_squared, 0, 1)
    closest = start + projections[:, None] * direction
    return np.linalg.norm(coordinates - closest, axis=1)


def _simplified_indices(coordinates: NDArray[np.float64], tolerance: float) -> tuple[int, ...]:
    """Douglas-Peucker itératif, avec conservation des extrémités et de l'ordre."""
    if len(coordinates) < 3:
        return tuple(range(len(coordinates)))
    keep = {0, len(coordinates) - 1}
    pending = [(0, len(coordinates) - 1)]
    while pending:
        start, end = pending.pop()
        if end - start < 2:
            continue
        distances = _segment_distances(coordinates[start + 1:end], coordinates[start], coordinates[end])
        offset = int(np.argmax(distances))
        if distances[offset] > tolerance:
            split = start + 1 + offset
            keep.add(split)
            pending.extend(((start, split), (split, end)))
    indices = sorted(keep)
    # Douglas-Peucker peut conserver deux sommets très proches d'un angle
    # raster arrondi. Enlever un sommet redondant reste permis seulement si
    # la sous-chaîne complète respecte la même tolérance au segment fusionné.
    index = 1
    while index < len(indices) - 1:
        start, end = indices[index - 1], indices[index + 1]
        errors = _segment_distances(coordinates[start:end + 1], coordinates[start], coordinates[end])
        if float(errors.max()) <= tolerance:
            indices.pop(index)
            index = max(1, index - 1)
        else:
            index += 1
    return tuple(indices)


def simplify_polyline(points: Sequence[Point], tolerance_px: float = 2.0) -> tuple[Point, ...]:
    """Simplifie une polyligne ouverte avec une erreur maximale en pixels.

    La distance est mesurée au segment fini, pas à sa droite infinie. Zéro
    n'autorise aucun écart mais permet d'enlever les points strictement
    colinéaires. Une boucle fermée est rejetée : son traitement viendra plus
    tard. Les coordonnées et la tolérance doivent être finies.
    """
    if not math.isfinite(tolerance_px) or tolerance_px < 0:
        raise ValueError("La tolérance de polyligne doit être finie et positive ou nulle.")
    source = tuple(points)
    if len(source) > 2 and source[0] == source[-1]:
        raise ValueError("La simplification requiert une polyligne ouverte.")
    coordinates = np.array([(p.x, p.y) for p in source], dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(coordinates).all():
        raise ValueError("Les coordonnées doivent être finies.")
    return tuple(source[index] for index in _simplified_indices(coordinates, tolerance_px))


def _polyline_approximation(
    points: tuple[Point, ...], edge_id: int, tolerance: float
) -> Polyline | None:
    coordinates = np.array([(p.x, p.y) for p in points], dtype=np.float64)
    indices = _simplified_indices(coordinates, tolerance)
    if len(indices) == len(points):
        return None  # Aucun gain : conserver la branche brute.
    errors = np.zeros(len(points), dtype=np.float64)
    for start, end in zip(indices, indices[1:]):
        errors[start:end + 1] = _segment_distances(
            coordinates[start:end + 1], coordinates[start], coordinates[end]
        )
    return Polyline(
        tuple(points[index] for index in indices), float(errors.max()),
        float(np.sqrt(np.mean(errors ** 2))), edge_id, points, indices,
    )


def fit_line(points: Sequence[Point]) -> LineFit | None:
    """Ajuste tous les points, sans privilégier x ou y ni rejeter de points.

    Retourne None pour moins de deux points ou des points tous identiques.
    Les coordonnées non finies sont rejetées. Les données sont centrées avant
    SVD pour limiter les erreurs numériques. La direction suit le sens de la
    branche lorsque la projection du premier au dernier point est non nulle.
    """
    if not points:
        return None
    coordinates = np.array([(point.x, point.y) for point in points], dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise ValueError("Les coordonnées doivent être finies.")
    if len(points) < 2:
        return None
    origin = coordinates.mean(axis=0)
    centered = coordinates - origin
    _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
    if singular_values[0] == 0:
        return None
    direction = axes[0]
    traversal = float(np.dot(coordinates[-1] - coordinates[0], direction))
    if traversal < 0 or (traversal == 0 and direction[np.argmax(np.abs(direction))] < 0):
        direction = -direction
    projections = centered @ direction
    # Distance perpendiculaire, par produit avec la normale unitaire.
    normal = np.array([-direction[1], direction[0]])
    errors = np.abs(centered @ normal)
    start = origin + float(projections.min()) * direction
    end = origin + float(projections.max()) * direction
    return LineFit(
        origin=Point(float(origin[0]), float(origin[1])),
        direction=Point(float(direction[0]), float(direction[1])),
        start=Point(float(start[0]), float(start[1])),
        end=Point(float(end[0]), float(end[1])),
        fit_error_max=float(errors.max()),
        fit_error_rms=float(np.sqrt(np.mean(errors ** 2))),
    )


def _curves_between_corners(points: tuple[Point, ...], edge_id: int,
                            tolerance: float, polyline_tolerance: float,
                            is_loop: bool = False) -> BezierPath | None:
    """Conserve les sommets marqués et ajuste le reste sur les points sources."""
    if tolerance == 0 or len(points) < 5:
        return None
    polyline = _polyline_approximation(points, edge_id, polyline_tolerance)
    if polyline is None or len(polyline.points) < 3:
        return None
    vectors = np.diff([(p.x, p.y) for p in polyline.points], axis=0)
    turns = np.arctan2(np.abs(vectors[:-1, 0]*vectors[1:, 1]-vectors[:-1, 1]*vectors[1:, 0]),
                       np.sum(vectors[:-1]*vectors[1:], axis=1))
    corner_angle_rad = math.pi/4
    cuts = [polyline.source_indices[i+1] for i, turn in enumerate(turns) if turn >= corner_angle_rad]
    intervals = sorted({0, len(points)-1, *cuts})
    if not is_loop and not any(
        (fit := fit_line(points[a:b+1])) is not None and fit.fit_error_max > tolerance
        for a, b in zip(intervals, intervals[1:])
    ):
        return None
    result = fit_piecewise_bezier(points, tolerance, cuts, source_edge_id=edge_id, is_loop=is_loop)
    if result is None or 3*len(result.segments)+1 >= len(points):
        return None
    # Les portions réellement droites/anguleuses restent des polylignes. La
    # sélection favorise les courbes sur les pixels denses, sans exiger moins
    # de poignées qu'une polyligne calculée avec une autre tolérance.
    for curve in result.segments:
        a = np.array((curve.start.x, curve.start.y))
        b = np.array((curve.end.x, curve.end.y))
        controls = np.array([(p.x, p.y) for p in (curve.control1, curve.control2)])
        if _segment_distances(controls, a, b).max() > max(0.25, tolerance/4):
            return result
    return None


def reconstruct_geometry(
    graph: SkeletonGraph,
    geometry_tolerance_px: float = 2.0,
    polyline_tolerance_px: float | None = 2.0,
    *,
    detect_arcs: bool = True,
    detect_beziers: bool = True,
) -> GeometryResult:
    """Reconstruit les cercles fermés, segments, arcs et polylignes sous tolérance.

    Critère : erreur orthogonale maximale <= geometry_tolerance_px. Les
    boucles explicitement fermées sont ajustées à un cercle si leur erreur
    radiale maximale respecte la tolérance et si elles font un tour complet.
    Les autres boucles peuvent devenir des Bézier fermés. Une branche ouverte est d'abord ajustée
    à une droite, puis à un arc si detect_arcs=True, puis simplifiée avec
    polyline_tolerance_px. None désactive cette simplification des branches ouvertes.
    Les angles >= 45° d'une polyligne servent de contacts imposés pour ajuster
    séparément les portions Bézier, sans arrondir ces sommets. La borne continue
    respecte geometry_tolerance_px et le nombre de points, poignées comprises,
    doit être réduit par rapport aux sources raster.
    Les boucles incompatibles et les branches dégénérées restent brutes.
    La topologie et les nœuds d'origine ne sont pas modifiés ni ajustés.
    """
    if not math.isfinite(geometry_tolerance_px) or geometry_tolerance_px < 0:
        raise ValueError("La tolérance géométrique doit être finie et positive ou nulle.")
    if polyline_tolerance_px is not None and (
        not math.isfinite(polyline_tolerance_px) or polyline_tolerance_px < 0
    ):
        raise ValueError("La tolérance de polyligne doit être finie et positive ou nulle.")
    primitives: list[LineSegment | Polyline | RawPolyline | Circle | CircularArc | BezierPath] = []
    for edge in graph.edges:
        closed = edge.is_loop or (len(edge.points) > 1 and edge.points[0] == edge.points[-1])
        if closed:
            circle_fit = None
            if len(edge.points) > 3 and edge.points[0] == edge.points[-1]:
                circle_fit = fit_circle(edge.points)
            if (circle_fit is not None and circle_fit.fit_error_max <= geometry_tolerance_px
                    and _makes_full_turn(edge.points, circle_fit.center)):
                primitives.append(Circle(
                    circle_fit.center, circle_fit.radius, circle_fit.fit_error_max,
                    circle_fit.fit_error_rms, edge.id, edge.points,
                ))
            else:
                bezier = (_curves_between_corners(edge.points, edge.id, geometry_tolerance_px,
                                                  polyline_tolerance_px, True)
                          if (detect_beziers and polyline_tolerance_px is not None
                              and edge.points[0] == edge.points[-1]
                              and len(set(edge.points[:-1])) == len(edge.points)-1) else None)
                if bezier is not None:
                    primitives.append(bezier)
                    continue
                primitives.append(RawPolyline(
                    edge.points, edge.id, True,
                    None if circle_fit is None else circle_fit.fit_error_max,
                    None if circle_fit is None else circle_fit.fit_error_rms,
                ))
            continue
        fit = None if closed else fit_line(edge.points)
        if fit is not None and fit.fit_error_max <= geometry_tolerance_px:
            primitives.append(LineSegment(
                fit.start, fit.end, fit.fit_error_max, fit.fit_error_rms,
                edge.id, edge.points,
            ))
        else:
            arc = fit_arc(edge.points, geometry_tolerance_px) if detect_arcs else None
            if arc is not None:
                primitives.append(CircularArc(
                    arc.center, arc.radius, arc.start_angle_rad, arc.end_angle_rad,
                    arc.fit_error_max, arc.fit_error_rms, edge.id, edge.points,
                ))
                continue
            polyline = None
            if fit is not None and not closed and polyline_tolerance_px is not None:
                polyline = _polyline_approximation(edge.points, edge.id, polyline_tolerance_px)
            if polyline is not None:
                if detect_beziers:
                    bezier = _curves_between_corners(edge.points, edge.id, geometry_tolerance_px,
                                                      polyline_tolerance_px)
                    if bezier is not None:
                        primitives.append(bezier)
                        continue
                primitives.append(polyline)
                continue
            primitives.append(RawPolyline(
                edge.points, edge.id, closed,
                None if fit is None else fit.fit_error_max,
                None if fit is None else fit.fit_error_rms,
            ))
    return GeometryResult(graph.width, graph.height, tuple(primitives))
