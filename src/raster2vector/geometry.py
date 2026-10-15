"""Reconstruction de segments, polylignes, cercles, arcs et Bézier sous tolérance."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
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
    is_loop: bool = False

    @property
    def segment_count(self) -> int:
        """Nombre de segments dans la polyligne."""
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


Primitive = LineSegment | Polyline | RawPolyline | Circle | CircularArc | BezierPath


def primitive_source_edge_ids(primitive: Primitive) -> tuple[int, ...]:
    """Retourne les identifiants de branches sources de la primitive."""
    if hasattr(primitive, "source_edge_ids"):
        return tuple(getattr(primitive, "source_edge_ids"))
    return (primitive.source_edge_id,)


@dataclass(frozen=True)
class GeometryResult:
    """Géométrie indépendante ; localement une primitive par branche.

    La reconstruction globale peut ensuite regrouper plusieurs branches.
    """

    width: int
    height: int
    primitives: tuple[Primitive, ...]


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


def _closed_polyline_approximation(
    points: tuple[Point, ...], edge_id: int, tolerance: float
) -> Polyline | None:
    """Approximation Douglas-Peucker d'une boucle fermée en polygone régularisé."""
    if len(points) < 4 or points[0] != points[-1]:
        return None
    ring = points[:-1]
    ring_coords = np.array([(p.x, p.y) for p in ring], dtype=np.float64)
    if not np.isfinite(ring_coords).all():
        return None
    centroid = ring_coords.mean(axis=0)
    dists_to_centroid = np.linalg.norm(ring_coords - centroid, axis=1)
    pivot1 = int(np.argmax(dists_to_centroid))

    rotated = (*ring[pivot1:], *ring[:pivot1], ring[pivot1])
    rotated_coords = np.array([(p.x, p.y) for p in rotated], dtype=np.float64)

    dists_to_p0 = np.linalg.norm(rotated_coords[:-1] - rotated_coords[0], axis=1)
    max_d2 = float(np.max(dists_to_p0))
    if max_d2 == 0:
        return None
    pivot2 = int(np.argmax(dists_to_p0))

    idx1 = _simplified_indices(rotated_coords[:pivot2 + 1], tolerance)
    idx2 = _simplified_indices(rotated_coords[pivot2:], tolerance)
    indices = list(idx1[:-1]) + [pivot2 + j for j in idx2]

    while True:
        changed = False
        k = 1
        while k < len(indices) - 1:
            start, end = indices[k - 1], indices[k + 1]
            errors = _segment_distances(
                rotated_coords[start:end + 1], rotated_coords[start], rotated_coords[end]
            )
            if float(errors.max()) <= tolerance:
                indices.pop(k)
                changed = True
                k = max(1, k - 1)
            else:
                k += 1

        if len(indices) >= 5:
            start_idx = indices[-2]
            end_idx = indices[1]
            seam_coords = np.vstack([rotated_coords[start_idx:-1], rotated_coords[:end_idx + 1]])
            seam_errors = _segment_distances(
                seam_coords, rotated_coords[start_idx], rotated_coords[end_idx]
            )
            if float(seam_errors.max()) <= tolerance:
                shift = indices[1]
                ring_r = rotated[:-1]
                rotated = (*ring_r[shift:], *ring_r[:shift], ring_r[shift])
                rotated_coords = np.array([(p.x, p.y) for p in rotated], dtype=np.float64)
                indices = [i - shift for i in indices[1:-1]] + [len(rotated) - 1]
                changed = True
                continue
        if not changed:
            break

    if len(indices) < 4 or len(indices) >= len(points):
        return None

    poly_points = tuple(rotated[i] for i in indices)
    if len(set(poly_points[:-1])) < 3:
        return None

    coords = np.array([(p.x, p.y) for p in poly_points], dtype=np.float64)
    shoelace = 0.5 * abs(np.sum(coords[:-1, 0] * coords[1:, 1] - coords[:-1, 1] * coords[1:, 0]))
    if math.isclose(shoelace, 0.0, abs_tol=1e-6):
        return None

    errors = np.zeros(len(rotated), dtype=np.float64)
    for start, end in zip(indices, indices[1:]):
        errors[start:end + 1] = _segment_distances(
            rotated_coords[start:end + 1], rotated_coords[start], rotated_coords[end]
        )
    return Polyline(
        poly_points,
        float(errors.max()),
        float(np.sqrt(np.mean(errors ** 2))),
        edge_id,
        rotated,
        tuple(indices),
        is_loop=True,
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
    if is_loop:
        polyline = _closed_polyline_approximation(points, edge_id, polyline_tolerance)
    else:
        polyline = _polyline_approximation(points, edge_id, polyline_tolerance)
    if polyline is None or len(polyline.points) < 3:
        return None
    vectors = np.diff([(p.x, p.y) for p in polyline.points], axis=0)
    turns = np.arctan2(np.abs(vectors[:-1, 0]*vectors[1:, 1]-vectors[:-1, 1]*vectors[1:, 0]),
                       np.sum(vectors[:-1]*vectors[1:], axis=1))
    corner_angle_rad = math.pi/4

    if is_loop:
        cross_0 = abs(vectors[-1, 0] * vectors[0, 1] - vectors[-1, 1] * vectors[0, 0])
        dot_0 = float(np.sum(vectors[-1] * vectors[0]))
        turn_0 = math.atan2(cross_0, dot_0)

        # Si points[0] est déjà un coin géométrique dans la source, pivoter pour conserver l'origine canonique du graphe
        v_in = np.array([points[0].x - points[-2].x, points[0].y - points[-2].y])
        v_out = np.array([points[1].x - points[0].x, points[1].y - points[0].y])
        c = abs(v_in[0] * v_out[1] - v_in[1] * v_out[0])
        d = float(np.dot(v_in, v_out))
        start_is_corner = math.atan2(c, d) >= corner_angle_rad

        k_start = next((i for i, p in enumerate(polyline.points[:-1]) if p == points[0]), None)
        if start_is_corner and k_start is not None:
            target_k = k_start
        else:
            corner_indices = [i + 1 for i, turn in enumerate(turns) if turn >= corner_angle_rad]
            target_k = corner_indices[0] if (turn_0 < corner_angle_rad and corner_indices) else 0

        if target_k != 0:
            shift = polyline.source_indices[target_k]
            poly_ring = polyline.points[:-1]
            new_poly_points = (*poly_ring[target_k:], *poly_ring[:target_k], poly_ring[target_k])
            src_ring = polyline.source_points[:-1]
            new_source_points = (*src_ring[shift:], *src_ring[:shift], src_ring[shift])
            n_src_ring = len(src_ring)
            new_source_indices = (
                [idx - shift for idx in polyline.source_indices[target_k:-1]]
                + [idx + n_src_ring - shift for idx in polyline.source_indices[:target_k]]
                + [len(new_source_points) - 1]
            )
            polyline = replace(
                polyline,
                points=new_poly_points,
                source_points=new_source_points,
                source_indices=tuple(new_source_indices),
            )
            vectors = np.diff([(p.x, p.y) for p in polyline.points], axis=0)
            turns = np.arctan2(
                np.abs(vectors[:-1, 0] * vectors[1:, 1] - vectors[:-1, 1] * vectors[1:, 0]),
                np.sum(vectors[:-1] * vectors[1:], axis=1),
            )

    cuts = [polyline.source_indices[i+1] for i, turn in enumerate(turns) if turn >= corner_angle_rad]
    source_pts = polyline.source_points
    intervals = sorted({0, len(source_pts)-1, *cuts})
    if not any(
        (fit := fit_line(source_pts[a:b+1])) is not None and fit.fit_error_max > tolerance
        for a, b in zip(intervals, intervals[1:])
    ):
        return None
    result = fit_piecewise_bezier(source_pts, tolerance, cuts, source_edge_id=edge_id, is_loop=is_loop)
    if result is None or 3*len(result.segments)+1 >= len(source_pts):
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


def _trimmed_junction_slice(points: tuple[Point, ...], start_is_junction: bool,
                            end_is_junction: bool, tolerance_px: float) -> slice | None:
    """Calcule la tranche interne (cœur) en rognant les pixels d'inflexion aux jonctions."""
    if tolerance_px <= 0 or (not start_is_junction and not end_is_junction) or len(points) < 10:
        return None
    trim = min(max(1, int(math.ceil(tolerance_px))), len(points) // 5)
    start_idx = trim if start_is_junction else 0
    end_idx = len(points) - (trim if end_is_junction else 0)
    if end_idx - start_idx >= 4:
        return slice(start_idx, end_idx)
    return None


def _debias_line_at_junctions(points: tuple[Point, ...], edge_id: int,
                              core_slice: slice, tolerance_px: float) -> LineSegment | None:
    """Ajuste une droite sur le cœur et vérifie que la déviation totale reste modérée."""
    core_fit = fit_line(points[core_slice])
    if core_fit is None or core_fit.fit_error_max > tolerance_px:
        return None
    coordinates = np.array([(p.x, p.y) for p in points], dtype=np.float64)
    origin = np.array([core_fit.origin.x, core_fit.origin.y], dtype=np.float64)
    direction = np.array([core_fit.direction.x, core_fit.direction.y], dtype=np.float64)
    centered = coordinates - origin
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    all_errors = np.abs(centered @ normal)
    if float(all_errors.max()) > 2.0 * tolerance_px:
        return None
    projections = centered @ direction
    start_xy = origin + float(projections.min()) * direction
    end_xy = origin + float(projections.max()) * direction
    start = Point(float(start_xy[0]), float(start_xy[1]))
    end = Point(float(end_xy[0]), float(end_xy[1]))
    return LineSegment(start, end, core_fit.fit_error_max, core_fit.fit_error_rms, edge_id, points)


def _debias_arc_at_junctions(points: tuple[Point, ...], edge_id: int,
                             core_slice: slice, tolerance_px: float) -> CircularArc | None:
    """Ajuste un arc sur le cœur et vérifie le balayage et les erreurs radiales totales."""
    core_arc = fit_arc(points[core_slice], tolerance_px)
    if core_arc is None:
        return None
    coordinates = np.array([(p.x, p.y) for p in points], dtype=np.float64)
    center = np.array([core_arc.center.x, core_arc.center.y], dtype=np.float64)
    vectors = coordinates - center
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 1e-9):
        return None
    angles = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
    sweep = float(angles[-1] - angles[0])
    if sweep == 0 or abs(sweep) >= 2 * math.pi:
        return None
    if (sweep > 0) != (core_arc.sweep_angle_rad > 0):
        return None
    errors = np.abs(norms - core_arc.radius)
    if float(errors.max()) > 2.0 * tolerance_px:
        return None
    return CircularArc(core_arc.center, core_arc.radius, float(angles[0]), float(angles[-1]),
                       core_arc.fit_error_max, core_arc.fit_error_rms, edge_id, points)


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
    nodes_by_id = {node.id: node for node in graph.nodes}
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
                if detect_beziers and polyline_tolerance_px is not None and edge.points[0] == edge.points[-1]:
                    poly = _closed_polyline_approximation(edge.points, edge.id, polyline_tolerance_px)
                    if poly is not None:
                        primitives.append(poly)
                        continue
                primitives.append(RawPolyline(
                    edge.points, edge.id, True,
                    None if circle_fit is None else circle_fit.fit_error_max,
                    None if circle_fit is None else circle_fit.fit_error_rms,
                ))
            continue

        start_node = graph.nodes[edge.start_node_id] if 0 <= edge.start_node_id < len(graph.nodes) else nodes_by_id.get(edge.start_node_id)
        end_node = graph.nodes[edge.end_node_id] if 0 <= edge.end_node_id < len(graph.nodes) else nodes_by_id.get(edge.end_node_id)
        start_is_junction = start_node is not None and start_node.kind == "junction"
        end_is_junction = end_node is not None and end_node.kind == "junction"
        core_slice = _trimmed_junction_slice(edge.points, start_is_junction, end_is_junction,
                                             geometry_tolerance_px)

        fit = fit_line(edge.points)
        if fit is not None and fit.fit_error_max <= geometry_tolerance_px:
            primitives.append(LineSegment(
                fit.start, fit.end, fit.fit_error_max, fit.fit_error_rms,
                edge.id, edge.points,
            ))
            continue
        if core_slice is not None:
            debiased_line = _debias_line_at_junctions(edge.points, edge.id, core_slice,
                                                      geometry_tolerance_px)
            if debiased_line is not None:
                primitives.append(debiased_line)
                continue

        arc = fit_arc(edge.points, geometry_tolerance_px) if detect_arcs else None
        if arc is not None:
            primitives.append(CircularArc(
                arc.center, arc.radius, arc.start_angle_rad, arc.end_angle_rad,
                arc.fit_error_max, arc.fit_error_rms, edge.id, edge.points,
            ))
            continue
        if detect_arcs and core_slice is not None:
            debiased_arc = _debias_arc_at_junctions(edge.points, edge.id, core_slice,
                                                    geometry_tolerance_px)
            if debiased_arc is not None:
                primitives.append(debiased_arc)
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
