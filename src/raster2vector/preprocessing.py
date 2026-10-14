"""Niveaux de gris, binarisation et nettoyage sans modifier la géométrie.

Tous les tableaux s'indexent en (row, col). Les masques sont booléens :
False (0) pour le fond, True (1) pour le trait.
"""

from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

Inversion = Literal["auto", "yes", "no"]


def automatic_threshold(grayscale: NDArray[np.uint8]) -> int:
    """Seuil global d'Otsu sur les gris 8 bits, sans filtrage ni changement de polarité.

    La polarité reste choisie séparément par binarize. Un fond non uniforme
    peut nécessiter un seuil manuel ; aucun seuillage adaptatif n'est appliqué.
    """
    if grayscale.ndim != 2 or grayscale.dtype != np.uint8 or grayscale.size == 0:
        raise ValueError("Une image en niveaux de gris uint8 non vide est requise.")
    threshold, _ = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return int(threshold)


def to_grayscale(image: NDArray[np.uint8] | NDArray[np.uint16]) -> NDArray[np.uint8]:
    """Convertit un raster gris, BGR ou BGRA en niveaux de gris 8 bits.

    Les PNG 16 bits sont ramenés de [0, 65535] à [0, 255]. L'alpha est
    composé sur du blanc afin que les pixels transparents ne deviennent pas
    des traits noirs. Aucun lissage ni changement de dimensions n'est appliqué.
    """
    if image.dtype not in (np.uint8, np.uint16):
        raise ValueError("L'image doit utiliser des pixels uint8 ou uint16.")
    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError("Dimensions d'image invalides.")
    if image.dtype == np.uint16:
        image = np.rint(image.astype(np.float64) / 257.0).astype(np.uint8)
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float64) / 255.0
        image = np.rint(image[:, :, :3] * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    elif image.shape[2] != 3:
        raise ValueError("L'image doit avoir 3 canaux BGR ou 4 canaux BGRA.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binarize(
    grayscale: NDArray[np.uint8],
    threshold: int = 128,
    invert: Inversion = "auto",
) -> NDArray[np.bool_]:
    """Isole les traits avec un seuil inclusif : sombre = intensité <= seuil.

    ``yes`` inverse les intensités binaires : les pixels sombres sont le trait.
    ``no`` conserve les pixels clairs comme trait. ``auto`` considère la classe
    majoritaire sur le bord de l'image comme fond. En cas d'égalité, il suppose
    un fond clair. Cette heuristique peut être forcée si le dessin remplit le bord.
    """
    if grayscale.ndim != 2 or grayscale.dtype != np.uint8 or grayscale.size == 0:
        raise ValueError("Une image en niveaux de gris uint8 non vide est requise.")
    if not 0 <= threshold <= 255:
        raise ValueError("Le seuil doit être compris entre 0 et 255.")
    if invert not in ("auto", "yes", "no"):
        raise ValueError("L'inversion doit être auto, yes ou no.")
    dark = grayscale <= threshold
    if invert == "auto":
        border = np.zeros(grayscale.shape, dtype=np.bool_)
        border[0, :] = border[-1, :] = True
        border[:, 0] = border[:, -1] = True
        invert = "no" if np.count_nonzero(dark[border]) > border.sum() / 2 else "yes"
    return dark if invert == "yes" else ~dark


def remove_small_objects(
    mask: NDArray[np.bool_], min_component_size: int = 0
) -> NDArray[np.bool_]:
    """Supprime les composantes de surface strictement inférieure au minimum.

    La connexité est de 8 voisins pour préserver les traits diagonaux. 0
    désactive le nettoyage. Aucun raccordement, comblement de trou, érosion ou
    dilatation n'est réalisé : les composantes conservées restent identiques.
    """
    if mask.ndim != 2 or mask.dtype != np.bool_ or mask.size == 0:
        raise ValueError("Un masque booléen 2D non vide est requis.")
    if min_component_size < 0:
        raise ValueError("La taille minimale des composantes doit être positive ou nulle.")
    if min_component_size <= 1:
        return mask.copy()
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    keep = stats[:, cv2.CC_STAT_AREA] >= min_component_size
    keep[0] = False  # L'étiquette 0 représente le fond, jamais un objet.
    return keep[labels]
