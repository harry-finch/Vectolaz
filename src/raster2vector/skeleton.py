"""Squelettisation 2D des masques de traits, sans reconstruction géométrique."""

import numpy as np
from numpy.typing import NDArray
from skimage.morphology import skeletonize


def skeletonize_mask(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Réduit un masque de traits à un squelette booléen avec Zhang-Suen.

    Entrée et sortie : fond=False, trait=True, indices (row, col). La méthode
    déterministe de scikit-image conserve la topologie en amincissant les
    composantes. La sortie conserve les dimensions et ne modifie pas l'entrée.
    Un masque sans trait produit un squelette vide.

    Il s'agit d'un amincissement topologique, pas d'un calcul exact de l'axe
    médian : un décalage local d'environ un pixel peut apparaître, notamment
    pour une épaisseur paire. Aucun raccordement ou élagage n'est effectué.
    """
    if mask.ndim != 2 or mask.dtype != np.bool_ or mask.size == 0:
        raise ValueError("Un masque booléen 2D non vide est requis.")
    return skeletonize(np.ascontiguousarray(mask), method="zhang")
