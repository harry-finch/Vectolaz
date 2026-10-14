"""Chargement raster, sans prétraitement ni conversion géométrique."""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


def load_image(path: str | Path) -> NDArray[np.uint8] | NDArray[np.uint16]:
    """Charge une image avec ses canaux et sa profondeur d'origine.

    Les dimensions du tableau sont (hauteur, largeur[, canaux]). Les indices
    sont donc (row, col), pas des coordonnées géométriques (x, y).
    OpenCV retourne les canaux couleur dans l'ordre BGR ou BGRA.
    La lecture via pathlib permet les chemins Unicode sur les trois plateformes.

    Raises:
        OSError: fichier absent, inaccessible ou chemin pointant vers un dossier.
        ValueError: fichier vide ou non décodable par OpenCV.
    """
    image_path = Path(path)
    encoded = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError(f"Image vide : {image_path}")
    try:
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:
        raise ValueError(f"Image illisible : {image_path}") from exc
    if image is None:
        raise ValueError(f"Image illisible : {image_path}")
    return image


def save_png(path: str | Path, image: NDArray[np.uint8]) -> None:
    """Écrit un PNG 8 bits, y compris vers un chemin Unicode.

    Le dossier parent doit exister. Les erreurs d'écriture remontent à l'appelant.
    """
    try:
        success, encoded = cv2.imencode(".png", image)
    except cv2.error as exc:
        raise ValueError(f"Impossible d'encoder le PNG : {path}") from exc
    if not success:
        raise ValueError(f"Impossible d'encoder le PNG : {path}")
    Path(path).write_bytes(encoded.tobytes())

