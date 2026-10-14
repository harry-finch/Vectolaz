"""Génère les quatre PNG de validation de l'étape 5.

Depuis la racine du projet : python samples/generate_polyline_samples.py
"""

from pathlib import Path

import cv2
import numpy as np

from raster2vector.image_io import save_png


def main() -> None:
    """Écrit un L, un zigzag, un trait presque droit et une courbe douce."""
    directory = Path(__file__).resolve().parent / "polyline_cases"
    directory.mkdir(parents=True, exist_ok=True)
    x = np.arange(15, 146)
    paths = {
        "L": np.array([(20, 15), (20, 65), (140, 65)], dtype=np.int32),
        "zigzag": np.array([(15, 65), (45, 20), (80, 65), (115, 20), (145, 65)], dtype=np.int32),
        "almost_straight": np.column_stack((x, np.rint(40 + np.sin(0.3 * x)))).astype(np.int32),
        "smooth_curve": np.column_stack((x, np.rint(40 + 20 * np.sin(np.pi * (x - 15) / 130)))).astype(np.int32),
    }
    for name, points in paths.items():
        image = np.full((91, 161), 255, dtype=np.uint8)
        cv2.polylines(image, [points], False, 0, 5)
        save_png(directory / f"{name}.png", image)
    print(f"Quatre images générées dans {directory}")


if __name__ == "__main__":
    main()
