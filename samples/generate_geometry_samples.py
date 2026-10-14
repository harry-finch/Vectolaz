"""Génère cinq PNG pour inspecter la reconstruction de l'étape 4.

Depuis la racine du projet : python samples/generate_geometry_samples.py
"""

from pathlib import Path

import cv2
import numpy as np

from raster2vector.image_io import save_png


def main() -> None:
    """Écrit des traits synthétiques épais de 5 pixels, sur fond blanc."""
    directory = Path(__file__).resolve().parent / "geometry_cases"
    directory.mkdir(parents=True, exist_ok=True)
    x = np.arange(15, 146)
    paths = {
        "horizontal": np.array([(15, 40), (145, 40)], dtype=np.int32),
        "vertical": np.array([(80, 10), (80, 70)], dtype=np.int32),
        "diagonal": np.array([(15, 10), (145, 70)], dtype=np.int32),
        "noisy_line": np.column_stack((x, np.rint(40 + np.sin(0.3 * x)))).astype(np.int32),
        "curve": np.column_stack((x, np.rint(20 + 35 * np.sin(np.pi * (x - 15) / 130)))).astype(np.int32),
    }
    for name, points in paths.items():
        image = np.full((81, 161), 255, dtype=np.uint8)
        cv2.polylines(image, [points], False, 0, 5)
        save_png(directory / f"{name}.png", image)
    print(f"Cinq images générées dans {directory}")


if __name__ == "__main__":
    main()
