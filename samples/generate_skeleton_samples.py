"""Génère les six PNG synthétiques de validation de l'étape 2.

Depuis la racine du projet : python samples/generate_skeleton_samples.py
"""

from pathlib import Path

import cv2
import numpy as np

from raster2vector.image_io import save_png


def main() -> None:
    """Écrit les exemples en traits noirs sur fond blanc, épaisseur 7 pixels."""
    directory = Path(__file__).resolve().parent / "skeleton_cases"
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "horizontal": [[(15, 40), (85, 40)]],
        "diagonal": [[(20, 15), (80, 65)]],
        "L": [[(25, 15), (25, 60), (80, 60)]],
        "T": [[(15, 20), (85, 20)], [(50, 20), (50, 65)]],
        "cross": [[(15, 40), (85, 40)], [(50, 15), (50, 65)]],
    }
    for name in (*paths, "circle"):
        image = np.full((81, 101), 255, dtype=np.uint8)
        if name == "circle":
            cv2.circle(image, (50, 40), 25, 0, 7)
        else:
            for points in paths[name]:
                cv2.polylines(image, [np.array(points, dtype=np.int32)], False, 0, 7)
        save_png(directory / f"{name}.png", image)
    print(f"Six images générées dans {directory}")


if __name__ == "__main__":
    main()
