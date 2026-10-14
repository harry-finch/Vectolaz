"""Génère les exemples raster de l'étape 8 : arcs circulaires et contre-exemple."""

from pathlib import Path

import cv2
import numpy as np

from raster2vector.image_io import save_png


def main() -> None:
    directory = Path(__file__).resolve().parent / "arc_cases"
    directory.mkdir(parents=True, exist_ok=True)
    cases = {"quarter": (0, 90), "half": (180, 360), "short": (-20, 25),
             "large": (20, 290), "non_arc": (180, 360)}
    for name, (start, end) in cases.items():
        image = np.full((181, 201), 255, dtype=np.uint8)
        axes = (75, 25) if name == "non_arc" else (65, 65)
        cv2.ellipse(image, (100, 90), axes, 0, start, end, 0, 5)
        save_png(directory / f"{name}.png", image)
    print(f"Cinq images générées dans {directory}")


if __name__ == "__main__":
    main()
