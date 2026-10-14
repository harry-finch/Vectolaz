"""Génère cercle raster, cercle légèrement bruité et ellipse pour l'étape 7."""

from pathlib import Path

import cv2
import numpy as np

from raster2vector.image_io import save_png


def main() -> None:
    """Écrit trois dessins de 5 pixels d'épaisseur, avec fermeture explicite."""
    directory = Path(__file__).resolve().parent / "circle_cases"
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("circle", "noisy_circle", "ellipse"):
        image = np.full((161, 201), 255, dtype=np.uint8)
        if name == "circle":
            cv2.circle(image, (100, 80), 55, 0, 5)
        elif name == "ellipse":
            cv2.ellipse(image, (100, 80), (75, 40), 0, 0, 360, 0, 5)
        else:
            angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
            radii = 55 + 0.6 * np.sin(7 * angles)
            points = np.rint(np.column_stack((100 + radii * np.cos(angles),
                                              80 + radii * np.sin(angles)))).astype(np.int32)
            cv2.polylines(image, [points], True, 0, 5)
        save_png(directory / f"{name}.png", image)
    print(f"Trois images générées dans {directory}")


if __name__ == "__main__":
    main()
