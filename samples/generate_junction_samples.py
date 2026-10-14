"""Intersections raster reproductibles pour l'étape 11."""

from pathlib import Path
import cv2
import numpy as np
from raster2vector.image_io import save_png


def main() -> None:
    directory = Path(__file__).resolve().parent / "junction_cases"
    directory.mkdir(parents=True, exist_ok=True)
    cases = {
        "T": [((15, 70), (185, 70)), ((100, 20), (100, 70))],
        "cross": [((15, 70), (185, 70)), ((100, 20), (100, 125))],
        "double_T": [((15, 45), (185, 45)), ((15, 100), (185, 100)), ((100, 45), (100, 100))],
        "oblique": [((20, 120), (175, 25)), ((30, 25), (175, 120))],
    }
    for name, segments in cases.items():
        image = np.full((151, 201), 255, dtype=np.uint8)
        for a, b in segments:
            cv2.line(image, a, b, 0, 5)
        save_png(directory / f"{name}.png", image)
    image = np.full((151, 201), 255, dtype=np.uint8)
    cv2.ellipse(image, (100, 95), (55, 55), 0, 180, 270, 0, 5)
    cv2.line(image, (45, 95), (185, 95), 0, 5)
    cv2.line(image, (100, 40), (100, 125), 0, 5)
    save_png(directory / "loop_contacts.png", image)
    image = np.full((151, 201), 255, dtype=np.uint8)
    cv2.ellipse(image, (100, 95), (55, 55), 0, 180, 270, 0, 5)
    cv2.line(image, (15, 95), (185, 95), 0, 5)
    save_png(directory / "arc_contacts.png", image)
    print(f"Six images générées dans {directory}")


if __name__ == "__main__":
    main()
