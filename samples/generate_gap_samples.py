"""Exemples de coupures et contre-exemples pour le raccordement optionnel."""

from pathlib import Path
import cv2
import numpy as np
from raster2vector.image_io import save_png


def main() -> None:
    directory = Path(__file__).resolve().parent / "gap_cases"
    directory.mkdir(parents=True, exist_ok=True)
    cases = {
        "aligned": [((15, 60), (95, 60)), ((105, 60), (185, 60))],
        "diagonal": [((20, 100), (95, 62)), ((105, 57), (180, 20))],
        "perpendicular": [((15, 55), (95, 55)), ((105, 60), (105, 110))],
        "parallel": [((20, 55), (180, 55)), ((20, 65), (180, 65))],
        "obstacle": [((15, 60), (95, 60)), ((105, 60), (185, 60)), ((100, 20), (100, 100))],
        "ambiguous": [((15, 60), (95, 60)), ((105, 56), (180, 32)), ((105, 64), (180, 88))],
    }
    for name, segments in cases.items():
        image = np.full((131, 201), 255, dtype=np.uint8)
        for start, end in segments:
            cv2.line(image, start, end, 0, 3)
        save_png(directory / f"{name}.png", image)
    print(f"Six images générées dans {directory}")


if __name__ == "__main__":
    main()
