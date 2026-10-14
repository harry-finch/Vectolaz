"""Courbes libres et témoin anguleux pour l'étape 9."""

from pathlib import Path
import cv2
import numpy as np
from raster2vector.image_io import save_png


def main() -> None:
    directory = Path(__file__).resolve().parent / "bezier_cases"
    directory.mkdir(parents=True, exist_ok=True)
    x = np.linspace(20, 260, 961)
    cases = {
        "s_curve": np.column_stack((x, 110+40*np.sin((x-20)*2*np.pi/240))),
        "wave": np.column_stack((x, 110+15*np.sin((x-20)*4*np.pi/240))),
        "corner": np.array([(20, 30), (20, 180), (260, 180)]),
    }
    for name, points in cases.items():
        image = np.full((221, 281), 255, dtype=np.uint8)
        cv2.polylines(image, [np.rint(points).astype(np.int32)], False, 0, 5)
        save_png(directory / f"{name}.png", image)
    print(f"Trois images générées dans {directory}")


if __name__ == "__main__":
    main()
