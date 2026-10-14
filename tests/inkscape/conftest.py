"""L'adaptateur est importable sans installer inkex dans le moteur."""

import os
from pathlib import Path
import sys

ADAPTER = Path(__file__).resolve().parents[2]/"extensions"/"inkscape"
sys.path.insert(0, str(ADAPTER))
if os.environ.get("INKSCAPE_EXTENSIONS_DIR"):
    sys.path.insert(0, os.environ["INKSCAPE_EXTENSIONS_DIR"])
