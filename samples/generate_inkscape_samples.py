"""Exécute l'adaptateur réel sur une rosace embarquée, placée et tournée.

INKSCAPE_EXTENSIONS_DIR doit désigner le dossier contenant le paquet inkex.
"""

import base64
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"extensions"/"inkscape"), os.environ["INKSCAPE_EXTENSIONS_DIR"]]

from vectolaz import VectolazExtension

target = ROOT/"debug_output"/"inkscape"
target.mkdir(parents=True, exist_ok=True)
png = ROOT/"samples"/"rosace.png"
source = target/"rosace_source.svg"
uri = "data:image/png;base64,"+base64.b64encode(png.read_bytes()).decode()
source.write_text(
    '<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="210mm" viewBox="0 0 800 800">'
    '<g id="placed-image" transform="rotate(12 400 400)">'
    f'<image id="rosace" x="100" y="100" width="600" height="600" opacity="0.3" href="{uri}"/>'
    '</g></svg>', encoding="utf-8")
output = target/"rosace_vectorized.svg"
with output.open("wb") as stream:
    VectolazExtension().run(["--id=rosace", f"--engine-python={sys.executable}",
                            "--threshold-auto=false", "--threshold=160", "--geometry-tolerance=5",
                            "--global-arcs=true", str(source)], output=stream)
root = ET.fromstring(output.read_bytes())
group = root[0][1][0]
assert len(group) == 7
assert sum(e.tag.endswith("}circle") for e in group) == 1
assert sum(e.tag.endswith("}path") for e in group) == 6
print(f"Contrôle Inkscape : {output} (1 cercle + 6 arcs, image conservée, rotation 12°)")
