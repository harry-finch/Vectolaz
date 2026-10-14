"""Tests sur l'API inkex réelle, activés avec INKSCAPE_EXTENSIONS_DIR."""

import base64
import io
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

if not os.environ.get("INKSCAPE_EXTENSIONS_DIR"):
    pytest.skip("Définir INKSCAPE_EXTENSIONS_DIR pour tester l'API Inkscape réelle.", allow_module_level=True)
inkex = pytest.importorskip("inkex")
import vectolaz
from vectolaz import VectolazExtension, image_transform, run_engine, validated_vectors

ROOT = Path(__file__).resolve().parents[2]


def document(attributes: str = "", parent: str = "", href: str | None = None) -> bytes:
    href = href or "data:image/png;base64,"+base64.b64encode((ROOT/"samples"/"line.png").read_bytes()).decode()
    from xml.sax.saxutils import quoteattr
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" viewBox="0 0 210 297">'
            f'<g id="parent" {parent}><image id="source" href={quoteattr(href)} x="10" y="20" '
            f'width="160" height="100" {attributes}/></g></svg>').encode()


def prepare(tmp_path: Path, data: bytes, ids: tuple[str, ...] = ("source",)) -> VectolazExtension:
    source = tmp_path/"input.svg"
    source.write_bytes(data)
    extension = VectolazExtension()
    extension.parse_arguments([*(f"--id={identifier}" for identifier in ids),
                               f"--engine-python={sys.executable}", str(source)])
    extension.load_raw()
    return extension


@pytest.mark.parametrize("transform", ["translate(4,7)", "rotate(30)", "scale(2,0.5)",
                                       "scale(-1,1)", "matrix(1,0.2,0.4,1,3,5)"])
def test_affine_and_parent_mapping(transform: str) -> None:
    root = inkex.load_svg(document(f'transform="{transform}"', 'transform="translate(30,40) rotate(15)"')).getroot()
    image = root[0][0]
    mapping = image_transform(image, 160, 100)
    source = (13.5, 17.5)
    expected = image.transform.apply_to_point((23.5, 37.5))
    assert tuple(mapping.apply_to_point(source)) == pytest.approx(tuple(expected))
    assert root.get("viewBox") == "0 0 210 297"


def test_physical_units_and_meet() -> None:
    root = inkex.load_svg(document()).getroot()
    image = root[0][0]
    image.set("width", "300")
    image.set("height", "300")
    mapping = image_transform(image, 200, 100)
    assert tuple(mapping.apply_to_point((0.5, 0.5))) == pytest.approx((10.75, 95.75))
    image.set("x", "1in")
    assert image_transform(image, 200, 100).apply_to_point((0, 0)).x == 96


@pytest.mark.parametrize("attributes,parent", [
    ('style="width:120px"', ''), ('width="20%"', ''),
    ('transform="scale(0)"', ''), ('', 'transform="scale(0)"'),
    ('style="transform:rotate(20deg)"', ''), ('preserveAspectRatio="xMidYMid slice"', ''),
    ('clip-path="url(#clip)"', ''), ('', 'style="filter:url(#f)"'),
    ('mask="url(#m)"', ''),
])
def test_unsupported_placement(attributes: str, parent: str) -> None:
    data = document("", parent)
    root = inkex.load_svg(data).getroot()
    # Set attributes through XML to avoid duplicate width in the fixture.
    if attributes:
        extra = ET.fromstring(f"<image {attributes}/>").attrib
        for name, value in extra.items():
            root[0][0].set(name, value)
    with pytest.raises(ValueError):
        image_transform(root[0][0], 160, 100)


@pytest.mark.parametrize("ids", [(), ("parent",), ("source", "parent")])
def test_selection_rejected_without_document_mutation(tmp_path: Path, ids: tuple[str, ...]) -> None:
    extension = prepare(tmp_path, document(), ids)
    before = inkex.etree.tostring(extension.svg)
    with pytest.raises(inkex.AbortExtension, match="exactement"):
        extension.effect()
    assert inkex.etree.tostring(extension.svg) == before


@pytest.mark.parametrize("linked", [False, True])
def test_real_engine_insertion_unique_ids_and_image_retained(tmp_path: Path, monkeypatch, linked: bool) -> None:
    href = (ROOT/"samples"/"line.png").as_uri() if linked else None
    extension = prepare(tmp_path, document('transform="rotate(15)"', 'transform="scale(0.5)"', href=href))
    parent = extension.svg[0]
    before = inkex.etree.tostring(parent[0])
    # Ensure the engine does not inherit Inkscape's Python configuration.
    monkeypatch.setenv("PYTHONHOME", "/invalid/python/home")
    extension.effect()
    assert len(parent) == 2 and inkex.etree.tostring(parent[0]) == before
    wrapper = parent[1]
    assert wrapper.get("data-source-image") == "source"
    assert tuple(wrapper[0].transform.apply_to_point((0, 0))) == pytest.approx((0.5, 0.5))
    assert len(wrapper[0]) > 0 and all(isinstance(e, (inkex.PathElement, inkex.Circle)) for e in wrapper[0])
    extension.effect()
    ids = [element.get("id") for element in extension.svg.iter() if element.get("id")]
    assert len(parent) == 3 and len(ids) == len(set(ids))


def test_cli_returns_only_modified_svg(tmp_path: Path) -> None:
    source = tmp_path/"input.svg"
    source.write_bytes(document())
    output = io.BytesIO()
    VectolazExtension().run(["--id=source", f"--engine-python={sys.executable}", str(source)], output=output)
    root = ET.fromstring(output.getvalue())
    assert len(root[0]) == 2


def test_relative_link_uses_original_document_path(tmp_path: Path, monkeypatch) -> None:
    original = tmp_path/"original"
    original.mkdir()
    (original/"image.png").write_bytes((ROOT/"samples"/"line.png").read_bytes())
    monkeypatch.setenv("DOCUMENT_PATH", str(original/"drawing.svg"))
    extension = prepare(tmp_path, document(href="image.png"))
    extension.effect()
    assert len(extension.svg[0]) == 2


def options(**overrides) -> SimpleNamespace:
    values = dict(threshold_auto=True, threshold=128, geometry_tolerance=2, gap_tolerance=0,
                  stroke_width=1, min_component_size=0, global_arcs=False)
    return SimpleNamespace(**(values | overrides))


@pytest.mark.parametrize("failure", ["timeout", "exit", "missing", "launch"])
def test_engine_failures(tmp_path: Path, monkeypatch, failure: str) -> None:
    def fake(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args[0], 120)
        if failure == "launch":
            raise OSError("missing executable")
        return SimpleNamespace(returncode=1 if failure == "exit" else 0, stdout=b"", stderr=b"bad engine")
    monkeypatch.setattr(vectolaz.subprocess, "run", fake)
    with pytest.raises(ValueError):
        run_engine(sys.executable, tmp_path/"input.png", tmp_path/"out.svg", options())


def test_failed_engine_does_not_mutate_document(tmp_path: Path, monkeypatch) -> None:
    extension = prepare(tmp_path, document())
    before = inkex.etree.tostring(extension.svg)
    def fail(*args):
        raise ValueError("moteur indisponible")
    monkeypatch.setattr(vectolaz, "run_engine", fail)
    with pytest.raises(inkex.AbortExtension, match="indisponible"):
        extension.effect()
    assert inkex.etree.tostring(extension.svg) == before


@pytest.mark.parametrize("data", [b"invalid", b'<svg xmlns="http://www.w3.org/2000/svg"/>',
    b'<!DOCTYPE svg><svg/>', b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"><g id="vectors" transform="translate(0.5 0.5)"><script/></g></svg>'])
def test_malformed_engine_output(data: bytes) -> None:
    with pytest.raises(ValueError):
        validated_vectors(data, 1, 1)


def test_inx_descriptor() -> None:
    from inkex.inx import InxFile
    descriptor = InxFile(str(ROOT/"extensions"/"inkscape"/"vectolaz.inx"))
    assert descriptor.ident == "org.vectolaz.centerline"
