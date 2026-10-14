"""Tests XML, alignement, fermeture et export CLI de SVG autonomes."""

import base64
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

from raster2vector import (
    Point, GeometryResult, LineSegment, Polyline, RawPolyline, export_svg, svg_document,
)
from raster2vector.cli import main

NS = {"svg": "http://www.w3.org/2000/svg"}


def geometry_fixture() -> GeometryResult:
    segment = LineSegment(Point(2.25, 4.5), Point(35.5, 4.5), 0.1, 0.05, 0, ())
    vertices = (Point(5, 10), Point(5, 20), Point(30, 20))
    polyline = Polyline(vertices, 0, 0, 1, vertices, (0, 1, 2))
    loop = (Point(10, 10), Point(20, 10), Point(20, 20), Point(10, 20), Point(10, 10))
    return GeometryResult(40, 30, (segment, polyline, RawPolyline(loop, 2, True)))


def png_bytes(width: int = 40, height: int = 30) -> bytes:
    image = np.full((height, width, 4), 255, dtype=np.uint8)
    image[12:16, 5:35, :3] = 0
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def test_svg_uses_strokes_and_original_dimensions() -> None:
    document = svg_document(geometry_fixture(), stroke_width=1.75)
    root = ET.fromstring(document)
    assert root.tag == f"{{{NS['svg']}}}svg"
    assert root.attrib["viewBox"] == "0 0 40 30"
    assert root.attrib["width"] == "40px" and root.attrib["height"] == "30px"
    assert root.find("svg:image", NS) is None
    group = root.find("svg:g", NS)
    assert group is not None
    assert group.attrib["fill"] == "none"
    assert group.attrib["stroke"] == "black"
    assert group.attrib["stroke-width"] == "1.75"
    assert group.attrib["transform"] == "translate(0.5 0.5)"
    paths = group.findall("svg:path", NS)
    assert len(paths) == 3
    assert all(path.attrib["fill"] == "none" and path.attrib["stroke"] == "black" for path in paths)
    assert paths[0].attrib["d"] == "M 2.25 4.5 L 35.5 4.5"
    assert paths[1].attrib["d"] == "M 5 10 L 5 20 L 30 20"
    assert paths[2].attrib["d"] == "M 10 10 L 20 10 L 20 20 L 10 20 Z"
    assert {path.attrib["id"] for path in paths} == {"edge-0", "edge-1", "edge-2"}


def test_topological_loop_does_not_invent_a_contact_segment() -> None:
    points = (Point(10, 10), Point(30, 10), Point(30, 20), Point(10, 11))
    root = ET.fromstring(svg_document(GeometryResult(40, 30, (RawPolyline(points, 0, True),))))
    path = root.find("svg:g/svg:path", NS)
    assert path is not None
    assert path.attrib["d"].endswith("L 10 11")
    assert "Z" not in path.attrib["d"]


def test_overlay_embeds_png_before_vectors() -> None:
    background = png_bytes()
    root = ET.fromstring(svg_document(geometry_fixture(), background_png=background))
    raster = root.find("svg:image", NS)
    assert raster is not None
    assert list(root)[0] is raster
    assert raster.attrib["width"] == "40" and raster.attrib["height"] == "30"
    assert raster.attrib["x"] == raster.attrib["y"] == "0"
    assert raster.attrib["opacity"] == "0.35"
    uri = raster.attrib["href"]
    assert uri == raster.attrib["{http://www.w3.org/1999/xlink}href"]
    assert base64.b64decode(uri.split(",", 1)[1]) == background


def test_export_unicode_path_and_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "résultats" / "dessin été.svg"
    export_svg(geometry_fixture(), destination)
    before = destination.read_text(encoding="utf-8")
    assert before.startswith('<?xml version="1.0" encoding="utf-8"?>')
    export_svg(GeometryResult(40, 30, ()), destination)
    assert ET.parse(destination).getroot().findall("svg:g/svg:path", NS) == []


@pytest.mark.parametrize("stroke_width", [0, -1, float("nan"), float("inf")])
def test_invalid_width_does_not_replace_existing_file(tmp_path: Path, stroke_width: float) -> None:
    destination = tmp_path / "keep.svg"
    destination.write_text("original", encoding="utf-8")
    with pytest.raises(ValueError, match="épaisseur"):
        export_svg(geometry_fixture(), destination, stroke_width)
    assert destination.read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize("width, height", [(0, 30), (40, 0), (-1, 30)])
def test_invalid_dimensions(width: int, height: int) -> None:
    with pytest.raises(ValueError, match="dimensions"):
        svg_document(GeometryResult(width, height, ()))


def test_nonfinite_coordinates() -> None:
    segment = LineSegment(Point(float("nan"), 1), Point(10, 1), 0, 0, 0, ())
    with pytest.raises(ValueError, match="coordonnées"):
        svg_document(GeometryResult(40, 30, (segment,)))


@pytest.mark.parametrize("background", [b"not a png", png_bytes(20, 30)])
def test_invalid_background(background: bytes) -> None:
    with pytest.raises(ValueError):
        svg_document(geometry_fixture(), background_png=background)


def test_cli_output_and_overlay(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(png_bytes())
    before = source.read_bytes()
    output = tmp_path / "exports" / "result.svg"
    debug = tmp_path / "debug"
    assert main([str(source), "--output", str(output), "--stroke-width", "2.5",
                 "--debug-dir", str(debug), "--debug-svg"]) == 0
    root = ET.parse(output).getroot()
    group = root.find("svg:g", NS)
    assert group is not None and group.attrib["stroke-width"] == "2.5"
    assert len(group.findall("svg:path", NS)) == 1
    assert root.find("svg:image", NS) is None
    overlay = ET.parse(debug / "overlay.svg").getroot()
    assert overlay.find("svg:image", NS) is not None
    assert source.read_bytes() == before
    assert "SVG de comparaison" in capsys.readouterr().out


def test_cli_requires_debug_directory(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main([str(tmp_path / "source.png"), "--debug-svg"])
    assert exc.value.code == 2


def test_cli_cannot_replace_source(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(png_bytes())
    before = source.read_bytes()
    assert main([str(source), "--output", str(source)]) == 1
    assert "écraserait" in capsys.readouterr().err
    assert source.read_bytes() == before


@pytest.mark.parametrize("name, debug_svg", [("geometry.png", False), ("overlay.svg", True)])
def test_cli_rejects_conflicting_destinations(tmp_path: Path, name: str, debug_svg: bool) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(png_bytes())
    debug = tmp_path / "debug"
    arguments = [str(source), "--output", str(debug / name), "--debug-dir", str(debug)]
    if debug_svg:
        arguments.append("--debug-svg")
    assert main(arguments) == 1
    assert not debug.exists()


def test_cli_directory_output_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(png_bytes())
    assert main([str(source), "--output", str(tmp_path)]) == 1
    assert "Erreur :" in capsys.readouterr().err
