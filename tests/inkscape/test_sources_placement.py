import base64
from pathlib import Path
import struct

import pytest

from image_source import read_png
from placement import image_mapping

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jH1EAAAAASUVORK5CYII=")


def test_embedded_png_and_whitespace() -> None:
    href = "data:image/png;base64,"+base64.encodebytes(PNG).decode()
    assert read_png(href, None) == (PNG, 1, 1)


@pytest.mark.parametrize("kind", ["absolute", "relative", "uri", "encoded_relative"])
def test_linked_unicode_png(tmp_path: Path, kind: str) -> None:
    source = tmp_path/"rosace été.png"
    source.write_bytes(PNG)
    href = {"absolute": str(source), "relative": source.name, "uri": source.as_uri(),
            "encoded_relative": "rosace%20%C3%A9t%C3%A9.png"}[kind]
    assert read_png(href, str(tmp_path/"document.svg")) == (PNG, 1, 1)


@pytest.mark.parametrize("href", ["", "https://example.org/a.png", "file://remote/a.png",
                                   "data:image/jpeg;base64,AAAA", "data:image/png;base64,!",
                                   "data:image/png;base64,AAAA"])
def test_invalid_sources(href: str) -> None:
    with pytest.raises(ValueError):
        read_png(href, None)


def test_missing_and_unsaved_link(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Enregistrez"):
        read_png("relative.png", None)
    with pytest.raises(ValueError, match="introuvable"):
        read_png(str(tmp_path/"missing.png"), None)
    data = PNG[:16]+struct.pack(">II", 0, 1)+PNG[24:]
    with pytest.raises(ValueError, match="dimensions"):
        read_png("data:image/png;base64,"+base64.b64encode(data).decode(), None)


def test_meet_none_and_default() -> None:
    assert image_mapping(10, 20, 300, 300, 200, 100) == (1.5, 1.5, 10, 95)
    assert image_mapping(10, 20, 300, 300, 200, 100, "none") == (1.5, 3, 10, 20)


@pytest.mark.parametrize("x", ["Min", "Mid", "Max"])
@pytest.mark.parametrize("y,factor", [("Min", 0), ("Mid", 0.5), ("Max", 1)])
def test_all_alignments(x: str, y: str, factor: float) -> None:
    assert image_mapping(10, 20, 300, 300, 200, 100, f"x{x}Y{y} meet") == (1.5, 1.5, 10, 20+150*factor)
    assert image_mapping(10, 20, 300, 300, 100, 200, f"x{y}Y{x} meet") == (1.5, 1.5, 10+150*factor, 20)


@pytest.mark.parametrize("aspect", ["xMidYMid slice", "invalid", "xMidYMid foo", "none meet extra"])
def test_unsupported_aspect(aspect: str) -> None:
    with pytest.raises(ValueError):
        image_mapping(0, 0, 100, 100, 200, 100, aspect)


@pytest.mark.parametrize("width", [0, -1, float("nan"), float("inf")])
def test_bad_dimensions(width: float) -> None:
    with pytest.raises(ValueError):
        image_mapping(0, 0, width, 100, 200, 100)
