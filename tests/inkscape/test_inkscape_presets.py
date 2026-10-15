"""Tests des préréglages (presets), de l'élagage des barbes et de l'interface Inkscape."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

import vectolaz
from vectolaz import PRESETS, VectolazExtension, resolve_options, run_engine

INX_PATH = Path(__file__).resolve().parents[2] / "extensions" / "inkscape" / "vectolaz.inx"
NS = {"inx": "http://www.inkscape.org/namespace/inkscape/extension"}


def test_inx_xml_validity() -> None:
    """Le fichier .inx est un document XML valide avec l'id attendu."""
    assert INX_PATH.is_file()
    tree = ET.parse(INX_PATH)
    root = tree.getroot()
    assert root.tag == f"{{{NS['inx']}}}inkscape-extension"
    id_elem = root.find("inx:id", NS)
    assert id_elem is not None
    assert id_elem.text == "org.vectolaz.centerline"


def test_inx_parameters_presence() -> None:
    """L'onglet Vectorisation contient le preset et spur_tolerance avec leurs options."""
    root = ET.parse(INX_PATH).getroot()
    params = {p.get("name"): p for p in root.findall(".//inx:param", NS)}

    # Vérification du sélecteur de préréglages
    assert "preset" in params
    preset_param = params["preset"]
    assert preset_param.get("type") == "optiongroup"
    assert preset_param.get("appearance") == "combo"
    options = {opt.get("value"): opt.text for opt in preset_param.findall("inx:option", NS)}
    expected_options = {"custom", "technical", "rosace", "organic", "detailed", "scanned"}
    assert set(options.keys()) == expected_options
    assert "Personnalisé" in options["custom"]
    assert "Dessin technique" in options["technical"]
    assert "Rosace" in options["rosace"]
    assert "Tracé libre" in options["organic"]
    assert "Motif fin" in options["detailed"]
    assert "Croquis scanné" in options["scanned"]

    # Vérification de spur_tolerance
    assert "spur_tolerance" in params
    spur_param = params["spur_tolerance"]
    assert spur_param.get("type") == "float"
    assert float(spur_param.get("min", "-1")) == 0.0
    assert float(spur_param.get("max", "-1")) == 50.0
    assert spur_param.text.strip() == "0.00"

    # Vérification des paramètres existants conservés
    for name in (
        "threshold-auto",
        "threshold",
        "min-component-size",
        "gap-tolerance",
        "geometry-tolerance",
        "global-arcs",
        "stroke-width",
        "engine-python",
    ):
        assert name in params, f"Paramètre manquant : {name}"


def test_inx_groups_and_help() -> None:
    """L'onglet Vectorisation est structuré en 3 groupes et l'Aide contient le guide complet."""
    root = ET.parse(INX_PATH).getroot()
    pages = {page.get("name"): page for page in root.findall(".//inx:page", NS)}

    assert "vectorization" in pages
    vec_page = pages["vectorization"]
    headers = [label.text for label in vec_page.findall("inx:label", NS) if label.get("appearance") == "header"]
    assert any("1. Extraction du trait" in h for h in headers)
    assert any("2. Nettoyage" in h and "Continuité" in h for h in headers)
    assert any("3. Reconnaissance des formes" in h for h in headers)

    assert "help" in pages
    help_page = pages["help"]
    help_headers = [label.text for label in help_page.findall("inx:label", NS) if label.get("appearance") == "header"]
    assert any("Guide par type de motif" in h for h in help_headers)
    assert any("Mémo de dépannage rapide" in h for h in help_headers)

    help_texts = " ".join(label.text or "" for label in help_page.findall("inx:label", NS))
    for keyword in ("Dessin technique", "Rosace", "Tracé libre", "Motif fin", "Croquis scanné"):
        assert keyword in help_texts
    for memo in ("Traits clairs manquants", "Droites ou arcs morcelés", "Angles vifs arrondis", "épines parasites"):
        assert memo in help_texts


def test_presets_immutability() -> None:
    """Le dictionnaire PRESETS et ses sous-dictionnaires sont immuables."""
    assert isinstance(PRESETS, dict)
    with pytest.raises(TypeError, match="immuable"):
        PRESETS["new_preset"] = {}  # type: ignore[index]
    with pytest.raises(TypeError, match="immuable"):
        PRESETS["technical"]["geometry_tolerance"] = 99.0  # type: ignore[index]


def test_preset_technical_application() -> None:
    """Le profil 'technical' applique les valeurs de dessin technique et préserve stroke_width."""
    raw = SimpleNamespace(preset="technical", stroke_width=2.5)
    opts = resolve_options(raw)
    assert opts.auto_threshold is True
    assert opts.threshold_auto is True
    assert opts.geometry_tolerance == 2.5
    assert opts.spur_tolerance == 3.0
    assert opts.gap_tolerance == 0.0
    assert opts.min_component_size == 5
    assert opts.global_arcs is False
    assert opts.stroke_width == 2.5


def test_preset_rosace_application() -> None:
    """Le profil 'rosace' applique le seuil 160, tolérance 5.0, barbes 3.0 et regroupement global."""
    raw = SimpleNamespace(preset="rosace", stroke_width=1.5)
    opts = resolve_options(raw)
    assert opts.auto_threshold is False
    assert opts.threshold_auto is False
    assert opts.threshold == 160
    assert opts.geometry_tolerance == 5.0
    assert opts.spur_tolerance == 3.0
    assert opts.gap_tolerance == 0.0
    assert opts.min_component_size == 10
    assert opts.global_arcs is True
    assert opts.stroke_width == 1.5


def test_preset_organic_application() -> None:
    """Le profil 'organic' applique les paramètres de courbes libres."""
    raw = SimpleNamespace(preset="organic")
    opts = resolve_options(raw)
    assert opts.auto_threshold is True
    assert opts.geometry_tolerance == 2.0
    assert opts.spur_tolerance == 2.5
    assert opts.gap_tolerance == 0.0
    assert opts.min_component_size == 5
    assert opts.global_arcs is False


def test_preset_detailed_application() -> None:
    """Le profil 'detailed' applique les paramètres pour motifs très fins."""
    raw = SimpleNamespace(preset="detailed")
    opts = resolve_options(raw)
    assert opts.auto_threshold is False
    assert opts.threshold == 165
    assert opts.geometry_tolerance == 1.0
    assert opts.spur_tolerance == 1.0
    assert opts.gap_tolerance == 0.0
    assert opts.min_component_size == 2
    assert opts.global_arcs is False


def test_preset_scanned_application() -> None:
    """Le profil 'scanned' applique les paramètres pour croquis scannés bruités."""
    raw = SimpleNamespace(preset="scanned")
    opts = resolve_options(raw)
    assert opts.auto_threshold is False
    assert opts.threshold == 175
    assert opts.geometry_tolerance == 3.0
    assert opts.spur_tolerance == 4.5
    assert opts.gap_tolerance == 5.0
    assert opts.min_component_size == 25
    assert opts.global_arcs is False


def test_preset_custom_preserves_interface_values() -> None:
    """Le mode 'custom' respecte fidèlement tous les paramètres personnalisés."""
    raw = SimpleNamespace(
        preset="custom",
        threshold_auto=False,
        threshold=142,
        geometry_tolerance=3.7,
        spur_tolerance=1.8,
        gap_tolerance=4.0,
        min_component_size=12,
        stroke_width=0.8,
        global_arcs=True,
    )
    opts = resolve_options(raw)
    assert opts.preset == "custom"
    assert opts.threshold_auto is False
    assert opts.threshold == 142
    assert opts.geometry_tolerance == 3.7
    assert opts.spur_tolerance == 1.8
    assert opts.gap_tolerance == 4.0
    assert opts.min_component_size == 12
    assert opts.stroke_width == 0.8
    assert opts.global_arcs is True


def test_invalid_preset_raises() -> None:
    """Un profil inconnu lève une ValueError."""
    raw = SimpleNamespace(preset="inconnu")
    with pytest.raises(ValueError, match="Profil de préréglage inconnu"):
        resolve_options(raw)


def test_run_engine_propagates_preset_and_spur_tolerance(tmp_path: Path, monkeypatch) -> None:
    """run_engine résout le preset et transmet --spur-tolerance à l'appel CLI de raster2vector."""
    captured_command: list[str] = []

    def mock_run(command, *args, **kwargs):
        captured_command.extend(command)
        out = Path(command[command.index("--output") + 1])
        out.write_bytes(b"dummy")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(vectolaz.subprocess, "run", mock_run)

    # 1. Custom avec spur_tolerance
    opts_custom = SimpleNamespace(
        preset="custom",
        threshold_auto=True,
        threshold=128,
        geometry_tolerance=2.0,
        spur_tolerance=3.2,
        gap_tolerance=0.0,
        min_component_size=0,
        stroke_width=1.0,
        global_arcs=False,
    )
    run_engine(sys.executable, tmp_path / "in.png", tmp_path / "out1.svg", opts_custom)
    assert "--spur-tolerance" in captured_command
    assert captured_command[captured_command.index("--spur-tolerance") + 1] == "3.2"

    # 2. Preset 'rosace'
    captured_command.clear()
    opts_rosace = SimpleNamespace(preset="rosace", stroke_width=2.0)
    run_engine(sys.executable, tmp_path / "in.png", tmp_path / "out2.svg", opts_rosace)
    assert "--threshold" in captured_command
    assert captured_command[captured_command.index("--threshold") + 1] == "160"
    assert "--geometry-tolerance" in captured_command
    assert captured_command[captured_command.index("--geometry-tolerance") + 1] == "5.0"
    assert "--spur-tolerance" in captured_command
    assert captured_command[captured_command.index("--spur-tolerance") + 1] == "3.0"
    assert "--stroke-width" in captured_command
    assert captured_command[captured_command.index("--stroke-width") + 1] == "2.0"
    assert "--global-arcs" in captured_command


def test_extension_argument_parser_options() -> None:
    """L'extension expose et parse correctement les arguments --preset et --spur-tolerance."""
    ext = VectolazExtension()
    ext.parse_arguments(["--preset=rosace", "--spur-tolerance=3.5", "--stroke-width=2.0"])
    assert ext.options.preset == "rosace"
    assert ext.options.spur_tolerance == 3.5
    assert ext.options.stroke_width == 2.0

    ext_underscore = VectolazExtension()
    ext_underscore.parse_arguments(["--spur_tolerance=4.2"])
    assert ext_underscore.options.spur_tolerance == 4.2
