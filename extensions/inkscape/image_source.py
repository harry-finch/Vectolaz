"""Lecture d'un PNG embarqué ou lié, sans dépendance au moteur."""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
import re
import struct
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname


def read_png(href: str, document_path: str | None) -> tuple[bytes, int, int]:
    """Retourne les octets et dimensions ; le moteur valide ensuite le décodage."""
    if not href:
        raise ValueError("L'image sélectionnée n'a pas de source PNG.")
    if href.startswith("data:"):
        header, separator, payload = href.partition(",")
        if not separator or header.lower() != "data:image/png;base64":
            raise ValueError("Seuls les PNG embarqués en base64 sont pris en charge.")
        try:
            data = base64.b64decode(re.sub(r"\s+", "", payload), validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("Les données base64 du PNG sont invalides.") from error
    else:
        # Une lettre de lecteur Windows ne constitue pas un protocole URI.
        windows_drive = bool(re.match(r"^[A-Za-z]:[\\/]", href))
        uri = urlsplit(href) if not windows_drive else None
        if uri is not None and uri.scheme:
            if uri.scheme.lower() != "file" or uri.netloc not in ("", "localhost") or uri.query or uri.fragment:
                raise ValueError("L'image doit être un fichier PNG local ou un PNG embarqué.")
            source = Path(url2pathname(uri.path))
        else:
            source = Path(href if windows_drive else unquote(href))
        if not source.is_absolute():
            if not document_path:
                raise ValueError("Enregistrez le document ou embarquez le PNG pour résoudre son lien relatif.")
            source = Path(document_path).parent / source
        if not source.is_file():
            raise ValueError(f"Le fichier image est introuvable : {source}")
        try:
            data = source.read_bytes()
        except OSError as error:
            raise ValueError(f"Impossible de lire le PNG : {error}") from error
    if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[8:16] != b"\x00\x00\x00\rIHDR":
        raise ValueError("L'image sélectionnée n'est pas un PNG valide.")
    width, height = struct.unpack(">II", data[16:24])
    if not width or not height:
        raise ValueError("Les dimensions du PNG doivent être positives.")
    return data, width, height
