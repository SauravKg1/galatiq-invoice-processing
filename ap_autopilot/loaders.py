"""Turn any file into a SourceDocument: raw text plus parsed data for structured
formats, or page images for scans (image-only PDFs and photos)."""

from __future__ import annotations

import csv
import io
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

STRUCTURED_FORMATS = {"json", "csv", "xml"}
TEXT_FORMATS = {"txt", "eml", "md", "text", "log"}
IMAGE_FORMATS = {"png", "jpg", "jpeg", "tif", "tiff", "webp", "bmp"}
SUPPORTED_EXTENSIONS = {".pdf", ".json", ".csv", ".xml"} | {f".{f}" for f in TEXT_FORMATS | IMAGE_FORMATS}

MIN_TEXT_CHARS = 20      # a PDF with less extractable text than this is treated as a scan
MAX_PAGES = 3            # pages sent to vision / OCR; invoices longer than this are rare
RENDER_DPI = 200         # enough for small print without huge payloads
MAX_IMAGE_SIDE = 2000    # pixels; larger photos are downscaled


class DocumentLoadError(RuntimeError):
    pass


@dataclass
class SourceDocument:
    path: str
    format: str
    text: str
    data: Any = None
    warnings: list[str] = field(default_factory=list)
    images: list[bytes] = field(default_factory=list)   # PNG pages, only for scans

    @property
    def is_structured(self) -> bool:
        return self.format in STRUCTURED_FORMATS and self.data is not None

    @property
    def is_scanned(self) -> bool:
        return bool(self.images)


def _normalize_image(raw: bytes) -> bytes:
    """Any photo -> upright RGB PNG within MAX_IMAGE_SIDE (phone EXIF rotation applied)."""
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(raw)) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        out = io.BytesIO()
        img.save(out, format="PNG", optimize=True)
        return out.getvalue()


def render_pdf_pages(path: Path, max_pages: int = MAX_PAGES, dpi: int = RENDER_DPI) -> list[bytes]:
    import pymupdf

    with pymupdf.open(str(path)) as doc:
        return [_normalize_image(page.get_pixmap(dpi=dpi).tobytes("png")) for page in list(doc)[:max_pages]]


def _xml_to_dict(element: ET.Element) -> Any:
    children = list(element)
    if not children:
        return (element.text or "").strip()
    grouped: dict[str, Any] = {}
    for child in children:
        value = _xml_to_dict(child)
        if child.tag in grouped:
            if not isinstance(grouped[child.tag], list):
                grouped[child.tag] = [grouped[child.tag]]
            grouped[child.tag].append(value)
        else:
            grouped[child.tag] = value
    return grouped


def _pdf_text(path: Path) -> tuple[str, list[str]]:
    warnings: list[str] = []
    text = ""
    try:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            text = "\n".join((page.extract_text() or "") for page in pdf.pages)
    except Exception as exc:
        warnings.append(f"pdfplumber failed ({type(exc).__name__}); trying PyMuPDF")
    if not text.strip():
        try:
            import pymupdf

            with pymupdf.open(str(path)) as doc:
                text = "\n".join(page.get_text() for page in doc)
        except Exception as exc:
            warnings.append(f"PyMuPDF failed ({type(exc).__name__})")
    if not text.strip():
        warnings.append("PDF has no text layer (scanned image).")
    return text, warnings


def load_document(path: str | Path) -> SourceDocument:
    p = Path(path)
    if not p.exists():
        raise DocumentLoadError(f"file not found: {p}")
    if p.stat().st_size == 0:
        raise DocumentLoadError(f"file is empty: {p}")
    fmt = p.suffix.lower().lstrip(".") or "txt"

    if fmt == "pdf":
        text, warnings = _pdf_text(p)
        if len("".join(text.split())) >= MIN_TEXT_CHARS:
            return SourceDocument(str(p), "pdf", text, warnings=warnings)
        try:
            images = render_pdf_pages(p)
        except Exception as exc:
            raise DocumentLoadError(f"PDF could not be opened or rendered: {exc}") from exc
        if not images:
            raise DocumentLoadError("PDF has no pages")
        notes = [w for w in warnings if "no text layer" not in w]
        return SourceDocument(str(p), "pdf", "", warnings=notes + [
            f"Scanned PDF (no text layer): {len(images)} page(s) rendered for vision/OCR"], images=images)

    if fmt in IMAGE_FORMATS:
        try:
            image = _normalize_image(p.read_bytes())
        except Exception as exc:
            raise DocumentLoadError(f"image could not be read: {exc}") from exc
        return SourceDocument(str(p), "image", "", warnings=["Image invoice: sent to vision/OCR"], images=[image])

    raw = p.read_text(encoding="utf-8", errors="replace")
    data: Optional[Any] = None
    warnings: list[str] = []
    try:
        if fmt == "json":
            data = json.loads(raw)
        elif fmt == "csv":
            data = list(csv.reader(io.StringIO(raw)))
        elif fmt == "xml":
            data = _xml_to_dict(ET.fromstring(raw))
    except (json.JSONDecodeError, ET.ParseError, csv.Error) as exc:
        # Malformed structured file: keep the text so the LLM / text parser can still try.
        warnings.append(f"{fmt.upper()} could not be parsed ({exc}); treating as free text")
        data = None
    return SourceDocument(str(p), fmt if fmt in STRUCTURED_FORMATS | {"pdf"} else "txt", raw, data, warnings)
