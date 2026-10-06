"""Optional local OCR with Tesseract, the offline fallback for scanned invoices.

Uses the `tesseract` command directly (no Python wrapper dependency). If it is
not installed, `available()` is False and scans go to Grok vision, or to the
human queue when there is no model either.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path


def available() -> bool:
    return shutil.which("tesseract") is not None


_CACHE: dict[str, str] = {}   # same image bytes -> same text; avoids re-OCR on retries and re-runs


def ocr_images(images: list[bytes], timeout_s: float = 60.0) -> str:
    """OCR each PNG page and join the text. `--psm 6` treats a page as one uniform block,
    which keeps invoice table rows on single lines."""
    key = hashlib.sha256(b"".join(images)).hexdigest()
    if key in _CACHE:
        return _CACHE[key]
    pages: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, png in enumerate(images):
            path = Path(tmp) / f"page{i}.png"
            path.write_bytes(png)
            result = subprocess.run(["tesseract", str(path), "stdout", "--psm", "6"],
                                    capture_output=True, text=True, timeout=timeout_s, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"tesseract failed: {result.stderr.strip()[:200]}")
            pages.append(result.stdout)
    _CACHE[key] = "\n".join(pages).strip()
    return _CACHE[key]
