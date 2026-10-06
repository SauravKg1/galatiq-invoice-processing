"""Scanned invoices: detection, Grok vision with self-correction, OCR fallback, fail-safe."""

from __future__ import annotations

import base64
import io
import json

import pytest
from PIL import Image

from ap_autopilot import ocr
from ap_autopilot.graph import InvoicePipeline
from ap_autopilot.llm import ChatLLM
from ap_autopilot.loaders import load_document
from ap_autopilot.models import Invoice, LineItem, VisionExtraction
from ap_autopilot.security import UNTRUSTED_DATA_RULE
from conftest import EXTRA, INVOICES, FakeChatClient, text_reply

SCAN = EXTRA / "invoice_2010_scanned.pdf"
PHOTO = EXTRA / "invoice_2011_photo.jpg"


# ------------------------------------------------------------------ loading
def test_image_only_pdf_is_detected_as_scan():
    doc = load_document(SCAN)
    assert doc.is_scanned and doc.text == "" and len(doc.images) == 1
    assert doc.images[0][:8] == b"\x89PNG\r\n\x1a\n"


def test_text_pdf_is_not_treated_as_scan():
    assert not load_document(INVOICES / "invoice_1011.pdf").is_scanned


def test_photo_is_loaded_upright_and_downscaled(tmp_path):
    big = tmp_path / "big.jpg"
    Image.new("RGB", (4000, 3000), "white").save(big, exif=_exif_rotated_90())
    doc = load_document(big)
    with Image.open(io.BytesIO(doc.images[0])) as img:
        assert max(img.size) <= 2000
        assert img.height > img.width  # EXIF orientation applied: landscape pixels shown as portrait


def _exif_rotated_90():
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90 CW
    return exif


def test_unreadable_image_fails_clearly(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    from ap_autopilot.loaders import DocumentLoadError
    with pytest.raises(DocumentLoadError):
        load_document(bad)


# ------------------------------------------------------------ fail-safe path
def test_scan_without_vision_or_ocr_goes_to_a_person(pipeline, db, monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: False)
    out = pipeline.process(SCAN)
    assert out.status == "FAILED"
    assert "needs Grok vision" in out.error
    assert db.review_queue()[0]["source_file"] == "invoice_2010_scanned.pdf"
    assert not db.payments()


@pytest.mark.skipif(not ocr.available(), reason="Tesseract not installed")
@pytest.mark.parametrize("path, total", [(SCAN, 2750.0), (PHOTO, 2000.0)])
def test_offline_ocr_reads_scans(pipeline, path, total):
    out = pipeline.process(path)
    assert out.extraction.method == "ocr"
    assert out.extraction.invoice.total == total
    assert out.extraction.invoice.invoice_number.isupper()
    assert out.status == "PAID"


# ---------------------------------------------------------- Grok vision path
def _good_read() -> VisionExtraction:
    inv = Invoice(vendor_name="Precision Parts Ltd.", invoice_number="INV-2010", due_date_raw="2026-03-20",
                  due_date="2026-03-20", invoice_date="2026-02-18", subtotal=2750, tax_amount=0, total=2750,
                  line_items=[LineItem(description="WidgetA", item="WidgetA", quantity=5, unit_price=250, amount=1250),
                              LineItem(description="GadgetX", item="GadgetX", quantity=2, unit_price=750, amount=1500)])
    return VisionExtraction(transcription="INVOICE Vendor: Precision Parts Ltd. Invoice Number: INV-2010 ...", invoice=inv)


def vision_model(requests_seen: list):
    """Fake Grok vision: misreads the total on the first look, then corrects itself.
    Downstream agents behave like a careful model."""

    def respond(kw, i):
        requests_seen.append(kw)
        system = kw["messages"][0]["content"]
        if "scanned invoice or a photo" in system:
            read = _good_read()
            if "verification pass flagged" not in json.dumps(kw["messages"][1]["content"]):
                read.invoice.total = 2570.0  # transposed digits
            return text_reply(read.model_dump_json())
        if "Validation Agent" in system:
            from conftest import tool_reply
            return tool_reply(("submit_result", {"additional_findings": [], "fraud_risk": 0, "summary": "ok"}))
        if "VP of Finance" in system:
            return text_reply(json.dumps({"decision": "APPROVE", "rationale": "clean", "cited_findings": []}))
        return text_reply(json.dumps({"agrees": True, "issues": [], "recommended_decision": "APPROVE"}))

    return respond


def test_grok_vision_reads_scan_and_self_corrects(settings, monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: False)  # prove vision works on its own
    seen: list = []
    llm = ChatLLM(provider="fake-grok", api_key="x", base_url="http://fake", model="grok-text",
                  vision_model="grok-vision", client=FakeChatClient(vision_model(seen)))
    out = InvoicePipeline(settings, llm=llm).process(SCAN)

    vision_calls = [r for r in seen if isinstance(r["messages"][1]["content"], list)]
    assert len(vision_calls) == 2 and all(r["model"] == "grok-vision" for r in vision_calls)
    image_part = vision_calls[0]["messages"][1]["content"][1]
    assert image_part["type"] == "image_url"
    assert base64.b64decode(image_part["image_url"]["url"].split(",", 1)[1])[:4] == b"\x89PNG"
    assert UNTRUSTED_DATA_RULE in vision_calls[0]["messages"][0]["content"]

    assert out.extraction.method == "vision" and out.extraction.attempts == 2
    assert any("total is 2570.00" in c for c in out.extraction.corrections)
    assert out.extraction.invoice.total == 2750 and out.status == "PAID"
    assert out.extraction.raw_text.startswith("INVOICE Vendor")  # transcription is the audit text


def test_vision_outage_degrades_to_ocr(settings, monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: True)
    monkeypatch.setattr(ocr, "ocr_images", lambda images: open(INVOICES / "invoice_1011.txt").read())

    def down(kw, i):
        if isinstance(kw["messages"][1]["content"], list):
            raise ConnectionError("vision endpoint unavailable")
        return text_reply(json.dumps({"decision": "APPROVE", "rationale": "ok", "cited_findings": [], "agrees": True,
                                      "issues": [], "recommended_decision": "APPROVE"}))

    llm = ChatLLM(provider="fake-grok", api_key="x", base_url="http://fake", model="g", api_retries=0,
                  client=FakeChatClient(down))
    out = InvoicePipeline(settings, llm=llm).process(SCAN)
    assert out.extraction.method == "ocr"
    assert out.extraction.invoice.vendor_name == "Summit Manufacturing Co."
    assert any("DEGRADED" in c["note"] for c in out.llm_calls)
