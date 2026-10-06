"""Tools exposed to LLM agents through function calling.

Agents get read-only access to systems of record. They can look things up,
they cannot change inventory, vendors or payments. Writes happen only in the
deterministic payment step after policy checks.
"""

from __future__ import annotations

import difflib
from typing import Any

from .db import Database
from .llm import ToolSpec
from .normalize import invoice_key, normalize_po, normalize_sku


def build_validation_tools(db: Database) -> list[ToolSpec]:
    def lookup_inventory(item_name: str) -> dict[str, Any]:
        row = db.find_item(item_name)
        if row:
            return {"found": True, **row}
        names = [r["item"] for r in db.catalog()]
        close = difflib.get_close_matches(normalize_sku(item_name), [normalize_sku(n) for n in names], n=3, cutoff=0.6)
        return {"found": False, "similar_catalog_items": [n for n in names if normalize_sku(n) in close],
                "note": "Similar is not the same. Do not treat a similar item as a match."}

    def list_catalog() -> dict[str, Any]:
        return {"items": db.catalog()}

    def get_vendor(vendor_name: str) -> dict[str, Any]:
        row = db.find_vendor(vendor_name)
        return {"found": True, **row} if row else {"found": False, "approved_vendors": [v["name"] for v in db.vendors()]}

    def invoice_history(invoice_number: str, vendor_name: str) -> dict[str, Any]:
        key = invoice_key(vendor_name, invoice_number)
        return {"invoice_key": key, "previous_submissions": db.invoice_history(key), "payment": db.payment_for(key)}

    def get_purchase_order(po_number: str) -> dict[str, Any]:
        po = db.get_po(normalize_po(po_number))
        return {"found": True, **po} if po else {"found": False, "note": f"No purchase order {po_number} exists."}

    def vendor_history_summary(vendor_name: str) -> dict[str, Any]:
        from .anomaly import amount_profile, price_history
        history = db.vendor_history(vendor_name)
        profile = amount_profile([h["total"] for h in history])
        prices = {k: {"min": min(v), "max": max(v), "count": len(v)} for k, v in price_history(history).items()}
        return {"invoices": len(history), "amount_baseline": profile and {k: round(profile[k], 2) for k in ("median", "p95", "normal_high")},
                "unit_prices": prices, "last_invoice_date": history[-1]["invoice_date"] if history else None}

    string = {"type": "string"}
    return [
        ToolSpec("lookup_inventory", "Look up one item in Acme's inventory: stock level and catalog unit price.",
                 {"type": "object", "properties": {"item_name": string}, "required": ["item_name"]}, lookup_inventory),
        ToolSpec("list_catalog", "List every item Acme stocks, with stock and catalog price.",
                 {"type": "object", "properties": {}}, list_catalog),
        ToolSpec("get_vendor", "Look up a vendor in Acme's approved vendor master (status, currency, address on file).",
                 {"type": "object", "properties": {"vendor_name": string}, "required": ["vendor_name"]}, get_vendor),
        ToolSpec("invoice_history", "Previous submissions and payments for this vendor + invoice number.",
                 {"type": "object", "properties": {"invoice_number": string, "vendor_name": string},
                  "required": ["invoice_number", "vendor_name"]}, invoice_history),
        ToolSpec("get_purchase_order", "A purchase order with its lines (ordered, agreed price, already invoiced) and goods receipts.",
                 {"type": "object", "properties": {"po_number": string}, "required": ["po_number"]}, get_purchase_order),
        ToolSpec("vendor_history_summary", "This vendor's past 12 months: invoice count, typical amount range, unit prices charged.",
                 {"type": "object", "properties": {"vendor_name": string}, "required": ["vendor_name"]}, vendor_history_summary),
    ]
