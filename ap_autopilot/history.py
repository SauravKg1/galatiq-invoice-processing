"""Synthetic 12-month invoice history, so the anomaly detectors have a "normal" to compare against.

SYNTHETIC DATA. The case brief contains no history. This module generates a
reproducible past (fixed seed) per vendor in the vendor master: invoice dates,
totals and line prices drawn around a per-vendor profile. It lives in its own
table (`historical_invoices`), so it never interacts with duplicate checks,
payments or the brief's scenarios. In production this table is a read of the
ERP's paid-invoice history.
"""

from __future__ import annotations

import json
import math
import random
from datetime import date, timedelta
from typing import Any

SEED = 20260101
HISTORY_END = date(2026, 1, 10)      # history ends just before the brief's invoices begin
HISTORY_DAYS = 365
CATALOG = {"WidgetA": 250.0, "WidgetB": 500.0, "GadgetX": 750.0}

# vendor -> (invoices per year, median total, spread (log sd), items they sell, price overrides)
PROFILES: dict[str, dict[str, Any]] = {
    "Widgets Inc.":                 {"n": 48, "median": 5_500, "spread": 0.55, "items": ["WidgetA", "WidgetB", "GadgetX"]},
    "Gadgets Co.":                  {"n": 36, "median": 6_000, "spread": 0.50, "items": ["GadgetX", "WidgetA", "WidgetB"]},
    "Precision Parts Ltd.":         {"n": 30, "median": 2_200, "spread": 0.40, "items": ["WidgetA", "WidgetB", "GadgetX"]},
    "Acme Industrial Supplies":     {"n": 40, "median": 2_500, "spread": 0.25, "items": ["WidgetA", "WidgetB"]},
    "MegaWidgets Corp":             {"n": 80, "median": 8_000, "spread": 0.15, "items": ["WidgetA", "WidgetB", "GadgetX"],
                                     "fabricated": True},   # amounts invented in a narrow band: fails Benford
    "Consolidated Materials Group": {"n": 44, "median": 5_000, "spread": 0.50, "items": ["WidgetA", "WidgetB", "GadgetX"]},
    "Summit Manufacturing Co.":     {"n": 36, "median": 2_500, "spread": 0.40, "items": ["WidgetA", "WidgetB"]},
    "FastShip Ltd.":                {"n": 24, "median": 7_000, "spread": 0.45, "items": ["WidgetA", "WidgetB", "GadgetX"]},
    "Atlas Industrial Supply":      {"n": 30, "median": 15_000, "spread": 0.45, "items": ["WidgetA", "WidgetB", "GadgetX"]},
    "TechParts International":      {"n": 20, "median": 4_000, "spread": 0.40, "items": ["WidgetA", "WidgetB"],
                                     "prices": {"WidgetA": 225.0, "WidgetB": 475.0}},   # EUR prices
    "Reliable Components Inc.":     {"n": 40, "median": 4_000, "spread": 0.50, "items": ["WidgetA", "WidgetB", "GadgetX"]},
}


def _lines_for_total(rng: random.Random, target: float, items: list[str], prices: dict[str, float]) -> list[dict[str, Any]]:
    """Build line items whose total lands near `target` using the vendor's real unit prices."""
    lines, remaining = [], target
    for item in rng.sample(items, k=min(len(items), rng.randint(1, len(items)))):
        price = prices.get(item, CATALOG[item])
        qty = max(1, round(remaining / price / rng.uniform(1.0, 2.0)))
        lines.append({"item": item, "quantity": qty, "unit_price": price})
        remaining -= qty * price
        if remaining <= price:
            break
    return lines


def generate() -> list[tuple[str, str, str, float, str]]:
    """Rows of (vendor, invoice_number, invoice_date, total, lines_json). Deterministic."""
    rng = random.Random(SEED)
    rows = []
    for vendor, p in PROFILES.items():
        prices = {**CATALOG, **p.get("prices", {})}
        prefix = "".join(w[0] for w in vendor.replace(".", "").split()[:3]).upper()
        for i in range(p["n"]):
            day = HISTORY_END - timedelta(days=int(HISTORY_DAYS * (p["n"] - i) / p["n"]) - rng.randint(0, 5))
            if p.get("fabricated"):
                target = rng.uniform(6_000, 9_999)          # invented numbers: narrow band, leading digits 6-9
            else:
                target = p["median"] * math.exp(rng.gauss(0, p["spread"]))
            lines = _lines_for_total(rng, target, p["items"], prices)
            total = round(sum(l["quantity"] * l["unit_price"] for l in lines), 2)
            if p.get("fabricated"):
                total = round(target, 2)
            rows.append((vendor, f"H-{prefix}-{i + 1:04d}", day.isoformat(), total, json.dumps(lines)))
    return rows
