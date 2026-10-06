"""Create (or recreate) inventory.db with the case's seed data plus our extensions.

    python scripts/setup_db.py

Seed = the brief's minimum (WidgetA 15, WidgetB 10, GadgetX 5, FakeItem 0),
extended with catalog unit prices and a vendor master. See ap_autopilot/db.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ap_autopilot.config import Settings  # noqa: E402
from ap_autopilot.db import Database  # noqa: E402

if __name__ == "__main__":
    settings = Settings()
    db = Database(settings.db_path)
    db.init(reset=True)
    print(f"Created {settings.db_path}")
    for row in db.catalog():
        print(f"  {row['item']:10} stock={row['stock']:<3} price={row['unit_price']}")
    print(f"  {len(db.vendors())} approved vendors")
