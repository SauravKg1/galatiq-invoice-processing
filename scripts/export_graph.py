"""Write the compiled LangGraph as a Mermaid diagram to docs/graph.mmd.

The diagram is generated from the real graph, so it can't drift from the code.

    python scripts/export_graph.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ap_autopilot.config import Settings  # noqa: E402
from ap_autopilot.graph import InvoicePipeline  # noqa: E402
from ap_autopilot.llm import OfflineLLM  # noqa: E402


def mermaid() -> str:
    tmp = Path(tempfile.mkdtemp())
    pipeline = InvoicePipeline(Settings(db_path=str(tmp / "g.db"), log_dir=str(tmp / "logs")), llm=OfflineLLM())
    return pipeline.graph.get_graph().draw_mermaid()


if __name__ == "__main__":
    out = ROOT / "docs" / "graph.mmd"
    out.write_text(mermaid())
    print(f"Wrote {out.relative_to(ROOT)}")
