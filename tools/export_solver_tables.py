"""Write the solver's Maltese Gear Cube move tables to shared/maltese/solver_tables.json.

The browser side derives and tests the dictionary between the 3D engine and the solver's
144-entry array from this file (web/tools/derive.gen.ts), so it never needs Python.

    python tools/export_solver_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "cube_solver"))

from puzzles.maltese_gear import tables as md  # noqa: E402


def main() -> None:
    idx, adds = md.build_tables()
    out = {
        "source": "third_party/cube_solver/puzzles/maltese_gear/tables.py",
        "layout": "next[i] = (state[idx[m][i]] + add[m][i]) mod (4 for i >= 120)",
        "moves": list(md.MOVE_LIST),
        "idx": idx.tolist(),
        "add": adds.tolist(),
        "goal": md.goal_state().tolist(),
    }
    path = ROOT / "shared" / "maltese" / "solver_tables.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
