"""Command line.

    python -m scanner serve [--port 8000] [--warm] [--open]       the app: http://127.0.0.1:8000
    python -m scanner solve --scramble "U R' F2 D"                 Maltese Gear Cube: a scramble in the 3D model's moves
    python -m scanner solve --array "0 0 1 ..."                    Maltese Gear Cube: the solver's 144-entry array
    python -m scanner solve --puzzle tangram --scramble "U2 F D' R"   Tangram Cube Extreme, in the simulator's notation
"""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser

import numpy as np


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from scanner.paths import WEB_DIST
    from scanner.puzzles.maltese.solver import SolverService
    from scanner.server import create_app

    url = f"http://{args.host}:{args.port}/"
    if not (WEB_DIST / "index.html").exists():
        print("note: the web app is not built, so only the API is served. Build it once with:\n"
              "    cd web && npm install && npm run build\n"
              "or run the development server (cd web && npm run dev) and open http://localhost:5173", flush=True)
    else:
        print(f"Cube Scanner: open {url} in your browser (Ctrl+C stops the server)", flush=True)
    solver = SolverService(beam=args.beam)
    app = create_app(solver)
    if args.warm:
        for pid, service in app.state.services.items():
            print(f"loading the {service.title} solver ...", flush=True)
            print(" ", service.solver.load().detail, flush=True)
    if args.open:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _solve_maltese(args: argparse.Namespace) -> None:
    from scanner.puzzles.maltese.decode import decode_exact
    from scanner.puzzles.maltese.layout import load_layout
    from scanner.puzzles.maltese.solver import SolverService

    layout = load_layout()
    if args.array:
        obs = layout.from_solver(np.array(args.array.replace(",", " ").split(), dtype=np.int64))
    else:
        obs = layout.apply(layout.solved, args.scramble)
    decoded = decode_exact(obs, layout)
    if not decoded.legal:
        sys.exit("not a reachable cube:\n  " + "\n  ".join(p.message for p in decoded.problems))
    print("array:", " ".join(str(v) for v in decoded.state144))
    solver = SolverService(beam=args.beam)
    status = solver.load()
    if status.state != "ready":
        sys.exit(status.detail)
    s = solver.solve(decoded.state144)
    print(f"solution ({s.quarter_turns} quarter turns, {s.face_turns} face turns, {s.seconds:.1f} s, verified):")
    print(s.word)
    print("hold the cube blue up, yellow in front, purple on the left; U, L, F are those three faces")


def _solve_tangram(args: argparse.Namespace) -> None:
    from scanner.core.services import NotSolvable, SolverNotReady
    from scanner.puzzles.tangram.layout import load_layout
    from scanner.puzzles.tangram.solver import TangramSolverService, solver_state

    if not args.scramble:
        sys.exit("give the Tangram state as --scramble \"<moves from solved>\"")
    layout = load_layout()
    orient = layout.apply(layout.solved_orient, args.scramble)
    print("state (orientation per piece):", " ".join(map(str, orient)))
    print("solver vector (per position, 6 = empty):", " ".join(map(str, solver_state(layout, orient))))
    try:
        s = TangramSolverService(beam=args.beam).solve(orient)
    except (SolverNotReady, NotSolvable) as err:
        sys.exit(str(err))
    print(f"solution ({s.turns} moves, {s.steps} steps, {s.seconds:.1f} s, verified on the scanner's model):")
    print(s.word)


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m scanner", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run the local server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--beam", type=int, default=4096, help="beam width on the Maltese solver's learned rungs")
    serve.add_argument("--warm", action="store_true", help="load the solvers before accepting requests")
    serve.add_argument("--open", action="store_true", help="open the app in the browser")
    serve.set_defaults(run=_serve)
    solve = sub.add_parser("solve", help="solve one state from the command line")
    solve.add_argument("--puzzle", choices=["maltese", "tangram"], default="maltese")
    group = solve.add_mutually_exclusive_group(required=True)
    group.add_argument("--scramble", help="moves from solved: the 3D model's (Maltese) or the simulator's notation (Tangram)")
    group.add_argument("--array", help="Maltese only: the solver's 144 entries")
    solve.add_argument("--beam", type=int, default=4096)
    solve.set_defaults(run=lambda a: (_solve_tangram if a.puzzle == "tangram" else _solve_maltese)(a))
    args = ap.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
