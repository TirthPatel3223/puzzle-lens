"""The local server: everything the browser UI cannot do itself.

    python -m scanner serve            # http://127.0.0.1:8000  (serves web/dist when it is built)

Every puzzle's endpoints are mounted at /api/<puzzle id>/ (see scanner/puzzles/<key>/api.py):

    GET  /api/puzzles                       the puzzles this server knows
    GET  /api/health                        every puzzle's status (and, for older pages, the Maltese one's)
    GET  /api/maltese-gear-cube/...         the Maltese Gear Cube (also at /api/... for older pages)
    GET  /api/tangram-cube-extreme/...      the Tangram Cube Extreme
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from scanner import __version__
from scanner.paths import WEB_DIST
from scanner.puzzles import DEFAULT, REGISTRY, create_service


def create_app(solver=None, vision=None, services: dict | None = None) -> FastAPI:
    """``solver`` and ``vision`` stand in for the Maltese Gear Cube's (tests, ``serve --warm``);
    ``services`` maps puzzle ids to ready-made services (anything else is created with defaults)."""
    app = FastAPI(title="Puzzle Lens", version=__version__)
    services = dict(services or {})
    if DEFAULT not in services:
        services[DEFAULT] = create_service(DEFAULT, solver=solver, vision=vision)
    for pid in REGISTRY:
        if pid not in services:
            services[pid] = create_service(pid)
    app.state.services = services

    @app.get("/api/puzzles")
    def puzzles() -> dict:
        return {"default": DEFAULT, "puzzles": [s.describe() for s in services.values()]}

    @app.get("/api/health")
    def health() -> dict:
        # Older pages read the Maltese Gear Cube's status at the top level.
        return {"ok": True, "version": __version__, **services[DEFAULT].health(),
                "puzzles": {pid: s.health() for pid, s in services.items()}}

    for pid, s in services.items():
        app.include_router(s.router(), prefix=f"/api/{pid}", tags=[pid])
    # Older pages and scripts call the Maltese Gear Cube's endpoints at /api/... directly.
    app.include_router(services[DEFAULT].router(), prefix="/api", include_in_schema=False)

    if WEB_DIST.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    return app
