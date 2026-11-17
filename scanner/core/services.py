"""Things every puzzle's server side needs: its networks loaded lazily, and its solver's status.

``LazyVision``   the networks, loaded on first use (importing torch takes a few seconds and the rest of
                 the server does not need it), or in the background when the page asks.
``SolverStatus``, ``NotSolvable``, ``SolverNotReady``   what a solver service reports.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import HTTPException


@dataclass
class SolverStatus:
    state: str = "not-loaded"              # not-loaded | loading | ready | missing | error
    detail: str = ""
    log: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"state": self.state, "detail": self.detail}


class NotSolvable(Exception):
    """The state is not one the solver's moves can reach (or the search found nothing)."""


class SolverNotReady(RuntimeError):
    """The solver is missing, not built yet, or failed to load."""


class LazyVision:
    """One puzzle's vision pipeline, created by ``factory`` on first use.

    ``factory()`` returns an object with ``available() -> dict[str, bool]`` (which checkpoints exist),
    ``load()``, ``dir`` and ``device``. ``train_hint`` is shown when a network is missing.
    """

    def __init__(self, factory: Callable[[], Any], train_hint: str, vision: Any = None) -> None:
        self.factory = factory
        self.train_hint = train_hint
        self.lock = threading.Lock()
        self.vision = vision
        self.error: str | None = None

    def get(self) -> Any:
        with self.lock:
            if self.vision is None:
                self.vision = self.factory()
            v = self.vision
            have = v.available()
            if not all(have.values()):
                missing = [k for k, ok in have.items() if not ok]
                raise HTTPException(503, f"No trained {' or '.join(missing)} in {v.dir}. Train them first: {self.train_hint}")
            v.load()
            return v

    def load_in_background(self) -> None:
        def work() -> None:
            try:
                self.get()
                self.error = None
            except HTTPException as err:
                self.error = err.detail
            except Exception as err:          # pragma: no cover - reported to the page
                self.error = f"{type(err).__name__}: {err}"

        threading.Thread(target=work, daemon=True).start()

    def status(self, checkpoints: dict[str, bool]) -> dict:
        """For /health: which checkpoints exist, whether the networks are in memory (a pipeline is loaded
        once its ``reader`` is), on which device, and why loading failed."""
        v = self.vision
        return {**checkpoints, "loaded": bool(v is not None and getattr(v, "reader", None) is not None),
                "device": str(v.device) if v is not None else None, "error": self.error}
