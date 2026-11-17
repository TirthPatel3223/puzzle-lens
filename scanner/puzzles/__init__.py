"""The puzzles the scanner knows.

Each puzzle is a package ``scanner.puzzles.<key>`` with the same parts:

    layout.py       what the puzzle shows, read from shared/<key>/ (generated from its 3D engine)
    decode.py       what it shows -> the most likely legal state (decode_probs), or what is wrong (decode_exact)
    solver.py       its solver, loaded once and kept warm
    api.py          its endpoints, as a service: create_service(**options) -> an object with
                    id, key, title, describe(), health() and router()
    vision/         its networks: geometry, dataset, models, pipeline, train, evaluate

Adding a puzzle: write those parts, then add it to REGISTRY below. The server mounts its router at
/api/<id>/, ``python -m scanner.vision.train --puzzle <key>`` and ``python -m scanner.vision.evaluate
--puzzle <key>`` find its training and evaluation, and the web app lists it (web/src/app/puzzles.ts).
"""
from __future__ import annotations

import importlib

#: id (as the web app and the API name it) -> key (its package and its folders under data/).
REGISTRY: dict[str, str] = {
    "maltese-gear-cube": "maltese",
    "tangram-cube-extreme": "tangram",
}
KEYS: dict[str, str] = {key: pid for pid, key in REGISTRY.items()}
DEFAULT = "maltese-gear-cube"


def resolve(name: str) -> str:
    """A puzzle id from its id or key ("tangram" -> "tangram-cube-extreme")."""
    if name in REGISTRY:
        return name
    if name in KEYS:
        return KEYS[name]
    raise KeyError(f"unknown puzzle {name!r}; known: {', '.join(KEYS)}")


def key_of(name: str) -> str:
    return REGISTRY[resolve(name)]


def module(name: str, part: str):
    """``module("tangram", "vision.train")`` -> scanner.puzzles.tangram.vision.train."""
    return importlib.import_module(f"scanner.puzzles.{key_of(name)}.{part}")


def create_service(name: str, **options):
    return module(name, "api").create_service(**options)
