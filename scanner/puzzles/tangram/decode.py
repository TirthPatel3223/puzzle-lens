"""From what the Tangram Cube Extreme shows to a legal state of its 45 pieces.

Two entry points share one decoder, as for the Maltese Gear Cube:

``decode_exact``  for a hand-edited observation: nothing is changed; what is impossible is reported in
                  words, with the positions involved.
``decode_probs``  for the vision model's per-position probabilities: the most likely *legal* state.

How it works:

1. **Assignment.** Every piece fills exactly one of the 45 cells, and its orientation decides which cell
   and which positions its stickers show on (a placement). The evidence for a placement is the sum over
   its stickers of log P(that colour there) - log P(empty there), so the log-likelihood of a whole state
   is a constant plus the sum over pieces. The best arrangement is a linear assignment of pieces to
   cells (Hungarian algorithm), each piece taking its best orientation in its cell.
2. **Reachability.** Not every arrangement can be reached by moves. It is tested exactly with the move
   group's stabiliser chain (``scanner.core.permgroup``). If the best arrangement is unreachable, the
   cheapest single changes (a piece turned within its cell, two pieces swapped), then pairs of them,
   are tried in order of cost until one is reachable.

The frame is the core's (the hidden middle piece never moves), so a state is unique.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.optimize import linear_sum_assignment

from scanner.core.permgroup import StabiliserChain, build_chain

from .layout import EMPTY, N_COLOURS, TangramLayout, load_layout

#: Probability given to the value of a hand-edited position.
_SURE = 1.0 - 1e-9
#: A repair that costs more than this (nats) is not made silently; the scan is flagged instead.
MAX_REPAIR_NATS = 8.0
#: How many single changes, and pairs of them, the repair search tries.
REPAIR_SINGLES = 1500
REPAIR_PAIRS = 60
_IMPOSSIBLE = 1e6


@dataclass
class Problem:
    code: str
    message: str
    positions: list[int] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"code": self.code, "message": self.message, "positions": self.positions}


@dataclass
class Correction:
    """A position where the decoded value is not the reader's most likely one."""

    position: int
    read: int
    decoded: int
    probability: float

    def to_json(self) -> dict:
        return {"position": self.position, "read": self.read, "decoded": self.decoded, "probability": round(float(self.probability), 6)}


@dataclass
class TangramDecode:
    #: Orientation of every piece (core frame), or None when no arrangement fits.
    orient: np.ndarray | None
    #: What that state shows (120 values), or None.
    observation: np.ndarray | None
    legal: bool
    problems: list[Problem]
    corrections: list[Correction]
    #: Probability the reader gave the decoded value at each position (1 for hand-edited input).
    confidence: np.ndarray
    #: Log-likelihood of the decoded state under the evidence (nats).
    loglik: float
    #: Changes the repair made to reach a legal state, in words (empty when none were needed).
    repairs: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "orient": self.orient.tolist() if self.orient is not None else None,
            "observation": self.observation.tolist() if self.observation is not None else None,
            "legal": self.legal,
            "problems": [p.to_json() for p in self.problems],
            "corrections": [c.to_json() for c in self.corrections],
            "confidence": [round(float(x), 4) for x in self.confidence],
            "loglik": round(float(self.loglik), 3),
            "repairs": self.repairs,
        }


class _Tables:
    """Placements as flat arrays, and the move group's stabiliser chain."""

    def __init__(self, layout: TangramLayout) -> None:
        self.layout = layout
        pl = [p for d in layout.placements for p in d.values()]
        self.piece = np.array([p.piece for p in pl], dtype=np.int64)
        self.ori = np.array([p.ori for p in pl], dtype=np.int64)
        self.cell = np.array([p.cell for p in pl], dtype=np.int64)
        k = max(len(p.positions) for p in pl)
        self.pos = np.full((len(pl), k), -1, dtype=np.int64)
        self.col = np.zeros((len(pl), k), dtype=np.int64)
        for j, p in enumerate(pl):
            self.pos[j, : len(p.positions)] = p.positions
            self.col[j, : len(p.positions)] = layout.piece_colours[p.piece]
        self.mask = self.pos >= 0
        self.index = {(p.piece, p.ori): j for j, p in enumerate(pl)}
        self.movable = [i for i in range(layout.n_pieces) if i != layout.core]
        self.cells = [c for c in range(layout.n_cells) if c != layout.placements[layout.core][layout.group_identity].cell]
        row = {p: i for i, p in enumerate(self.movable)}
        col = {c: j for j, c in enumerate(self.cells)}
        keep = self.piece != layout.core
        #: Placements of movable pieces, as (row, column) of the piece-by-cell assignment matrix.
        self.assign_placements = np.nonzero(keep)[0]
        self.assign_rc = np.array([row[int(self.piece[j])] * len(self.cells) + col[int(self.cell[j])] for j in self.assign_placements], dtype=np.int64)
        self.chain: StabiliserChain = build_chain(layout.generators, list(layout.sticker_homes))

    def reachable(self, orient: np.ndarray) -> bool:
        """Can moves reach this arrangement (core fixed)? False too when a piece is somewhere it never goes."""
        lay = self.layout
        if any(int(o) not in lay.placements[i] for i, o in enumerate(orient)):
            return False
        return self.chain.contains(lay.sticker_homes, lay.sticker_positions(orient))


@lru_cache(maxsize=2)
def _tables(layout_id: int) -> _Tables:
    return _Tables(_LAYOUTS[layout_id])


_LAYOUTS: dict[int, TangramLayout] = {}


def tables(layout: TangramLayout | None = None) -> _Tables:
    layout = layout or load_layout()
    _LAYOUTS[id(layout)] = layout
    return _tables(id(layout))


def _logp(probs: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0))


def placement_scores(t: _Tables, logp: np.ndarray) -> np.ndarray:
    """Evidence for every placement: sum over its stickers of log P(colour) - log P(empty)."""
    pos = np.where(t.mask, t.pos, 0)
    gain = logp[pos, t.col] - logp[pos, EMPTY]
    return np.where(t.mask, gain, 0.0).sum(1)


def _assign(t: _Tables, score: np.ndarray) -> np.ndarray:
    """The best arrangement (ignoring reachability): orientation per piece."""
    n = len(t.movable)
    js = t.assign_placements[np.argsort(-score[t.assign_placements], kind="stable")]
    rc = t.assign_rc[np.searchsorted(t.assign_placements, js)]
    uniq, first = np.unique(rc, return_index=True)              # best placement for each (piece, cell)
    best = np.full(n * n, -_IMPOSSIBLE)
    arg = np.full(n * n, -1, dtype=np.int64)
    best[uniq] = score[js[first]]
    arg[uniq] = js[first]
    rows, cols = linear_sum_assignment(-best.reshape(n, n))
    orient = t.layout.solved_orient.copy()
    chosen = arg.reshape(n, n)[rows, cols]
    for r, j in zip(rows, chosen):
        orient[t.movable[r]] = t.ori[j]
    return orient


def _total(t: _Tables, score: np.ndarray, orient: np.ndarray) -> float:
    return float(sum(score[t.index[(i, int(o))]] for i, o in enumerate(orient)))


def _changes(t: _Tables, score: np.ndarray, orient: np.ndarray) -> list[tuple[float, list[tuple[int, int]], str]]:
    """Single changes to an arrangement that keep it physical, cheapest first: (cost, [(piece, new ori)], words)."""
    lay = t.layout
    cell = lay.cell_of(orient)
    cur = np.array([score[t.index[(i, int(o))]] for i, o in enumerate(orient)])
    by_cell: dict[tuple[int, int], list[int]] = {}
    for (p, o), j in t.index.items():
        by_cell.setdefault((p, int(t.cell[j])), []).append(o)
    out: list[tuple[float, list[tuple[int, int]], str]] = []
    for i in t.movable:
        for o in by_cell.get((i, int(cell[i])), []):
            if o != orient[i]:
                out.append((cur[i] - score[t.index[(i, o)]], [(i, o)], f"the {piece_name(lay, i)} turned in its place"))
    for a in t.movable:
        for b in t.movable:
            if b <= a:
                continue
            oa = by_cell.get((a, int(cell[b])))
            ob = by_cell.get((b, int(cell[a])))
            if not oa or not ob:
                continue
            for x in oa:
                for y in ob:
                    cost = cur[a] + cur[b] - score[t.index[(a, x)]] - score[t.index[(b, y)]]
                    out.append((cost, [(a, x), (b, y)], f"the {piece_name(lay, a)} and the {piece_name(lay, b)} swapped"))
    out.sort(key=lambda c: c[0])
    return out


def _repair(t: _Tables, score: np.ndarray, orient: np.ndarray) -> tuple[np.ndarray, float, list[str]] | None:
    """The cheapest reachable arrangement within one or two changes of ``orient``: (orient, cost, words)."""
    changes = _changes(t, score, orient)
    for cost, ch, words in changes[:REPAIR_SINGLES]:
        o = orient.copy()
        for i, x in ch:
            o[i] = x
        if t.reachable(o):
            return o, float(cost), [words]
    top = changes[:REPAIR_PAIRS]
    pairs = []
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            pa, pb = {i for i, _ in top[a][1]}, {i for i, _ in top[b][1]}
            if pa & pb:
                continue
            pairs.append((top[a][0] + top[b][0], top[a], top[b]))
    pairs.sort(key=lambda p: p[0])
    for cost, a, b in pairs:
        o = orient.copy()
        for i, x in [*a[1], *b[1]]:
            o[i] = x
        if t.reachable(o):
            return o, float(cost), [a[2], b[2]]
    return None


def decode_logp(logp: np.ndarray, layout: TangramLayout | None = None, max_repair: float = MAX_REPAIR_NATS) -> TangramDecode:
    """The most likely legal state under per-position log-probabilities ``logp`` (120, 7): colours 0-5, empty."""
    layout = layout or load_layout()
    t = tables(layout)
    logp = np.asarray(logp, dtype=np.float64)
    if logp.shape != (layout.n_positions, N_COLOURS + 1):
        raise ValueError(f"need ({layout.n_positions}, {N_COLOURS + 1}) log-probabilities, got {logp.shape}")
    score = placement_scores(t, logp)
    base = float(logp[:, EMPTY].sum())
    orient = _assign(t, score)
    problems: list[Problem] = []
    repairs: list[str] = []
    legal = t.reachable(orient)
    if not legal:
        fixed = _repair(t, score, orient)
        if fixed is not None:
            o2, cost, repairs = fixed
            if cost <= max_repair:
                orient, legal = o2, True
            else:
                problems.append(Problem(
                    "unreachable",
                    "Every piece is there, but moves cannot reach this arrangement. The closest reachable one needs "
                    f"{' and '.join(repairs)}; check the marked stickers.",
                    _positions_changed(layout, orient, o2),
                ))
                repairs = []
        else:
            problems.append(Problem("unreachable", "Every piece is there, but moves cannot reach this arrangement.", []))
    obs = layout.observe(orient)
    probs = np.exp(logp)
    conf = probs[np.arange(layout.n_positions), obs]
    read = probs.argmax(1)
    corrections = [Correction(int(p), int(read[p]), int(obs[p]), float(conf[p])) for p in np.nonzero(read != obs)[0]]
    return TangramDecode(orient, obs, legal, problems, corrections, conf, base + _total(t, score, orient), repairs)


def assign_logp(logp: np.ndarray, layout: TangramLayout | None = None) -> tuple[np.ndarray, float]:
    """The most likely arrangement ignoring reachability, and its log-likelihood: the fast first look a
    scan uses to rank the ways its photos could fit together."""
    layout = layout or load_layout()
    t = tables(layout)
    logp = np.asarray(logp, dtype=np.float64)
    score = placement_scores(t, logp)
    orient = _assign(t, score)
    return orient, float(logp[:, EMPTY].sum()) + _total(t, score, orient)


def reachable(orient: np.ndarray, layout: TangramLayout | None = None) -> bool:
    return tables(layout).reachable(np.asarray(orient, dtype=np.int64))


def decode_probs(probs: np.ndarray, layout: TangramLayout | None = None) -> TangramDecode:
    """``probs`` (120, 7): per position the probability of each colour and of empty."""
    return decode_logp(_logp(probs), layout)


def piece_name(layout: TangramLayout, piece: int) -> str:
    """"the orange-yellow-green piece", "a green piece" (without the article)."""
    cols = layout.piece_colours[piece]
    return "-".join(layout.colour_names[c] for c in cols) + " piece" if cols else "core"


def _positions_changed(layout: TangramLayout, a: np.ndarray, b: np.ndarray) -> list[int]:
    return [int(p) for p in np.nonzero(layout.observe(a) != layout.observe(b))[0]]


def exact_logp(observation: np.ndarray, layout: TangramLayout) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.int64)
    logp = np.full((layout.n_positions, N_COLOURS + 1), np.log((1 - _SURE) / N_COLOURS))
    logp[np.arange(layout.n_positions), obs] = np.log(_SURE)
    return logp


def decode_exact(observation: np.ndarray, layout: TangramLayout | None = None) -> TangramDecode:
    """A hand-edited observation (120 values): legal exactly when it is a reachable state as given."""
    layout = layout or load_layout()
    obs = np.asarray(observation, dtype=np.int64)
    if obs.shape != (layout.n_positions,) or obs.min() < 0 or obs.max() > EMPTY:
        raise ValueError(f"an observation is {layout.n_positions} values, colours 0-5 or {EMPTY} for empty")
    d = decode_logp(exact_logp(obs, layout), layout, max_repair=0.0)
    problems = _explain(layout, obs, d)
    legal = not problems and d.legal and np.array_equal(d.observation, obs)
    return TangramDecode(d.orient, d.observation, legal, problems, [], np.ones(layout.n_positions), d.loglik, [])


def _explain(layout: TangramLayout, obs: np.ndarray, d: TangramDecode) -> list[Problem]:
    """What is wrong with a hand-edited observation, in words."""
    out: list[Problem] = []
    names = layout.colour_names
    for c in range(N_COLOURS):
        n = int((obs == c).sum())
        if n != 11:
            where = [int(p) for p in np.nonzero(obs == c)[0]] if n > 11 else []
            out.append(Problem("colour-count", f"There are {n} {names[c]} stickers; a Tangram Cube Extreme has 11 of each colour.", where))
    if d.observation is not None and not np.array_equal(d.observation, obs):
        wrong = [int(p) for p in np.nonzero(d.observation != obs)[0]]
        out.append(Problem(
            "no-piece",
            f"{len(wrong)} position{'s' if len(wrong) != 1 else ''} cannot be what is marked: no arrangement of the real pieces shows that "
            "(a sticker where nothing can be, a gap where a sticker must be, or colours no piece has together).",
            wrong,
        ))
    elif not d.legal:
        out.extend(d.problems or [Problem("unreachable", "Every piece is there, but moves cannot reach this arrangement.", [])])
    return out
