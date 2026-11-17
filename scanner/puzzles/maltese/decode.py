"""From what the cube shows to a legal arrangement of its 50 pieces.

Two entry points share one decoder:

``decode_exact``  for a hand-edited observation (the 3D editor): nothing is changed, every
                  impossibility is reported in words with the cells involved.
``decode_probs``  for the vision model's per-cell probabilities: the most likely *legal*
                  arrangement is found, and what had to be overruled is reported.

A visible arrangement can be reached by moves exactly when (``tools/derive_invariants.py``
proves this from the order of the move group):

1. every piece appears exactly once (8 corners, 6 centres, 12 edges, 4 gears per colour);
2. the corner twists add up to 0 mod 3;
3. the permutation of the 18 centre and edge pieces is even;
4. the gear quarter turns add up to 0 mod 4.

Corner parity and edge turns carry no further visible constraint: a centre's own turn and
which of four same-coloured gears is which cannot be seen, and they absorb the rest.

Everything is decoded in the solver's frame: the cube is first turned (as a whole) so that
the white-green-red corner is home, wherever it was when it was scanned.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .layout import N_CELLS, N_COLOURS, N_GEARS, Layout, Observation, Slot, load_layout

#: Probability given to the label of a hand-edited cell (the rest is shared by the other labels).
_SURE = 1.0 - 1e-9
#: A repair that needs a read this confident to be wrong is not made silently; the scan is flagged instead.
MAX_REPAIR_NATS = 6.0


@dataclass
class Problem:
    code: str
    message: str
    cells: list[int] = field(default_factory=list)     # cell indices, in the frame the input was given in
    gears: list[int] = field(default_factory=list)     # gear places, same frame

    def to_json(self) -> dict:
        return {"code": self.code, "message": self.message, "cells": self.cells, "gears": self.gears}


@dataclass
class Correction:
    """A place where the decoder overruled the most likely single read."""

    kind: str            # "cell", "gearColour" or "gearQuarter"
    index: int           # in the frame the input was given in
    read: int
    decoded: int
    probability: float   # the probability the reader gave the decoded value

    def to_json(self) -> dict:
        return {"kind": self.kind, "index": self.index, "read": self.read, "decoded": self.decoded,
                "probability": round(float(self.probability), 6)}


@dataclass
class DecodeResult:
    #: The arrangement shown, in the solver's frame (None if no arrangement of real pieces fits).
    observation: Observation | None
    #: Whole-cube rotation (cube-group index) from the input frame to the solver's frame.
    rotation: int
    #: Engine poses of the 50 pieces in the solver's frame: slot and orientation, by piece id.
    slot: list[int] | None
    ori: list[int] | None
    #: The solver's 144-entry array (None unless legal).
    state144: list[int] | None
    legal: bool
    problems: list[Problem]
    corrections: list[Correction]
    #: Probability the reader gave each decoded value, in the input frame (1.0 for hand-edited input).
    cell_confidence: list[float]
    gear_confidence: list[float]
    #: For each solver-frame cell / gear place, its index in the input frame.
    cell_source: list[int]
    gear_source: list[int]

    def to_json(self) -> dict:
        return {
            "observation": self.observation.to_json() if self.observation else None,
            "rotation": self.rotation,
            "slot": self.slot,
            "ori": self.ori,
            "state144": self.state144,
            "legal": self.legal,
            "problems": [p.to_json() for p in self.problems],
            "corrections": [c.to_json() for c in self.corrections],
            "cellConfidence": [round(float(x), 4) for x in self.cell_confidence],
            "gearConfidence": [round(float(x), 4) for x in self.gear_confidence],
            "cellSource": self.cell_source,
            "gearSource": self.gear_source,
        }


# ---------------------------------------------------------------------------------
# Structure derived once from the layout
# ---------------------------------------------------------------------------------

class _Structure:
    """Twist bookkeeping and home slots, derived from the layout's moves and solved state."""

    def __init__(self, layout: Layout) -> None:
        self.layout = layout
        solved = layout.solved.cells
        # A cyclic order of each corner slot's three cells with the same handedness everywhere:
        # start from one slot and let the moves (rotations, so they keep handedness) carry it round.
        slot_of_cell = {c: s.index for s in layout.corners for c in s.cells}
        cyc: dict[int, list[int]] = {layout.corners[0].index: list(layout.corners[0].cells)}
        grew = True
        while grew:
            grew = False
            for m in layout.moves.values():
                to = np.empty(N_CELLS, dtype=np.int64)
                to[m.cells] = np.arange(N_CELLS)           # where the content of each cell goes
                for s, order in list(cyc.items()):
                    img = [int(to[c]) for c in order]
                    s2 = slot_of_cell[img[0]]
                    if s2 not in cyc:
                        cyc[s2] = img
                        grew = True
        if len(cyc) != len(layout.corners):
            raise RuntimeError("the moves do not connect all corner slots")
        self.corner_cycle = cyc
        self.piece_home_slot: dict[int, int] = {}
        # Corners: the piece at home shows the solved colours of its slot.
        for s in layout.corners:
            for o in s.options:
                if all(solved[c] == col for c, col in zip(s.cells, o.colours)):
                    self.piece_home_slot[o.piece] = s.index
        for s in layout.radial:
            for o in s.options:
                if o.turn == 0 and all(solved[c] == col for c, col in zip(s.cells, o.colours)):
                    self.piece_home_slot.setdefault(o.piece, s.index)
        # twist[slot][option index]: how far the piece is turned from the way it sits at home.
        self.corner_twist: list[list[int]] = []
        for s in layout.corners:
            order = cyc[s.index]
            pos = [s.cells.index(c) for c in order]
            row = []
            for o in s.options:
                shown = [o.colours[i] for i in pos]
                home = [int(solved[c]) for c in cyc[self.piece_home_slot[o.piece]]]
                j = next(j for j in range(3) if all(shown[(i + j) % 3] == home[i] for i in range(3)))
                row.append(j)
            self.corner_twist.append(row)


_STRUCTURES: dict[int, _Structure] = {}


def _structure(layout: Layout) -> _Structure:
    s = _STRUCTURES.get(id(layout))
    if s is None:
        s = _STRUCTURES[id(layout)] = _Structure(layout)
    return s


def _parity(perm: list[int]) -> int:
    """0 for an even permutation, 1 for an odd one."""
    seen = [False] * len(perm)
    odd = 0
    for i in range(len(perm)):
        n = 0
        j = i
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            n += 1
        if n and n % 2 == 0:
            odd ^= 1
    return odd


# ---------------------------------------------------------------------------------
# Assignment of pieces to slots
# ---------------------------------------------------------------------------------

def _slot_scores(slots: list[Slot], logp: np.ndarray, pieces: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """score[slot, piece] = best log-probability of that piece in that slot, and the option that gives it."""
    col = {p: j for j, p in enumerate(pieces)}
    score = np.full((len(slots), len(pieces)), -np.inf)
    best = np.full((len(slots), len(pieces)), -1, dtype=np.int64)
    for i, s in enumerate(slots):
        cells = np.array(s.cells)
        for k, o in enumerate(s.options):
            v = float(logp[cells, np.array(o.colours)].sum())
            j = col[o.piece]
            if v > score[i, j]:
                score[i, j] = v
                best[i, j] = k
    return score, best


def _assign(score: np.ndarray) -> np.ndarray:
    rows, cols = linear_sum_assignment(-score)
    out = np.empty(score.shape[0], dtype=np.int64)
    out[rows] = cols
    return out


def _residue_repair(costs: np.ndarray, need: int, modulus: int) -> tuple[float, np.ndarray]:
    """Cheapest changes d_i (0..modulus-1) with sum = need (mod modulus); costs[i, d] is the price of d_i."""
    n = costs.shape[0]
    best = np.full((n + 1, modulus), np.inf)
    choice = np.zeros((n + 1, modulus), dtype=np.int64)
    best[0, 0] = 0.0
    for i in range(n):
        for r in range(modulus):
            if not np.isfinite(best[i, r]):
                continue
            for d in range(modulus):
                c = best[i, r] + costs[i, d]
                r2 = (r + d) % modulus
                if c < best[i + 1, r2]:
                    best[i + 1, r2] = c
                    choice[i + 1, r2] = d
    d = np.zeros(n, dtype=np.int64)
    r = need % modulus
    total = float(best[n, r])
    for i in range(n, 0, -1):
        d[i - 1] = choice[i, r]
        r = (r - d[i - 1]) % modulus
    return total, d


# ---------------------------------------------------------------------------------
# The decoder
# ---------------------------------------------------------------------------------

def decode_probs(cell_p: np.ndarray, gear_colour_p: np.ndarray, gear_quarter_p: np.ndarray,
                 layout: Layout | None = None, strict: bool = False) -> DecodeResult:
    """Most likely legal arrangement for per-cell probabilities given in any whole-cube orientation.

    cell_p: (96, 6), gear_colour_p: (24, 6), gear_quarter_p: (24, 4); rows sum to 1.
    With ``strict`` nothing is overruled: every disagreement becomes a problem.
    """
    layout = layout or load_layout()
    st = _structure(layout)
    cell_p = np.asarray(cell_p, dtype=np.float64).reshape(N_CELLS, N_COLOURS)
    gear_colour_p = np.asarray(gear_colour_p, dtype=np.float64).reshape(N_GEARS, N_COLOURS)
    gear_quarter_p = np.asarray(gear_quarter_p, dtype=np.float64).reshape(N_GEARS, 4)
    eps = 1e-12
    read = Observation(cell_p.argmax(1), gear_colour_p.argmax(1), gear_quarter_p.argmax(1))

    corner_pieces = [p["id"] for p in layout.pieces if p["kind"] == "corner"]
    radial_pieces = [p["id"] for p in layout.pieces if p["kind"] in ("centre", "edge")]
    gear_pieces = [p["id"] for p in layout.pieces if p["kind"] == "gear"]
    gear_colour_of = np.array([layout.pieces[i]["colours"][0] for i in gear_pieces])

    # 1. Find the white-green-red corner in the frame given, and turn the cube to bring it home.
    logp = np.log(np.clip(cell_p, eps, 1.0))
    score, best = _slot_scores(layout.corners, logp, corner_pieces)
    pick = _assign(score)
    peg = layout.pegged["piece"]
    peg_slot = int(np.nonzero(pick == corner_pieces.index(peg))[0][0])
    peg_ori = layout.corners[peg_slot].options[best[peg_slot, corner_pieces.index(peg)]].turn
    rotation = int(layout.group_mul[layout.pegged["ori"], layout.group_inv[peg_ori]])
    rot = layout.rotations[rotation]
    cell_source = rot.cells
    gear_source = rot.gears
    cell_p = cell_p[cell_source]
    gear_colour_p = gear_colour_p[gear_source]
    gear_quarter_p = np.stack([np.roll(gear_quarter_p[g], int(rot.dk[i])) for i, g in enumerate(gear_source)])
    logp = np.log(np.clip(cell_p, eps, 1.0))
    log_gc = np.log(np.clip(gear_colour_p, eps, 1.0))
    log_gq = np.log(np.clip(gear_quarter_p, eps, 1.0))

    problems: list[Problem] = []
    src_cells = lambda cells: [int(cell_source[c]) for c in cells]          # noqa: E731

    # 2. Pieces into slots.
    c_score, c_best = _slot_scores(layout.corners, logp, corner_pieces)
    c_pick = _assign(c_score)
    c_opt = [int(c_best[i, c_pick[i]]) for i in range(len(layout.corners))]
    r_score, r_best = _slot_scores(layout.radial, logp, radial_pieces)
    r_pick = _assign(r_score)
    g_pick = _assign(log_gc[:, gear_colour_of])                                # place -> gear piece
    quarter = log_gq.argmax(1)

    # 3. The three invariants, repaired where the evidence allows.
    def unlikely(cost: float) -> bool:
        return strict or cost > MAX_REPAIR_NATS

    # 3a. corner twists add up to 0 mod 3
    twist = sum(st.corner_twist[i][c_opt[i]] for i in range(len(layout.corners))) % 3
    if twist:
        costs = np.full((len(layout.corners), 3), np.inf)
        alt: dict[tuple[int, int], int] = {}
        for i, s in enumerate(layout.corners):
            piece = corner_pieces[c_pick[i]]
            base = st.corner_twist[i][c_opt[i]]
            cells = np.array(s.cells)
            here = float(logp[cells, np.array(s.options[c_opt[i]].colours)].sum())
            for k, o in enumerate(s.options):
                if o.piece != piece:
                    continue
                d = (st.corner_twist[i][k] - base) % 3
                costs[i, d] = here - float(logp[cells, np.array(o.colours)].sum())
                alt[(i, d)] = k
        if layout.pegged["slot"] < len(layout.corners):
            costs[layout.pegged["slot"], 1:] = np.inf                              # the pegged corner defines the frame
        cost, d = _residue_repair(costs, -twist, 3)
        if unlikely(cost):
            problems.append(Problem(
                "corner-twist",
                f"The corner twists do not add up: one corner is turned {'a third' if twist == 1 else 'two thirds'} "
                "of a turn too far. Check each corner's three stars (or a photo that was taken rotated).",
                cells=src_cells([c for s in layout.corners for c in s.cells])))
        else:
            for i in range(len(layout.corners)):
                if d[i]:
                    c_opt[i] = alt[(i, int(d[i]))]

    # 3b. the centre/edge permutation is even
    home_of = [st.piece_home_slot[radial_pieces[r_pick[i]]] for i in range(len(layout.radial))]
    if _parity(home_of):
        # The cheapest way to an even permutation: exchange the pieces of the two slots that mind it least.
        best_swap, best_cost = None, np.inf
        for a in range(len(layout.radial)):
            for b in range(a + 1, len(layout.radial)):
                c = r_score[a, r_pick[a]] + r_score[b, r_pick[b]] - r_score[a, r_pick[b]] - r_score[b, r_pick[a]]
                if c < best_cost:
                    best_swap, best_cost = (a, b), float(c)
        if best_swap is None or unlikely(best_cost):
            problems.append(Problem(
                "ring-parity",
                "Two centre or edge pieces are swapped: no sequence of moves swaps just two of them. "
                "Check the small square stickers on the centres and edges.",
                cells=src_cells([c for s in layout.radial for c in s.cells])))
        else:
            a, b = best_swap
            r_pick[a], r_pick[b] = r_pick[b], r_pick[a]
    r_opt = [int(r_best[i, r_pick[i]]) for i in range(len(layout.radial))]

    # 3c. the gear quarter turns add up to 0 mod 4
    total = int(quarter.sum()) % 4
    if total:
        costs = np.stack([log_gq[i, quarter[i]] - log_gq[i, (quarter[i] + np.arange(4)) % 4] for i in range(N_GEARS)])
        cost, d = _residue_repair(costs, -total, 4)
        if unlikely(cost):
            problems.append(Problem(
                "gear-sum",
                f"The gear quarter turns add up to {int(quarter.sum())}, which is not a multiple of 4: "
                "at least one gear is turned differently from what is shown.",
                gears=[int(g) for g in gear_source]))
        else:
            quarter = (quarter + d) % 4

    # 4. What that arrangement shows, and whether it is what was read.
    cells = np.full(N_CELLS, -1, dtype=np.int64)
    slot = [-1] * len(layout.pieces)
    ori = [0] * len(layout.pieces)
    for i, s in enumerate(layout.corners):
        o = s.options[c_opt[i]]
        cells[list(s.cells)] = o.colours
        slot[o.piece], ori[o.piece] = s.index, o.turn
    for i, s in enumerate(layout.radial):
        o = s.options[r_opt[i]]
        cells[list(s.cells)] = o.colours
        slot[o.piece], ori[o.piece] = s.index, o.turn
    gear_colour = gear_colour_of[g_pick]
    for i in range(N_GEARS):
        piece = gear_pieces[g_pick[i]]
        slot[piece], ori[piece] = int(layout.gear_slot[i]), int(quarter[i])
    decoded = Observation(cells, gear_colour, quarter)

    corrections: list[Correction] = []
    for i in range(N_CELLS):
        src = int(cell_source[i])
        if decoded.cells[i] != read.cells[src]:
            corrections.append(Correction("cell", src, int(read.cells[src]), int(decoded.cells[i]), cell_p[i, decoded.cells[i]]))
    for i in range(N_GEARS):
        src = int(gear_source[i])
        if decoded.gear_colour[i] != read.gear_colour[src]:
            corrections.append(Correction("gearColour", src, int(read.gear_colour[src]), int(decoded.gear_colour[i]),
                                          gear_colour_p[i, decoded.gear_colour[i]]))
        read_q = (int(read.gear_quarter[src]) + int(rot.dk[i])) % 4
        if decoded.gear_quarter[i] != read_q:
            corrections.append(Correction("gearQuarter", src, int(read.gear_quarter[src]),
                                          int((decoded.gear_quarter[i] - rot.dk[i]) % 4), gear_quarter_p[i, decoded.gear_quarter[i]]))

    if strict:
        # Hand-edited input is never overruled. If its pieces are not a real set, say which and stop
        # there (twists and parities mean nothing until the pieces exist); otherwise the invariant
        # problems found above are the answer.
        missing = _explain(layout, read)
        if missing:
            problems = missing
        elif corrections and not problems:
            problems.append(Problem("impossible", "These stickers cannot be a real cube.",
                                    cells=[c.index for c in corrections if c.kind == "cell"],
                                    gears=[c.index for c in corrections if c.kind != "cell"]))

    cell_conf = np.ones(N_CELLS)
    gear_conf = np.ones(N_GEARS)
    for i in range(N_CELLS):
        cell_conf[int(cell_source[i])] = cell_p[i, decoded.cells[i]]
    for i in range(N_GEARS):
        gear_conf[int(gear_source[i])] = min(gear_colour_p[i, decoded.gear_colour[i]], gear_quarter_p[i, decoded.gear_quarter[i]])

    legal = not problems
    shown = decoded if (legal or not strict) else None
    return DecodeResult(
        observation=shown,
        rotation=rotation,
        slot=slot if shown is not None else None,
        ori=ori if shown is not None else None,
        state144=layout.to_solver(decoded).tolist() if legal else None,
        legal=legal,
        problems=problems,
        corrections=[] if strict else corrections,
        cell_confidence=cell_conf.tolist(),
        gear_confidence=gear_conf.tolist(),
        cell_source=[int(c) for c in cell_source],
        gear_source=[int(g) for g in gear_source],
    )


def _explain(layout: Layout, read: Observation) -> list[Problem]:
    """Why an observation is not an arrangement of real pieces, slot by slot, in the frame it was given in."""
    names = layout.colour_names
    out: list[Problem] = []
    used: dict[int, Slot] = {}
    for s in [*layout.corners, *layout.radial]:
        colours = tuple(int(read.cells[c]) for c in s.cells)
        match = [o for o in s.options if o.colours == colours]
        what = {"corner": "corner", "centre": "centre slot", "edge": "edge slot"}[s.kind]
        if not match:
            out.append(Problem(
                "no-such-piece",
                f"The {what} {s.name} shows {', '.join(names[c] for c in colours)}: no piece has those colours like that.",
                cells=list(s.cells)))
            continue
        piece = match[0].piece
        if piece in used:
            other = used[piece]
            out.append(Problem(
                "piece-twice",
                f"The same {layout.pieces[piece]['kind']} piece ({'-'.join(names[c] for c in layout.pieces[piece]['colours'])}) "
                f"appears twice: at {other.name} and at {s.name}.",
                cells=list(s.cells) + list(other.cells)))
        else:
            used[piece] = s
    for colour in range(N_COLOURS):
        places = [i for i in range(N_GEARS) if read.gear_colour[i] == colour]
        if len(places) != 4:
            out.append(Problem(
                "gear-count",
                f"There are {len(places)} {names[colour]} gears; a cube has four of each colour.",
                gears=places))
    return out


def decode_exact(obs: Observation, layout: Layout | None = None) -> DecodeResult:
    """Check a hand-edited observation (any whole-cube orientation) and turn it to the solver's frame."""
    layout = layout or load_layout()
    rest_c = (1.0 - _SURE) / (N_COLOURS - 1)
    cell_p = np.full((N_CELLS, N_COLOURS), rest_c)
    cell_p[np.arange(N_CELLS), obs.cells] = _SURE
    gear_colour_p = np.full((N_GEARS, N_COLOURS), rest_c)
    gear_colour_p[np.arange(N_GEARS), obs.gear_colour] = _SURE
    gear_quarter_p = np.full((N_GEARS, 4), (1.0 - _SURE) / 3)
    gear_quarter_p[np.arange(N_GEARS), obs.gear_quarter] = _SURE
    return decode_probs(cell_p, gear_colour_p, gear_quarter_p, layout, strict=True)
