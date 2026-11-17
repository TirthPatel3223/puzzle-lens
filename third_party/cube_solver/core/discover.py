"""Choosing the order of the rungs from measurements (Part V, phase 2 of the design note).

Phase 1 (``core/structure.py``) lists what could be a rung: every feature of every orbit.
This module orders them with the note's rules, each turned into something measured on the
move tables. Nothing about a particular puzzle is assumed.

Rule 1, period hierarchy (orders the kinds of piece)
    The order of a generator on an orbit is its period there. If a generator has period 4
    on one orbit and 8 on another, its 4th power fixes the first and still moves the
    second: solving the first orbit first leaves short moves for the rest. Orbits are taken
    in ascending period. Orbits whose pieces move identically (a gear's colour and its
    rotation) are one kind of piece and stay together.

Rule 2, cheapest pure macros last (breaks ties between kinds of piece)
    The last kind solved needs words that move it and nothing else. The shortest such
    words are found exhaustively by meet in the middle; the kind with the shortest goes last.

Rule 4, bits per macro length (orders the features of one kind of piece)
    score(feature) = bits it resolves / mean length of the shortest macros it leaves behind.
    Bits are log2 of the feature's coset count under the moves then allowed, exact when a
    table fits the budget. Macros left are the shortest words that keep it (and everything
    before it) solved and still change something visible, again found exhaustively.
    When no feature of the kind leaves short macros, the finest one is taken as a single
    rung: splitting further only lengthens the macros of what follows.

A rung is given an exact table when its coset count fits the budget and a value network
otherwise. The result is a list of ``RungSpec`` that ``core.ladder.Ladder`` builds as is.

Completeness check
    A feature that no allowed short word changes is not necessarily solved: it may only need
    longer words. (On the Tangram Cube Extreme, with everything else solved, a turning corner can
    still be turned 120° and two middle edges flipped, but no word of up to 12 moves does it.)
    So at the end every orbit is tested exactly: does fixing the chosen features force it to its
    goal? The test is a pointwise stabiliser (sympy), which is exact when each feature's goal
    value pins its pieces down (all pieces look different, or the only look-alikes are empty
    places). An orbit that is not forced becomes part of a first rung, solved while the
    primitive moves are still allowed.

The scores are greedy and look one rung ahead. They reproduce the ladder used for the
Maltese Gear Cube, but they are a proposal to be checked by building it, not a guarantee.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from core.coset_table import build_table_auto
from core.harvest import ball, collisions
from core.ladder import Ladder, RungSpec
from core.perm_model import PointModel, Word
from core.quotient import ActionSet, Quotient
from core.structure import Feature, Orbit, Structure


@dataclass
class Candidate:
    spec: tuple                  # (orbit, kind)
    name: str
    cosets: int                  # coset count under the current moves (the budget, if it was exceeded)
    exact: bool
    macro_len: float             # mean length of the shortest macros left behind (inf: none within 2R)
    n_macros: int

    @property
    def bits(self) -> float:
        return math.log2(max(self.cosets, 1))

    @property
    def score(self) -> float:
        return 0.0 if math.isinf(self.macro_len) else self.bits / self.macro_len


def generator_periods(model: PointModel, structure: Structure) -> list[list[int]]:
    """For each orbit, the order of each primitive move on that orbit's points."""
    out = []
    for orb in structure.orbits:
        row = []
        for m in range(model.num_moves):
            g = model.gens[m]
            cur = g.copy()
            k = 1
            while not np.array_equal(cur[orb.points], orb.points):
                cur = cur[g]
                k += 1
            row.append(k)
        out.append(row)
    return out


def _slot_perms(model: PointModel, orb: Orbit) -> np.ndarray:
    """(M, n_pieces): the slot each slot is filled from, for every primitive move."""
    return np.stack([orb.slot_of[model.gens[m][orb.ref_pos]] for m in range(model.num_moves)])


def linked_orbits(model: PointModel, structure: Structure) -> list[list[int]]:
    """Groups of orbits whose pieces are permuted identically by every move. They describe
    the same physical pieces (for instance a counter and the colour that travels with it).
    An orbit with a single piece (a piece that only turns in place) is permuted trivially,
    which says nothing, so it is never grouped with another."""
    perms = [_slot_perms(model, o) for o in structure.orbits]
    groups: list[list[int]] = []
    for i, orb in enumerate(structure.orbits):
        placed = False
        for group in groups:
            j = group[0]
            if structure.orbits[j].n_pieces != orb.n_pieces or orb.n_pieces == 1:
                continue
            # look for a relabelling of slots that turns one action into the other
            n = orb.n_pieces
            for image_of_zero in range(n):
                pi = np.full(n, -1)
                pi[0] = image_of_zero
                stack, ok = [0], True
                while stack and ok:
                    x = stack.pop()
                    for m in range(model.num_moves):
                        y, target = int(perms[i][m][x]), int(perms[j][m][pi[x]])
                        if pi[y] < 0:
                            pi[y] = target
                            stack.append(y)
                        elif pi[y] != target:
                            ok = False
                            break
                if ok and (pi >= 0).all():
                    group.append(i)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            groups.append([i])
    return groups


def _fixes(features: Sequence[Feature], tables: np.ndarray) -> np.ndarray:
    ok = np.ones(len(tables), dtype=bool)
    for f in features:
        ok &= (f.compute(tables) == f.goal).all(axis=-1)
    return ok


def stabiliser_points(feature: Feature) -> list[int] | None:
    """Points whose pointwise stabiliser is exactly the set of elements that leave ``feature`` at
    its goal value, or None if no such set exists.

    A slot is pinned when only its own home piece shows the goal class there. Fixing the feature
    then fixes the pinned slots' points, and the unpinned pieces fill the unpinned slots. This is
    the whole stabiliser when every unpinned piece shows the goal class in every unpinned slot it
    can reach: the unpinned pieces are look-alikes of one kind (for example empty places), so any
    arrangement of them shows the goal. Otherwise (identical pieces of several colours, say) None.
    """
    orb = feature.orbit
    hits = feature.table == feature.goal[:, None]                   # (slots, labels): shows the goal class
    possible = orb.arrangement[:, :, 0] >= 0
    pinned = [b for b in range(orb.n_pieces) if int(hits[b].sum()) == 1]
    loose = [b for b in range(orb.n_pieces) if int(hits[b].sum()) > 1]
    loose_labels = [int(x) for b in loose for x in orb.pieces[b]]
    for b in loose:
        for c in loose_labels:
            if possible[b, c] and not hits[b, c]:
                return None
    return sorted(int(p) for b in pinned for p in orb.pieces[b])


def forced_all(model: PointModel, fixed: Sequence[Feature], features: Sequence[Feature]) -> list[bool] | None:
    """For each of ``features``: does every element that keeps ``fixed`` at their goal values keep it
    at its goal too? Exact (one pointwise stabiliser in sympy); None when ``stabiliser_points``
    cannot express one of the fixed features."""
    from sympy.combinatorics import Permutation, PermutationGroup

    points: list[int] = []
    for f in fixed:
        p = stabiliser_points(f)
        if p is None:
            return None
        points.extend(p)
    group = PermutationGroup([Permutation(np.argsort(g).tolist()) for g in model.gens])   # image form
    stab = group.pointwise_stabilizer(sorted(set(points))) if points else group
    labels = [np.argsort(np.asarray(k.array_form)) for k in stab.generators]  # the labelled state each produces
    return [all((f.compute(lab) == f.goal).all() for lab in labels) for f in features]


def forced(model: PointModel, fixed: Sequence[Feature], feature: Feature) -> bool | None:
    """``forced_all`` for one feature."""
    out = forced_all(model, fixed, [feature])
    return None if out is None else out[0]


def propose_ladder(model: PointModel, structure: Structure, radius: int = 6, table_budget: int = 10_000_000,
                   n_shortest: int = 12, max_actions: int = 64,
                   log: Callable[[str], None] | None = None) -> tuple[list[RungSpec], dict]:
    """Rung order from the rules above. Returns the rung specifications and the measurements."""
    log = log or (lambda msg: print(msg, flush=True))
    prim = ActionSet.primitives(model)
    tables, words, _ = ball(model, prim, radius)
    everything = [structure.feature(o.index, "full") for o in structure.orbits]
    helper = Ladder(model, structure, [], mitm_radius=radius, log=lambda _msg: None)

    def stabiliser_words(features: Sequence[Feature]) -> list[Word]:
        """Words within 2R that fix ``features`` and change something visible, shortest first."""
        if not features:
            return [(m,) for m in range(model.num_moves)]
        found = collisions(model, tables, words, features)
        if not found:
            return []
        keys = list(found.keys())
        tabs = np.frombuffer(b"".join(keys), dtype=np.int16).reshape(-1, model.n).astype(np.int32)
        change = ~_fixes(everything, tabs)
        return sorted((found[k] for k, ch in zip(keys, change) if ch), key=lambda w: (len(w), w))

    # -- rules 1 and 2: order the kinds of piece ---------------------------------------
    periods = generator_periods(model, structure)
    groups = linked_orbits(model, structure)
    report: dict = {"periods": periods, "groups": groups, "steps": []}
    info = []
    for group in groups:
        others = [everything[o] for o in range(len(structure.orbits)) if o not in group]
        pure = stabiliser_words(others) if others else []
        pure_len = len(pure[0]) if pure else math.inf
        info.append((max(max(periods[o]) for o in group), pure_len, group))
    # ascending period; among equal periods, the cheapest pure macros go last
    info.sort(key=lambda t: (t[0], -t[1] if not math.isinf(t[1]) else -1e9))
    report["group_order"] = [(g, p, pl) for p, pl, g in info]
    for period, pure_len, group in info:
        pure = "none within %d" % (2 * radius) if math.isinf(pure_len) else f"{pure_len} moves"
        log(f"orbits {group}: generator period {period}, shortest pure macro {pure}")

    # -- rule 4: order the features inside each kind ------------------------------------
    fixed: list[Feature] = []
    pool: list[Word] = [(m,) for m in range(model.num_moves)]
    specs: list[RungSpec] = []
    all_candidates = [(f.orbit.index, f.name.split(".", 1)[1]) for f in structure.candidate_features()]
    for _, _, group in info:
        while True:
            pool_tables = model.compile_many(pool[:600]) if pool else np.empty((0, model.n), dtype=np.int32)
            open_specs = []
            for spec in all_candidates:
                if spec[0] not in group:
                    continue
                f = structure.feature(*spec)
                if len(pool_tables) and not _fixes([f], pool_tables).all():      # some allowed move still changes it
                    open_specs.append((spec, f))
            if not open_specs:
                break
            scored: list[Candidate] = []
            for spec, f in open_specs:
                actions = helper._select_actions(RungSpec(f.name, [spec], max_actions=max_actions), [f], pool)
                quotient = Quotient([f], actions)
                if quotient.packable:
                    table = build_table_auto(quotient, max_states=table_budget)
                    cosets, exact = len(table), table.complete
                else:                                   # too many feature vectors to index: certainly over budget
                    cosets, exact = table_budget, False
                left = stabiliser_words(fixed + [f])
                shortest = left[:n_shortest]
                macro_len = float(np.mean([len(w) for w in shortest])) if shortest else math.inf
                scored.append(Candidate(spec, f.name, cosets, exact, macro_len, len(left)))
            scored.sort(key=lambda c: (-c.score, -c.bits))
            best = scored[0]
            if best.score == 0.0:                       # nothing leaves short macros: take the finest feature whole
                fulls = [c for c in scored if c.spec[1] == "full"]
                best = max(fulls or scored, key=lambda c: c.bits)
            log(f"orbits {group}, {len(fixed)} feature(s) fixed, {len(pool):,} allowed moves of length "
                f"{min(map(len, pool))}..{max(map(len, pool))}:")
            for c in scored:
                size = f"{c.cosets:,}" if c.exact else f">{table_budget:,}"
                macros = "none within %d" % (2 * radius) if math.isinf(c.macro_len) else f"{c.macro_len:.1f} ({c.n_macros:,} words)"
                mark = "  <- chosen" if c is best else ""
                log(f"    {c.name:18s} cosets {size:>14s}  bits {c.bits:5.1f}  macros left: {macros:26s} score {c.score:5.2f}{mark}")
            report["steps"].append({"group": group, "candidates": scored, "chosen": best.name})
            mode = "table" if best.exact else "learned"
            specs.append(RungSpec(best.name.replace(".", "_"), [best.spec], mode=mode,
                                  max_actions=600 if mode == "table" else 128))
            fixed.append(structure.feature(*best.spec))
            pool = stabiliser_words(fixed)
            if not pool:
                break
        if not pool:
            break

    # -- completeness: an orbit no short word changes may still be unsolved --------------------
    verdicts = forced_all(model, fixed, everything)
    if verdicts is None:
        loose, undecided = [], [o.index for o in structure.orbits]
    else:
        loose = [o.index for o, ok in zip(structure.orbits, verdicts) if not ok]
        undecided = []
    report["not_forced"], report["undecided"] = loose, undecided
    if undecided:
        log(f"orbits {undecided}: identical-looking pieces, so whether the rungs above solve them is not "
            "decided here; building the ladder and replaying its solutions checks it")
    if loose:
        feats = [(o, "full") for o in loose]
        q = Quotient([structure.feature(*fs) for fs in feats], prim)
        table = build_table_auto(q, max_states=table_budget) if q.packable else None
        mode = "table" if table is not None and table.complete else "learned"
        size = f"{len(table):,} cosets" if table is not None and table.complete else "too many cosets for a table"
        log(f"orbits {loose} are not forced by the rungs above: no word of up to {2 * radius} moves reaches them "
            f"once those are solved. They go first, with the primitive moves ({size}, {mode}).")
        specs.insert(0, RungSpec("first", feats, mode=mode, max_actions=600 if mode == "table" else 128))
    return specs, report
