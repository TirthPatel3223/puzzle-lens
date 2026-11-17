"""Finding short elements of a stabiliser subgroup: the macros of the next rung.

A rung may only use moves that keep every earlier rung solved. Those moves form the
subgroup H that fixes the earlier rungs' goal features. This module finds short words in H.

Two sources, both exact (no heuristics, every result is verified by the caller):

Meet in the middle (``ball`` + ``collisions``)
    Enumerate every element reachable within a cost budget. Two elements u, v that show
    the same features differ by an element of H:  u * v^-1  fixes the goal features. With a
    ball of radius R in a unit-cost alphabet this finds *every* element of H of length <= 2R.

Schreier generators (``schreier_elements``)
    For a rung with an exact table, every coset c has a cheapest word t_c from the goal.
    For every coset c and action a,  t_c * a * t_(c.a)^-1  lies in H, and by Schreier's
    lemma these elements generate H. The cheap ones make good macros; a random sample of
    all of them generates H with high probability.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from core.coset_table import CosetTable
from core.perm_model import PointModel, Word
from core.quotient import ActionSet, Quotient
from core.structure import Feature


# -- words -----------------------------------------------------------------------------

class WordAlgebra:
    """Free reduction of primitive words, with runs of one move reduced modulo its order."""

    def __init__(self, model: PointModel) -> None:
        from core.perm_model import move_orders

        self.model = model
        self.inverse = [int(x) for x in model.inverse]
        self.order = move_orders(model)
        # A "face" is a move together with its inverse; pick the smaller index as the base.
        self.base = [min(m, self.inverse[m]) for m in range(model.num_moves)]
        self.sign = [1 if m == self.base[m] else -1 for m in range(model.num_moves)]
        gens = [np.asarray(g) for g in model.gens]
        # commutes[a][b]: moves a and b commute (the Tangram Cube's two corner layers; no Maltese pair).
        self.commutes = [[bool(np.array_equal(ga[gb], gb[ga])) for gb in gens] for ga in gens]

    def simplify(self, word: Sequence[int]) -> Word:
        """``reduce``, and also merge a turn into an earlier turn of its face when every turn between
        them commutes with it: ``D' U' D`` is ``U'`` because the two corner layers commute. Only group
        identities are used, so the result is the same element; it is never longer."""
        runs: list[list[int]] = []                    # [base move, exponent modulo the order]
        for m in self.reduce(word):
            b, s = self.base[m], self.sign[m]
            if self.inverse[b] == b:
                s = 1
            j = len(runs) - 1
            while j >= 0 and runs[j][0] != b and self.commutes[runs[j][0]][b]:
                j -= 1
            if j >= 0 and runs[j][0] == b:
                runs[j][1] = (runs[j][1] + s) % self.order[b]
                if runs[j][1] == 0:
                    del runs[j]                       # every later run commutes with b, so none can merge now
            else:
                runs.append([b, s % self.order[b]])
        out: list[int] = []
        for b, e in runs:
            k = self.order[b]
            out.extend([b] * e if e <= k - e else [self.inverse[b]] * (k - e))
        return self.reduce(out)

    def reduce(self, word: Sequence[int]) -> Word:
        """Shortest word obtainable by merging adjacent turns of the same face."""
        runs: list[list[int]] = []                    # [base move, net exponent]
        for m in word:
            m = int(m)
            b, s = self.base[m], self.sign[m]
            if self.inverse[b] == b:                  # an involution: exponent modulo 2
                s = 1
            if runs and runs[-1][0] == b:
                runs[-1][1] += s
            else:
                runs.append([b, s])
            while runs:
                b2, e = runs[-1]
                e %= self.order[b2]
                if e == 0:
                    runs.pop()
                    if len(runs) >= 2 and runs[-1][0] == runs[-2][0]:
                        runs[-2][1] += runs[-1][1]
                        runs.pop()
                        continue
                else:
                    runs[-1][1] = e
                break
        out: list[int] = []
        for b, e in runs:
            k = self.order[b]
            if e <= k - e:
                out.extend([b] * e)
            else:
                out.extend([self.inverse[b]] * (k - e))
        return tuple(out)

    def invert(self, word: Sequence[int]) -> Word:
        return tuple(self.inverse[int(m)] for m in reversed(word))

    def runs(self, word: Sequence[int]) -> list[tuple[int, int]]:
        """The reduced word as (base move, signed exponent) runs, e.g. U U U' L' -> [(U, 1), (L, -1)]."""
        out: list[tuple[int, int]] = []
        for m in self.reduce(word):
            b, s = self.base[m], self.sign[m]
            if out and out[-1][0] == b:
                out[-1] = (b, out[-1][1] + s)
            else:
                out.append((b, s))
        return out

    def face_turns(self, word: Sequence[int]) -> int:
        """Length in the face-turn metric used for this family of puzzles: a single or double
        turn of a face counts one, a triple or quadruple turn counts two."""
        return sum(1 if abs(e) <= 2 else 2 for _, e in self.runs(word))

    def format(self, word: Sequence[int]) -> str:
        """Compact notation: U2 for two clockwise quarter turns, U'3 for three anticlockwise."""
        names = self.model.move_names
        parts = []
        for b, e in self.runs(word):
            name = names[b] if e > 0 else names[self.inverse[b]]
            parts.append(name if abs(e) == 1 else f"{name}{abs(e)}")
        return " ".join(parts)

    def concat(self, *words: Sequence[int]) -> Word:
        flat: list[int] = []
        for w in words:
            flat.extend(int(m) for m in w)
        return self.reduce(flat)


def invert_tables(tables: np.ndarray) -> np.ndarray:
    out = np.empty_like(tables)
    rows = np.arange(tables.shape[0])[:, None]
    out[rows, tables] = np.arange(tables.shape[1], dtype=tables.dtype)[None, :]
    return out


# -- meet in the middle ------------------------------------------------------------------

def ball(model: PointModel, actions: ActionSet, max_cost: int, max_states: int = 2_000_000,
         log: Callable[[str], None] | None = None) -> tuple[np.ndarray, list[Word], np.ndarray]:
    """Every element reachable from the identity with total action cost <= ``max_cost``.

    Returns tables (N, n) int16, a cheapest primitive word for each, and its cost. Stops
    early at ``max_states`` (the ball is then complete only up to the last finished cost).
    """
    alg = WordAlgebra(model)
    ident = model.identity().astype(np.int16)
    tables: list[np.ndarray] = [ident]
    words: list[Word] = [()]
    costs: list[int] = [0]
    index: dict[bytes, int] = {ident.tobytes(): 0}
    by_cost: dict[int, list[int]] = {0: [0]}
    act_tables = actions.tables.astype(np.int16)
    c = 0
    while c <= max_cost and len(tables) < max_states:
        for i in by_cost.get(c, []):
            if costs[i] != c:
                continue                               # reached more cheaply after being queued
            for a in range(len(actions)):
                nc = c + int(actions.costs[a])
                if nc > max_cost:
                    continue
                t = tables[i][act_tables[a]]
                key = t.tobytes()
                j = index.get(key)
                if j is None:
                    index[key] = len(tables)
                    tables.append(t)
                    words.append(alg.concat(words[i], actions.words[a]))
                    costs.append(nc)
                    by_cost.setdefault(nc, []).append(len(tables) - 1)
                elif nc < costs[j]:
                    costs[j] = nc
                    words[j] = alg.concat(words[i], actions.words[a])
                    by_cost.setdefault(nc, []).append(j)
        if log is not None and c in by_cost:
            log(f"  ball cost {c}: {len(tables):,} elements")
        c += 1
    return np.array(tables, dtype=np.int16), words, np.array(costs, dtype=np.int64)


def collisions(model: PointModel, tables: np.ndarray, words: list[Word], features: Sequence[Feature],
               max_pairs: int = 40_000_000) -> dict[bytes, Word]:
    """All products u * v^-1 of ball elements that show the same ``features``.

    Returns {table bytes (int16): shortest word found}. The identity is excluded. If the ball
    has more colliding pairs than ``max_pairs``, the costliest elements are dropped first.
    """
    alg = WordAlgebra(model)
    n_ball = len(tables)
    lengths = np.array([len(w) for w in words])
    feat = np.concatenate([f.compute(tables.astype(np.int32)) for f in features], axis=1)
    _, cls = np.unique(feat, axis=0, return_inverse=True)
    cls = cls.reshape(-1)

    keep = np.ones(n_ball, dtype=bool)
    while True:
        counts = np.bincount(cls[keep])
        pairs = int((counts.astype(np.int64) * (counts - 1)).sum())
        if pairs <= max_pairs:
            break
        keep &= lengths < lengths[keep].max()          # drop the outermost shell
    idx_all = np.flatnonzero(keep)
    order = idx_all[np.argsort(cls[idx_all], kind="stable")]
    sorted_cls = cls[order]
    bounds = np.flatnonzero(np.r_[True, sorted_cls[1:] != sorted_cls[:-1], True])

    inv_tables = invert_tables(tables)
    ident = model.identity().astype(np.int16).tobytes()
    best: dict[bytes, Word] = {}
    for g in range(len(bounds) - 1):
        members = order[bounds[g]:bounds[g + 1]]
        if len(members) < 2:
            continue
        inv_block = inv_tables[members]
        for ai, a in enumerate(members):
            prods = tables[a][inv_block]                # u, then v^-1, for every v in the class
            for bj, b in enumerate(members):
                if bj == ai:
                    continue
                key = prods[bj].tobytes()
                bound = lengths[a] + lengths[b]
                old = best.get(key)
                if old is not None and len(old) <= bound and len(old) <= abs(int(lengths[a]) - int(lengths[b])):
                    continue
                w = alg.concat(words[a], alg.invert(words[b]))
                if old is None or len(w) < len(old):
                    best[key] = w
    best.pop(ident, None)
    return best


# -- Schreier generators -------------------------------------------------------------------

def schreier_elements(model: PointModel, q: Quotient, table: CosetTable, max_cost: int,
                      max_count: int = 20_000, n_random: int = 0, seed: int = 0,
                      chunk: int = 1 << 18) -> list[Word]:
    """Words t_c * a * t_(c.a)^-1 from an exact table.

    Returns the cheapest ones (transversal cost + action cost + transversal cost <= max_cost,
    at most ``max_count``), followed by ``n_random`` drawn uniformly from all (coset, action)
    pairs. Each word fixes the rung's goal features. Trivial ones (tree edges) are skipped.
    """
    alg = WordAlgebra(model)
    acts = q.actions
    inv = acts.inverse
    n_states = len(table)
    dist = table.dist.astype(np.int64)

    # Candidate (coset, action) pairs, cheapest first. The threshold tightens as pairs
    # accumulate, so memory stays bounded however large the table is.
    tot_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    act_parts: list[np.ndarray] = []
    held = 0
    thr = int(max_cost)

    def compact() -> None:
        nonlocal tot_parts, row_parts, act_parts, held, thr
        tot = np.concatenate(tot_parts)
        row = np.concatenate(row_parts)
        act = np.concatenate(act_parts)
        if len(tot) > max_count:
            thr = int(np.partition(tot, max_count - 1)[max_count - 1])
            keep = np.flatnonzero(tot <= thr)
            if len(keep) > 2 * max_count:                     # many ties at the threshold
                keep = keep[np.argsort(tot[keep], kind="stable")[:max_count]]
            tot, row, act = tot[keep], row[keep], act[keep]
        tot_parts, row_parts, act_parts, held = [tot], [row], [act], len(tot)

    eligible = np.flatnonzero(dist <= max_cost)
    eligible = eligible[np.argsort(dist[eligible], kind="stable")]      # cheap cosets first: the threshold tightens at once
    for s in range(0, len(eligible), chunk):
        rows = eligible[s:s + chunk]
        # dist(c.a) >= dist(c) - cost(a), because the inverse action leads back at the same
        # cost. So the generator costs at least 2 * dist(c), and cosets are sorted by dist.
        if 2 * int(dist[rows[0]]) > thr:
            break
        states = q.unpack(table.keys[rows])
        for a in range(q.n_actions):
            ca = int(q.costs[a])
            room = (2 * dist[rows] <= thr) & (dist[rows] + ca <= thr)
            if not room.any():
                continue
            sub = rows[room]
            nxt = q.pack(q.step(states[room], a))
            pos, found = table.lookup(nxt)
            total = dist[sub] + ca + dist[pos]
            # tree edges give the identity: c -> c.a is c's own first step, or c.a's first step back
            tree = (table.action[sub] == a) | ((table.action[pos] == inv[a]) & (dist[pos] == dist[sub] + ca))
            ok = found & ~tree & (total <= thr)
            if ok.any():
                tot_parts.append(total[ok])
                row_parts.append(sub[ok])
                act_parts.append(np.full(int(ok.sum()), a, dtype=np.int64))
                held += int(ok.sum())
                if held > 4 * max_count:
                    compact()
    cand: list[tuple[int, int, int]] = []
    if tot_parts:
        compact()
        order = np.argsort(tot_parts[0], kind="stable")[:max_count]
        cand = list(zip(tot_parts[0][order].tolist(), row_parts[0][order].tolist(), act_parts[0][order].tolist()))
    if n_random:
        rng = np.random.default_rng(seed)
        for r, a in zip(rng.integers(0, n_states, n_random).tolist(), rng.integers(0, q.n_actions, n_random).tolist()):
            cand.append((-1, r, a))

    if not cand:
        return []
    rows = np.array([c[1] for c in cand], dtype=np.int64)
    acts_taken = np.array([c[2] for c in cand], dtype=np.int64)
    states = q.unpack(table.keys[rows])
    nxt_states = q.step_each(states, acts_taken)
    rows2 = table.lookup(q.pack(nxt_states))[0]

    down = table_paths(q, table, rows)                 # c -> goal
    back = table_paths(q, table, rows2)                # c.a -> goal

    def as_primitives(actions_seq: Sequence[int]) -> list[int]:
        flat: list[int] = []
        for a in actions_seq:
            flat.extend(acts.words[a])
        return flat

    out: list[Word] = []
    seen: set[Word] = set()
    for k in range(len(cand)):
        # goal -> c is the inverse of ``down``; then the action; then c.a -> goal
        w = alg.concat(alg.invert(as_primitives(down[k])), acts.words[int(acts_taken[k])], as_primitives(back[k]))
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def table_paths(q: Quotient, table: CosetTable, rows: np.ndarray) -> list[list[int]]:
    """For each table row, the actions the table prescribes from that coset to the goal."""
    rows = np.asarray(rows, dtype=np.int64)
    paths: list[list[int]] = [[] for _ in range(len(rows))]
    live = np.arange(len(rows))
    cur = rows.copy()
    while len(live):
        a = table.action[cur].astype(np.int64)
        going = a >= 0
        live, cur, a = live[going], cur[going], a[going]
        if not len(live):
            break
        for i, ai in zip(live.tolist(), a.tolist()):
            paths[i].append(ai)
        states = q.step_each(q.unpack(table.keys[cur]), a)
        cur = table.lookup(q.pack(states))[0]
    return paths
