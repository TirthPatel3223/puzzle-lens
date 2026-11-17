"""Exact cost-to-go tables for rungs whose coset space is small enough to enumerate.

Dijkstra from the goal over the quotient puzzle, with integer action costs. The result is
the exact value function of the rung: the quantity a value network approximates on the
rungs that are too large for a table. It also yields, for every coset, the action that
starts a cheapest path to the goal, so solving the rung is a table walk with no search.

States are stored as sorted 63-bit packed keys. Lookup is a binary search.

The search runs backwards from the goal: an edge "state --a--> next" lowers
dist[state] to dist[next] + cost[a]. Predecessors of ``next`` under action ``a`` are
obtained with the inverse action, so the action set must be closed under inverses.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from core.quotient import Quotient


@dataclass
class CosetTable:
    keys: np.ndarray        # (N,) sorted uint64 packed states
    dist: np.ndarray        # (N,) exact cost to the goal
    action: np.ndarray      # (N,) action that starts a cheapest path (-1 at the goal)
    complete: bool          # False if the build stopped at its state budget

    def __len__(self) -> int:
        return len(self.keys)

    def lookup(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Row index of each key and whether it was found."""
        keys = np.asarray(keys, dtype=np.uint64)
        pos = np.searchsorted(self.keys, keys)
        pos = np.minimum(pos, len(self.keys) - 1)
        return pos, self.keys[pos] == keys

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, keys=self.keys, dist=self.dist, action=self.action, complete=np.array(self.complete))

    @staticmethod
    def load(path: str | Path) -> "CosetTable":
        z = np.load(Path(path))
        return CosetTable(keys=z["keys"], dist=z["dist"], action=z["action"], complete=bool(z["complete"]))


def build_table(q: Quotient, max_states: int = 50_000_000, chunk: int = 1 << 18,
                log: Callable[[str], None] | None = None) -> CosetTable:
    """Exact table of the quotient puzzle, or an incomplete one if it exceeds ``max_states``.

    Dial's algorithm: buckets of states by tentative distance, processed in increasing
    order. A state is settled the first time its bucket is processed.
    """
    assert q.packable, "rung state does not fit an integer key"
    assert q.actions.is_symmetric(), "exact tables need an inverse-closed action set"
    inv = q.actions.inverse
    costs = q.costs
    max_cost = int(costs.max())

    goal_key = q.pack(q.goal[None, :])
    settled_keys = [goal_key]                       # list of sorted arrays, merged lazily
    settled_dist = [np.zeros(1, dtype=np.int32)]
    settled_act = [np.full(1, -1, dtype=np.int16)]
    all_keys = goal_key.copy()                      # sorted union of settled keys
    buckets: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    frontier_keys = goal_key
    d = 0
    total = 1
    complete = True

    def push(keys: np.ndarray, acts: np.ndarray, dist: int) -> None:
        buckets.setdefault(dist, []).append((keys, acts))

    while True:
        # expand the states settled at distance d
        for s in range(0, len(frontier_keys), chunk):
            states = q.unpack(frontier_keys[s:s + chunk])
            for a in range(q.n_actions):
                # predecessor p with p --a--> state, found by applying the inverse action
                pred = q.pack(q.step(states, int(inv[a])))
                push(pred, np.full(len(pred), a, dtype=np.int16), d + int(costs[a]))
        # next non-empty bucket
        pending = [k for k in buckets if k > d]
        if not pending:
            break
        d = min(pending)
        parts = buckets.pop(d)
        keys = np.concatenate([p[0] for p in parts])
        acts = np.concatenate([p[1] for p in parts])
        order = np.argsort(keys, kind="stable")
        keys, acts = keys[order], acts[order]
        first = np.ones(len(keys), dtype=bool)
        first[1:] = keys[1:] != keys[:-1]
        keys, acts = keys[first], acts[first]
        pos = np.searchsorted(all_keys, keys)
        pos = np.minimum(pos, len(all_keys) - 1)
        new = all_keys[pos] != keys
        keys, acts = keys[new], acts[new]
        if len(keys) == 0:
            frontier_keys = keys
            continue
        settled_keys.append(keys)
        settled_dist.append(np.full(len(keys), d, dtype=np.int32))
        settled_act.append(acts)
        all_keys = np.union1d(all_keys, keys)
        frontier_keys = keys
        total += len(keys)
        if log is not None:
            log(f"  cost {d:4d}: {len(keys):>12,} new cosets, {total:>13,} total")
        if total > max_states:
            complete = False
            break
        _ = max_cost

    keys = np.concatenate(settled_keys)
    dist = np.concatenate(settled_dist)
    act = np.concatenate(settled_act)
    order = np.argsort(keys)
    return CosetTable(keys=keys[order], dist=dist[order], action=act[order], complete=complete)


def build_table_torch(q: Quotient, device: str = "cuda", max_states: int = 60_000_000,
                      chunk: int = 1 << 20, compact_at: int = 32_000_000,
                      log: Callable[[str], None] | None = None) -> CosetTable:
    """The same table as ``build_table``, computed on a torch device.

    Distances are identical to the NumPy version. The stored first action can differ where
    several actions start a cheapest path.

    The table is filled one cost level at a time. The cosets at cost d are the unsettled
    predecessors, by an action of cost c, of the cosets at cost d - c. Everything cheaper
    than d is settled by then, so every candidate is final: only the level being built is
    held in memory, whatever the spread of the action costs. (Pushing candidates forward
    into one bucket per future cost, as ``build_table`` does, runs a 12 GB card out of
    memory on a 60-million-coset rung with costs from 8 to 76.)
    """
    import torch

    from core.quotient import TorchQuotient

    assert q.packable, "rung state does not fit an integer key"
    assert q.actions.is_symmetric(), "exact tables need an inverse-closed action set"
    tq = TorchQuotient(q, device)
    dev = tq.device
    inv = [int(x) for x in q.actions.inverse]
    by_cost: dict[int, list[int]] = {}
    for a, c in enumerate(q.costs.tolist()):
        by_cost.setdefault(int(c), []).append(a)
    assert min(by_cost) > 0, "action costs must be positive"
    max_cost = max(by_cost)

    def dedupe(keys, acts):
        """Sort by key and keep one (key, action) per distinct key."""
        keys, order = torch.sort(keys, stable=True)
        acts = acts[order]
        first = torch.ones(keys.shape[0], dtype=torch.bool, device=dev)
        first[1:] = keys[1:] != keys[:-1]
        return keys[first], acts[first]

    def unsettled(keys):
        pos = torch.searchsorted(all_keys, keys).clamp_(max=all_keys.shape[0] - 1)
        return all_keys[pos] != keys

    goal_key = tq.pack(tq.goal[None, :])
    all_keys = goal_key.clone()                                    # sorted keys settled so far
    out_keys = [goal_key.cpu().numpy()]
    out_dist = [np.zeros(1, dtype=np.int32)]
    out_act = [np.full(1, -1, dtype=np.int16)]
    levels = {0: goal_key}              # cost -> keys settled at that cost, kept while a later level can use them
    d, last, total, complete = 0, 0, 1, True

    with torch.no_grad():
        while complete and d < last + max_cost:
            d += 1
            parts: list[tuple] = []
            held = base = 0
            for c, acts_c in by_cost.items():
                source = levels.get(d - c)
                if source is None:
                    continue
                for s in range(0, source.shape[0], chunk):
                    states = tq.unpack(source[s:s + chunk])
                    for a in acts_c:
                        pred = tq.pack(tq.step_one(states, inv[a]))   # p with p --a--> state
                        pred = pred[unsettled(pred)]
                        if pred.numel() == 0:
                            continue
                        parts.append((pred, torch.full((pred.shape[0],), a, dtype=torch.int16, device=dev)))
                        held += pred.shape[0]
                        if held > base + compact_at:
                            keys, acts = dedupe(torch.cat([x[0] for x in parts]), torch.cat([x[1] for x in parts]))
                            parts = [(keys, acts)]
                            held = base = keys.shape[0]
            levels.pop(d - max_cost, None)                         # no later level reaches back this far
            if not parts:
                continue
            keys, acts = dedupe(torch.cat([x[0] for x in parts]), torch.cat([x[1] for x in parts]))
            del parts
            levels[d] = keys
            last = d
            out_keys.append(keys.cpu().numpy())
            out_dist.append(np.full(keys.shape[0], d, dtype=np.int32))
            out_act.append(acts.cpu().numpy())
            all_keys = torch.sort(torch.cat([all_keys, keys]))[0]
            total += keys.shape[0]
            if log is not None:
                log(f"  cost {d:4d}: {keys.shape[0]:>12,} new cosets, {total:>13,} total")
            if total > max_states:
                complete = False

    keys = np.concatenate(out_keys).astype(np.uint64)
    order = np.argsort(keys)
    return CosetTable(keys=keys[order], dist=np.concatenate(out_dist)[order],
                      action=np.concatenate(out_act)[order], complete=complete)


def build_table_auto(q: Quotient, max_states: int = 60_000_000, log: Callable[[str], None] | None = None) -> CosetTable:
    """GPU build when CUDA is available, NumPy otherwise."""
    try:
        import torch

        if torch.cuda.is_available():
            return build_table_torch(q, "cuda", max_states=max_states, log=log)
    except ImportError:
        pass
    return build_table(q, max_states=max_states, log=log)


def table_solve(q: Quotient, table: CosetTable, state: np.ndarray, max_steps: int = 10_000) -> list[int] | None:
    """Cheapest action sequence taking ``state`` to the goal, read off the table.
    Returns None if the state is not in the table."""
    cur = np.asarray(state, dtype=np.uint8)[None, :]
    out: list[int] = []
    for _ in range(max_steps):
        pos, found = table.lookup(q.pack(cur))
        if not found[0]:
            return None
        a = int(table.action[pos[0]])
        if a < 0:
            return out
        out.append(a)
        cur = q.step(cur, a)
    raise RuntimeError("table walk did not terminate")
