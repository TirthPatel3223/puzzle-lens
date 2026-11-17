"""Quotient puzzles: one rung of a ladder as a small puzzle of its own.

A rung has a set of actions (words in the primitive moves, each compiled to one table) and
a list of features. Its state is the concatenated feature vector. Because every feature is
closed under every permutation, an action acts on feature vectors by

    new[s] = lut[a, s, old[src[a, s]]]

without reference to the rest of the puzzle. The reachable feature vectors are the cosets
the rung has to tell apart, and this graph is the Schreier coset graph of the design note.

`ActionSet`  words, their costs and their point tables.
`Quotient`   NumPy dynamics, goal test and integer packing of states.
`TorchQuotient`  the same dynamics on a torch device, for search and training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core.perm_model import PointModel, Word
from core.structure import Feature


@dataclass
class ActionSet:
    words: list[Word]
    costs: np.ndarray            # (A,) cost of each action, in primitive moves
    tables: np.ndarray           # (A, n) gather tables on the points
    inverse: np.ndarray          # (A,) index of the inverse action, or -1 if it is not in the set

    def __len__(self) -> int:
        return len(self.words)

    @staticmethod
    def from_words(model: PointModel, words: Sequence[Sequence[int]], costs: Sequence[int] | None = None) -> "ActionSet":
        words = [tuple(int(m) for m in w) for w in words]
        tables = model.compile_many(words)
        if costs is None:
            costs = [len(w) for w in words]
        index = {tables[a].tobytes(): a for a in range(len(words))}
        inverse = np.full(len(words), -1, dtype=np.int64)
        for a in range(len(words)):
            inv = np.empty(model.n, dtype=np.int32)
            inv[tables[a]] = np.arange(model.n)
            inverse[a] = index.get(inv.tobytes(), -1)
        return ActionSet(words=words, costs=np.asarray(costs, dtype=np.int64), tables=tables, inverse=inverse)

    @staticmethod
    def primitives(model: PointModel) -> "ActionSet":
        return ActionSet.from_words(model, [(m,) for m in range(model.num_moves)])

    def is_symmetric(self) -> bool:
        return bool((self.inverse >= 0).all())


def symmetrise(model: PointModel, words: Sequence[Sequence[int]]) -> list[Word]:
    """Add the inverse of every word, and drop words that act identically (keeping the shortest)."""
    best: dict[bytes, Word] = {}
    for w in words:
        w = tuple(int(m) for m in w)
        for cand in (w, model.invert_word(w)):
            key = model.compile(cand).tobytes()
            if key not in best or len(cand) < len(best[key]):
                best[key] = cand
    ident = model.identity().tobytes()
    best.pop(ident, None)
    return sorted(best.values(), key=lambda w: (len(w), w))


class Quotient:
    def __init__(self, features: Sequence[Feature], actions: ActionSet, name: str = "") -> None:
        self.name = name or "+".join(f.name for f in features)
        self.features = list(features)
        self.actions = actions
        self.n_actions = len(actions)
        self.costs = actions.costs.copy()
        self.radix = np.concatenate([f.radix for f in self.features]).astype(np.int64)
        self.n_slots = len(self.radix)
        self.n_classes = int(self.radix.max())
        self.goal = np.concatenate([f.goal for f in self.features]).astype(np.uint8)

        src = np.empty((self.n_actions, self.n_slots), dtype=np.int64)
        lut = np.zeros((self.n_actions, self.n_slots, self.n_classes), dtype=np.int16)
        base = 0
        for f in self.features:
            s, c = f.n_slots, f.n_classes
            for a in range(self.n_actions):
                f_src, f_lut = f.transition(actions.tables[a])
                src[a, base:base + s] = f_src + base
                lut[a, base:base + s, :c] = f_lut
            base += s
        self.src = src
        self.lut = lut                        # -1 marks a class the slot can never hold
        self._lut_u8 = np.maximum(lut, 0).astype(np.uint8)

        total = 1
        for r in self.radix.tolist():
            total *= int(r)
        self.state_count_bound = total       # number of feature vectors, reachable or not
        self.packable = total < (1 << 63)
        if self.packable:
            mult = np.ones(self.n_slots, dtype=np.uint64)
            for s in range(1, self.n_slots):
                mult[s] = mult[s - 1] * np.uint64(self.radix[s - 1])
            self._mult = mult

    # -- dynamics ---------------------------------------------------------------
    def step(self, states: np.ndarray, action: int) -> np.ndarray:
        """Apply one action to states (N, S)."""
        return self._lut_u8[action, np.arange(self.n_slots), states[:, self.src[action]]]

    def step_each(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Apply actions[i] to states[i]. (N, S), (N,) -> (N, S)."""
        actions = np.asarray(actions, dtype=np.int64)
        rows = np.arange(states.shape[0])[:, None]
        return self._lut_u8[actions[:, None], np.arange(self.n_slots)[None, :], states[rows, self.src[actions]]]

    def step_all(self, states: np.ndarray) -> np.ndarray:
        """Every state under every action: (N, S) -> (N, A, S)."""
        a = np.arange(self.n_actions)[None, :, None]
        s = np.arange(self.n_slots)[None, None, :]
        return self._lut_u8[a, s, states[:, self.src]]

    def apply_word(self, state: np.ndarray, actions: Sequence[int]) -> np.ndarray:
        cur = np.asarray(state, dtype=np.uint8)[None, :]
        for a in actions:
            cur = self.step(cur, int(a))
        return cur[0]

    def is_goal(self, states: np.ndarray) -> np.ndarray:
        return (states == self.goal).all(axis=-1)

    # -- from the full puzzle -----------------------------------------------------
    def project(self, lab: np.ndarray) -> np.ndarray:
        """Feature vector(s) of labelled state(s): (..., n) -> (..., S)."""
        return np.concatenate([f.compute(lab) for f in self.features], axis=-1).astype(np.uint8)

    # -- integer keys -------------------------------------------------------------
    def pack(self, states: np.ndarray) -> np.ndarray:
        assert self.packable, "feature vectors of this rung do not fit in 63 bits"
        return (states.astype(np.uint64) * self._mult).sum(axis=-1, dtype=np.uint64)

    def unpack(self, keys: np.ndarray) -> np.ndarray:
        keys = np.asarray(keys, dtype=np.uint64)
        out = np.empty(keys.shape + (self.n_slots,), dtype=np.uint8)
        rem = keys.copy()
        for s in range(self.n_slots):
            r = np.uint64(self.radix[s])
            out[..., s] = (rem % r).astype(np.uint8)
            rem //= r
        return out


class TorchQuotient:
    """Quotient dynamics on a torch device. States are (N, S) uint8 tensors."""

    def __init__(self, quotient: Quotient, device: str = "cuda") -> None:
        import torch

        self.q = quotient
        self.device = torch.device(device)
        self.n_actions = quotient.n_actions
        self.n_slots = quotient.n_slots
        self.n_classes = quotient.n_classes
        a, s, c = self.n_actions, self.n_slots, self.n_classes
        self.src = torch.as_tensor(quotient.src, dtype=torch.long, device=self.device)                 # (A, S)
        self.lut = torch.as_tensor(quotient._lut_u8.reshape(-1), dtype=torch.uint8, device=self.device)  # (A*S*C,)
        self.base = (torch.arange(a, device=self.device)[:, None] * (s * c)
                     + torch.arange(s, device=self.device)[None, :] * c)                                # (A, S)
        self.goal = torch.as_tensor(quotient.goal, dtype=torch.uint8, device=self.device)
        self.costs = torch.as_tensor(quotient.costs, dtype=torch.float32, device=self.device)
        self.inverse = torch.as_tensor(quotient.actions.inverse, dtype=torch.long, device=self.device)
        gen = torch.Generator().manual_seed(20260713)
        self.hash_vec = torch.randint(1, 2**62, (s,), dtype=torch.int64, generator=gen).to(self.device)

    def step_all(self, states):
        """(N, S) -> (N, A, S): every state under every action."""
        idx = states[:, self.src].long() + self.base[None]
        return self.lut[idx]

    def step(self, states, actions):
        """Apply actions[i] to states[i]. (N, S), (N,) -> (N, S)."""
        idx = states.gather(1, self.src[actions]).long() + self.base[actions]
        return self.lut[idx]

    def step_one(self, states, action: int):
        """Apply one action to every state. (N, S) -> (N, S)."""
        idx = states[:, self.src[action]].long() + self.base[action]
        return self.lut[idx]

    def pack(self, states):
        """63-bit integer key of each state (requires ``q.packable``)."""
        return (states.long() * self._mult()).sum(dim=-1)

    def unpack(self, keys):
        import torch

        out = torch.empty((keys.shape[0], self.n_slots), dtype=torch.uint8, device=keys.device)
        rem = keys.clone()
        radix = self.q.radix.tolist()
        for s in range(self.n_slots):
            out[:, s] = (rem % radix[s]).to(torch.uint8)
            rem = rem // radix[s]
        return out

    def _mult(self):
        import torch

        if not hasattr(self, "_mult_t"):
            assert self.q.packable, "feature vectors of this rung do not fit in 63 bits"
            self._mult_t = torch.as_tensor(self.q._mult.astype(np.int64), device=self.device)
        return self._mult_t

    def is_goal(self, states):
        return (states == self.goal).all(dim=-1)

    def goal_batch(self, n: int):
        return self.goal.expand(n, self.n_slots).clone()

    def hash(self, states):
        """64-bit hash of each state. Collisions are possible but rare; callers that need
        certainty replay their result on the reference simulator."""
        return (states.long() * self.hash_vec).sum(dim=-1)
