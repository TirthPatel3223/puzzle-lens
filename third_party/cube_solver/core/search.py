"""Batched beam search on a rung's quotient puzzle, guided by a cost-to-go function.

One step expands every beam state under every action, drops duplicates and states already
expanded, scores the children by  f = weight * g + h  (g = cost so far, h = estimated cost
to go) and keeps the ``width`` best. Action costs differ, so the first goal found need not
be the cheapest: the search keeps going while some beam state could still beat it.

Duplicates are detected with a 64-bit hash. A collision can only drop a state, never
produce a wrong path, because the returned actions are replayed by the caller.
"""
from __future__ import annotations

from typing import Callable

import torch

from core.quotient import TorchQuotient

ValueFn = Callable[[torch.Tensor], torch.Tensor]      # (N, S) uint8 -> (N,) cost-to-go

@torch.no_grad()
def beam_search(tq: TorchQuotient, value_fn: ValueFn, start: torch.Tensor, width: int = 4096,
                max_steps: int = 200, weight: float = 1.0, expand_chunk: int = 1 << 15) -> tuple[list[int], float] | None:
    """Search from ``start`` (S,) to the rung's goal. Returns (actions, cost) or None."""
    dev = tq.device
    start = start.to(dev).reshape(1, -1)
    if bool(tq.is_goal(start)[0]):
        return [], 0.0
    n_act = tq.n_actions
    costs = tq.costs

    beam = start
    g = torch.zeros(1, device=dev)
    visited = tq.hash(beam)                           # sorted hashes of every expanded state
    parents: list[torch.Tensor] = []                  # per step: parent index of each kept child
    actions: list[torch.Tensor] = []
    best_cost = float("inf")
    best_path: tuple[int, int, int] | None = None     # (step, parent index, action)

    for step in range(max_steps):
        b = beam.shape[0]
        child_states, child_g, child_parent, child_action = [], [], [], []
        for s in range(0, b, expand_chunk):
            part = beam[s:s + expand_chunk]
            kids = tq.step_all(part)                                      # (n, A, S)
            kg = g[s:s + expand_chunk, None] + costs[None, :]
            n = part.shape[0]
            kids = kids.reshape(n * n_act, -1)
            kg = kg.reshape(-1)
            par = torch.arange(s, s + n, device=dev).repeat_interleave(n_act)
            act = torch.arange(n_act, device=dev).repeat(n)
            solved = tq.is_goal(kids)
            if solved.any():
                cand = torch.where(solved, kg, torch.full_like(kg, float("inf")))
                i = int(torch.argmin(cand))
                if float(kg[i]) < best_cost:
                    best_cost = float(kg[i])
                    best_path = (step, int(par[i]), int(act[i]))
            keep = ~solved & (kg < best_cost)                             # cannot improve otherwise
            child_states.append(kids[keep]); child_g.append(kg[keep])
            child_parent.append(par[keep]); child_action.append(act[keep])
        kids = torch.cat(child_states); kg = torch.cat(child_g)
        par = torch.cat(child_parent); act = torch.cat(child_action)
        del child_states, child_g, child_parent, child_action  # free GPU memory before hashing
        if kids.shape[0] == 0:
            break

        # one copy of each state (the cheapest), and nothing that was expanded before
        h = tq.hash(kids)
        order = torch.argsort(kg, stable=True)
        order = order[torch.argsort(h[order], stable=True)]
        hs = h[order]
        first = torch.ones(hs.shape[0], dtype=torch.bool, device=dev)
        first[1:] = hs[1:] != hs[:-1]
        order = order[first]
        hs = hs[first]
        pos = torch.searchsorted(visited, hs).clamp_(max=visited.shape[0] - 1)
        fresh = visited[pos] != hs
        order = order[fresh]
        if order.shape[0] == 0:
            break
        kids, kg, par, act = kids[order], kg[order], par[order], act[order]

        f = weight * kg + value_fn(kids).reshape(-1)
        if f.shape[0] > width:
            top = torch.topk(f, width, largest=False).indices
            kids, kg, par, act = kids[top], kg[top], par[top], act[top]
        beam, g = kids, kg
        parents.append(par.cpu()); actions.append(act.cpu())
        visited = torch.sort(torch.cat([visited, tq.hash(beam)]))[0]
        if best_path is not None and float(g.min()) + float(costs.min()) >= best_cost:
            break

    if best_path is None:
        return None
    step, idx, action = best_path
    out = [action]
    for t in range(step - 1, -1, -1):
        out.append(int(actions[t][idx]))
        idx = int(parents[t][idx])
    out.reverse()
    return out, best_cost


@torch.no_grad()
def greedy_rollout(tq: TorchQuotient, value_fn: ValueFn, states: torch.Tensor, max_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Follow argmin_a (cost(a) + h(child)) from every state. Returns (solved, cost) per state.
    A cheap test of a heuristic: with a perfect one, every state is solved at optimal cost."""
    dev = tq.device
    states = states.to(dev).clone()
    n = states.shape[0]
    done = tq.is_goal(states)
    cost = torch.zeros(n, device=dev)
    for _ in range(max_steps):
        if bool(done.all()):
            break
        live = torch.nonzero(~done).reshape(-1)
        kids = tq.step_all(states[live])                                  # (m, A, S)
        m = kids.shape[0]
        flat = kids.reshape(m * tq.n_actions, -1)
        h = value_fn(flat).reshape(m, tq.n_actions)
        h = torch.where(tq.is_goal(flat).reshape(m, tq.n_actions), torch.zeros_like(h), h)
        a = torch.argmin(tq.costs[None, :] + h, dim=1)
        states[live] = kids[torch.arange(m, device=dev), a]
        cost[live] += tq.costs[a]
        done[live] = tq.is_goal(states[live])
    return done, cost
