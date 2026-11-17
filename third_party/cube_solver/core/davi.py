"""Coset value iteration: learn a rung's cost-to-go by deep approximate value iteration.

This is the training loop of DeepCubeA / DeepXube (Agostinelli et al.; DeepXube is MIT
licensed, https://github.com/forestagostinelli/deepxube) applied to one rung's Schreier
coset graph instead of the whole puzzle. The structure follows DeepXube's
``Train.update_step``:

    repeat
        generate states by random walks from the goal            (updater: data generation)
        targets  y = 0                          if state is the goal
                 y = min_a cost(a) + V_target(state . a)   otherwise      (Bellman backup)
        train the network on (state, y) for some iterations             (trainer)
        if the loss is below a threshold, or it has not been for a while,
            copy the network into the target network                    (target update)

What is different here:

* States are feature vectors of the rung, so the network only ever sees the pieces this
  rung is responsible for. Walks start from the goal coset and use the rung's own actions,
  which keep all earlier rungs solved, so every training state is a state the solver can
  actually meet.
* Actions are macros with different costs; the backup uses each macro's length in
  primitive moves, so the value is a length in primitive moves.
* Everything runs on the GPU as tensors. DeepXube's worker processes, shared memory and
  per-state Python objects are not needed, because states are small uint8 vectors.
* ``ub_walk`` caps each target by the cost of the walk that produced the state, which is a
  true upper bound on its cost-to-go (DeepXube's ``ub_heur_solns`` idea). It makes early
  targets informative before the bootstrap has propagated.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from core.nnet import ResnetFC, evaluate
from core.quotient import Quotient, TorchQuotient
from core.search import beam_search, greedy_rollout


@dataclass
class DaviConfig:
    res_dim: int = 1000
    num_blocks: int = 4
    batch_norm: bool = True
    batch_size: int = 8192
    states_per_update: int = 400_000     # states generated per update
    itrs_per_update: int = 100           # training iterations per update
    max_updates: int = 100
    lr: float = 1e-3
    lr_decay: float = 0.99995            # per training iteration, as in DeepXube
    walk_max: int = 30                   # longest random walk, in actions; must be long enough to mix
    loss_thresh: float = 0.06            # target update when the (scaled) loss is below this
    force_sync: int = 2                  # ... or after this many updates without one
    ub_walk: bool = True
    amp: bool = True
    target_chunk: int = 8192             # states per chunk when computing targets
    test_states: int = 500
    seed: int = 0


class RungValue:
    """A trained (or training) value network for one rung, with search on top."""

    def __init__(self, quotient: Quotient, cfg: DaviConfig | None = None, device: str = "cuda") -> None:
        self.q = quotient
        self.cfg = cfg or DaviConfig()
        self.tq = TorchQuotient(quotient, device)
        self.device = self.tq.device
        torch.manual_seed(self.cfg.seed)
        self.scale = float(quotient.costs.mean())
        self.net = ResnetFC(quotient.n_slots, quotient.n_classes, res_dim=self.cfg.res_dim,
                            num_blocks=self.cfg.num_blocks, batch_norm=self.cfg.batch_norm,
                            scale=self.scale).to(self.device)
        self.target = copy.deepcopy(self.net).eval()
        self.target_is_zero = True       # as in DeepXube: the target returns 0 until its first update
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        self.itr = 0
        self.update = 0
        self.target_updates = 0
        self.since_sync = 0
        self.history: list[dict] = []
        self.beam_widths = (256, 4096, 32768)

    # -- data generation -----------------------------------------------------------
    @torch.no_grad()
    def scramble(self, n: int, walk_max: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """States reached by random walks from the goal, and the cost of each walk.
        Walk lengths are uniform on 0..walk_max. A walk never undoes its previous action."""
        tq = self.tq
        walk_max = self.cfg.walk_max if walk_max is None else walk_max
        states = tq.goal_batch(n)
        length = torch.randint(0, walk_max + 1, (n,), device=self.device)
        cost = torch.zeros(n, device=self.device)
        last = torch.full((n,), -1, dtype=torch.long, device=self.device)
        for t in range(walk_max):
            a = torch.randint(0, tq.n_actions, (n,), device=self.device)
            undo = (last >= 0) & (a == tq.inverse[last.clamp(min=0)])
            a = torch.where(undo, (a + 1) % tq.n_actions, a)
            active = t < length
            nxt = tq.step(states, a)
            states = torch.where(active[:, None], nxt, states)
            cost += torch.where(active, tq.costs[a], torch.zeros_like(cost))
            last = torch.where(active, a, last)
        return states, cost

    # -- targets ---------------------------------------------------------------------
    @torch.no_grad()
    def bellman_targets(self, states: torch.Tensor, net: torch.nn.Module | None = None) -> torch.Tensor:
        """min over actions of cost + value(child), 0 at the goal. Uses the target network."""
        net = self.target if net is None else net
        tq = self.tq
        out = torch.empty(states.shape[0], device=self.device)
        for s in range(0, states.shape[0], self.cfg.target_chunk):
            part = states[s:s + self.cfg.target_chunk]
            kids = tq.step_all(part)                                          # (n, A, S)
            n = part.shape[0]
            flat = kids.reshape(n * tq.n_actions, -1)
            if net is self.target and self.target_is_zero:
                v = torch.zeros((n, tq.n_actions), device=self.device)
            else:
                v = evaluate(net, flat, amp=self.cfg.amp).reshape(n, tq.n_actions).clamp_(min=0)
                v = torch.where(tq.is_goal(flat).reshape(n, tq.n_actions), torch.zeros_like(v), v)
            y = (tq.costs[None, :] + v).min(dim=1).values
            out[s:s + n] = torch.where(tq.is_goal(part), torch.zeros_like(y), y)
        return out

    # -- value function ----------------------------------------------------------------
    def value(self, states: torch.Tensor) -> torch.Tensor:
        """Estimated cost-to-go of states (N, S). Goal states are exactly 0."""
        v = evaluate(self.net, states, amp=self.cfg.amp).reshape(-1).clamp_(min=0)
        return torch.where(self.tq.is_goal(states), torch.zeros_like(v), v)

    # -- training ------------------------------------------------------------------------
    def train(self, max_updates: int | None = None, log: Callable[[str], None] | None = None,
              checkpoint: str | Path | None = None, time_limit: float | None = None) -> None:
        cfg = self.cfg
        log = log or (lambda msg: print(msg, flush=True))
        max_updates = cfg.max_updates if max_updates is None else max_updates
        use_amp = cfg.amp and self.device.type == "cuda"
        t_start = time.time()
        while self.update < max_updates:
            t0 = time.time()
            states, walk_cost = self.scramble(cfg.states_per_update)
            y = self.bellman_targets(states)
            if cfg.ub_walk:
                y = torch.minimum(y, walk_cost)
            t_data = time.time() - t0

            self.net.train()
            loss_sum, n_states = 0.0, states.shape[0]
            for _ in range(cfg.itrs_per_update):
                idx = torch.randint(0, n_states, (cfg.batch_size,), device=self.device)
                for group in self.opt.param_groups:
                    group["lr"] = cfg.lr * (cfg.lr_decay ** self.itr)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    pred = self.net(states[idx]).reshape(-1)
                loss = torch.nn.functional.mse_loss(pred.float() / self.scale, y[idx] / self.scale)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
                loss_sum += float(loss.detach())
                self.itr += 1
            loss_mean = loss_sum / cfg.itrs_per_update
            last_loss = float(loss.detach())

            self.update += 1
            self.since_sync += 1
            synced = False
            if last_loss < cfg.loss_thresh or self.since_sync >= cfg.force_sync:
                self.target.load_state_dict(self.net.state_dict())
                self.target_is_zero = False
                self.target_updates += 1
                self.since_sync = 0
                synced = True
            row = dict(update=self.update, itr=self.itr, loss=loss_mean, target_mean=float(y.mean()),
                       target_max=float(y.max()), synced=synced, target_updates=self.target_updates,
                       seconds=time.time() - t0)
            if self.update % 5 == 0 or self.update == max_updates:
                test, _ = self.scramble(cfg.test_states)
                solved, cost = greedy_rollout(self.tq, self.value, test, max_steps=4 * cfg.walk_max)
                row["greedy_solved"] = float(solved.float().mean())
                row["greedy_cost"] = float(cost[solved].mean()) if bool(solved.any()) else float("nan")
            self.history.append(row)
            extra = f"  greedy {100 * row['greedy_solved']:5.1f}% cost {row['greedy_cost']:.1f}" if "greedy_solved" in row else ""
            log(f"    update {self.update:4d}  loss {loss_mean:.4f}  target mean {row['target_mean']:6.2f} max {row['target_max']:6.1f}"
                f"  sync {self.target_updates:3d}{'*' if synced else ' '}  data {t_data:4.1f}s  total {row['seconds']:4.1f}s{extra}")
            if checkpoint is not None and (self.update % 5 == 0 or self.update == max_updates):
                self.save(checkpoint)
            if time_limit is not None and time.time() - t_start > time_limit:
                log(f"    stopped at the time limit after {self.update} updates")
                break
        if checkpoint is not None:
            self.save(checkpoint)

    # -- solving -------------------------------------------------------------------------
    def solve(self, vec, width: int | None = None, weight: float = 0.3, max_steps: int | None = None) -> list[int] | None:
        """Actions taking the feature vector ``vec`` to the rung's goal, or None.

        Beam search with the given width; if it finds nothing, the width is multiplied by 8
        (twice at most). Without ``width`` the default schedule ``beam_widths`` is used.
        ``weight`` scales the accumulated cost g in f = weight * g + h (default: 0.3).
        """
        start = torch.as_tensor(vec, dtype=torch.uint8, device=self.device)
        max_w = self.beam_widths[-1]  # 32,768 upper bound to fit comfortably in GPU memory
        widths = self.beam_widths if width is None else tuple(dict.fromkeys(min(w, max_w) for w in (width, 8 * width, 64 * width)))
        steps = max(200, 6 * self.cfg.walk_max) if max_steps is None else max_steps
        for w in widths:
            found = beam_search(self.tq, self.value, start, width=w, max_steps=steps, weight=weight)
            if found is not None:
                return found[0]
        return None

    # -- persistence ---------------------------------------------------------------------
    @staticmethod
    def from_checkpoint(quotient: Quotient, path: str | Path, device: str = "cuda") -> "RungValue":
        """Rebuild a trained rung value from its checkpoint (network size comes from the file)."""
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        return RungValue(quotient, DaviConfig(**data["cfg"]), device).load(path)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"net": self.net.state_dict(), "target": self.target.state_dict(), "opt": self.opt.state_dict(),
                    "cfg": asdict(self.cfg), "itr": self.itr, "update": self.update,
                    "target_updates": self.target_updates, "since_sync": self.since_sync,
                    "target_is_zero": self.target_is_zero,
                    "history": self.history, "n_actions": self.q.n_actions, "n_slots": self.q.n_slots}, path)

    def load(self, path: str | Path) -> "RungValue":
        data = torch.load(Path(path), map_location=self.device, weights_only=False)
        assert data["n_actions"] == self.q.n_actions and data["n_slots"] == self.q.n_slots, "checkpoint is for another rung"
        self.net.load_state_dict(data["net"])
        self.target.load_state_dict(data["target"])
        self.opt.load_state_dict(data["opt"])
        self.itr, self.update = data["itr"], data["update"]
        self.target_updates, self.since_sync = data["target_updates"], data["since_sync"]
        self.target_is_zero = data.get("target_is_zero", False)
        self.history = data["history"]
        return self
