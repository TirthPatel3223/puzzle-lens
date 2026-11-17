"""GPU version of GatherPuzzle: the same gather-plus-add semantics, batched on CUDA.

Results must match ``GatherPuzzle.apply`` exactly; tests/test_torch_engine.py checks this.
"""
from __future__ import annotations

import torch

from core.gather_engine import GatherPuzzle


class TorchGatherPuzzle:
    def __init__(self, puzzle: GatherPuzzle, device: str | torch.device = "cuda") -> None:
        self.puzzle = puzzle
        self.device = torch.device(device)
        self.num_moves = puzzle.num_moves
        self.dim = puzzle.dim
        if any(int(q) & (int(q) - 1) for q in puzzle.modulus):
            raise ValueError("uint8 fast path needs power-of-two moduli")
        self.idx = torch.as_tensor(puzzle.idx, dtype=torch.long, device=self.device)
        # uint8 addition wraps mod 256; for a power-of-two modulus q, `& (q - 1)` gives mod q.
        self.add = torch.as_tensor(puzzle.add % 256, dtype=torch.uint8, device=self.device)
        self.mask = torch.as_tensor(puzzle.modulus - 1, dtype=torch.uint8, device=self.device)
        self.goal = torch.as_tensor(puzzle.goal, dtype=torch.uint8, device=self.device)
        self.inverse = torch.as_tensor(puzzle.inverse, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def apply(self, states: torch.Tensor, move: int) -> torch.Tensor:
        """states: (N, D) uint8 on self.device -> (N, D) uint8."""
        out = states[:, self.idx[move]]  # advanced indexing returns a fresh tensor
        out += self.add[move]            # wraps mod 256
        out &= self.mask                 # reduces mod each power-of-two modulus
        return out

    @torch.no_grad()
    def apply_all(self, states: torch.Tensor) -> torch.Tensor:
        """(N, D) -> (M, N, D): every state under every move."""
        return torch.stack([self.apply(states, m) for m in range(self.num_moves)])

    def goal_batch(self, n: int) -> torch.Tensor:
        return self.goal.expand(n, self.dim).clone()
