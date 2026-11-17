"""Which face each photo shows, and which colour layout the cube has: one search for both.

For six readings, every pair (colour layout, way of giving the six photos the six faces) is a candidate:
120 layout classes x 720 face assignments for the Tangram Cube Extreme. Each candidate's score is the
log-likelihood of its best arrangement of pieces (the assignment of ``decode.assign_logp``) minus the
layout's prior. Doing 86,400 assignments would take seconds, so:

1. **Batched scoring.** The six readings are assembled for all 720 face assignments at once. A layout
   only relabels colours, so it is applied while scoring, not by assembling again. The 30 one-colour
   pieces' best fit in each cell depends on the layout only through that piece's colour, so it is
   worked out once for each of the six colours; only the 14 corner and edge pieces are scored for each
   layout.
2. **A bound.** For every candidate, the sum over pieces of each piece's best fit anywhere (and the
   same over cells) bounds the assignment from above. Exact assignments are done in order of that
   bound, and stop once no remaining candidate can enter the top ones.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from scanner.core.colours import ColourLayout

from ..decode import EMPTY, assign_logp, tables
from ..layout import N_COLOURS, TangramLayout
from .geometry import FaceGeometry

PERMS = np.array(list(itertools.permutations(range(6))), dtype=np.int64)   # PERMS[k, f]: the photo shown as face f
_NONE = -1e6


@dataclass
class Candidate:
    #: Log-likelihood of the best arrangement, minus the layout's prior.
    score: float
    loglik: float
    #: Index into the layouts searched.
    layout: int
    #: The face each photo shows (index into the layout's faces).
    faces: tuple[int, ...]


def _runs(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For sorted ``keys``: (the distinct keys, where each run starts)."""
    uniq, start = np.unique(keys, return_index=True)
    assert (np.diff(keys) >= 0).all()
    return uniq, start


def _sorted_runs(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(an order that sorts ``keys``, where each run starts in that order, the distinct keys)."""
    order = np.argsort(keys, kind="stable")
    uniq, start = np.unique(keys[order], return_index=True)
    return order, start, uniq


class FaceSearch:
    def __init__(self, layout: TangramLayout, geom: FaceGeometry) -> None:
        self.layout, self.geom = layout, geom
        t = tables(layout)
        self.t = t
        n = self.n = len(t.movable)
        ap = t.assign_placements
        rc = t.assign_rc
        piece_row = {p: i for i, p in enumerate(t.movable)}
        n_stickers = t.mask.sum(1)
        single_piece = np.array([len(layout.piece_colours[p]) == 1 for p in t.movable])
        self.single_rows = np.nonzero(single_piece)[0]
        self.multi_rows = np.nonzero(~single_piece)[0]

        # One-colour pieces: groups of placements sharing (piece, cell); each placement shows one position.
        sp = [j for j in ap if single_piece[piece_row[int(t.piece[j])]]]
        sp.sort(key=lambda j: int(rc[np.searchsorted(ap, j)]))
        sp = np.array(sp, dtype=np.int64)
        s_rc = rc[np.searchsorted(ap, sp)]
        self.s_groups, s_start = np.unique(s_rc, return_index=True)
        self.s_start = s_start
        self.s_pos = t.pos[sp, 0]
        assert (n_stickers[sp] == 1).all()
        self.s_colour = np.array([layout.piece_colours[t.movable[g // n]][0] for g in self.s_groups])   # model colour per group
        self.s_row = self.s_groups // n
        self.s_col = self.s_groups % n

        # Corner and edge pieces: scored per layout.
        mp = [j for j in ap if not single_piece[piece_row[int(t.piece[j])]]]
        mp.sort(key=lambda j: int(rc[np.searchsorted(ap, j)]))
        mp = np.array(mp, dtype=np.int64)
        m_rc = rc[np.searchsorted(ap, mp)]
        self.m_groups, self.m_start = np.unique(m_rc, return_index=True)
        self.m_pos = np.where(t.mask[mp], t.pos[mp], layout.n_positions)       # unused sticker slots -> a zero row
        self.m_col = t.col[mp]
        self.m_mask = t.mask[mp]
        self.m_row = self.m_groups // n
        self.m_col_of_group = self.m_groups % n
        # For the bound: groups by piece (they are sorted that way already) and by cell.
        self.s_by_row = _runs(self.s_row)
        # model colour of each one-colour piece's row run
        self.s_row_colour = np.array([layout.piece_colours[t.movable[r]][0] for r in self.s_by_row[0]])
        self.m_by_row = _runs(self.m_row)
        self.s_by_col = _sorted_runs(self.s_col)
        self.m_by_col = _sorted_runs(self.m_col_of_group)

    # -- the batched part -----------------------------------------------------------------------
    def _assembled(self, evid: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """For every face assignment: (B, base). B (720, positions, 6) is log P(colour) - log P(empty)
        in the cube's own colours; base (720,) is the sum of log P(empty)."""
        g = self.geom
        npos = self.layout.n_positions
        es = np.stack(evid)                                         # (photo, 29, 7)
        s = np.zeros((6, 6, npos, N_COLOURS + 1))                   # (face, photo, position, value)
        for f in range(6):
            s[f][:, g.position_of[f], :] = es
        a = s[0, PERMS[:, 0]]
        for f in range(1, 6):
            a = a + s[f, PERMS[:, f]]
        m = a.max(2, keepdims=True)
        a = a - (m + np.log(np.exp(a - m).sum(2, keepdims=True)))
        return a[:, :, :N_COLOURS] - a[:, :, EMPTY:EMPTY + 1], a[:, :, EMPTY].sum(1)

    def rank(self, evid: list[np.ndarray], layouts: list[ColourLayout], prior: list[float], top: int = 36,
             perms: np.ndarray | None = None) -> tuple[list[Candidate], float]:
        """The ``top`` best candidates, best first, and an upper bound on the score of every other one.
        ``perms`` limits the face assignments (rows of PERMS) searched."""
        n = self.n
        b, base = self._assembled(evid)
        rows = np.arange(len(PERMS)) if perms is None else np.asarray(perms, dtype=np.int64)
        # Face assignments last, so that every gather below copies whole contiguous rows.
        bt = np.zeros((self.layout.n_positions + 1, N_COLOURS, len(rows)))               # (positions + a zero row, 6, P)
        bt[:-1] = b[rows].transpose(1, 2, 0)
        base = base[rows]
        P = len(rows)
        # One-colour pieces, for each colour the cube could show: best over each group's placements.
        g_all = np.maximum.reduceat(bt[self.s_pos], self.s_start, axis=0)                 # (groups, 6, P)
        ns = len(self.s_groups)
        order_s, starts_s, cells_s = self.s_by_col
        order_m, starts_m, cells_m = self.m_by_col
        # Sum over one-colour pieces of their best fit anywhere, by (model colour, cube colour).
        row_best = np.maximum.reduceat(g_all, self.s_by_row[1], axis=0)                  # (pieces, 6, P)
        row_by = np.zeros((N_COLOURS, N_COLOURS, P))
        for c in range(N_COLOURS):
            row_by[c] = row_best[self.s_row_colour == c].sum(0)

        bounds, per_layout = [], []
        for li, lay in enumerate(layouts):
            paint = np.array(lay.paint)
            gs = g_all[np.arange(ns), paint[self.s_colour]]                               # (groups, P)
            # corner and edge pieces
            v = bt[self.m_pos, paint[self.m_col]].sum(1)                                   # (placements, P)
            gm = np.maximum.reduceat(v, self.m_start, axis=0)                             # (groups, P)
            row_sum = row_by[np.arange(N_COLOURS), paint].sum(0) + np.maximum.reduceat(gm, self.m_by_row[1], axis=0).sum(0)
            col_max = np.full((n, P), _NONE)
            col_max[cells_s] = np.maximum.reduceat(gs[order_s], starts_s, axis=0)
            col_max[cells_m] = np.maximum(col_max[cells_m], np.maximum.reduceat(gm[order_m], starts_m, axis=0))
            ub = base + np.minimum(row_sum, col_max.sum(0)) - prior[li]
            bounds.append(ub)
            per_layout.append((gs, gm))
        ub = np.concatenate(bounds)
        order = np.argsort(-ub, kind="stable")
        found: list[tuple[float, int, int]] = []
        kth = -np.inf
        rest = -np.inf
        for k, i in enumerate(order):
            if len(found) >= top and ub[i] <= kth:
                rest = float(ub[i])
                break
            li, pi = divmod(int(i), P)
            gs, gm = per_layout[li]
            dense = np.full(n * n, _NONE)
            dense[self.s_groups] = gs[:, pi]
            dense[self.m_groups] = gm[:, pi]
            mat = dense.reshape(n, n)
            r, c = linear_sum_assignment(-mat)
            exact = float(base[pi] + mat[r, c].sum()) - prior[li]
            found.append((exact, li, pi))
            if len(found) >= top:
                found.sort(key=lambda x: -x[0])
                del found[top:]
                kth = found[-1][0]
        found.sort(key=lambda x: -x[0])
        out = []
        for score, li, pi in found[:top]:
            faces = tuple(int(f) for f in np.argsort(PERMS[rows[pi]]))
            out.append(Candidate(score, score + prior[li], li, faces))
        return out, rest

    def arrangement(self, evid: list[np.ndarray], layout: ColourLayout, faces: tuple[int, ...]) -> tuple[np.ndarray, float, np.ndarray]:
        """For one candidate: the arrangement (orientation per piece), its log-likelihood, and the
        assembled log-probabilities in the model's colours (positions, 7)."""
        from .pipeline import assemble

        logp = assemble([layout.model_columns(e) for e in evid], faces, self.geom, self.layout.n_positions)
        orient, ll = assign_logp(logp, self.layout)
        return orient, ll, logp
