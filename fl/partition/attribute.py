"""Attribute-based ("affinity") non-IID partitioner (Section 3).

For a TRAIN sample i with coordinate u_i in [0,1] along the partition variable P, client k gets
sampling weight

    w_k(u) = eps + (1 - 2*eps) * max(0, 1 - |t_k - u| / h),   t_k = (k-1)/(K-1)

  * discrete P (L ordered levels, ascending TRAIN base rate): u = level_index / (L-1), h = 1/(L-1)
  * continuous P (CaC racepctblack):                          u = (tie-broken rank + 0.5) / n,
                                                                h = 1/(K-1)

`eps` is a support floor: every client keeps weight >= eps for every coordinate, so no client is
ever single-group by construction. Labels are NOT stratified. Randomness (tie-breaking + the
client draw) is a function of (seed, K, eps, attempt) only, per Section 3.

Feasibility: a draw is accepted only if every client has >= MIN_CLIENT_TRAIN training samples
and >= MIN_CELL samples in every (A, Y) cell; otherwise the attempt index increases and the
whole assignment is redrawn (fl.config.MAX_PARTITION_ATTEMPTS caps this; every attempt, accepted
or not, is logged by the caller).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from fl import config as C


def coordinates(kind: str, values: np.ndarray, order: Optional[list], K: int, rng: np.random.Generator):
    """Returns (u, h) for the given TRAIN-row partition-variable values."""
    if kind == "continuous":
        n = len(values)
        tie_break = rng.random(n)
        rank = np.empty(n, dtype=np.int64)
        rank[np.lexsort((tie_break, values))] = np.arange(n)  # ascending value, random tie-break
        u = (rank + 0.5) / n
        h = 1.0 / (K - 1)
        return u, h
    if kind == "categorical":
        L = len(order)
        codes = pd.Categorical(values, categories=order).codes
        assert (codes >= 0).all(), "partition value outside the declared level order"
        u = codes / (L - 1)
        h = 1.0 / (L - 1)
        return u, h
    raise ValueError(f"unknown partition kind {kind!r}")


def affinity_weights(u: np.ndarray, K: int, h: float, eps: float) -> np.ndarray:
    """Returns an (n, K) row-stochastic matrix of client sampling weights."""
    t = np.linspace(0.0, 1.0, K)
    g = np.clip(1.0 - np.abs(t[None, :] - u[:, None]) / h, 0.0, None)
    w = eps + (1.0 - 2.0 * eps) * g
    return w / w.sum(axis=1, keepdims=True)


def attribute_partition(kind: str, values: np.ndarray, order: Optional[list], K: int,
                         eps: float, seed: int, attempt: int = 0) -> list:
    """One draw (no feasibility check). Returns a list of K 1-D index arrays into `values`."""
    rng = np.random.default_rng([seed, K, int(round(eps * 1000)), attempt])
    u, h = coordinates(kind, values, order, K, rng)
    weights = affinity_weights(u, K, h, eps)
    cdf = np.cumsum(weights, axis=1)
    draw = rng.random(len(u))[:, None]
    client = np.minimum((draw > cdf).sum(axis=1), K - 1)
    return [np.nonzero(client == k)[0] for k in range(K)]


def partition_stats(parts: list, y: np.ndarray, a: np.ndarray) -> dict:
    """y, a must be the TRAIN-row arrays (same length/order as the values partitioned)."""
    n_total = len(y)
    global_a_share = a.mean()
    sizes = np.array([len(p) for p in parts])
    a_share = np.array([a[p].mean() if len(p) else np.nan for p in parts])
    pos_rate = np.array([y[p].mean() if len(p) else np.nan for p in parts])
    cells = np.array([
        [int(((a[p] == ai) & (y[p] == yi)).sum()) for ai in (0, 1) for yi in (0, 1)]
        for p in parts
    ])
    return dict(
        sizes=sizes, a_share=a_share, pos_rate=pos_rate, cells=cells,
        min_size=int(sizes.min()), min_cell=int(cells.min()),
        size_ratio=float(sizes.max() / max(sizes.min(), 1)),
        a_share_range=float(np.nanmax(a_share) - np.nanmin(a_share)),
        pos_rate_range=float(np.nanmax(pos_rate) - np.nanmin(pos_rate)),
        tv_mean=float((sizes / n_total * np.abs(a_share - global_a_share)).sum()),
    )


def is_feasible(stats: dict, min_size: int = C.MIN_CLIENT_TRAIN, min_cell: int = C.MIN_CELL) -> bool:
    return stats["min_size"] >= min_size and stats["min_cell"] >= min_cell


def partition_with_guard(kind: str, values: np.ndarray, order: Optional[list],
                          y: np.ndarray, a: np.ndarray, K: int, eps: float, seed: int,
                          max_attempts: int = C.MAX_PARTITION_ATTEMPTS):
    """Redraws (incrementing `attempt`) until feasible or `max_attempts` is exhausted.

    Returns (parts, stats, n_attempts_used, feasible_bool). Every attempt is implicitly logged
    by the caller via n_attempts_used (>1 means at least one redraw happened)."""
    parts, stats = None, None
    for attempt in range(max_attempts):
        parts = attribute_partition(kind, values, order, K, eps, seed, attempt)
        stats = partition_stats(parts, y, a)
        if is_feasible(stats):
            return parts, stats, attempt + 1, True
    return parts, stats, max_attempts, False
