"""Seeded IID partitioner (Section 3: "IID with K = 3, 5, 10 (seeded random equal split)").

A single deterministic permutation of the TRAIN-row indices, split into K (near-)equal,
sorted, disjoint chunks. Randomness depends only on (seed, K).
"""
from __future__ import annotations

import numpy as np


def iid_partition(n: int, K: int, seed: int) -> list:
    """Returns a list of K sorted, disjoint 1-D index arrays covering range(n) exactly once."""
    rng = np.random.default_rng([seed, K, 99_991])  # distinct stream from the attribute partitioner
    perm = rng.permutation(n)
    chunks = np.array_split(perm, K)
    return [np.sort(chunk) for chunk in chunks]
