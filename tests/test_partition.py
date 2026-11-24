"""Unit tests for fl.partition (IID + attribute-based affinity partitioner) and the
partitioner's own feasibility guard (Section 3 "Client settings" + Section 5.4 property tests).
"""
import numpy as np
import pytest

from fl.partition.iid import iid_partition
from fl.partition.attribute import (
    attribute_partition, partition_stats, is_feasible, partition_with_guard, coordinates,
)


def test_iid_partition_is_disjoint_and_covers_everything():
    n, K, seed = 1000, 5, 0
    parts = iid_partition(n, K, seed)
    assert len(parts) == K
    sizes = [len(p) for p in parts]
    assert max(sizes) - min(sizes) <= 1
    union = np.sort(np.concatenate(parts))
    assert np.array_equal(union, np.arange(n))
    for p in parts:
        assert np.array_equal(p, np.sort(p))  # sorted, as required by the spec


def test_iid_partition_is_seed_deterministic_and_seed_sensitive():
    p1 = iid_partition(1000, 5, seed=0)
    p2 = iid_partition(1000, 5, seed=0)
    p3 = iid_partition(1000, 5, seed=1)
    for a, b in zip(p1, p2):
        assert np.array_equal(a, b)
    assert not all(np.array_equal(a, b) for a, b in zip(p1, p3))


def test_attribute_partition_disjoint_and_covers_everything_categorical():
    rng = np.random.default_rng(0)
    n = 2000
    a = rng.integers(0, 2, size=n)  # categorical 2-level partition variable (P == A case)
    parts = attribute_partition("categorical", a, [0, 1], K=3, eps=0.20, seed=0)
    assert len(parts) == 3
    union = np.sort(np.concatenate(parts))
    assert np.array_equal(union, np.arange(n))
    assert len(set.union(*[set(p.tolist()) for p in parts])) == n
    for i in range(3):
        for j in range(i + 1, 3):
            assert set(parts[i].tolist()).isdisjoint(set(parts[j].tolist()))


def test_attribute_partition_disjoint_and_covers_everything_continuous():
    rng = np.random.default_rng(1)
    n = 2000
    values = rng.normal(size=n)
    parts = attribute_partition("continuous", values, None, K=5, eps=0.20, seed=0)
    union = np.sort(np.concatenate(parts))
    assert np.array_equal(union, np.arange(n))


def test_support_floor_eps_means_no_client_is_ever_single_group():
    """At any eps > 0, extreme coordinates (u=0 or u=1) must still receive non-trivial
    weight on every client (the whole point of the support floor)."""
    rng = np.random.default_rng(0)
    u = np.array([0.0, 1.0])
    K, h, eps = 5, 1.0 / 4, 0.20
    from fl.partition.attribute import affinity_weights
    w = affinity_weights(u, K, h, eps)
    # eps is a floor on the UN-normalised weight (before the row is rescaled to sum to 1), so
    # after normalisation the guarantee is strict positivity, not literally >= eps: no client
    # ever has exactly zero probability, i.e. no client is single-group by construction.
    assert np.all(w > 0.0)
    assert np.allclose(w.sum(axis=1), 1.0)


def test_randomness_depends_only_on_seed_K_eps_attempt():
    rng = np.random.default_rng(0)
    n = 1000
    values = rng.integers(0, 2, size=n)
    p_a = attribute_partition("categorical", values, [0, 1], K=3, eps=0.2, seed=5, attempt=0)
    p_b = attribute_partition("categorical", values, [0, 1], K=3, eps=0.2, seed=5, attempt=0)
    for x, y in zip(p_a, p_b):
        assert np.array_equal(x, y)
    p_c = attribute_partition("categorical", values, [0, 1], K=3, eps=0.2, seed=5, attempt=1)
    assert not all(np.array_equal(x, y) for x, y in zip(p_a, p_c))


def test_continuous_coordinate_is_tie_broken_rank_quantile():
    rng = np.random.default_rng(0)
    values = np.array([1.0, 1.0, 1.0, 2.0, 0.0])  # ties at value 1.0
    u, h = coordinates("continuous", values, None, K=3, rng=rng)
    assert h == pytest.approx(1.0 / 2)
    # smallest value gets smallest u, largest value gets largest u; ties are *some* permutation
    assert u[4] < u[0] and u[0] < u[3]  # value 0.0 < (one of the 1.0's) < value 2.0
    assert np.array_equal(np.sort(u), (np.arange(5) + 0.5) / 5)


def test_partition_with_guard_reproduces_feasibility_at_eps_020():
    """Sanity check for Phase 1: with a reasonably balanced attribute and plenty of samples,
    eps=0.20 is feasible on the FIRST draw (attempt == 1), matching
    reference/analysis/AUDIT_AND_PROTOCOL.md Section 8 ("all 50 combinations feasible at eps=0.20")."""
    rng = np.random.default_rng(0)
    n = 5000
    a = rng.integers(0, 2, size=n)
    y = rng.integers(0, 2, size=n)
    parts, stats, n_attempts, feasible = partition_with_guard(
        "categorical", a, [0, 1], y, a, K=5, eps=0.20, seed=0,
    )
    assert feasible
    assert n_attempts == 1
    assert is_feasible(stats)
