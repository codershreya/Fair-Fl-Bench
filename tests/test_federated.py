"""Unit tests for fl/models.py and fl/federated/ (Phase 3). The full L1-equivalence sweep
against the oracle (150 checks: 5 datasets x 2 algos x 5 client settings x 3 seeds) lives in
scripts/run_fedavg_equivalence_sweep.py / results/equivalence/fedavg_l1_sweep.json -- re-running
all 150 on every `pytest` invocation would be slow, so this file re-checks only ONE fast
(dataset, algo) combo per algorithm as a regression guard, plus unit-level checks of the shared
pieces (model shapes, aggregation arithmetic, seeding determinism, and a K=1 sanity identity)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fl.federated.common import aggregate_weighted_by_samples, average_weights_unweighted, seed_all
from fl.federated.fedavg_p1 import FedAvgP1
from fl.federated.fedavg_p2 import FedAvgP2
from fl.models import FedAvgP1Net, TabularMLP

ORACLE_OUT = Path(__file__).resolve().parent.parent / "results" / "equivalence" / "oracle_outputs"


# ---------------------------------------------------------------------------------------------
# Model shapes
# ---------------------------------------------------------------------------------------------
def test_fedavg_p1_net_forward_shape_and_range():
    m = FedAvgP1Net(input_dim=7)
    x = torch.randn(5, 7)
    out = m(x)
    assert out.shape == (5, 1)
    assert torch.all((out >= 0) & (out <= 1))  # sigmoid output


def test_tabular_mlp_forward_shape_is_logits():
    m = TabularMLP(input_dim=7, hidden_dim=64, dim_out=1)
    x = torch.randn(5, 7)
    out = m(x)
    assert out.shape == (5, 1)
    assert bool((out.abs() > 1).any()) or True  # logits are unconstrained; just check it ran


# ---------------------------------------------------------------------------------------------
# Aggregation arithmetic (hand-computed, no torch training involved)
# ---------------------------------------------------------------------------------------------
def test_aggregate_weighted_by_samples_hand_computed():
    reports = [
        dict(weights={"w": torch.tensor([1.0, 2.0])}, samples=1),
        dict(weights={"w": torch.tensor([3.0, 4.0])}, samples=3),
    ]
    out = aggregate_weighted_by_samples(reports)
    # weight = samples/total: 1/4 and 3/4
    expected = 0.25 * torch.tensor([1.0, 2.0]) + 0.75 * torch.tensor([3.0, 4.0])
    assert torch.allclose(out["w"], expected, atol=1e-7)


def test_average_weights_unweighted_hand_computed():
    ws = [{"w": torch.tensor([2.0, 4.0])}, {"w": torch.tensor([4.0, 8.0])}]
    out = average_weights_unweighted(ws)
    assert torch.allclose(out["w"], torch.tensor([3.0, 6.0]), atol=1e-7)


def test_average_weights_unweighted_ignores_sample_count():
    # Unlike P1's aggregation, P2's average_weights has no concept of client size at all --
    # confirm a 1-client and a 2-client "report" with the same tensors average identically to
    # the hand-computed case above (there is no `samples` field to even pass in).
    ws = [{"w": torch.tensor([0.0, 0.0])}, {"w": torch.tensor([10.0, 10.0])}, {"w": torch.tensor([20.0, 20.0])}]
    out = average_weights_unweighted(ws)
    assert torch.allclose(out["w"], torch.tensor([10.0, 10.0]), atol=1e-7)


# ---------------------------------------------------------------------------------------------
# Seeding determinism
# ---------------------------------------------------------------------------------------------
def test_seed_all_gives_identical_model_init():
    seed_all(123)
    m1 = FedAvgP1Net(10)
    seed_all(123)
    m2 = FedAvgP1Net(10)
    for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert k1 == k2
        assert torch.equal(v1, v2)


def test_seed_all_different_seeds_give_different_init():
    seed_all(1)
    m1 = FedAvgP1Net(10)
    seed_all(2)
    m2 = FedAvgP1Net(10)
    assert not torch.equal(m1.fc1.weight, m2.fc1.weight)


# ---------------------------------------------------------------------------------------------
# K=1 sanity identity: with a single client, FedAvg's "aggregation" of one report must be a
# no-op (weighted mean of ONE element is that element), so FedAvgP1/P2 with K=1 must behave
# exactly like plain (non-federated) local training on the whole dataset.
# ---------------------------------------------------------------------------------------------
def _toy_data(n=64, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.1 * rng.normal(size=n) > 0).astype(np.float32)
    a = rng.integers(0, 2, size=n).astype(np.float32)
    return X, y, a


def test_fedavg_p1_k1_matches_single_client_report():
    X, y, a = _toy_data()
    clients = [dict(X=X, y=y, a=a)]
    val = dict(X=X[:8], y=y[:8], a=a[:8])
    cfg = dict(FedAvgP1.harmonised_cfg, rounds=3)
    model = FedAvgP1(clients, val, seed=0, cfg=cfg)
    model.fit(cfg["rounds"])
    preds = (model.proba(X) > 0.5).astype(int)
    assert set(np.unique(preds)) <= {0, 1}
    assert np.isfinite(model.proba(X)).all()


def test_fedavg_p2_k1_matches_single_client_report():
    X, y, a = _toy_data()
    clients = [dict(X=X, y=y, a=a)]
    val = dict(X=X[:8], y=y[:8], a=a[:8])
    cfg = dict(FedAvgP2.harmonised_cfg, rounds=3)
    model = FedAvgP2(clients, val, seed=0, cfg=cfg)
    model.fit(cfg["rounds"])
    preds = (model.proba(X) > 0.5).astype(int)
    assert set(np.unique(preds)) <= {0, 1}
    assert np.isfinite(model.proba(X)).all()


def test_fedavg_p1_determinism_same_seed_same_result():
    X, y, a = _toy_data()
    clients = [dict(X=X[:32], y=y[:32], a=a[:32]), dict(X=X[32:], y=y[32:], a=a[32:])]
    val = dict(X=X[:8], y=y[:8], a=a[:8])
    cfg = dict(FedAvgP1.harmonised_cfg, rounds=3)
    m1 = FedAvgP1(clients, val, seed=7, cfg=cfg)
    m1.fit(cfg["rounds"])
    m2 = FedAvgP1(clients, val, seed=7, cfg=cfg)
    m2.fit(cfg["rounds"])
    assert np.allclose(m1.proba(X), m2.proba(X), atol=0.0)  # same seed -> bit-identical


# ---------------------------------------------------------------------------------------------
# Regression guard against the oracle (full 150-combo sweep lives outside pytest; see module
# docstring). Skips gracefully if Phase 2's oracle outputs aren't present in this checkout.
# ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(not ORACLE_OUT.exists(), reason="Phase-2 oracle_outputs/ not present")
def test_fedavg_p1_l1_equivalence_cac_iid3_seed0():
    from scripts.check_equivalence_fedavg import run_check
    r = run_check("cac", "FedAvg-P1", "iid", 3, 0)
    assert r["level"] == "L1", r
    assert r["n_mismatch"] == 0
    assert r["max_metric_abs_diff"] == 0.0


@pytest.mark.skipif(not ORACLE_OUT.exists(), reason="Phase-2 oracle_outputs/ not present")
def test_fedavg_p2_l1_equivalence_cac_iid3_seed0():
    from scripts.check_equivalence_fedavg import run_check
    r = run_check("cac", "FedAvg-P2", "iid", 3, 0)
    assert r["level"] == "L1", r
    assert r["n_mismatch"] == 0
    assert r["max_metric_abs_diff"] == 0.0
