"""Unit tests for Phase 4: TrustFed-SP/-EO, FedMinMax, EOFedMinMax. Mirrors the structure of
tests/test_federated.py (Phase 3): unit-level checks of the new shared pieces
(project_simplex, the TrustFed constraint-loss M matrices), a K=1/determinism sanity pass per
algorithm, and a one-fast-combo regression guard against the oracle per algorithm (the full
24-combo-per-algorithm sweep lives in scripts/run_equivalence_sweep.py /
results/equivalence/full_l1_sweep.json, too slow to re-run on every `pytest` invocation)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fl.federated.common import project_simplex, seed_all
from fl.federated.eofedminmax import EOFedMinMax, _compute_eo_p1
from fl.federated.fedminmax import FedMinMax
from fl.federated.trustfed import DemographicParityLoss, EqualizedOddsLoss, TrustFedEO, TrustFedSP, loss_trustfed
from fl.models import FedMinMaxNet

ORACLE_OUT = Path(__file__).resolve().parent.parent / "results" / "equivalence" / "oracle_outputs"


# ---------------------------------------------------------------------------------------------
# project_simplex (shared by FedMinMax and EOFedMinMax)
# ---------------------------------------------------------------------------------------------
def test_project_simplex_already_on_simplex_is_a_no_op():
    v = torch.tensor([0.5, 0.5])
    out = project_simplex(v, eps=0.0)
    assert torch.allclose(out, v, atol=1e-6)


def test_project_simplex_respects_eps_floor_and_sums_to_one():
    v = torch.tensor([1.0, 0.0])
    out = project_simplex(v, eps=0.1)
    assert out.min().item() >= 0.1 - 1e-6
    assert abs(out.sum().item() - 1.0) < 1e-6


def test_project_simplex_negative_values_still_project_validly():
    v = torch.tensor([-2.0, 5.0, 0.3])
    out = project_simplex(v, eps=0.0)
    assert out.min().item() >= -1e-6
    assert abs(out.sum().item() - 1.0) < 1e-6


# ---------------------------------------------------------------------------------------------
# TrustFed constraint losses: shapes and basic sanity (hand-checkable properties, not full
# bit-exactness -- that's covered by the oracle regression guard below).
# ---------------------------------------------------------------------------------------------
def test_demographic_parity_loss_is_zero_when_groups_agree():
    # Every row in both groups has IDENTICAL output -> groups' means are equal -> gap = 0.
    out_logits = torch.zeros(8)  # sigmoid(0) = 0.5 for everyone
    sensitive = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    loss = DemographicParityLoss(sensitive_classes=(0, 1), alpha=1.0, p_norm=2)
    val = loss(out_logits, sensitive)
    assert val.item() == pytest.approx(0.0, abs=1e-6)


def test_equalized_odds_loss_runs_and_is_nonnegative():
    torch.manual_seed(0)
    out_logits = torch.randn(16)
    sensitive = torch.tensor([0, 0, 1, 1] * 4, dtype=torch.float32)
    y = torch.tensor([0, 1, 0, 1] * 4, dtype=torch.float32)
    loss = EqualizedOddsLoss(sensitive_classes=(0, 1), y_classes=(0, 1), alpha=1.0, p_norm=2)
    val = loss(out_logits, sensitive, y=y)
    assert val.item() >= 0.0
    assert np.isfinite(val.item())


def test_loss_trustfed_falls_back_to_base_loss_if_fairness_term_is_nan():
    # A group with zero members (sensitive_classes names a value absent from the batch) makes
    # out[mask].mean() a NaN-producing empty-tensor mean; loss_trustfed must fall back to the
    # (finite) base_loss rather than propagate NaN (repo:p1 TrustFedLoss.py's own fallback).
    preds = torch.full((4, 1), 0.5)
    targets = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    sensitive = torch.zeros((4, 1))  # every row is group 0 -> group "1" is empty
    total, fairness = loss_trustfed(preds, targets, sensitive, fairness_notion="SP",
                                    sensitive_classes=(0, 1))
    assert np.isfinite(total.item())


# ---------------------------------------------------------------------------------------------
# K=1 / determinism sanity, one per algorithm (matches tests/test_federated.py's pattern)
# ---------------------------------------------------------------------------------------------
def _toy_data(n=64, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.1 * rng.normal(size=n) > 0).astype(np.float32)
    a = rng.integers(0, 2, size=n).astype(np.float32)
    return X, y, a


@pytest.mark.parametrize("cls", [TrustFedSP, TrustFedEO, FedMinMax, EOFedMinMax])
def test_k1_runs_and_produces_valid_predictions(cls):
    X, y, a = _toy_data()
    clients = [dict(X=X, y=y, a=a)]
    val = dict(X=X[:8], y=y[:8], a=a[:8])
    cfg = dict(cls.harmonised_cfg, rounds=3)
    model = cls(clients, val, seed=0, cfg=cfg)
    model.fit(cfg["rounds"])
    proba = model.proba(X)
    assert np.isfinite(proba).all()
    preds = (proba > 0.5).astype(int)
    assert set(np.unique(preds)) <= {0, 1}


@pytest.mark.parametrize("cls", [TrustFedSP, TrustFedEO, FedMinMax, EOFedMinMax])
def test_determinism_same_seed_same_result(cls):
    X, y, a = _toy_data()
    clients = [dict(X=X[:32], y=y[:32], a=a[:32]), dict(X=X[32:], y=y[32:], a=a[32:])]
    val = dict(X=X[:8], y=y[:8], a=a[:8])
    cfg = dict(cls.harmonised_cfg, rounds=3)
    m1 = cls(clients, val, seed=5, cfg=cfg)
    m1.fit(cfg["rounds"])
    m2 = cls(clients, val, seed=5, cfg=cfg)
    m2.fit(cfg["rounds"])
    assert np.array_equal(m1.proba(X), m2.proba(X))


def test_fedminmax_and_eofedminmax_have_a_native_final_state_distinct_from_last_round():
    X, y, a = _toy_data(n=128)
    clients = [dict(X=X[:64], y=y[:64], a=a[:64]), dict(X=X[64:], y=y[64:], a=a[64:])]
    val = dict(X=X[:8], y=y[:8], a=a[:8])
    for cls in (FedMinMax, EOFedMinMax):
        cfg = dict(cls.harmonised_cfg, rounds=4)
        m = cls(clients, val, seed=1, cfg=cfg)
        m.fit(cfg["rounds"])
        nf = m.native_final_state()
        assert nf is not None
        last = m.get_state()
        # the running average is generally NOT identical to the final round's own weights
        assert any(not torch.equal(nf[k], last[k]) for k in nf)


def test_fedminmax_net_outputs_logits_not_probabilities():
    m = FedMinMaxNet(input_dim=6)
    x = torch.randn(10, 6) * 50  # large inputs -> pre-activation logits can exceed [0,1]
    out = m(x)
    assert out.shape == (10, 1)
    assert bool((out.abs() > 1.0).any())  # would be impossible if a sigmoid were applied


def test_compute_eo_p1_returns_zero_not_nan_when_a_group_is_empty():
    # Section 3/D15 says OUR OWN fl.metrics module must return NaN here; EOFedMinMax's internal
    # lambda controller intentionally reproduces the ORIGINAL (0.0, not NaN) behaviour instead
    # -- this test pins that intentional difference so it is never "fixed" by accident.
    preds = torch.tensor([1.0, 0.0, 1.0, 1.0])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    sensitive = torch.zeros(4)  # only group 0 present -> compute_equalized_odds returns 0.0
    assert _compute_eo_p1(preds, labels, sensitive) == 0.0


# ---------------------------------------------------------------------------------------------
# Regression guard against the oracle (full sweep lives outside pytest; see module docstring).
# ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(not ORACLE_OUT.exists(), reason="Phase-2 oracle_outputs/ not present")
@pytest.mark.parametrize("algo", ["TrustFed-SP", "TrustFed-EO", "FedMinMax", "EOFedMinMax"])
def test_l1_equivalence_cac_iid3_seed0(algo):
    from scripts.check_equivalence import run_check
    r = run_check("cac", algo, "iid", 3, 0)
    assert r["level"] == "L1", r
    for name, st in r["per_state"].items():
        assert st["level"] == "L1", (name, st)
