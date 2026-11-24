"""Unit tests for Phase 5: GlobalGroup-SP/-EO. Mirrors tests/test_phase4.py's structure: unit
checks of the new shared pieces (`_YSet`, `_distance_kernel`), a K=1/determinism sanity pass per
algorithm, a pinning test for the GlobalGroup-EO joint-group-masking audit finding documented in
fl/federated/globalgroup.py's module docstring, and a one-fast-combo regression guard against the
oracle per algorithm (the full 75-combo-per-algorithm sweep lives in
scripts/run_equivalence_sweep.py / results/equivalence/full_l1_sweep.json)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fl.federated.common import seed_all
from fl.federated.globalgroup import GlobalGroupEO, GlobalGroupSP, _ClientEO, _distance_kernel, _YSet
from fl.models import FedAvgP1Net

ORACLE_OUT = Path(__file__).resolve().parent.parent / "results" / "equivalence" / "oracle_outputs"


# ---------------------------------------------------------------------------------------------
# _YSet / _distance_kernel (shared by GlobalGroup-SP and -EO)
# ---------------------------------------------------------------------------------------------
def test_yset_drop_with_mu_1_always_empties_but_still_consumes_rng():
    seed_all(0)
    ys = _YSet(ny=10)
    assert len(ys.values) == 10
    state_before = np.random.get_state()
    ys.drop(mu=1.0)
    state_after = np.random.get_state()
    assert len(ys.values) == 0  # mu=1.0 -> int((1-1.0)*10) == 0 points kept
    # The RNG draw still happened (np.random.choice(10, 0, replace=False) is not a no-op).
    assert not np.array_equal(state_before[1], state_after[1]) or state_before[2] != state_after[2]


def test_yset_drop_respects_mu_less_than_one():
    seed_all(0)
    ys = _YSet(ny=20)
    ys.drop(mu=0.5)
    assert len(ys.values) == 10  # int((1-0.5)*20) == 10 points kept


def test_yset_update_concatenates_onto_whatever_remains():
    ys = _YSet(ny=4)
    ys.drop(mu=1.0)  # empties it
    assert len(ys.values) == 0
    ys.update([torch.tensor([1.0, 2.0]), torch.tensor([3.0])])
    assert torch.equal(ys.values, torch.tensor([1.0, 2.0, 3.0]))


def test_distance_kernel_is_symmetric_and_matches_hand_computed_value():
    a = torch.tensor([0.3])
    b = torch.tensor([0.7])
    # (|a-1|+|b-1|-|a-b|) + (|a|+|b|-|a-b|), all divided by 4.
    expected = ((abs(0.3 - 1) + abs(0.7 - 1) - abs(0.3 - 0.7)) + (abs(0.3) + abs(0.7) - abs(0.3 - 0.7))) / 4
    out = _distance_kernel(a, b)
    assert out.item() == pytest.approx(expected, abs=1e-6)
    assert _distance_kernel(a, b).item() == pytest.approx(_distance_kernel(b, a).item(), abs=1e-6)


# ---------------------------------------------------------------------------------------------
# K=1 / determinism sanity, one per algorithm (matches tests/test_phase4.py's pattern). Uses a
# larger toy dataset than Phase 3/4's helper -- GlobalGroup's per-group resampling can request
# more points than a tiny toy client has if group proportions are very skewed, which is a
# property of the ORIGINAL algorithm (repo:p1 Global_Group/clients.py::sample_C_update uses
# replace=False), not a bug in this port; a 256-row, roughly-balanced toy set avoids tripping it.
# ---------------------------------------------------------------------------------------------
def _toy_data(n=256, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.1 * rng.normal(size=n) > 0).astype(np.float32)
    a = rng.integers(0, 2, size=n).astype(np.float32)
    return X, y, a


@pytest.mark.parametrize("cls", [GlobalGroupSP, GlobalGroupEO])
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


@pytest.mark.parametrize("cls", [GlobalGroupSP, GlobalGroupEO])
def test_determinism_same_seed_same_result(cls):
    X, y, a = _toy_data()
    clients = [dict(X=X[:128], y=y[:128], a=a[:128]), dict(X=X[128:], y=y[128:], a=a[128:])]
    val = dict(X=X[:8], y=y[:8], a=a[:8])
    cfg = dict(cls.harmonised_cfg, rounds=3)
    m1 = cls(clients, val, seed=5, cfg=cfg)
    m1.fit(cfg["rounds"])
    m2 = cls(clients, val, seed=5, cfg=cfg)
    m2.fit(cfg["rounds"])
    assert np.array_equal(m1.proba(X), m2.proba(X))


def test_globalgroup_model_is_fedavg_p1_net_shape():
    # GlobalGroup reuses FedAvgP1Net's architecture verbatim (Linear(d,64)->ReLU->Linear(64,1)
    # ->Sigmoid); this pins that no divergent model class crept in.
    X, y, a = _toy_data(n=32, d=6)
    clients = [dict(X=X, y=y, a=a)]
    val = dict(X=X[:8], y=y[:8], a=a[:8])
    cfg = dict(GlobalGroupSP.harmonised_cfg, rounds=1)
    m = GlobalGroupSP(clients, val, seed=0, cfg=cfg)
    assert isinstance(m.model, FedAvgP1Net)
    out = m.model(torch.as_tensor(X))
    assert out.shape == (32, 1)
    assert bool(((out >= 0) & (out <= 1)).all())  # sigmoid output


# ---------------------------------------------------------------------------------------------
# Pinning test for the GlobalGroup-EO audit finding (module docstring in
# fl/federated/globalgroup.py): the joint-group encoding means groups (A=1,Y=0) and (A=0,Y=1)
# are ALWAYS empty, and the s=1 subpopulation never enters the fairness term at all. This test
# fails loudly if that behaviour is ever "fixed" by accident, which would silently break L1
# equivalence against the oracle (which wraps the same original, buggy code).
# ---------------------------------------------------------------------------------------------
def test_globalgroup_eo_joint_coding_leaves_two_groups_permanently_empty():
    # s=0,y=0 -> joint 0; s=0,y=1 -> joint 1; s=1,y=0 -> joint 2; s=1,y=1 -> joint 3.
    X = torch.zeros((8, 2))
    Y = torch.tensor([0., 0., 1., 1., 0., 0., 1., 1.])
    s_raw = torch.tensor([0., 0., 0., 0., 1., 1., 1., 1.])
    a_joint = s_raw * 2 + Y
    cl = _ClientEO(X, Y, a_joint, stepsize=0.005, batchsize=32, epochs=1, lambda_=0.2)
    cl.set_N(8)
    cl.set_alpha_kay(Pa00=0.25, Pa10=0.0, Pa01=0.0, Pa11=0.25)  # Pa10/Pa01 are always 0 (the bug)
    model = FedAvgP1Net(2)
    theta = {k: v.detach().clone() for k, v in model.state_dict().items()}
    p00, p10, p01, p11 = cl.sample_C_update([10, 10, 10, 10], model, theta)
    # (A=0,Y=0) and (A=1,Y=1) are real, non-degenerate groups (joint 0 and joint 1 resp.).
    assert len(p00) > 0
    assert len(p11) > 0
    # (A=1,Y=0) and (A=0,Y=1) are permanently empty: no row can satisfy the contradiction.
    assert len(p10) == 0
    assert len(p01) == 0
    # The s=1 subpopulation (joint codes 2 and 3) never appears in ANY of the four groups above
    # -- confirm directly that get_Pkay never sees it either (Section "Audit finding").
    assert cl.get_Pkay(0, 0) == pytest.approx(2 / 8)  # s=0,y=0: rows 0,1
    assert cl.get_Pkay(1, 1) == pytest.approx(2 / 8)  # s=0,y=1: rows 2,3
    assert cl.get_Pkay(1, 0) == 0.0  # contradiction -> always 0, regardless of s=1 rows 4,6
    assert cl.get_Pkay(0, 1) == 0.0  # contradiction -> always 0, regardless of s=1 rows 5,7


# ---------------------------------------------------------------------------------------------
# Regression guard against the oracle (full sweep lives outside pytest; see module docstring).
# ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(not ORACLE_OUT.exists(), reason="Phase-2 oracle_outputs/ not present")
@pytest.mark.parametrize("algo", ["GlobalGroup-SP", "GlobalGroup-EO"])
def test_l1_equivalence_cac_iid3_seed0(algo):
    from scripts.check_equivalence import run_check
    r = run_check("cac", algo, "iid", 3, 0)
    assert r["level"] == "L1", r
    for name, st in r["per_state"].items():
        assert st["level"] == "L1", (name, st)
