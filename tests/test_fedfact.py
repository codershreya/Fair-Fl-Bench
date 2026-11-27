"""Unit tests for FedFACT-In and FedFACT-Post (fl/federated/fedfact.py; DEVIATIONS.md D13-D16).

Unit checks of the calibration matrix, determinism, dual-variable behaviour and that the fairness
constraint actually acts, plus one fast regression guard against the saved Phase-2 oracle output
(the full sweep is Phase 4). Toy data only, except the oracle guard."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fl.federated.fedfact import FedFACTIn, FedFACTLogistic
from fl.metrics import compute_metrics

ORACLE_OUT = Path(__file__).resolve().parent.parent / "results" / "equivalence" / "oracle_outputs"


def _toy_clients(K=3, n=300, d=6, seed=0, bias=1.5):
    """Labels correlated with the sensitive attribute, so an unconstrained model is unfair."""
    rng = np.random.default_rng(seed)
    clients = []
    for _ in range(K):
        a = rng.integers(0, 2, size=n).astype(np.float32)
        X = rng.normal(size=(n, d)).astype(np.float32)
        X[:, 0] += bias * a                      # a proxy for A inside X (A itself is not an input)
        y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0.75).astype(np.float32)
        clients.append(dict(X=X, y=y, a=a))
    return clients


def _cfg(**kw):
    return dict(FedFACTIn.harmonised_cfg, **kw)


def _dp(model, clients):
    X = np.concatenate([c["X"] for c in clients])
    y = np.concatenate([c["y"] for c in clients])
    a = np.concatenate([c["a"] for c in clients])
    p = (model.proba(X) > 0.5).astype(int)
    return compute_metrics(y, p, a)["sp_abs"]


def test_model_is_single_linear_layer_with_two_logits():
    m = FedFACTLogistic(7)
    assert [tuple(p.shape) for p in m.parameters()] == [(2, 7), (2,)]
    flat = m.get_flat_params()
    m2 = FedFACTLogistic(7)
    m2.set_flat_params(flat)
    assert torch.equal(m2.get_flat_params(), flat)
    assert m(torch.zeros(7)).shape == (1, 2)       # 1-D input is one row, as in the code


def test_calibration_matrix_zero_duals_is_identity():
    alg = FedFACTIn(_toy_clients(), None, 0, _cfg(rounds=1))
    M = alg._calibration_matrices(0, torch.zeros(2), torch.zeros(2))
    for g in (0, 1):
        assert torch.allclose(M[g], torch.eye(2, dtype=M[g].dtype))


def test_calibration_matrix_global_part_is_client_independent():
    """1/p_{a,k} * p_{k|a} = n / n_a for every client (DEVIATIONS.md D13), so with mu = 0 every
    client gets the same M(a) up to float rounding."""
    clients = _toy_clients(K=4)
    alg = FedFACTIn(clients, None, 0, _cfg(rounds=1))
    lamb = torch.tensor([0.7, 0.2])
    Ms = [alg._calibration_matrices(k, lamb, torch.zeros(2)) for k in range(4)]
    a = np.concatenate([c["a"] for c in clients])
    J = torch.tensor([[0., 1.], [0., 1.]], dtype=torch.float64)
    raw = {g: torch.eye(2, dtype=torch.float64) - len(a) / (a == g).sum() * 0.5 * (2 * g - 1) * J for g in (0, 1)}
    low = min(float(m.min()) for m in raw.values())
    shift = -low + 0.001 if low < 0 else 0.0              # the code's "positive calibration"
    for g in (0, 1):
        for M in Ms:
            assert torch.allclose(M[g].double(), raw[g] + shift, atol=1e-6)


def test_deterministic_given_seed_and_different_across_seeds():
    clients = _toy_clients()
    runs = []
    for seed in (0, 0, 1):
        alg = FedFACTIn(clients, None, seed, _cfg(rounds=3))
        alg.fit(3)
        runs.append(alg.get_state()["theta"])
    assert torch.equal(runs[0], runs[1])
    assert not torch.equal(runs[0], runs[2])


def test_duals_stay_in_bounds_and_vanish_when_constraint_never_binds():
    clients = _toy_clients()
    loose = FedFACTIn(clients, None, 0, _cfg(rounds=8, global_constraints=1.0, local_constraints=1.0))
    loose.fit(8)
    assert torch.all(loose.global_lamb == 0)
    assert all(torch.all(c.local_mu == 0) for c in loose.clients)
    tight = FedFACTIn(clients, None, 0, _cfg(rounds=8))
    tight.fit(8)
    for t in [tight.global_lamb] + [c.local_mu for c in tight.clients]:
        assert torch.all(t >= 0) and torch.all(t <= tight.cfg["dual_params_bound"])


def test_fairness_constraint_reduces_demographic_parity():
    clients = _toy_clients(bias=2.0)
    loose = FedFACTIn(clients, None, 0, _cfg(rounds=30, global_constraints=1.0, local_constraints=1.0))
    loose.fit(30)
    tight = FedFACTIn(clients, None, 0, _cfg(rounds=30))
    tight.fit(30)
    assert _dp(tight, clients) < _dp(loose, clients) - 0.05


def test_proba_is_a_probability_and_consumes_no_rng():
    alg = FedFACTIn(_toy_clients(), None, 0, _cfg(rounds=1))
    alg.fit(1)
    state = torch.get_rng_state()
    p = alg.proba(np.random.default_rng(1).normal(size=(50, 6)).astype(np.float32))
    assert torch.equal(state, torch.get_rng_state())
    assert p.shape == (50,) and np.all((p >= 0) & (p <= 1))


def test_only_dp_is_implemented():
    with pytest.raises(NotImplementedError):
        FedFACTIn(_toy_clients(), None, 0, _cfg(rounds=1, fairness_constraints="EO"))


@pytest.mark.skipif(not (ORACLE_OUT / "cac" / "cac__iid3__FedFACT-In__harmonised__s0.json").exists(),
                    reason="Phase-2 FedFACT oracle outputs not present")
def test_matches_oracle_cac_iid3_seed0():
    from scripts.check_equivalence import run_check
    r = run_check("cac", "FedFACT-In", "iid", 3, 0)
    assert r["level"] == "L1", r


# ---------------------------------------------------------------------------------------------
# FedFACT-Post (global-only variant, DEVIATIONS.md D13)
# ---------------------------------------------------------------------------------------------
from fl.federated.fedfact import FedFACTPost, _calibration_matrices  # noqa: E402


def _toy_val(n=240, d=6, seed=5, bias=1.5):
    c = _toy_clients(K=1, n=n, d=d, seed=seed, bias=bias)[0]
    return dict(X=c["X"], y=c["y"], a=c["a"])


def _pcfg(**kw):
    return dict(FedFACTPost.harmonised_cfg, **kw)


def test_post_calibration_matrix_has_no_positive_shift():
    p_a = {0: 0.5, 1: 0.5}
    p_ka = {0: 0.25, 1: 0.25}
    lamb = torch.tensor([3.0, 0.0])
    M_post = _calibration_matrices(p_a, p_ka, lamb, torch.zeros(2), positive_shift=False)
    M_in = _calibration_matrices(p_a, p_ka, lamb, torch.zeros(2), positive_shift=True)
    assert float(min(m.min() for m in M_post.values())) < 0          # negative entries kept
    assert float(min(m.min() for m in M_in.values())) > 0            # In shifts them positive
    # the shift adds the same constant everywhere, so it cannot change any argmax decision
    assert torch.allclose(M_in[0] - M_post[0], M_in[1] - M_post[1])


def test_post_needs_the_sensitive_attribute():
    alg = FedFACTPost(_toy_clients(), _toy_val(), 0, _pcfg(rounds=1, post_round=1))
    alg.fit(1)
    with pytest.raises(ValueError):
        alg.proba(_toy_val()["X"])
    p = alg.proba(_toy_val()["X"], _toy_val()["a"])
    assert set(np.unique(p)) <= {0.0, 1.0}


def test_post_mu_stays_zero_and_lambda_in_bounds():
    alg = FedFACTPost(_toy_clients(), _toy_val(), 0, _pcfg(rounds=3, post_round=10))
    alg.fit(3)
    assert all(torch.all(c.local_mu == 0) for c in alg.clients)
    assert torch.all(alg.global_lamb >= 0) and torch.all(alg.global_lamb <= alg.cfg["dual_params_bound"])


def test_post_deterministic_given_seed():
    states = []
    for _ in range(2):
        alg = FedFACTPost(_toy_clients(), _toy_val(), 3, _pcfg(rounds=3, post_round=5))
        alg.fit(3)
        states.append(alg.get_state())
    assert torch.equal(states[0]["theta"], states[1]["theta"]) and torch.equal(states[0]["lamb"], states[1]["lamb"])


def test_post_calibration_reduces_demographic_parity_on_validation():
    clients, val = _toy_clients(bias=2.0), _toy_val(bias=2.0)
    alg = FedFACTPost(clients, val, 0, _pcfg(rounds=20, post_round=30))
    dps = []
    alg.fit(20, hook=lambda t: dps.append(compute_metrics(
        val["y"], alg.proba(val["X"], val["a"]).astype(int), val["a"])["sp_abs"]))
    assert dps[-1] < dps[0] - 0.05


@pytest.mark.skipif(not (ORACLE_OUT / "cac" / "cac__iid3__FedFACT-Post__harmonised__s0.json").exists(),
                    reason="Phase-2 FedFACT oracle outputs not present")
def test_post_matches_oracle_cac_iid3_seed0():
    from scripts.check_equivalence import run_check
    r = run_check("cac", "FedFACT-Post", "iid", 3, 0)
    assert r["level"] == "L1", r
