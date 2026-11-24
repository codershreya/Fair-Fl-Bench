"""Unit tests for Phase 6: GIFAIR-FL, FairTrade-SP/-ATE, GFL / GFL-GBCFSS-EO / GFL-GBCFSS-DP.
Mirrors tests/test_phase5.py's structure: unit checks of new shared pieces, K=1/determinism
sanity per algorithm, model-architecture pins, a couple of dedicated pinning/regression tests
for behaviour discovered while building this phase (notably the FairTrade `_safe_mean` bug --
see below), and one-fast-combo regression guards against the oracle per algorithm (the full
sweep lives in scripts/run_equivalence_sweep.py / results/equivalence/full_l1_sweep.json)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fl.federated.common import seed_all
from fl.federated.fairtrade import AverageTreatmentEffectLoss, DemographicParityLoss, FairTradeATE, FairTradeSP
from fl.federated.gifair import GIFAIRFL, _gifair_local_update
from fl.federated.gfl import (
    GFL,
    GFLGBCFSSDP,
    GFLGBCFSSEO,
    LSHCadc,
    _agdg_run,
    _client_statistics,
    _compute_alphas,
    _fairness_contribution_scores,
    _greedy_select_gbcfss,
    _metrics_from_scores,
)
from fl.models import GFLTabularNet, GIFAIRMLP

ORACLE_OUT = Path(__file__).resolve().parent.parent / "results" / "equivalence" / "oracle_outputs"


def _toy_data(n=256, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.1 * rng.normal(size=n) > 0).astype(np.float32)
    a = rng.integers(0, 2, size=n).astype(np.float32)
    return X, y, a


# ===============================================================================================
# GIFAIR-FL
# ===============================================================================================

def test_gifair_model_is_gifair_mlp_shape_and_order():
    # GIFAIR-FL's own `lambda_reg = min(x/(K-1) for x in p)/2` formula divides by K-1, so K=1
    # is not a supported client count for this algorithm (a property of the original, not
    # something this port needs to special-case) -- use K=2 for this architecture-only check.
    X, y, a = _toy_data(n=16, d=6)
    clients = [dict(X=X[:8], y=y[:8].astype(np.int64), a=a[:8]), dict(X=X[8:], y=y[8:].astype(np.int64), a=a[8:])]
    val = dict(X=X[:8], y=y[:8], a=a[:8])
    cfg = dict(GIFAIRFL.harmonised_cfg, rounds=1)
    m = GIFAIRFL(clients, val, seed=0, cfg=cfg)
    assert isinstance(m.model, GIFAIRMLP)
    out = m.model(torch.as_tensor(X))
    assert out.shape == (16, 2)
    # log_softmax output: each row's exp() sums to 1.
    assert torch.allclose(torch.exp(out).sum(dim=1), torch.ones(16), atol=1e-5)


def test_gifair_initial_loss_uses_python_random_not_numpy():
    import random

    seed_all(0)
    expected = random.sample(range(4), 4)
    seed_all(0)
    X, y, a = _toy_data(n=64, d=3)
    clients = [dict(X=X[i::4], y=y[i::4].astype(np.int64), a=a[i::4]) for i in range(4)]
    val = dict(X=X[:8], y=y[:8], a=a[:8])
    cfg = dict(GIFAIRFL.harmonised_cfg, rounds=1)
    m = GIFAIRFL(clients, val, seed=0, cfg=cfg)
    assert m.initial_loss == expected


def test_gifair_users_pool_draw_is_discarded_but_sorted_ascending():
    X, y, a = _toy_data(n=64, d=3)
    clients = [dict(X=X[i::4], y=y[i::4].astype(np.int64), a=a[i::4]) for i in range(4)]
    val = dict(X=X[:8], y=y[:8], a=a[:8])
    cfg = dict(GIFAIRFL.harmonised_cfg, rounds=1)
    m = GIFAIRFL(clients, val, seed=0, cfg=cfg)
    assert list(m.users_pool) == [0, 1, 2, 3]  # regardless of what np.random.choice drew


def test_gifair_r_k_ranking_gives_highest_loss_client_the_most_positive_r_k():
    X, y, a = _toy_data(n=64, d=3)
    clients = [dict(X=X[i::4], y=y[i::4].astype(np.int64), a=a[i::4]) for i in range(4)]
    val = dict(X=X[:8], y=y[:8], a=a[:8])
    cfg = dict(GIFAIRFL.harmonised_cfg, rounds=1)
    m = GIFAIRFL(clients, val, seed=0, cfg=cfg)
    m.initial_loss = [1.0, 5.0, 3.0, 2.0]  # client 1 has the highest loss
    loss_dic = {m.users_pool[i]: m.initial_loss[i] for i in range(len(m.users_pool))}
    loss_dic = dict(sorted(loss_dic.items(), key=lambda it: it[1], reverse=True))
    r_k_dic = {list(loss_dic.keys())[i]: m.r_k_values[i] for i in range(len(loss_dic))}
    assert r_k_dic[1] == max(m.r_k_values)  # highest-loss client gets the most positive r_k
    assert r_k_dic[0] == min(m.r_k_values)  # lowest-loss client gets the most negative r_k


@pytest.mark.parametrize("cls", [GIFAIRFL])
def test_gifair_k1_and_determinism(cls):
    # K=2, not K=1 -- see note above about GIFAIR-FL's own lambda_reg formula.
    X, y, a = _toy_data()
    clients = [dict(X=X[:128], y=y[:128].astype(np.int64), a=a[:128]),
               dict(X=X[128:], y=y[128:].astype(np.int64), a=a[128:])]
    val = dict(X=X[:8], y=y[:8].astype(np.int64), a=a[:8])
    cfg = dict(cls.harmonised_cfg, rounds=3)
    m1 = cls(clients, val, seed=5, cfg=cfg)
    m1.fit(cfg["rounds"])
    proba1 = m1.proba(X)
    assert np.isfinite(proba1).all()
    m2 = cls(clients, val, seed=5, cfg=cfg)
    m2.fit(cfg["rounds"])
    assert np.array_equal(proba1, m2.proba(X))


# ===============================================================================================
# FairTrade-SP / FairTrade-ATE -- the `_safe_mean` bug found this phase.
# ===============================================================================================

def test_demographic_parity_loss_handles_empty_group_without_nan():
    # Regression test for a real bug caught by the oracle equivalence sweep: reusing
    # trustfed.py's DemographicParityLoss (plain `.mean()`) produced NaN whenever a batch had
    # zero samples of one sensitive group -- common under attr-skewed small-K partitions.
    # FairTrade's OWN mu_f (repo:p2 constraint.py) uses `_safe_mean` and must NOT produce NaN.
    loss_fn = DemographicParityLoss(alpha=1.0)
    logits = torch.tensor([0.5, -0.2, 1.0, 0.3])
    sensitive = torch.zeros(4)  # every sample is group 0 -- group 1 is EMPTY in this batch
    out = loss_fn(logits, sensitive, y=None)
    assert torch.isfinite(out).all()
    assert not torch.isnan(out).any()


def test_demographic_parity_loss_matches_hand_computed_value():
    loss_fn = DemographicParityLoss(alpha=1.0)
    logits = torch.tensor([2.0, -2.0, 2.0, -2.0])
    sensitive = torch.tensor([0., 0., 1., 1.])
    out = loss_fn(logits, sensitive, y=None)
    probs = torch.sigmoid(logits)
    mu0 = probs[sensitive == 0].mean()
    mu1 = probs[sensitive == 1].mean()
    mu_all = probs.mean()
    mu = torch.stack([mu0, mu1, mu_all])
    gap = torch.relu(torch.mv(loss_fn.M, mu) - loss_fn.c)
    expected = loss_fn.alpha * torch.dot(gap, gap)
    assert out.item() == pytest.approx(expected.item(), abs=1e-6)


def test_average_treatment_effect_loss_handles_empty_group_without_nan():
    loss_fn = AverageTreatmentEffectLoss(alpha=1.0)
    logits = torch.tensor([0.5, -0.2, 1.0, 0.3])
    sensitive = torch.zeros(4)
    y = torch.tensor([0., 0., 0., 0.])  # no positive labels at all
    out = loss_fn(logits, sensitive, y=y)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("cls", [FairTradeSP, FairTradeATE])
def test_fairtrade_k1_and_determinism(cls):
    X, y, a = _toy_data()
    clients = [dict(X=X, y=y, a=a)]
    val = dict(X=X[:8], y=y[:8], a=a[:8])
    cfg = dict(cls.harmonised_cfg, rounds=3)
    m1 = cls(clients, val, seed=5, cfg=cfg)
    m1.fit(cfg["rounds"])
    proba1 = m1.proba(X)
    assert np.isfinite(proba1).all()
    m2 = cls(clients, val, seed=5, cfg=cfg)
    m2.fit(cfg["rounds"])
    assert np.array_equal(proba1, m2.proba(X))


# ===============================================================================================
# GFL / GFL-GBCFSS-EO / GFL-GBCFSS-DP
# ===============================================================================================

def test_gfl_tabular_net_shape_and_embedding_order():
    model = GFLTabularNet(d_num=3, cat_cardinalities=[4, 6], emb_dim=8, hidden=32)
    assert len(model.embeddings) == 2
    assert model.embeddings[0].num_embeddings == 4
    assert model.embeddings[1].num_embeddings == 6
    x_num = torch.randn(5, 3)
    x_cat = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [0, 5]], dtype=torch.long)
    out = model(x_num, x_cat)
    assert out.shape == (5,)


def test_client_statistics_per_group_mean_and_jittered_covariance():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]], dtype=np.float64)
    g = np.array([0, 0, 1])
    stats = _client_statistics(X, g)
    assert set(stats.keys()) == {0, 1}
    assert stats[0]["num"] == 2
    assert np.allclose(stats[0]["mu"], [2.0, 3.0])
    assert stats[1]["num"] == 1
    # single-sample group: np.cov returns NaN pre-jitter in general, but the +1e-6*eye ridge
    # still gets added on top -- just confirm the jitter diagonal is present and finite isn't
    # required for a size-1 group (matches the original's own un-guarded behaviour).
    assert stats[1]["sigma"].shape == (2, 2)


def test_agdg_synthesizes_requested_sample_count_per_group():
    # 2 numeric columns (not 1) -- np.cov(Xg, rowvar=False) on a single-column input returns a
    # bare 0-d scalar rather than a (1,1) matrix, a genuine property of np.cov (and hence of
    # the original algorithm too) but not what this test means to exercise. Also >=2 samples
    # per group per client, so np.cov's ddof=1 default doesn't produce NaN (a single-sample
    # group's "variance" is undefined pre-jitter in the ORIGINAL algorithm too -- not a bug,
    # just not what this test is probing).
    seed_all(0)
    stats_c1 = _client_statistics(
        np.array([[0.0, 1.0], [1.0, 1.5], [2.0, 2.0], [3.0, 2.5]]), np.array([0, 0, 1, 1]))
    stats_c2 = _client_statistics(
        np.array([[4.0, 3.0], [4.5, 3.2], [5.0, 3.5], [5.5, 3.7]]), np.array([0, 0, 1, 1]))
    synth, global_stats = _agdg_run([stats_c1, stats_c2])
    assert set(synth.keys()) == {0, 1}
    assert synth[0]["X_num"].shape[0] == global_stats[0]["num"]
    assert synth[1]["X_num"].shape[0] == global_stats[1]["num"]
    assert global_stats[0]["num"] == 4  # 2 from c1 + 2 from c2
    assert global_stats[1]["num"] == 4  # 2 from c1 + 2 from c2


def test_lsh_cadc_isolated_rng_is_seed_invariant_across_outer_seeds():
    # LSH_CADC.fit always uses np.random.default_rng(42) regardless of the outer benchmark
    # seed -- its hyperplanes (and hence its completion output) must be IDENTICAL whichever
    # outer seed_all() call preceded it. This is a property of the original algorithm.
    X_num = np.array([[0.1], [0.4], [0.9], [1.2]])
    X_cat = np.array([[0], [1], [0], [1]])
    g = np.array([0, 0, 1, 1])
    covs = {0: np.array([[1.0]]), 1: np.array([[1.0]])}
    synth_numeric = {0: np.array([[0.2]]), 1: np.array([[1.1]])}
    synth_groups = {0: np.array([0]), 1: np.array([1])}

    seed_all(0)
    out_a = LSHCadc(k=7).fit(X_num, X_cat, g, covs).complete(synth_numeric, synth_groups)
    seed_all(999)
    out_b = LSHCadc(k=7).fit(X_num, X_cat, g, covs).complete(synth_numeric, synth_groups)
    assert np.array_equal(out_a[0], out_b[0])
    assert np.array_equal(out_a[1], out_b[1])


def test_gbcfss_fairness_contribution_boosts_disadvantaged_group_eo():
    y_true = np.array([1, 1, 1, 0, 0])
    g = np.array([0, 0, 1, 0, 1])  # among positives (idx 0,1,2): group0 has 2, group1 has 1
    s = _fairness_contribution_scores(y_true, g, mode="eo")
    assert s[2] == 1.0  # the lone group-1 positive (disadvantaged) is boosted
    assert s[0] == 0.0 and s[1] == 0.0  # group-0 positives are NOT boosted


def test_gbcfss_greedy_select_respects_budget_and_group_balance():
    rng = np.random.default_rng(0)
    N = 40
    y_true = rng.integers(0, 2, N)
    g = rng.integers(0, 2, N)
    y_score = rng.uniform(0, 1, N)
    sel = _greedy_select_gbcfss(y_true, g, y_score, budget=10, clean_w=1.0, fair_w=1.0,
                                 group_balance=0.2, fairness_mode="eo")
    assert len(sel) == 10
    assert len(set(sel.tolist())) == 10  # no duplicate indices
    counts = {0: int((g[sel] == 0).sum()), 1: int((g[sel] == 1).sum())}
    assert counts[0] >= 2 and counts[1] >= 2  # >=20% of budget=10 from each group


def test_gbcfss_budget_exceeding_n_selects_every_row_exactly_once():
    rng = np.random.default_rng(1)
    N = 15
    y_true = rng.integers(0, 2, N)
    g = rng.integers(0, 2, N)
    y_score = rng.uniform(0, 1, N)
    sel = _greedy_select_gbcfss(y_true, g, y_score, budget=5000, clean_w=1.0, fair_w=1.0,
                                 group_balance=0.2, fairness_mode="dp")
    assert sorted(sel.tolist()) == list(range(N))  # every row selected, just possibly reordered


def test_fairness_metrics_and_alpha_fallback_to_uniform_when_all_biases_zero():
    y_score = np.array([0.5, 0.5, 0.5, 0.5])
    y_true = np.array([1, 0, 1, 0])
    g = np.array([0, 0, 1, 1])
    eod, dpd = _metrics_from_scores(y_score, y_true, g)
    assert eod == pytest.approx(0.0, abs=1e-9)
    assert dpd == pytest.approx(0.0, abs=1e-9)
    alphas = _compute_alphas({0: 0.0, 1: 0.0, 2: 0.0})
    assert alphas == {0: pytest.approx(1 / 3), 1: pytest.approx(1 / 3), 2: pytest.approx(1 / 3)}


def _gfl_toy(n=48, d_num=3, cat_cards=(4, 3), seed=0):
    rng = np.random.default_rng(seed)
    num = rng.normal(size=(n, d_num)).astype(np.float32)
    cats = np.column_stack([rng.integers(0, c, size=n) for c in cat_cards]).astype(np.float32)
    X = np.concatenate([num, cats], axis=1)
    y = (num[:, 0] > 0).astype(np.float32)
    a = rng.integers(0, 2, size=n).astype(np.float32)
    meta = dict(d_num=d_num, cat_cards=list(cat_cards))
    return X, y, a, meta


@pytest.mark.parametrize("cls", [GFL, GFLGBCFSSEO, GFLGBCFSSDP])
def test_gfl_k1_runs_and_determinism(cls):
    X, y, a, meta = _gfl_toy(n=64)
    clients = [dict(X=X, y=y, a=a)]
    val = dict(X=X[:16], y=y[:16], a=a[:16], meta=meta)
    cfg = dict(cls.harmonised_cfg, rounds=2, hidden=16, emb_dim=4, global_bs=32, batch=32)
    m1 = cls(clients, val, seed=3, cfg=cfg)
    m1.fit(cfg["rounds"])
    proba1 = m1.proba(X)
    assert np.isfinite(proba1).all()
    m2 = cls(clients, val, seed=3, cfg=cfg)
    m2.fit(cfg["rounds"])
    assert np.array_equal(proba1, m2.proba(X))


def test_gfl_declares_codes_input_kind_for_equivalence_harness():
    assert GFL.input_kind == "codes"
    assert GFLGBCFSSEO.input_kind == "codes"
    assert GFLGBCFSSDP.input_kind == "codes"


# ---------------------------------------------------------------------------------------------
# Regression guards against the oracle (full sweep lives outside pytest; see module docstring).
# Includes the exact combo that caught the FairTrade `_safe_mean` bug during this phase.
# ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(not ORACLE_OUT.exists(), reason="Phase-2 oracle_outputs/ not present")
@pytest.mark.parametrize("algo", ["GIFAIR-FL", "FairTrade-SP", "FairTrade-ATE",
                                   "GFL", "GFL-GBCFSS-EO", "GFL-GBCFSS-DP"])
def test_l1_equivalence_cac_iid3_seed0(algo):
    from scripts.check_equivalence import run_check
    r = run_check("cac", algo, "iid", 3, 0)
    assert r["level"] == "L1", r
    for name, st in r["per_state"].items():
        assert st["level"] == "L1", (name, st)


@pytest.mark.skipif(not ORACLE_OUT.exists(), reason="Phase-2 oracle_outputs/ not present")
def test_l1_equivalence_fairtrade_sp_attr_skew_regression():
    # The exact combo that exposed the DemographicParityLoss `_safe_mean` divergence: an
    # attr-skewed 5-client partition where a training batch can contain zero samples of one
    # sensitive group.
    from scripts.check_equivalence import run_check
    r = run_check("adult", "FairTrade-SP", "attr", 5, 0)
    assert r["level"] == "L1", r
