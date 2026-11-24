"""Unit tests for fl.metrics (Section 3 "Metrics" + Section 4 tests/test_metrics.py).

Cross-checks against the FORMULAS published in the two original papers' own metric code
(no import of either repo -- the formulas below are transcribed, by hand, from reading
  reference/paper1_repo/FairnessBenchmarkingFL/fairnessMetrics.py
  reference/paper2_repo/Fair_Federated_Learning/code/GIFAIR-FL/fairness_metrics.py
and are used here purely as an independent numpy re-derivation for agreement testing; see
reference/analysis/AUDIT_AND_PROTOCOL.md Section 7 / Section 2 D14).
"""
import numpy as np
import pytest

from fl.metrics import compute_metrics


# ---------------------------------------------------------------------------------------------
# Independent numpy re-derivations of each paper's own formulas (not imports), for cross-check
# ---------------------------------------------------------------------------------------------
def _p1_formulas(y, yhat, a):
    """Mirrors fairnessMetrics.py: compute_statistical_parity / compute_equalized_odds /
    compute_balanced_accuracy, with group_a=0, group_b=1 (sorted unique values of a)."""
    def rates(mask):
        if mask.sum() == 0:
            return 0.0, 0.0, 0.0
        ppr = yhat[mask].mean()
        pos, neg = mask & (y == 1), mask & (y == 0)
        tpr = yhat[pos].mean() if pos.sum() > 0 else 0.0
        fpr = yhat[neg].mean() if neg.sum() > 0 else 0.0
        return tpr, fpr, ppr

    tpr0, fpr0, ppr0 = rates(a == 0)
    tpr1, fpr1, ppr1 = rates(a == 1)
    sp = ppr1 - ppr0
    eo = 0.5 * (abs(tpr1 - tpr0) + abs(fpr1 - fpr0))
    tp, tn = ((yhat == 1) & (y == 1)).sum(), ((yhat == 0) & (y == 0)).sum()
    pos, neg = (y == 1).sum(), (y == 0).sum()
    tpr_all = tp / pos if pos > 0 else 0.0
    tnr_all = tn / neg if neg > 0 else 0.0
    ba = 0.5 * (tpr_all + tnr_all)
    return dict(sp=sp, eo=eo, ba=ba)


def _p2_formulas(y, yhat, a):
    """Mirrors fairness_metrics.py: compute_fairness_metrics (groups==1 is privileged)."""
    def safe_div(n, d):
        return float(n) / float(d) if d != 0 else 0.0

    mask_pos, mask_priv, mask_unpriv = (y == 1), (a == 1), (a == 0)
    tpr_priv = safe_div(((yhat == 1) & mask_pos & mask_priv).sum(), (mask_pos & mask_priv).sum())
    tpr_unpriv = safe_div(((yhat == 1) & mask_pos & mask_unpriv).sum(), (mask_pos & mask_unpriv).sum())
    eop = abs(tpr_priv - tpr_unpriv)
    p_pos_unpriv = safe_div(((yhat == 1) & mask_unpriv).sum(), mask_unpriv.sum())
    p_pos_priv = safe_div(((yhat == 1) & mask_priv).sum(), mask_priv.sum())
    di = safe_div(p_pos_unpriv, p_pos_priv)
    acc = (y == yhat).mean()
    return dict(eop=eop, di=di, acc=acc)


@pytest.mark.parametrize("trial", range(8))
def test_agrees_with_both_papers_formulas_on_random_data(trial):
    rng = np.random.default_rng(trial)
    n = 500
    y = rng.integers(0, 2, size=n)
    a = rng.integers(0, 2, size=n)
    yhat = rng.integers(0, 2, size=n)
    # guarantee every (a, y) cell is non-empty so the 0-vs-NaN "undefined" convention cannot
    # make the two papers' own code (which return 0.0) disagree with ours (which returns NaN)
    for ai in (0, 1):
        for yi in (0, 1):
            idx = np.nonzero((a == ai) & (y == yi))[0]
            if len(idx) == 0:
                a[trial % n] = ai
                y[trial % n] = yi

    ours = compute_metrics(y, yhat, a)
    p1 = _p1_formulas(y, yhat, a)
    p2 = _p2_formulas(y, yhat, a)

    assert ours["sp_signed"] == pytest.approx(p1["sp"], abs=1e-9)
    assert ours["eo"] == pytest.approx(p1["eo"], abs=1e-9)
    assert ours["balanced_accuracy"] == pytest.approx(p1["ba"], abs=1e-9)
    assert ours["eop"] == pytest.approx(p2["eop"], abs=1e-9)
    assert ours["di"] == pytest.approx(p2["di"], abs=1e-9)
    assert ours["accuracy"] == pytest.approx(p2["acc"], abs=1e-9)


def test_undefined_cases_are_nan_not_zero():
    # all predictions negative, a=1 group empty -> PPR(A=1) undefined -> DI must be NaN, not 0
    y = np.array([0, 1, 0, 1])
    yhat = np.array([0, 0, 0, 0])
    a = np.array([0, 0, 0, 0])
    m = compute_metrics(y, yhat, a)
    assert np.isnan(m["tpr_a1"]) and np.isnan(m["fpr_a1"]) and np.isnan(m["ppr_a1"])
    assert np.isnan(m["di"])


def test_known_toy_example():
    # a=0: 2 samples, both predicted 1 (ppr=1.0); a=1: 2 samples, 0 predicted 1 (ppr=0.0)
    y =    np.array([1, 0, 1, 0])
    yhat = np.array([1, 1, 0, 0])
    a =    np.array([0, 0, 1, 1])
    m = compute_metrics(y, yhat, a)
    assert m["ppr_a0"] == 1.0 and m["ppr_a1"] == 0.0
    assert m["sp_signed"] == pytest.approx(-1.0)
    assert m["sp_abs"] == pytest.approx(1.0)
    assert np.isnan(m["di"])  # ppr_a1 == 0 -> DI undefined by our own rule (NaN, never 0 or inf)
