"""Single metric module used by every algorithm, computed from the same test predictions
(threshold 0.5) (Section 3: "Metrics").

A = 1 is always the privileged / higher-TRAIN-base-rate group (fl.data.orientation); yhat = 1 is
the favourable predicted outcome.

    signed SP = PPR(A=1) - PPR(A=0)                    (>0: A=1 receives more favourable predictions)
    |SP|      = absolute value of the above
    EO        = 0.5 * (|TPR(A=1)-TPR(A=0)| + |FPR(A=1)-FPR(A=0)|)     (Paper-1 definition)
    EOp       = |TPR(A=1) - TPR(A=0)|                                  (Paper-2 definition)
    DI        = PPR(A=0) / PPR(A=1)                                    (Paper-2 definition; 1 = ideal)

Undefined quantities (an empty group, a group with no positives/negatives for TPR/FPR, or
PPR(A=1) == 0 for DI) are NaN, never 0 -- a run must never look "fair" just because a group or
class was empty (Section 3; reference/analysis/AUDIT_AND_PROTOCOL.md finding D15).
"""
from __future__ import annotations

import numpy as np

NAN = float("nan")


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else NAN


def _group_rates(y: np.ndarray, yhat: np.ndarray, mask: np.ndarray) -> dict:
    n = int(mask.sum())
    pos_mask = mask & (y == 1)
    neg_mask = mask & (y == 0)
    return dict(
        n=n,
        ppr=_safe_rate(int((yhat[mask] == 1).sum()), n),
        tpr=_safe_rate(int((yhat[pos_mask] == 1).sum()), int(pos_mask.sum())),
        fpr=_safe_rate(int((yhat[neg_mask] == 1).sum()), int(neg_mask.sum())),
    )


def compute_metrics(y_true, y_pred, a) -> dict:
    y = np.asarray(y_true).astype(int)
    yhat = np.asarray(y_pred).astype(int)
    a = np.asarray(a).astype(int)
    assert y.shape == yhat.shape == a.shape
    assert set(np.unique(a)) <= {0, 1}, f"sensitive attribute must be binary, got {np.unique(a)}"
    assert set(np.unique(yhat)) <= {0, 1}, f"predictions must be in {{0,1}}, got {np.unique(yhat)}"

    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    tpr_overall = _safe_rate(tp, n_pos)
    tnr_overall = _safe_rate(tn, n_neg)

    g1 = _group_rates(y, yhat, a == 1)
    g0 = _group_rates(y, yhat, a == 0)

    sp_signed = g1["ppr"] - g0["ppr"]
    eo = 0.5 * (abs(g1["tpr"] - g0["tpr"]) + abs(g1["fpr"] - g0["fpr"]))
    eop = abs(g1["tpr"] - g0["tpr"])
    di = (g0["ppr"] / g1["ppr"]) if (g1["ppr"] == g1["ppr"] and g1["ppr"] > 0
                                      and g0["ppr"] == g0["ppr"]) else NAN

    return dict(
        accuracy=float((yhat == y).mean()),
        balanced_accuracy=0.5 * (tpr_overall + tnr_overall),
        sp_signed=sp_signed, sp_abs=abs(sp_signed), eo=eo, eop=eop, di=di,
        tpr_a1=g1["tpr"], tpr_a0=g0["tpr"], fpr_a1=g1["fpr"], fpr_a0=g0["fpr"],
        ppr_a1=g1["ppr"], ppr_a0=g0["ppr"], n_a1=g1["n"], n_a0=g0["n"],
        positive_rate_pred=float(yhat.mean()), n=int(len(y)),
    )
