"""Mirrors `reference/oracle_fairbench`'s own `fairbench.runner.run_experiment` hook /
best-validation-model bookkeeping (Section 5.2), but drives OUR OWN `REGISTRY` instead of the
oracle's -- used by the Phase-3/4 equivalence harness (`scripts/check_equivalence.py`) to make
sure what gets compared against the oracle is genuinely "the same kind of result" (same three
saved states: last round, best-validation-accuracy round, and the algorithm's own
`native_final_state` when it has one), not just a convenient subset. Will be reused unchanged
by the real benchmark run once that exists (Phase 7)."""
from __future__ import annotations

import numpy as np

from fl.algorithms import REGISTRY
from fl.metrics import compute_metrics


def run_from_scratch(algo: str, clients: list[dict], val: dict, test: dict, seed: int, cfg: dict) -> dict:
    cls = REGISTRY[algo]
    ad = cls(clients, val, seed, cfg)
    val_hist: list[dict] = []
    best = dict(acc=-1.0, rnd=0, state=None)

    def hook(r: int) -> None:
        p = (ad.proba(val["X"]) > 0.5).astype(int)
        m = compute_metrics(val["y"], p, val["a"])
        m["round"] = r
        val_hist.append({k: m[k] for k in ("round", "accuracy", "balanced_accuracy", "sp_abs", "eo", "positive_rate_pred")})
        if m["accuracy"] > best["acc"]:
            best.update(acc=m["accuracy"], rnd=r, state=ad.get_state())

    ad.fit(cfg["rounds"], hook)
    last_state = ad.get_state()

    states = {"last_round": (last_state, cfg["rounds"]), "best_val_acc": (best["state"], best["rnd"])}
    native_final = getattr(ad, "native_final_state", lambda: None)()
    if native_final is not None:
        states["native_final_avg_iterate"] = (native_final, None)

    results, preds = {}, {}
    for name, (st, rnd) in states.items():
        ad.set_state(st)
        pr = ad.proba(test["X"])
        yhat = (pr > 0.5).astype(int)
        valid = bool(np.isfinite(pr).all() and set(np.unique(yhat)) <= {0, 1})
        m = compute_metrics(test["y"], yhat, test["a"])
        m.update(round=rnd, valid_predictions=valid, n_unique_pred=int(len(np.unique(yhat))))
        pv = (ad.proba(val["X"]) > 0.5).astype(int)
        m["val_accuracy_of_this_model"] = compute_metrics(val["y"], pv, val["a"])["accuracy"]
        results[name] = m
        preds[name] = yhat.astype(np.int8)
    ad.set_state(last_state)
    return dict(results=results, preds=preds, val_history=val_hist)
