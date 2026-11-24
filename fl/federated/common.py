"""Shared helpers for every from-scratch FL algorithm: deterministic seeding and the two
aggregation rules both papers use (Section 2). Independently re-derived from reading both
repos' server/aggregation code; no import from reference/paper1_repo or reference/paper2_repo.

`seed_all` matters beyond style: Section 5's L1 equivalence target (bit-exact, <=1e-6, vs the
oracle) is only reachable if our code consumes the global `random`/`numpy`/`torch` RNGs in
EXACTLY the same order the original repo's own code does for the same algorithm (model
construction order, per-round client order, per-epoch DataLoader reshuffling all draw from
these global generators) -- see reports/phase3_report.md for the full argument. This function
must be called at the exact point in each algorithm's `__init__` that the original code calls
its own equivalent (repo:p1 algos/base.py::seed_all / oracle's copy of the same).
"""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def aggregate_weighted_by_samples(reports: list[dict], device: str = "cpu") -> dict:
    """Paper-1 FedAvg aggregation (repo:p1 FedAvg/FedAvgLossStrategies.py::agg_fedavg):
    per-tensor weighted average, weight = client_samples / total_samples, accumulated in
    `reports` list order (order matters only for float rounding, not for the result's meaning).
    """
    total = sum(r["samples"] for r in reports)
    keys = reports[0]["weights"].keys()
    out = {k: torch.zeros_like(reports[0]["weights"][k], device=device) for k in keys}
    for r in reports:
        w = r["samples"] / total
        for k, v in r["weights"].items():
            out[k] += w * v.to(device)
    return out


def project_simplex(v: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Projects `v` onto the simplex {x : sum(x) = 1, x_i >= eps}. Re-derived identically from
    repo:p1 OrigFedMinMax/OriglossStrategiesFedMinMax.py::project_simplex and
    repo:p1 EOFedMinMax/lossStrategiesEOFedMinMax.py::project_simplex (byte-identical in both
    original files) -- used by FedMinMax/EOFedMinMax's adversarial group-reweighting step.
    Pure deterministic tensor arithmetic; consumes no RNG."""
    d = v.shape[0]
    if eps < 0:
        raise ValueError("eps must be >= 0")
    if d * eps > 1.0 + 1e-12:
        raise ValueError(f"Infeasible: d*eps={d * eps} > 1. Choose smaller eps.")
    mass = 1.0 - d * eps
    u = v.float() - eps
    if mass <= 1e-15:
        return torch.full((d,), eps, dtype=torch.float32)

    s, _ = torch.sort(u, descending=True)
    cssv = torch.cumsum(s, dim=0) - mass
    ind = torch.arange(1, d + 1).float()
    cond = s - cssv / ind > 0
    indices = torch.nonzero(cond)
    if indices.numel() == 0:
        y = torch.zeros(d)
        y[torch.argmax(u)] = mass
    else:
        rho_idx = indices[-1].item()
        theta = cssv[rho_idx] / (rho_idx + 1)
        y = torch.clamp(u - theta, min=0.0)

    x = y + eps
    x = torch.clamp(x, min=eps)
    x = x / x.sum()
    return x.detach().clone()


def average_weights_unweighted(weight_dicts: list[dict]) -> dict:
    """Paper-2's aggregation (repo:p2 code/GIFAIR-FL/utils.py::average_weights): a plain,
    UNWEIGHTED mean over the selected clients' state dicts, summed in list order then divided
    by the count -- client size never enters the P2 aggregation formula."""
    import copy

    out = copy.deepcopy(weight_dicts[0])
    for key in out.keys():
        for i in range(1, len(weight_dicts)):
            out[key] = out[key] + weight_dicts[i][key]
        out[key] = torch.div(out[key], len(weight_dicts))
    return out
