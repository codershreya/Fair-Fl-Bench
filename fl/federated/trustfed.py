"""TrustFed-SP / TrustFed-EO, re-derived from reading repo:p1 TrustFed/{TrustFedClient,
TrustFedServer,TrustFedLoss}.py and TrustFed/LossStrategies/loss.py (no import from
reference/paper1_repo). Same model/aggregation/construction-order pattern as FedAvg-P1
(fl/federated/fedavg_p1.py) -- TrustFed's own client/server code is "identical to the FedAvg
setup" per the original source's own comment, reusing `models.simpleModel` (== our
`FedAvgP1Net`) and `FedAvgLossStrategies.agg_fedavg`'s weighted-average formula verbatim --
plus two things on top: a cost-weighted BCE (FN=10x / FP=5x) + a fairness constraint term added
to the loss, and per-parameter, per-batch differential-privacy noise added to the gradient
after clipping.

RNG note (Section 5 L1 target): `torch.nn.utils.clip_grad_norm_` and the constraint-loss terms
are pure arithmetic (no RNG); only `torch.normal(...)` for the DP noise draws from the GLOBAL
torch RNG (no explicit generator is passed, matching the original), once per model PARAMETER
TENSOR (not per element) per batch, in `model.parameters()` iteration order (fc1.weight,
fc1.bias, fc2.weight, fc2.bias for our `FedAvgP1Net`, since that is also the order
`repo:p1 models.py::simpleModel`'s own two `nn.Linear` layers are registered in). Getting this
order right -- and reusing the identical per-client throwaway-model pre-construction quirk from
FedAvg-P1 -- is what lets the equivalence harness target L1 here too, not just L2.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fl.federated.common import aggregate_weighted_by_samples, seed_all
from fl.models import FedAvgP1Net


# ---------------------------------------------------------------------------------------------
# Fairness constraint terms (repo:p1 TrustFed/LossStrategies/loss.py) -- pure arithmetic, no RNG.
# ---------------------------------------------------------------------------------------------
class _ConstraintLoss(nn.Module):
    def __init__(self, M: torch.Tensor, c: torch.Tensor, alpha: float = 1.0, p_norm: int = 2):
        super().__init__()
        self.M, self.c, self.alpha, self.p_norm = M, c, alpha, p_norm

    def mu_f(self, out: torch.Tensor, sensitive: torch.Tensor, y) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, out_logits: torch.Tensor, sensitive: torch.Tensor, y=None) -> torch.Tensor:
        sensitive = sensitive.view(out_logits.shape)
        if isinstance(y, torch.Tensor):
            y = y.view(out_logits.shape)
        out = torch.sigmoid(out_logits)
        mu = self.mu_f(out, sensitive, y)
        gap = torch.relu(torch.mv(self.M, mu) - self.c)
        if self.p_norm == 2:
            return self.alpha * torch.dot(gap, gap)
        return self.alpha * torch.dot(gap.detach(), gap)


class DemographicParityLoss(_ConstraintLoss):
    """repo:p1 TrustFed/LossStrategies/loss.py::DemographicParityLoss (SP notion)."""

    def __init__(self, sensitive_classes=(0, 1), alpha: float = 1.0, p_norm: int = 2):
        self.sensitive_classes = list(sensitive_classes)
        n_class = len(self.sensitive_classes)
        n_constraints, dim_condition = 2 * n_class, n_class + 1
        M = torch.zeros((n_constraints, dim_condition))
        for i in range(n_constraints):
            j = i % 2
            if j == 0:
                M[i, j] = 1.0
                M[i, -1] = -1.0
            else:
                M[i, j - 1] = -1.0
                M[i, -1] = 1.0
        super().__init__(M, torch.zeros(n_constraints), alpha, p_norm)

    def mu_f(self, out, sensitive, y):
        vals = [out[sensitive == v].mean() for v in self.sensitive_classes]
        vals.append(out.mean())
        return torch.stack(vals)


class EqualizedOddsLoss(_ConstraintLoss):
    """repo:p1 TrustFed/LossStrategies/loss.py::EqualiedOddsLoss (EO notion). The nested-loop
    construction of M below -- including passing `y_1` for BOTH the y0 and y1 arguments of
    `_element` (the original's `y_0` loop variable is used only to index `i`, never inside the
    element formula itself) -- is reproduced exactly from the original, not simplified, since
    the loss value depends on this exact (if slightly redundant) M matrix."""

    def __init__(self, sensitive_classes=(0, 1), y_classes=(0, 1), alpha: float = 1.0, p_norm: int = 2):
        self.sensitive_classes = list(sensitive_classes)
        self.y_classes = list(y_classes)
        n_class, n_y = len(self.sensitive_classes), len(self.y_classes)
        n_constraints = n_class * n_y * 2
        dim_condition = n_y * (n_class + 1)
        M = torch.zeros((n_constraints, dim_condition))
        element_K_A = self.sensitive_classes + [None]

        def _element(a0, a1, y0, y1, s):
            if a0 is None or a1 is None:
                return -1.0 * s * float(y0 == y1)
            return s * float((a0 == a1) and (y0 == y1))

        for i_a, a0 in enumerate(self.sensitive_classes):
            for i_y in range(n_y):
                for i_s, s in enumerate([-1, 1]):
                    for j_y, y1 in enumerate(self.y_classes):
                        for j_a, a1 in enumerate(element_K_A):
                            i = i_a * (2 * n_y) + i_y * 2 + i_s
                            j = j_y + n_y * j_a
                            M[i, j] = _element(a0, a1, y1, y1, s)
        super().__init__(M, torch.zeros(n_constraints), alpha, p_norm)

    def mu_f(self, out, sensitive, y):
        vals = []
        for u in self.sensitive_classes:
            for v in self.y_classes:
                mask = (y == v) & (sensitive == u)
                vals.append(out[mask].mean())
        for v in self.y_classes:
            vals.append(out[y == v].mean())
        return torch.stack(vals)


def loss_trustfed(preds: torch.Tensor, targets: torch.Tensor, sensitive: torch.Tensor, *,
                   fairness_notion: str, alpha: float = 1.0, p_norm: int = 2,
                   sensitive_classes=(0, 1), cost_fn: float = 10.0, cost_fp: float = 5.0):
    """repo:p1 TrustFed/TrustFedLoss.py::loss_trustfed. Returns (total_loss, fairness_loss)."""
    eps = 1e-6
    preds = torch.clamp(preds, eps, 1.0 - eps)
    per_sample = nn.BCELoss(reduction="none")(preds, targets)
    weights = torch.where(targets >= 0.5,
                          torch.tensor(cost_fn, device=targets.device, dtype=targets.dtype),
                          torch.tensor(cost_fp, device=targets.device, dtype=targets.dtype))
    base_loss = (per_sample * weights).mean()

    out_logits = torch.log(preds / (1.0 - preds))  # _prob_to_logit, preds already clamped
    if fairness_notion == "SP":
        constraint = DemographicParityLoss(sensitive_classes, alpha, p_norm)
        fairness_loss = constraint(out_logits, sensitive)
    elif fairness_notion == "EO":
        constraint = EqualizedOddsLoss(sensitive_classes, alpha=alpha, p_norm=p_norm)
        fairness_loss = constraint(out_logits, sensitive, y=targets)
    else:
        raise ValueError("fairness_notion must be 'SP' or 'EO'")

    total_loss = base_loss + fairness_loss
    if not torch.isfinite(total_loss):
        total_loss = base_loss
    return total_loss, fairness_loss


def _trustfed_local_update(model: nn.Module, X, y, s, *, epochs, lr, batch_size, fairness_notion,
                            alpha, p_norm, sensitive_classes, cost_fn, cost_fp,
                            clip_norm=1.0, epsilon=3.0, delta=1.0):
    """repo:p1 TrustFed/TrustFedClient.py::train -- Adam, per-batch grad clip + DP noise."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X, y, s), batch_size=batch_size, shuffle=True)
    noise_scale = (clip_norm / epsilon) * math.sqrt(2.0 * math.log(1.25 / delta))

    epoch_losses = []
    for _ in range(epochs):
        batch_losses = []
        for batch_X, batch_y, batch_s in loader:
            optimizer.zero_grad()
            out = model(batch_X)
            total_loss, _fair = loss_trustfed(out, batch_y, batch_s, fairness_notion=fairness_notion,
                                              alpha=alpha, p_norm=p_norm, sensitive_classes=sensitive_classes,
                                              cost_fn=cost_fn, cost_fp=cost_fp)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            for p in model.parameters():
                if p.grad is None:
                    continue
                noise = torch.normal(mean=0.0, std=noise_scale, size=p.grad.shape, device=p.grad.device)
                p.grad.add_(noise)
            optimizer.step()
            batch_losses.append(total_loss.item())
        epoch_losses.append(sum(batch_losses) / len(batch_losses))

    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return state, sum(epoch_losses) / len(epoch_losses)


class TrustFed:
    family = "P1"
    notion = "SP"  # overridden by the TrustFedSP/TrustFedEO subclasses below
    harmonised_cfg = dict(rounds=50, local_epochs=1, lr=0.005, batch=128, optimizer="adam", C=1.0,
                          loss="cost-weighted BCE (FN 10x / FP 5x) + constraint",
                          dp="eps=3, delta=1.0, clip=1", alpha=1.0, p_norm=2, sensitive_classes=(0, 1))

    def __init__(self, clients: list[dict], val: dict, seed: int, cfg: dict):
        seed_all(seed)
        self.cfg = cfg
        d = clients[0]["X"].shape[1]
        self.K = len(clients)

        # Same construction-order quirk as FedAvg-P1 (see fl/federated/fedavg_p1.py docstring):
        # global model, THEN one throwaway local model per client, in order.
        self.global_model = FedAvgP1Net(d)
        self._client_models = []
        self._client_X, self._client_y, self._client_s, self._client_n = [], [], [], []
        for c in clients:
            self._client_models.append(FedAvgP1Net(d))
            self._client_X.append(torch.tensor(c["X"], dtype=torch.float32))
            self._client_y.append(torch.tensor(c["y"], dtype=torch.float32).view(-1, 1))
            self._client_s.append(torch.tensor(c["a"], dtype=torch.float32).view(-1, 1))
            self._client_n.append(len(c["y"]))
        self.val = val

    @property
    def model(self) -> nn.Module:
        return self.global_model

    def _round(self, r: int) -> None:
        gw = self.global_model.state_dict()
        reports = []
        for i in range(self.K):
            self._client_models[i].load_state_dict(gw)
            state, _loss = _trustfed_local_update(
                self._client_models[i], self._client_X[i], self._client_y[i], self._client_s[i],
                epochs=self.cfg["local_epochs"], lr=self.cfg["lr"], batch_size=self.cfg["batch"],
                fairness_notion=self.notion, alpha=self.cfg["alpha"], p_norm=self.cfg["p_norm"],
                sensitive_classes=self.cfg["sensitive_classes"],
                cost_fn=self.cfg.get("cost_fn", 10.0), cost_fp=self.cfg.get("cost_fp", 5.0),
            )
            reports.append(dict(weights=state, samples=self._client_n[i]))
        new_weights = aggregate_weighted_by_samples(reports)  # identical formula to agg_fedavg
        self.global_model.load_state_dict(new_weights)

    def fit(self, rounds: int, hook=lambda r: None) -> None:
        for r in range(rounds):
            self._round(r)
            hook(r + 1)

    def proba(self, X):
        import numpy as np

        was = self.global_model.training
        self.global_model.eval()
        with torch.no_grad():
            out = self.global_model(torch.as_tensor(np.asarray(X), dtype=torch.float32))
        self.global_model.train(was)
        return out.reshape(-1).detach().cpu().numpy().astype(np.float64)

    def get_state(self) -> dict:
        return {k: v.detach().clone() for k, v in self.global_model.state_dict().items()}

    def set_state(self, state: dict) -> None:
        self.global_model.load_state_dict(state)


class TrustFedSP(TrustFed):
    variant, notion = "TrustFed-SP", "SP"


class TrustFedEO(TrustFed):
    variant, notion = "TrustFed-EO", "EO"
