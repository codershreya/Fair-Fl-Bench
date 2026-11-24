"""FairTrade-SP / FairTrade-ATE, re-derived from reading repo:p2 code/FairTrade/
{fairtrade_federated,constraint}.py in full (no import from reference/paper2_repo). Reuses
FedAvg-P2's overall shape almost exactly -- the oracle adapter's own `FairTradeP2` class
SUBCLASSES `FedAvgP2` and overrides only `__init__`, inheriting `_round` unchanged -- plus an
extra fairness-constraint term added to the per-batch loss.

Model: `TabularMLP` (`fl/models.py`) -- repo:p2 FairTrade/fairtrade_federated.py's own
`TabularMLP` is Linear(d,64)->ReLU->Dropout(0.2)->Linear(64,1) [logits], BYTE-IDENTICAL in
layer order and shape to the one already re-derived for FedAvg-P2 in Phase 3, so it is reused
directly rather than re-defined. Aggregation: `average_weights_unweighted` (same P2 formula as
FedAvg-P2/GIFAIR-FL).

Fairness constraint terms: `fl/federated/trustfed.py::DemographicParityLoss`'s M-matrix
construction is STRUCTURALLY IDENTICAL to repo:p2 `FairTrade/constraint.py::
DemographicParityLoss` -- confirmed by comparing both repos' source line by line -- so this
module reuses the shared `_ConstraintLoss` BASE class (`forward`, which is identical in both
repos) from `trustfed.py`. Its `mu_f`, however, is NOT reused: repo:p1's TrustFed version
computes each group's conditional mean with a plain `out[sensitive == v].mean()` (NaN when a
batch has zero samples of group v -- a bug repo:p1 reproduces faithfully and the oracle wraps
unchanged, confirmed fine for TrustFed-SP in Phase 4), whereas repo:p2's FairTrade version
explicitly guards against this with its own `_safe_mean` (returns 0.0 for an empty slice
instead of NaN) -- found the hard way: reusing trustfed.py's `DemographicParityLoss` wholesale
passed every spot-check this phase (iid, low-skew attr) but diverged from the oracle under
attr-skewed small-K client partitions, where a training-batch can easily contain zero samples
of one sensitive group. FairTrade-SP therefore gets its OWN `DemographicParityLoss` here, with
the same M-matrix construction but `_safe_mean`-based `mu_f`, matching repo:p2 exactly. The
Average Treatment Effect loss (FairTrade-ATE only, no Paper-1 analogue) was already using
`_safe_mean` correctly and is unaffected; it subclasses the same shared `_ConstraintLoss` base.

RNG note (Section 5 L1 target): identical shape to FedAvg-P2's (fl/federated/fedavg_p2.py) --
ONE model construction at `__init__` (no per-client throwaways), then every round ONE
`np.random.choice(range(K), m, replace=False)` draw (unweighted, `m=K` at harmonised C=1.0 --
note this is NOT weighted by `p=`, unlike GIFAIR-FL's client sampling), then one
`DataLoader(..., shuffle=True)` draw per selected client per local epoch PLUS one
`Dropout(p=0.2)` draw per training-mode forward pass (once per batch), consumed in the order
`np.random.choice` returned. The fairness-constraint term itself (`DemographicParityLoss` /
`AverageTreatmentEffectLoss`) is pure deterministic tensor arithmetic -- building the M matrix
is `torch.zeros(...)` plus a nested Python loop over fixed class lists, no RNG at all.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fl.federated.common import average_weights_unweighted, seed_all
from fl.federated.trustfed import _ConstraintLoss
from fl.models import TabularMLP


def _safe_mean(t: torch.Tensor) -> torch.Tensor:
    """repo:p2 FairTrade/constraint.py::_safe_mean -- avoids NaN when a group is empty."""
    if t.numel() == 0:
        return torch.tensor(0.0, device=t.device)
    return t.mean()


class DemographicParityLoss(_ConstraintLoss):
    """repo:p2 FairTrade/constraint.py::DemographicParityLoss. Same M-matrix construction as
    `trustfed.py`'s version (confirmed identical by direct comparison), but its OWN `mu_f` --
    `_safe_mean`-based, unlike TrustFed's plain `.mean()` (see module docstring)."""

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

    def mu_f(self, out, sensitive, y=None):
        vals = [_safe_mean(out[sensitive == v]) for v in self.sensitive_classes]
        vals.append(_safe_mean(out))
        return torch.stack(vals)


class AverageTreatmentEffectLoss(_ConstraintLoss):
    """repo:p2 FairTrade/constraint.py::AverageTreatmentEffectLoss (FairTrade-ATE only; no
    Paper-1 analogue). Uses a single `y_classes=[1]` (only the positive outcome is considered),
    so despite the nested-loop construction mirroring EqualizedOddsLoss's shape, `y0`/`y1` are
    always equal here (there is only one y-class to iterate) -- not the same "redundant-but-
    load-bearing" quirk TrustFed-EO's M matrix has, just a genuinely single-valued loop."""

    def __init__(self, sensitive_classes=(0, 1), alpha: float = 1.0, p_norm: int = 2):
        self.sensitive_classes = list(sensitive_classes)
        self.y_classes = [1]
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
                vals.append(_safe_mean(out[mask]))
        for v in self.y_classes:
            vals.append(_safe_mean(out[y == v]))
        return torch.stack(vals)


def _fairtrade_local_update(model: nn.Module, X: torch.Tensor, y: torch.Tensor, a: torch.Tensor, *,
                             epochs: int, lr: float, batch_size: int, optimizer_name: str,
                             fairness_notion: str, fairness_lambda: float,
                             weight_decay: float = 1e-4) -> tuple[dict, float]:
    """repo:p2 FairTrade/fairtrade_federated.py::LocalUpdate.update_weights, under the oracle
    adapter's full-partition `train_val_test` override (same `FullFT` pattern as FedAvg-P2's
    `FullFed`)."""
    model.train()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"unsupported optimizer {optimizer_name!r}")
    criterion = nn.BCEWithLogitsLoss()
    if fairness_notion == "stat_parity":
        fairness_loss = DemographicParityLoss(alpha=1.0)
        needs_y = False
    elif fairness_notion == "ate":
        fairness_loss = AverageTreatmentEffectLoss(alpha=1.0)
        needs_y = True
    else:
        raise ValueError(f"unsupported fairness_notion {fairness_notion!r}")

    loader = DataLoader(TensorDataset(X, y, a), batch_size=batch_size, shuffle=True)
    epoch_losses = []
    for _ in range(epochs):
        batch_losses = []
        for batch_X, batch_y, batch_a in loader:
            optimizer.zero_grad()
            logits = model(batch_X).view(-1)
            primary_loss = criterion(logits, batch_y)
            fair = fairness_loss(logits, batch_a, y=batch_y if needs_y else None)
            loss = primary_loss + fairness_lambda * fair
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_losses.append(sum(batch_losses) / len(batch_losses))
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return state, sum(epoch_losses) / len(epoch_losses)


class FairTrade:
    family = "P2"
    notion = "stat_parity"  # overridden by FairTradeSP/FairTradeATE below
    harmonised_cfg = dict(rounds=50, local_epochs=1, lr=0.005, batch=128, optimizer="adam", C=1.0,
                          fairness_lambda=1.0, aggregation="unweighted mean (P2 code)")
    WEIGHT_DECAY = 1e-4  # hard-coded in repo:p2 LocalUpdate._make_optimizer for optimizer="adam"

    def __init__(self, clients: list[dict], val: dict, seed: int, cfg: dict):
        seed_all(seed)
        self.cfg = cfg
        self.K = len(clients)
        d = clients[0]["X"].shape[1]

        # ONE model construction (matches FedAvg-P2 -- no per-client throwaway models; local
        # models come from copy.deepcopy every round).
        self.global_model = TabularMLP(d, hidden_dim=64, dim_out=1)

        self._client_X = [torch.tensor(c["X"], dtype=torch.float32) for c in clients]
        self._client_y = [torch.tensor(c["y"], dtype=torch.float32) for c in clients]
        self._client_a = [torch.tensor(c["a"], dtype=torch.float32) for c in clients]
        self.val = val

    @property
    def model(self) -> nn.Module:
        return self.global_model

    def _round(self, r: int) -> None:
        m = max(int(self.cfg["C"] * self.K), 1)
        sel = np.random.choice(range(self.K), m, replace=False)  # repo:p2 FedAvgP2._round, inherited
        weight_dicts = []
        for u in sel:
            u = int(u)
            local_model = copy.deepcopy(self.global_model)
            state, _loss = _fairtrade_local_update(
                local_model, self._client_X[u], self._client_y[u], self._client_a[u],
                epochs=self.cfg["local_epochs"], lr=self.cfg["lr"], batch_size=self.cfg["batch"],
                optimizer_name=self.cfg["optimizer"], fairness_notion=self.notion,
                fairness_lambda=self.cfg.get("fairness_lambda", 1.0), weight_decay=self.WEIGHT_DECAY,
            )
            weight_dicts.append(state)
        self.global_model.load_state_dict(average_weights_unweighted(weight_dicts))

    def fit(self, rounds: int, hook=lambda r: None) -> None:
        for r in range(rounds):
            self._round(r)
            hook(r + 1)

    def proba(self, X):
        was = self.global_model.training
        self.global_model.eval()
        with torch.no_grad():
            logits = self.global_model(torch.as_tensor(np.asarray(X), dtype=torch.float32))
            out = torch.sigmoid(logits)
        self.global_model.train(was)
        return out.reshape(-1).detach().cpu().numpy().astype(np.float64)

    def get_state(self) -> dict:
        return {k: v.detach().clone() for k, v in self.global_model.state_dict().items()}

    def set_state(self, state: dict) -> None:
        self.global_model.load_state_dict(state)


class FairTradeSP(FairTrade):
    variant, notion = "FairTrade-SP", "stat_parity"


class FairTradeATE(FairTrade):
    variant, notion = "FairTrade-ATE", "ate"
