"""EOFedMinMax, re-derived from reading repo:p1 EOFedMinMax/{EOFedMinMaxServer,
EOFedMinMaxClient,lossStrategiesEOFedMinMax}.py (no import from reference/paper1_repo).

Same adversarial min-max shape as FedMinMax (fl/federated/fedminmax.py), but the group the
adversary tracks is the SENSITIVE ATTRIBUTE alone (not the joint (A,Y) group), and it keeps
TWO simplex variables -- `mu_y0` tracking soft-FPR risk (mean predicted P(y=1) among the
TRUE-NEGATIVE rows of each group: a "proxy" for FPR using *probabilities*, not hard
predictions) and `mu_y1` tracking soft-FNR risk (mean 1-P(y=1) among true-positive rows) --
and a single adaptive scalar `lambda_eo` that weights the fairness penalty against the
classification loss. `lambda_eo` is NOT part of the harmonised config: the server recomputes
it every round via proportional control against a target EO of ~0, using the ORIGINAL repo's
own (NaN-free, 0.0-for-undefined) `compute_equalized_odds` formula evaluated on the VALIDATION
set with the model's HARD (thresholded) predictions -- re-derived here as `_compute_eo_p1`,
kept deliberately separate from `fl/metrics.py` (which intentionally returns NaN for the same
undefined cases, per Section 3/D15): the NaN-vs-0.0 choice is a real difference in behaviour
(it changes what `lambda_eo` becomes when a group is briefly empty on a tiny validation slice),
not a detail we get to normalise away, since matching it is what the bit-exact comparison in
this phase is actually testing.

RNG note: identical shape to FedMinMax -- one throwaway per-client model at construction, one
(numerically inert, since batch_size==N) global-RNG draw per client per round for the shuffle.
`_update_lambda_from_val`, the simplex projections, and the risk aggregation are all pure
arithmetic forward passes; none of them touch the RNG.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fl.federated.common import aggregate_weighted_by_samples, project_simplex, seed_all
from fl.models import FedMinMaxNet


def _get_rates_p1(preds: torch.Tensor, labels, sensitive: torch.Tensor, s_value: int):
    """repo:p1 fairnessMetrics.py::get_rates -- 0.0 (never NaN) for an empty/undefined group;
    used ONLY for EOFedMinMax's own internal lambda-adaptation controller, never for this
    project's reported metrics (those always go through fl/metrics.py)."""
    mask = sensitive == int(s_value)
    if mask.sum() == 0:
        return 0.0, 0.0, 0.0
    g_preds = preds[mask]
    ppr = g_preds.float().mean().item()
    if labels is None:
        return 0.0, 0.0, ppr
    g_labels = labels[mask]
    mask_pos = g_labels == 1
    tpr = g_preds[mask_pos].float().mean().item() if mask_pos.sum() > 0 else 0.0
    mask_neg = g_labels == 0
    fpr = g_preds[mask_neg].float().mean().item() if mask_neg.sum() > 0 else 0.0
    return tpr, fpr, ppr


def _compute_eo_p1(preds: torch.Tensor, labels: torch.Tensor, sensitive: torch.Tensor) -> float:
    """repo:p1 fairnessMetrics.py::compute_equalized_odds."""
    groups = torch.unique(sensitive).sort()[0]
    if len(groups) < 2:
        return 0.0
    a, b = groups[0].item(), groups[1].item()
    tpr_a, fpr_a, _ = _get_rates_p1(preds, labels, sensitive, a)
    tpr_b, fpr_b, _ = _get_rates_p1(preds, labels, sensitive, b)
    return 0.5 * (abs(tpr_b - tpr_a) + abs(fpr_b - fpr_a))


def _evaluate_group_eo_stats(model: nn.Module, X: torch.Tensor, y: torch.Tensor, s: torch.Tensor):
    """repo:p1 EOFedMinMax/EOFedMinMaxClient.py::evaluate_group_eo_stats."""
    model.eval()
    s_flat, y_flat = s.view(-1).long(), y.view(-1).float()
    group_fpr, group_fnr, count_y0, count_y1 = {}, {}, {}, {}
    with torch.no_grad():
        p = torch.sigmoid(model(X).view(-1).float())
        for gid_t in torch.unique(s_flat):
            gid = int(gid_t.item())
            mask_g = s_flat == gid
            if not mask_g.any():
                continue
            m0 = mask_g & (y_flat < 0.5)
            n0 = int(m0.sum().item())
            count_y0[gid] = n0
            if n0 > 0:
                group_fpr[gid] = float(p[m0].mean().item())
            m1 = mask_g & (y_flat >= 0.5)
            n1 = int(m1.sum().item())
            count_y1[gid] = n1
            if n1 > 0:
                group_fnr[gid] = float((1.0 - p[m1].mean()).item())
    return group_fpr, group_fnr, count_y0, count_y1


def _loss_eofedminmax(outputs, targets, sensitive, *, group_weights_y0: dict, group_weights_y1: dict,
                       lambda_eo: float) -> torch.Tensor:
    """repo:p1 EOFedMinMax/lossStrategiesEOFedMinMax.py::loss_EOfedminmax."""
    s = sensitive.view(-1).long()
    y = targets.view(-1).float()
    logits = outputs.view(-1).float()
    utility_loss = nn.BCEWithLogitsLoss(reduction="mean")(logits, y)
    if lambda_eo <= 0.0:
        return utility_loss
    probs = torch.sigmoid(logits)
    group_keys = {int(k) for k in group_weights_y0} | {int(k) for k in group_weights_y1}
    if not group_keys:
        return utility_loss
    fairness_fpr = torch.tensor(0.0)
    fairness_fnr = torch.tensor(0.0)
    for gid in group_keys:
        mask_g = s == gid
        if not mask_g.any():
            continue
        mask0 = mask_g & (y < 0.5)
        if mask0.any():
            w0 = float(group_weights_y0.get(gid, 0.0))
            if w0 != 0.0:
                fairness_fpr = fairness_fpr + w0 * probs[mask0].mean()
        mask1 = mask_g & (y >= 0.5)
        if mask1.any():
            w1 = float(group_weights_y1.get(gid, 0.0))
            if w1 != 0.0:
                fairness_fnr = fairness_fnr + w1 * (1.0 - probs[mask1].mean())
    return utility_loss + lambda_eo * (fairness_fpr + fairness_fnr)


def _eofedminmax_local_update(model, X, y, s, *, epochs, lr, group_weights_y0, group_weights_y1, lambda_eo):
    """repo:p1 EOFedMinMax/EOFedMinMaxClient.py::train -- full-batch SGD."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X, y, s), batch_size=len(X), shuffle=True)
    epoch_losses = []
    for _ in range(epochs):
        batch_losses = []
        for batch_X, batch_y, batch_s in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = _loss_eofedminmax(outputs, batch_y, batch_s, group_weights_y0=group_weights_y0,
                                      group_weights_y1=group_weights_y1, lambda_eo=lambda_eo)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_losses.append(sum(batch_losses) / len(batch_losses))
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return state, sum(epoch_losses) / len(epoch_losses)


class EOFedMinMax:
    family, variant = "P1", "EOFedMinMax"
    outputs_logits = True
    harmonised_cfg = dict(rounds=50, local_epochs=1, lr=0.005, optimizer="sgd", C=1.0,
                          batch="full-batch", loss="BCE-with-logits + FPR/FNR minimax fairness loss",
                          adversary_lr=0.05, lambda0=3.0, lambda_max=5.0, lambda_lr=1.5,
                          eo_target=1e-5, final_model="average of round iterates")

    def __init__(self, clients: list[dict], val: dict, seed: int, cfg: dict):
        seed_all(seed)
        self.cfg = cfg
        d = clients[0]["X"].shape[1]
        self.K = len(clients)

        self.global_model = FedMinMaxNet(d)  # RNG draw #1

        self._client_models, self._client_X, self._client_y, self._client_s, self._client_n = [], [], [], [], []
        counts: dict[int, dict] = {}
        for c in clients:
            y_t = torch.tensor(c["y"], dtype=torch.float32).view(-1, 1)
            s_t = torch.tensor(c["a"], dtype=torch.float32).long().view(-1, 1)
            self._client_models.append(FedMinMaxNet(d))  # RNG draw per client, in order
            self._client_X.append(torch.tensor(c["X"], dtype=torch.float32))
            self._client_y.append(y_t)
            self._client_s.append(s_t)
            self._client_n.append(len(c["y"]))
            s_flat, y_flat = s_t.view(-1), y_t.view(-1).long()
            for gid in torch.unique(s_flat).tolist():
                m = s_flat == gid
                e = counts.setdefault(int(gid), {"y0": 0, "y1": 0})
                e["y0"] += int(((y_flat == 0) & m).sum())
                e["y1"] += int(((y_flat == 1) & m).sum())

        real_y0 = {g: v["y0"] for g, v in counts.items()}
        real_y1 = {g: v["y1"] for g, v in counts.items()}
        self.real_count_y0, self.real_count_y1 = real_y0, real_y1
        self.group_ids = sorted(set(real_y0) | set(real_y1))
        self.gid_to_idx = {g: i for i, g in enumerate(self.group_ids)}
        total0, total1 = sum(real_y0.values()), sum(real_y1.values())
        rho0 = [real_y0.get(g, 0) / (total0 + 1e-12) for g in self.group_ids]
        rho1 = [real_y1.get(g, 0) / (total1 + 1e-12) for g in self.group_ids]
        self.rho_y0 = torch.tensor(rho0, dtype=torch.float32)
        self.rho_y1 = torch.tensor(rho1, dtype=torch.float32)
        self.mu_y0 = self.rho_y0.clone()
        self.mu_y1 = self.rho_y1.clone()
        self.lr_mu = 0.05
        self.global_lambda = cfg.get("lambda0", 3.0)
        self.lambda_lr = cfg.get("lambda_lr", 1.5)
        self.lambda_max = cfg.get("lambda_max", 5.0)
        self.eo_target = cfg.get("eo_target", 1e-5)

        self._avg_state = None
        self._avg_t = 0
        self.val = val
        self._val_X = torch.tensor(val["X"], dtype=torch.float32)
        self._val_y = torch.tensor(val["y"], dtype=torch.float32).view(-1, 1)
        self._val_s = torch.tensor(val["a"], dtype=torch.float32).long().view(-1, 1)

    @property
    def model(self) -> nn.Module:
        return self.global_model

    def _initialize_weights(self):
        w0 = self.mu_y0 / (self.rho_y0 + 1e-10)
        w1 = self.mu_y1 / (self.rho_y1 + 1e-10)
        w0_dict = {g: float(w0[self.gid_to_idx[g]].item()) for g in self.group_ids}
        w1_dict = {g: float(w1[self.gid_to_idx[g]].item()) for g in self.group_ids}
        return w0_dict, w1_dict

    def _update_running_average(self, new_state: dict) -> None:
        if self._avg_state is None:
            self._avg_state = {k: v.clone() for k, v in new_state.items()}
            self._avg_t = 1
            return
        self._avg_t += 1
        t = self._avg_t
        for k, v in new_state.items():
            self._avg_state[k].add_((v - self._avg_state[k]) / t)

    def _update_lambda_from_val(self) -> None:
        """repo:p1 EOFedMinMaxServer.py::_update_lambda_from_val -- proportional control
        against the ORIGINAL repo's own NaN-free EO formula, evaluated post-aggregation."""
        self.global_model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self.global_model(self._val_X))
            preds = (probs > 0.5).float()
        eo = _compute_eo_p1(preds.view(-1), self._val_y.view(-1), self._val_s.view(-1))
        err = eo - self.eo_target
        new_lambda = self.global_lambda + self.lambda_lr * err
        self.global_lambda = float(min(max(new_lambda, 0.001), self.lambda_max))

    def _round(self, r: int) -> None:
        w0_dict, w1_dict = self._initialize_weights()
        lambda_eo = self.global_lambda
        gw = self.global_model.state_dict()
        reports = []
        for i in range(self.K):
            self._client_models[i].load_state_dict(gw)
            group_fpr, group_fnr, count_y0, count_y1 = _evaluate_group_eo_stats(
                self._client_models[i], self._client_X[i], self._client_y[i], self._client_s[i])
            state, _loss = _eofedminmax_local_update(
                self._client_models[i], self._client_X[i], self._client_y[i], self._client_s[i],
                epochs=self.cfg["local_epochs"], lr=self.cfg["lr"],
                group_weights_y0=w0_dict, group_weights_y1=w1_dict, lambda_eo=lambda_eo,
            )
            reports.append(dict(weights=state, samples=self._client_n[i], group_fpr=group_fpr,
                                group_fnr=group_fnr, count_y0=count_y0, count_y1=count_y1))
        new_weights = aggregate_weighted_by_samples(reports)
        self.global_model.load_state_dict(new_weights)
        self._update_running_average(new_weights)

        fpr_vec = torch.zeros(len(self.group_ids), dtype=torch.float32)
        fnr_vec = torch.zeros(len(self.group_ids), dtype=torch.float32)
        for idx, gid in enumerate(self.group_ids):
            num0 = sum(float(rep["group_fpr"][gid]) * int(rep["count_y0"][gid])
                      for rep in reports if gid in rep["group_fpr"] and rep["count_y0"].get(gid, 0) > 0)
            den0 = int(self.real_count_y0.get(gid, 0))
            if den0 > 0:
                fpr_vec[idx] = num0 / den0
            num1 = sum(float(rep["group_fnr"][gid]) * int(rep["count_y1"][gid])
                      for rep in reports if gid in rep["group_fnr"] and rep["count_y1"].get(gid, 0) > 0)
            den1 = int(self.real_count_y1.get(gid, 0))
            if den1 > 0:
                fnr_vec[idx] = num1 / den1
        self.mu_y0 = project_simplex(self.mu_y0 + self.lr_mu * fpr_vec, eps=1e-3)
        self.mu_y1 = project_simplex(self.mu_y1 + self.lr_mu * fnr_vec, eps=1e-3)
        self._update_lambda_from_val()

    def fit(self, rounds: int, hook=lambda r: None) -> None:
        for r in range(rounds):
            self._round(r)
            hook(r + 1)

    def proba(self, X):
        import numpy as np

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

    def native_final_state(self) -> dict:
        if self._avg_state is None:
            return self.get_state()
        return {k: v.detach().clone() for k, v in self._avg_state.items()}
