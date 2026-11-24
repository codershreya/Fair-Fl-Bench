"""FedMinMax, re-derived from reading repo:p1 OrigFedMinMax/{OrigFedMinMaxServer,
OrigFedMinMaxClient,OriglossStrategiesFedMinMax}.py (no import from reference/paper1_repo).

Adversarial group-reweighting FedAvg: the server keeps a probability vector `mu` over the
JOINT (sensitive x label) groups (4 groups for a binary sensitive attribute: a*2+y), initialised
to the groups' true global frequency `rho`. Each round, EVERY client first evaluates its own
per-group loss ("risk") on the model it is ABOUT TO train further (i.e. the current global
weights, before this round's local update), then trains with a loss re-weighted by
`mu / (rho + eps)` per joint group (so an over-represented-but-high-risk group gets up-weighted),
using full-batch SGD (one gradient step per client per round -- `batch_size == len(client data)`).
After FedAvg-averaging the client updates, the server does one step of PROJECTED GRADIENT ASCENT
on `mu` using the clients' pre-training risks, weighted by client group size -- an adversarial
min-max game between the model (minimising weighted loss) and `mu` (maximising it). The actual
benchmark result is the RUNNING AVERAGE of every round's post-aggregation global model, not the
last round's weights (`native_final_state`) -- the averaged iterate is the theoretically
justified solution to the min-max game, per the original paper.

RNG note (Section 5 L1 target): same pattern as FedAvg-P1/TrustFed -- one throwaway per-client
model at construction (repo:p1 `OrigFedMinMaxClient.__init__`), then one global-RNG draw per
client per round from the (here, trivial since batch_size==N) `shuffle=True` DataLoader. Every
other step (risk evaluation, SGD, the simplex projection) is pure arithmetic.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fl.federated.common import aggregate_weighted_by_samples, project_simplex, seed_all
from fl.models import FedMinMaxNet


def _evaluate_group_risks(model: nn.Module, X: torch.Tensor, y: torch.Tensor, s_joint: torch.Tensor):
    """repo:p1 OrigFedMinMax/OrigFedMinMaxClient.py::evaluate_group_risks."""
    model.eval()
    crit = nn.BCEWithLogitsLoss(reduction="sum")
    risks, counts = {}, {}
    with torch.no_grad():
        outputs = model(X)
        for gid_t in torch.unique(s_joint):
            gid = int(gid_t.item())
            mask = (s_joint == gid).squeeze(-1)
            if mask.sum() == 0:
                continue
            loss = crit(outputs[mask], y[mask])
            count = int(mask.sum().item())
            risks[gid] = (loss / count).item()
            counts[gid] = count
    return risks, counts


def _loss_fedminmax(outputs: torch.Tensor, targets: torch.Tensor, sensitive_joint: torch.Tensor,
                     group_weights: dict) -> torch.Tensor:
    """repo:p1 OrigFedMinMax/OriglossStrategiesFedMinMax.py::loss_Origfedminmax."""
    crit = nn.BCEWithLogitsLoss(reduction="none")
    s = sensitive_joint.view(-1).long()
    y = targets.view(-1)
    o = outputs.view(-1)
    loss_per_sample = crit(o, y)
    sample_weights = torch.ones_like(loss_per_sample)
    for gid, w in group_weights.items():
        mask = s == int(gid)
        if mask.any():
            sample_weights[mask] = float(w)
    return (loss_per_sample * sample_weights).mean()


def _fedminmax_local_update(model: nn.Module, X, y, s_joint, *, epochs: int, lr: float, group_weights: dict):
    """repo:p1 OrigFedMinMax/OrigFedMinMaxClient.py::train -- full-batch SGD."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X, y, s_joint), batch_size=len(X), shuffle=True)
    epoch_losses = []
    for _ in range(epochs):
        batch_losses = []
        for batch_X, batch_y, batch_s in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = _loss_fedminmax(outputs, batch_y, batch_s, group_weights)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_losses.append(sum(batch_losses) / len(batch_losses))
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return state, sum(epoch_losses) / len(epoch_losses)


class FedMinMax:
    family, variant = "P1", "FedMinMax"
    outputs_logits = True
    harmonised_cfg = dict(rounds=50, local_epochs=1, lr=0.005, optimizer="sgd", C=1.0,
                          batch="full-batch (1 gradient step / round / client)",
                          loss="BCE-with-logits, group-weighted", adversary_lr=0.05,
                          final_model="average of round iterates")

    def __init__(self, clients: list[dict], val: dict, seed: int, cfg: dict):
        seed_all(seed)
        self.cfg = cfg
        d = clients[0]["X"].shape[1]
        self.K = len(clients)

        self.global_model = FedMinMaxNet(d)  # RNG draw #1 (repo:p1 OrigFedMinMaxServer.__init__)

        self._client_models, self._client_X, self._client_y, self._client_s, self._client_n = [], [], [], [], []
        counts: dict[int, int] = {}
        for c in clients:
            y_t = torch.tensor(c["y"], dtype=torch.float32).view(-1, 1)
            s_raw_t = torch.tensor(c["a"], dtype=torch.float32).view(-1, 1)
            s_joint = (s_raw_t * 2 + y_t).long()  # joint (A,Y) group id, as in repo:p1's "s"
            self._client_models.append(FedMinMaxNet(d))  # RNG draw per client, in order
            self._client_X.append(torch.tensor(c["X"], dtype=torch.float32))
            self._client_y.append(y_t)
            self._client_s.append(s_joint)
            self._client_n.append(len(c["y"]))
            u, cnt = torch.unique(s_joint, return_counts=True)
            for g, n in zip(u.tolist(), cnt.tolist()):
                counts[int(g)] = counts.get(int(g), 0) + int(n)

        self.group_ids = sorted(counts.keys())
        self.gid_to_idx = {g: i for i, g in enumerate(self.group_ids)}
        total = sum(counts.values())
        rho_list = [counts[g] / total for g in self.group_ids]
        self.rho = torch.tensor(rho_list, dtype=torch.float32)
        self.mu = torch.tensor(rho_list, dtype=torch.float32)  # init mu = rho
        self.lr_mu = 0.05
        self.real_group_counts = counts
        self._avg_state = None
        self._avg_t = 0
        self.val = val

    @property
    def model(self) -> nn.Module:
        return self.global_model

    def _initialize_weights(self) -> dict:
        w_vector = self.mu / (self.rho + 1e-10)
        return {g: w_vector[self.gid_to_idx[g]].item() for g in self.group_ids}

    def _update_running_average(self, new_state: dict) -> None:
        if self._avg_state is None:
            self._avg_state = {k: v.clone() for k, v in new_state.items()}
            self._avg_t = 1
            return
        self._avg_t += 1
        t = self._avg_t
        for k, v in new_state.items():
            self._avg_state[k].add_((v - self._avg_state[k]) / t)

    def _round(self, r: int) -> None:
        group_weights = self._initialize_weights()
        gw = self.global_model.state_dict()
        reports = []
        for i in range(self.K):
            self._client_models[i].load_state_dict(gw)
            risks, counts = _evaluate_group_risks(self._client_models[i], self._client_X[i],
                                                   self._client_y[i], self._client_s[i])
            state, _loss = _fedminmax_local_update(
                self._client_models[i], self._client_X[i], self._client_y[i], self._client_s[i],
                epochs=self.cfg["local_epochs"], lr=self.cfg["lr"], group_weights=group_weights,
            )
            reports.append(dict(weights=state, samples=self._client_n[i], group_risks=risks, group_counts=counts))
        new_weights = aggregate_weighted_by_samples(reports)
        self.global_model.load_state_dict(new_weights)
        self._update_running_average(new_weights)

        risk_vector = torch.zeros(len(self.group_ids), dtype=torch.float32)
        for idx, gid in enumerate(self.group_ids):
            numerator = 0.0
            for rep in reports:
                cr, cc = rep["group_risks"], rep["group_counts"]
                if gid in cr and gid in cc and cc[gid] > 0:
                    numerator += float(cr[gid]) * int(cc[gid])
            denom = int(self.real_group_counts.get(gid, 0))
            if denom > 0:
                risk_vector[idx] = numerator / denom
        self.mu = project_simplex(self.mu + self.lr_mu * risk_vector, eps=1e-3)

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
        """The benchmark-reported model for FedMinMax: the running average of every round's
        post-aggregation global weights, not the last round's weights alone (repo:p1
        OrigFedMinMaxServer.py::get_averaged_state)."""
        if self._avg_state is None:
            return self.get_state()
        return {k: v.detach().clone() for k, v in self._avg_state.items()}
