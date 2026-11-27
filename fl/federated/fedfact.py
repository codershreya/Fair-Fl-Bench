"""FedFACT-In and FedFACT-Post (Zhang et al., NeurIPS 2025, Algorithms 1 and 2; Post is described
at the class, further down), re-derived from reading the released code
(github.com/liizhang/FedFACT @ 26e72f7: fedlearn/algorithm/{FedBase,FedFACT_In}.py,
fedlearn/models/{client,CGFFedclient,CGFFedclient_Post,models}.py, FedFACT_Post.py) in full. No code is imported from that repository.
Scope, configuration and every code-vs-paper difference: DEVIATIONS.md D13-D16. Only the setting
the benchmark runs is implemented: binary classification, binary A, Demographic Parity.

Per round (`FedFACTIn._round`), in the code's exact order:
  1. Client order: if r > 0, `np.random.seed(r)`; then `np.random.choice(K, K, replace=False)`
     (the code's `select_clients` re-seeds the global NumPy RNG with the round index; with
     C = 1 this only fixes the order clients are processed in).
  2. Each client, in that order (`_client_round`):
     a. Calibration matrices M(a) from the CURRENT (lambda, mu_k) (`_calibration_matrices`).
     b. Ensemble weight w_k <- w_k e^{-eta_w L(theta)} / (w_k e^{-eta_w L(theta)} + (1-w_k) e^{-eta_w L(phi_k)}),
        L = mean calibrated loss on the client's training data (two passes, shuffle=False).
     c. Local DP of the ENSEMBLE w_k softmax(theta) + (1-w_k) softmax(phi_k) on the client's
        training data (one pass of the client's SHUFFLED train loader), then
        mu_k <- clamp(mu_k + eta_d [dp - xi_l, -dp - xi_l], 0, bound). The ensemble's confusion
        counts are kept for the server's lambda step.
     d. 40 mini-batch steps of the personal model phi_k, then 40 steps of the client's copy of
        theta, both on the calibrated loss (SUM over the batch, not the mean) with M from step a,
        both reading the same persistent batch iterator.
  3. Server: unweighted mean of the client thetas, then theta <- (1-0.95) theta + 0.95 mean.
  4. Server: global DP from the pooled ensemble confusion counts,
     lambda <- clamp(lambda + eta_d [dp - xi_g, -dp - xi_g], 0, bound).
The evaluated model is the global theta (D13).

RNG: everything that draws from the global torch RNG in the code is a DataLoader iterator being
created (one int64 for its base seed) or a shuffled sampler starting (one more int64). Real
`torch.utils.data.DataLoader` objects are therefore used at exactly the points the code uses
them, so the draws happen in the same order: model init; one iterator per client at
construction; then per client per round two unshuffled loss passes, one shuffled evaluation
pass, and a fresh shuffled iterator whenever the persistent training iterator runs out.

Numerics are kept identical to the code, including its dtype mix: M(a) is float64 (its global
part is built from float64 probabilities), the loss multiplies float32 log-probabilities by float64
rows of M, the duals and the ensemble weight are float32 tensors, and DP is computed from int64
counts by true division (float32).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fl.federated.common import seed_all


class FedFACTLogistic(nn.Module):
    """The code's `Logistic` model with 2 outputs (`--model logistic --criterion multiclass`):
    a single Linear layer returning logits; 1-D inputs are treated as one row."""

    def __init__(self, d: int, n_class: int = 2):
        super().__init__()
        self.layer = nn.Linear(d, n_class)

    def forward(self, x):
        if x.dim() == 1:
            x = x.view(1, x.shape[0])
        x = x.view(-1, int(np.prod(x.shape[1:])))
        return self.layer(x)

    def get_flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self.parameters()]).clone().detach()

    def set_flat_params(self, flat: torch.Tensor) -> None:
        i = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat[i:i + n].view(p.size()))
            i += n


def _dataset(X, y, a) -> TensorDataset:
    """Same batches the code's `Fair_Dataset` collates to: X float32 (B, d), Y and A float32 (B, 1)."""
    return TensorDataset(torch.as_tensor(np.asarray(X, dtype=np.float32)),
                         torch.as_tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1)),
                         torch.as_tensor(np.asarray(a, dtype=np.float32).reshape(-1, 1)))


def _calibrated_rows(M_tensor: torch.Tensor, y: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Row y_i of M(a_i) for every sample: (n, m), float64."""
    n = y.shape[0]
    M_sel = M_tensor[a.view(-1).long()]
    return M_sel[torch.arange(n), y.view(-1).long()]


def _dp_counts(predicted: torch.Tensor, y: torch.Tensor, a: torch.Tensor, cm: dict, n_a: dict) -> None:
    """Accumulate the code's confusion counts cm[a][y][pred] and group sizes n_a (int64 tensors)."""
    for at in (0, 1):
        n_a[at] = n_a[at] + torch.sum(a == at)
        for yt in (0, 1):
            for pt in (0, 1):
                cm[at][yt][pt] = cm[at][yt][pt] + torch.sum((predicted.unsqueeze(1) == pt) * (y == yt) * (a == at))


def _signed_dp(cm: dict, n_a: dict):
    """P(yhat=1 | A=1) - P(yhat=1 | A=0), written exactly as the code does (float32)."""
    return - (cm[0][0][1] + cm[0][1][1]) / n_a[0] + (cm[1][0][1] + cm[1][1][1]) / n_a[1]


def _calibration_matrices(p_a_k: dict, p_k_given_a: dict, lamb: torch.Tensor, mu: torch.Tensor,
                          positive_shift: bool) -> dict:
    """The code's `get_cal_Matrix`, binary DP branch:
        M(a) = I - (1/p_{a,k}) (lambda_0 - lambda_1) D_g(a) - (1/p_{a,k}) (mu_0 - mu_1) D_l(a),
        D_g(a) = (2a-1) p_{k|a} [[0,1],[0,1]] (float64),  D_l(a) = (2a-1) [[0,1],[0,1]] (int64).
    The in-processing client then adds a constant so every entry is positive ("positive
    calibration"); the post-processing client's copy of this function does NOT (positive_shift=False).
    Written with the code's exact operations and operand types so results are bit-identical."""
    D_g = {g: (2 * g - 1) * torch.tensor([[0, p_k_given_a[g]], [0, p_k_given_a[g]]], dtype=torch.float64)
           for g in (0, 1)}
    D_l = {g: (2 * g - 1) * torch.tensor([[0, 1], [0, 1]]) for g in (0, 1)}
    M = {}
    for g in (0, 1):
        M[g] = (torch.eye(2, 2) - 1 / float(p_a_k[g]) * (lamb[0] - lamb[1]) * D_g[g]
                - 1 / float(p_a_k[g]) * (mu[0] - mu[1]) * D_l[g])
    if positive_shift:
        global_min = min([m.min().item() for m in M.values()])
        if global_min < 0:
            shift = -global_min + 0.001
            M = {g: m + shift for g, m in M.items()}
    return M


class _Client:
    """One FedFACT-In client: its data, models, optimisers, persistent batch iterator and duals."""

    def __init__(self, X, y, a, model: FedFACTLogistic, cfg: dict):
        self.data = _dataset(X, y, a)
        self.n = len(self.data)
        self.model = model                                   # the client's working copy of theta
        self.train_loader = DataLoader(self.data, batch_size=cfg["batch"], shuffle=True)
        self.train_iter = enumerate(self.train_loader)       # created at construction, as in the code
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        self.local_mu = torch.tensor([cfg["dual_init"], cfg["dual_init"]], requires_grad=False)
        self.ensemble_weight_global = 0.5
        self.personal_model = copy.deepcopy(model)
        self.personal_optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=cfg["lr"],
                                                   weight_decay=cfg["weight_decay"])
        self.confusion = None

    def next_batch(self):
        try:
            _, batch = self.train_iter.__next__()
        except StopIteration:
            self.train_iter = enumerate(self.train_loader)
            _, batch = self.train_iter.__next__()
        return batch


class FedFACTIn:
    family = "FedFACT"
    variant = "FedFACT-In"
    harmonised_cfg = dict(rounds=50, local_steps=40, batch=1024, lr=0.008, optimizer="adam", weight_decay=1e-4, C=1.0,
                          model="logistic (2 logits, softmax CE)", aggregation="unweighted mean + server momentum 0.95",
                          fairness_constraints="DP", global_constraints=0.01, local_constraints=0.01,
                          ensemble_params_lr=0.3, dual_params_lr=0.03, dual_params_bound=10.0, evaluated_model="global theta")
    SERVER_MOMENTUM = 0.95       # code default `--beta`
    DUAL_INIT = 0.1              # FedFACT_In / CGFFedClient initial lambda and mu (binary DP)
    EVAL_BATCH = 1024

    def __init__(self, clients: list[dict], val: dict, seed: int, cfg: dict):
        if cfg.get("fairness_constraints", "DP") != "DP":
            raise NotImplementedError("only the binary Demographic Parity constraint is implemented (DEVIATIONS.md D13)")
        seed_all(seed)
        self.cfg = cfg
        self.K = len(clients)
        d = clients[0]["X"].shape[1]
        ccfg = dict(batch=cfg["batch"], lr=cfg["lr"], weight_decay=cfg["weight_decay"], dual_init=self.DUAL_INIT)

        # statistics the code takes from `get_data_info` (training data only is used by FedFACT-In)
        a_all = [np.asarray(c["a"]).reshape(-1) for c in clients]
        self.train_num = int(sum(len(a) for a in a_all))
        n_a_total = {g: int(sum(int((a == g).sum()) for a in a_all)) for g in (0, 1)}
        self.p_a_k = [{g: int((a == g).sum()) / self.train_num for g in (0, 1)} for a in a_all]
        self.p_k_given_a = [{g: int((a == g).sum()) / n_a_total[g] for g in (0, 1)} for a in a_all]

        self.model = FedFACTLogistic(d)                      # model init: first torch RNG draws
        self.latest = self.model.get_flat_params()
        self.clients = [_Client(c["X"], c["y"], c["a"], copy.deepcopy(self.model), ccfg) for c in clients]
        self.global_lamb = torch.tensor([self.DUAL_INIT, self.DUAL_INIT], requires_grad=False)
        self._eval_model = copy.deepcopy(self.model)

    # ---- calibration matrices (binary DP) ---------------------------------------------------------
    def _calibration_matrices(self, k: int, lamb: torch.Tensor, mu: torch.Tensor) -> dict:
        return _calibration_matrices(self.p_a_k[k], self.p_k_given_a[k], lamb, mu, positive_shift=True)

    # ---- one client's local work -----------------------------------------------------------------
    def _calibrated_loss(self, c: _Client, model: nn.Module, M_tensor: torch.Tensor) -> float:
        total, num = 0, 0
        model.eval()
        with torch.no_grad():
            for x, y, a in DataLoader(c.data, batch_size=self.cfg["batch"], shuffle=False):
                outputs = -F.log_softmax(model(x), dim=1)
                rows = _calibrated_rows(M_tensor, y, a)
                total += torch.sum(outputs * rows, dim=1).sum().detach().cpu().item()
                num += outputs.shape[0]
        return total / num

    def _ensemble_dp(self, c: _Client):
        """One pass over the client's shuffled train loader with the ensemble; returns (signed DP, counts)."""
        cm = {at: {yt: {pt: 0 for pt in (0, 1)} for yt in (0, 1)} for at in (0, 1)}
        n_a = {0: 0, 1: 0}
        c.model.eval()
        c.personal_model.eval()
        w = c.ensemble_weight_global
        with torch.no_grad():
            for x, y, a in c.train_loader:
                y, a = y.reshape(-1, 1), a.reshape(-1, 1)
                pred = F.softmax(c.model(x), dim=1) * w + (1 - w) * F.softmax(c.personal_model(x), dim=1)
                _, predicted = torch.max(pred, 1)
                _dp_counts(predicted, y, a, cm, n_a)
        return _signed_dp(cm, n_a), cm

    def _local_steps(self, c: _Client, model: nn.Module, optimizer, M_tensor: torch.Tensor) -> None:
        for _ in range(self.cfg["local_steps"]):
            model.train()
            x, y, a = c.next_batch()
            x = torch.squeeze(x)
            optimizer.zero_grad()
            outputs = -F.log_softmax(model(x), dim=1)
            loss = torch.sum(outputs * _calibrated_rows(M_tensor, y, a))
            loss.backward()
            optimizer.step()

    def _client_round(self, k: int) -> torch.Tensor:
        c, cfg = self.clients[k], self.cfg
        c.model.set_flat_params(self.latest)
        lamb = self.global_lamb
        M = self._calibration_matrices(k, lamb, c.local_mu)
        M_tensor = torch.stack([M[g] for g in (0, 1)], dim=0)

        lg = self._calibrated_loss(c, c.model, M_tensor)
        ll = self._calibrated_loss(c, c.personal_model, M_tensor)
        eta_w = cfg["ensemble_params_lr"]
        w1 = c.ensemble_weight_global * torch.exp(-torch.tensor(eta_w * lg))
        w2 = (1 - c.ensemble_weight_global) * torch.exp(-torch.tensor(eta_w * ll))
        c.ensemble_weight_global = w1 / (w1 + w2)

        dp, c.confusion = self._ensemble_dp(c)
        c.local_mu[0] += cfg["dual_params_lr"] * (dp - cfg["local_constraints"])
        c.local_mu[1] += cfg["dual_params_lr"] * (-dp - cfg["local_constraints"])
        c.local_mu = torch.clamp(c.local_mu, min=0.0, max=cfg["dual_params_bound"])

        c.model.eval()
        self._local_steps(c, c.personal_model, c.personal_optimizer, M_tensor)
        c.personal_model.eval()
        self._local_steps(c, c.model, c.optimizer, M_tensor)
        return c.model.get_flat_params()

    # ---- server ----------------------------------------------------------------------------------
    def _round(self, r: int) -> None:
        if r:
            np.random.seed(r)
        order = np.random.choice(self.K, self.K, replace=False)
        solns, confusions = [], []
        for k in order:
            solns.append(self._client_round(int(k)))
            confusions.append(self.clients[int(k)].confusion)

        avg = torch.zeros_like(self.latest)
        for s in solns:
            avg += s
        avg /= len(solns)
        self.latest = ((1 - self.SERVER_MOMENTUM) * self.latest + self.SERVER_MOMENTUM * avg).detach()

        tp1 = tp0 = n1 = n0 = 0
        for cm in confusions:
            tp1 += sum(v for y_d in cm[1].values() for p, v in y_d.items() if p == 1)
            tp0 += sum(v for y_d in cm[0].values() for p, v in y_d.items() if p == 1)
            n1 += sum(v for y_d in cm[1].values() for v in y_d.values())
            n0 += sum(v for y_d in cm[0].values() for v in y_d.values())
        dp = tp1 / n1 - tp0 / n0
        self.global_lamb[0] += self.cfg["dual_params_lr"] * (dp - self.cfg["global_constraints"])
        self.global_lamb[1] += self.cfg["dual_params_lr"] * (-dp - self.cfg["global_constraints"])
        self.global_lamb = torch.clamp(self.global_lamb, min=0.0, max=self.cfg["dual_params_bound"])

    def fit(self, rounds: int, hook=lambda r: None) -> None:
        for r in range(rounds):
            self._round(r)
            hook(r + 1)

    # ---- evaluation / state ----------------------------------------------------------------------
    @property
    def model_for_eval(self) -> nn.Module:
        return self._eval_model

    def _eta(self, X) -> torch.Tensor:
        m = self._eval_model
        m.set_flat_params(self.latest)
        m.eval()
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        out = []
        with torch.no_grad():
            for i in range(0, len(X), self.EVAL_BATCH):
                out.append(F.softmax(m(X[i:i + self.EVAL_BATCH]).detach().clone().cpu(), dim=1))
        return torch.cat(out, dim=0)

    def proba(self, X, a=None):
        return self._eta(X)[:, 1].numpy().astype(np.float64)

    def get_state(self) -> dict:
        return {"theta": self.latest.detach().clone()}

    def set_state(self, state: dict) -> None:
        self.latest = state["theta"].detach().clone()


# =================================================================================================
# FedFACT-Post (Algorithm 2), global-only variant (DEVIATIONS.md D13), re-derived from
# fedlearn/algorithm/FedFACT_Post.py and fedlearn/models/CGFFedclient_Post.py.
#
#   Construction  : same model init and per-client persistent shuffled train iterator as In.
#   Pre-training  : `rounds` rounds; per client (code's selection order) 40 mini-batch steps of plain
#                   cross-entropy (mean), then one evaluation pass over a NEW shuffled train iterator
#                   (`local_train` -> `model_eval`; only its RNG draws matter here); server aggregates as
#                   in In (unweighted mean + momentum 0.95).
#   Scores        : every client scores its validation share with the final model (softmax, batches of
#                   1024, shuffle=False, x squeezed as in `get_val_test_score`).
#   Calibration   : `post_round` rounds; per client: mu stays 0 (post_local_round_mu = 0), one gradient
#                   step lambda_k = lambda - (eta_d * 5) * grad H_k(lambda), with
#                   H_k = mean_i max_j [M(a_i)^T eta_i]_j + xi_g ||lambda_0 + lambda_1||_1 + xi_l ||mu_0+mu_1||_1 / p_k
#                   (hard max; this client's M has NO positive shift); server: val-count-weighted mean of
#                   the lambda_k in selection order, clamp [0, bound].
#   Prediction    : yhat = argmax_j [M(a)^T eta(x)]_j with client 0's M (D16); needs A.
# =================================================================================================
class _PostClient:
    def __init__(self, X, y, a, Xv, yv, av, model: FedFACTLogistic, cfg: dict):
        self.data = _dataset(X, y, a)
        self.val = _dataset(Xv, yv, av)
        self.model = model
        self.train_loader = DataLoader(self.data, batch_size=cfg["batch"], shuffle=True)
        self.train_iter = enumerate(self.train_loader)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        self.local_mu = torch.zeros(2)               # mu fixed at 0 (global-only variant)
        self.val_score = None

    next_batch = _Client.next_batch


class FedFACTPost:
    family = "FedFACT"
    variant = "FedFACT-Post"
    needs_a = True
    harmonised_cfg = dict(rounds=50, local_steps=40, batch=1024, lr=0.008, optimizer="adam", weight_decay=1e-4, C=1.0,
                          model="logistic (2 logits, softmax CE)", aggregation="unweighted mean + server momentum 0.95",
                          fairness_constraints="DP", global_constraints=0.01, local_constraints=0.01,
                          local_fairness="off (mu=0)", post_round=30, dual_params_lr=0.03, dual_params_bound=10.0,
                          calibration_data="val")
    SERVER_MOMENTUM = 0.95
    LAMBDA_INIT = 0.01           # FedFACT_Post initial lambda (binary DP)
    EVAL_BATCH = 1024

    def __init__(self, clients: list[dict], val: dict, seed: int, cfg: dict):
        if cfg.get("fairness_constraints", "DP") != "DP":
            raise NotImplementedError("only the binary Demographic Parity constraint is implemented (DEVIATIONS.md D13)")
        seed_all(seed)
        self.cfg = cfg
        self.K = len(clients)
        d = clients[0]["X"].shape[1]
        ccfg = dict(batch=cfg["batch"], lr=cfg["lr"], weight_decay=cfg["weight_decay"])

        a_all = [np.asarray(c["a"]).reshape(-1) for c in clients]
        self.train_num = int(sum(len(a) for a in a_all))
        n_a_total = {g: int(sum(int((a == g).sum()) for a in a_all)) for g in (0, 1)}
        self.p_a_k = [{g: int((a == g).sum()) / self.train_num for g in (0, 1)} for a in a_all]
        self.p_k_given_a = [{g: int((a == g).sum()) / n_a_total[g] for g in (0, 1)} for a in a_all]
        self.p_k = [torch.tensor([len(a) / self.train_num], dtype=torch.float64) for a in a_all]

        shares = np.array_split(np.arange(len(val["y"])), self.K)      # contiguous val shares (D16)
        self.model = FedFACTLogistic(d)
        self.latest = self.model.get_flat_params()
        self.clients = [_PostClient(c["X"], c["y"], c["a"], val["X"][ix], val["y"][ix], val["a"][ix],
                                    copy.deepcopy(self.model), ccfg)
                        for c, ix in zip(clients, shares)]
        self.global_lamb = torch.tensor([self.LAMBDA_INIT, self.LAMBDA_INIT], requires_grad=False)
        self._eval_model = copy.deepcopy(self.model)

    # ---- pre-training ----------------------------------------------------------------------------
    def _pretrain_client(self, c: _PostClient) -> torch.Tensor:
        c.model.set_flat_params(self.latest)
        criterion = nn.CrossEntropyLoss()
        for _ in range(self.cfg["local_steps"]):
            c.model.train()
            x, y, a = c.next_batch()
            x, y = torch.squeeze(x), y.reshape(-1, 1)
            c.optimizer.zero_grad()
            loss = criterion(c.model(x), y.view(-1).long())
            loss.backward()
            c.optimizer.step()
        flat = c.model.get_flat_params()
        next(iter(c.train_loader))      # the code's post-training `model_eval` pass: same two RNG draws
        return flat

    def _pretrain_round(self, r: int) -> None:
        if r:
            np.random.seed(r)
        order = np.random.choice(self.K, self.K, replace=False)
        solns = [self._pretrain_client(self.clients[int(k)]) for k in order]
        avg = torch.zeros_like(self.latest)
        for s in solns:
            avg += s
        avg /= len(solns)
        self.latest = ((1 - self.SERVER_MOMENTUM) * self.latest + self.SERVER_MOMENTUM * avg).detach()

    def _score_validation(self) -> None:
        for c in self.clients:
            c.model.set_flat_params(self.latest)
            c.model.eval()
            score = torch.zeros(len(c.val), 2)
            start = 0
            with torch.no_grad():
                for x, y, a in DataLoader(c.val, batch_size=self.cfg["batch"], shuffle=False):
                    s = F.softmax(c.model(torch.squeeze(x)).detach().clone().cpu(), dim=1)
                    score[start:start + s.size(0), :] = s
                    start += s.size(0)
            c.val_score = score.detach().clone().cpu()

    # ---- calibration -----------------------------------------------------------------------------
    def _objective(self, k: int, lamb: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        c, cfg = self.clients[k], self.cfg
        M = _calibration_matrices(self.p_a_k[k], self.p_k_given_a[k], lamb, mu, positive_shift=False)
        M_stack = torch.stack([M[g].t() for g in (0, 1)], dim=0)
        M_i_T = M_stack[c.val.tensors[2].view(-1).long()]
        eta = c.val_score.unsqueeze(-1).double()
        logits = torch.bmm(M_i_T, eta).squeeze(-1)
        term_1 = logits.max(dim=1).values.mean()
        return (term_1 + cfg["global_constraints"] * torch.norm(lamb[0] + lamb[1], p=1)
                + cfg["local_constraints"] * torch.norm(mu[0] + mu[1], p=1) / self.p_k[k])

    def _calibration_round(self, t: int) -> None:
        if t:
            np.random.seed(t)
        order = np.random.choice(self.K, self.K, replace=False)
        lambs, n_val = [], []
        self.global_lamb.requires_grad_(False)
        for k in order:
            k = int(k)
            c = self.clients[k]
            lamb_k = self.global_lamb.clone().detach()
            lamb_k.requires_grad_(True)
            loss = self._objective(k, lamb_k, c.local_mu)
            loss.backward()
            with torch.no_grad():
                lamb_k -= self.cfg["dual_params_lr"] * 5 * lamb_k.grad
            lambs.append(lamb_k.detach().clone())
            n_val.append(len(c.val))
        new = sum([lambs[i] * (n_val[i] / sum(n_val)) for i in range(len(lambs))])
        self.global_lamb = torch.clamp(new, min=0.0, max=self.cfg["dual_params_bound"]).requires_grad_(False)

    def fit(self, rounds: int, hook=lambda r: None) -> None:
        for r in range(rounds):
            self._pretrain_round(r)
        self._score_validation()
        for t in range(self.cfg["post_round"]):
            self._calibration_round(t)
            hook(t + 1)

    # ---- evaluation / state ----------------------------------------------------------------------
    _eta = FedFACTIn._eta

    def proba(self, X, a=None):
        """Hard decision (0.0 / 1.0) of the post-processed classifier; needs the sensitive attribute."""
        if a is None:
            raise ValueError("FedFACT-Post's decision rule needs the sensitive attribute a")
        M = _calibration_matrices(self.p_a_k[0], self.p_k_given_a[0], self.global_lamb, torch.zeros(2),
                                  positive_shift=False)
        M_stack = torch.stack([M[g].t() for g in (0, 1)], dim=0)
        A = torch.tensor(np.asarray(a, dtype=np.float32).reshape(-1, 1))
        M_i_T = M_stack[A.view(-1).long()]
        pred = torch.bmm(M_i_T, self._eta(X).unsqueeze(-1).double()).squeeze(-1)
        _, yhat = torch.max(pred, 1)
        return (yhat == 1).numpy().astype(np.float64)

    def get_state(self) -> dict:
        return {"theta": self.latest.detach().clone(), "lamb": self.global_lamb.detach().clone()}

    def set_state(self, state: dict) -> None:
        self.latest = state["theta"].detach().clone()
        self.global_lamb = state["lamb"].detach().clone()
