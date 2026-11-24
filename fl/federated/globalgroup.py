"""GlobalGroup-SP / GlobalGroup-EO, re-derived from reading repo:p1 Global_Group/{server,clients,
yset,metrics,main}.py (SP) and Global_Group_Eodd/{server,clients,yset,metrics,main}.py (EO) in
full. No import from reference/paper1_repo.

Model: reuses `FedAvgP1Net` (Linear(d,64) -> ReLU -> Linear(64,1) -> Sigmoid) -- architecturally
identical to repo:p1 Global_Group/main.py::build_model_class's locally-defined `NNModel`, and to
the oracle adapter's own copy of it (`fairbench/algos/p1.py::GlobalGroupP1.__init__`). Loss:
plain `BCELoss` on probabilities (not logits).

Training loop: UNLIKE every other algorithm in this codebase, GlobalGroup's round loop is NOT
the shared per-round `_round()` hook pattern -- the original's own `Server.train()` drives
everything internally (Y-set bookkeeping, client sampling, local SGD, aggregation, the
once-per-round logging callback), and the oracle's own adapter comment says so explicitly
("original Server.train() drives the rounds; we hook per-round"). This module reproduces that
by putting the whole loop inside `fit()` instead of a `_round()` method.

RNG note (Section 5 L1 target): FIVE distinct global-RNG-consuming operations per run,
replicated in the EXACT order and with the EXACT arguments repo:p1 uses:

  1. Model construction (repo:p1 Global_Group{,_Eodd}/server.py::Server.__init__): ONE
     throwaway model per client, in client order (`clients.Client(dataset, modelclass().to
     (device), ...)` -- the `modelclass()` call is evaluated as an argument, so it runs BEFORE
     the Client object exists), THEN the global/server model LAST (`self.model =
     modelclass().to(device)`) -- i.e. K+1 draws, clients first, global model last. This is the
     REVERSE order from FedAvg-P1/TrustFed (which build the global model first, then the K
     client throwaways).
  2. Every round, `YSet.drop(mu)` (x2 for SP's Y_0/Y_1, x4 for EO's Y_00/Y_10/Y_01/Y_11), in
     that fixed order: `np.random.choice(len(self.yset), int((1-mu)*len(self.yset)),
     replace=False)`. The harmonised config's `mu=1.0` is never overridden by the oracle's
     adapter (confirmed by reading `fairbench/algos/p1.py::GlobalGroupP1.__init__`, which never
     sets `self.server.mu`), so `int((1-mu)*len(yset))` is always 0 -- `drop` always returns an
     EMPTY set. The RNG draw still happens, though, and ITS COST DEPENDS ON `len(self.yset)`
     (confirmed empirically: `np.random.choice(n, 0, replace=False)` advances the Mersenne
     Twister by an amount that varies with `n`), so `len(self.yset)` must be tracked exactly
     round to round (it is NOT always NY=100 -- only the very first round's `drop()` sees
     NY=100; every later round sees however many points the PREVIOUS round's `update()` added),
     never shortcut to "always empty, so just skip the call."
  3. `Client.sample_C_update`: for SP, 2 `np.random.choice(count, size, replace=False)` draws
     per client per round, in the exact field order repo:p1 uses (A=1 first, then A=0). For EO,
     the group-order is (0,0), (1,0), (0,1), (1,1), but (see the audit finding below) counts for
     (1,0) and (0,1) are ALWAYS zero, and `Global_Group_Eodd/clients.py::get_preds`'s own
     `if count == 0: return torch.tensor([])` guard SKIPS the `np.random.choice` call entirely
     in that case -- so EO draws RNG only twice per client per round too (for groups (0,0) and
     (1,1)), not four times, and EO's draw additionally floors the requested size to
     `max(1, size)` and samples WITH replacement whenever the requested size exceeds the
     available count (`replace=(size > count)`), unlike SP's always-`replace=False`.
  4. `Server.sample_clients`: ONE `np.random.choice(K, m, p=client_weights, replace=False)`
     draw per round. The oracle's adapter never overrides `m`, so `m=K` (every client always
     participates) -- but the RETURNED ORDER is still a random weighted permutation that
     determines both the order local SGD runs in (point 5) and the order terms are summed
     during aggregation (float addition is not bit-exact-commutative).
  5. `Client.client_step`: one `DataLoader(..., shuffle=True)` RNG draw per client per local
     epoch (`client_epochs=1` here, so one draw per client per round), consumed in the order
     `sample_clients()` returned -- NOT original client-index order.

Everything else (the MMD-style `distance_kernel`, the fairness-constraint arithmetic, the
`(N/(N-1))` bias-correction factor, weighted aggregation, alpha/Pa bookkeeping, the stepsize's
`*0.99` per-client-step decay) is pure deterministic tensor/float arithmetic with no RNG
involvement at all.

Audit finding (GlobalGroup-EO only -- documented here rather than "fixed"; see
reports/phase5_report.md for the full writeup): the oracle's adapter builds each client's third
dataset tensor as the JOINT group `s*2+y` for EO (repo:p1 `Global_Group_Eodd/main.py`'s own
`client_datasets` construction -- not an adapter invention; confirmed by reading that file
directly). But `Global_Group_Eodd/clients.py` (and `server.py::sync_Pay`) then tests that
joint-coded tensor against RAW 0/1 thresholds (`a == 0`, `a == 1`) while ALSO requiring the true
label via a separate `& (y == ...)` condition. Since joint = s*2+y, `a == 0` can only ever
co-occur with `y == 0` (both imply s=0,y=0) and `a == 1` can only ever co-occur with `y == 1`
(both imply s=0,y=1) -- so of the four `(A, Y)` combinations the code names, `A=1,Y=0` and
`A=0,Y=1` are ALWAYS empty selections (a genuine contradiction: `a==1` forces `y==1`, which
contradicts `y==0`, and symmetrically for the other), and the `s=1` subgroup is never selected
by ANY of the four named combinations at all (nothing ever tests `a == 2` or `a == 3`, the
joint codes that correspond to s=1). So GlobalGroup-EO's fairness-tracking term, exactly as
written in repo:p1, only ever contrasts the `s=0,y=0` and `s=0,y=1` subpopulations against each
other, and the `s=1` subpopulation never enters the fairness loss or the alpha/Y-set bookkeeping
at all. This is reproduced exactly here (including the two permanently-empty groups, via the
same `if len(p) == 0: return 0` / `if count == 0: return torch.tensor([])` guards the original
uses) -- not "fixed" to use all four joint groups the variable names suggest -- because the
oracle wraps this SAME original code, and any correction would silently break L1 equivalence.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fl.federated.common import aggregate_weighted_by_samples, seed_all
from fl.models import FedAvgP1Net


def _distance_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """repo:p1 Global_Group{,_Eodd}/clients.py::distance_kernel (byte-identical in both)."""
    return ((torch.abs(a - 1) + torch.abs(b - 1) - torch.abs(a - b))
            + (torch.abs(a) + torch.abs(b) - torch.abs(a - b))) / 4


class _YSet:
    """repo:p1 Global_Group{,_Eodd}/yset.py::YSet (byte-identical in both, aside from the `A`
    label argument's type -- a string in the EO version -- which is never read elsewhere)."""

    def __init__(self, ny: int):
        self.values = torch.zeros(ny)

    def drop(self, mu: float) -> None:
        n_keep = int((1 - mu) * len(self.values))
        keep_idxs = np.random.choice(len(self.values), n_keep, replace=False)
        self.values = self.values[keep_idxs]

    def update(self, updates: list) -> None:
        cat = torch.cat([u.flatten() for u in updates], 0)
        self.values = torch.cat((self.values, cat), 0)


# ================================================================================================
# GlobalGroup-SP
# ================================================================================================
class _ClientSP:
    """repo:p1 Global_Group/clients.py::Client."""

    def __init__(self, X: torch.Tensor, Y: torch.Tensor, A: torch.Tensor, *, stepsize: float,
                 batchsize: int, epochs: int, lambda_: float):
        self.X, self.Y, self.A = X, Y, A
        self.stepsize = stepsize
        self.batchsize = batchsize
        self.epochs = epochs
        self.lambda_ = lambda_
        self.alphak0 = None
        self.alphak1 = None
        self.N = None
        self.current_C = lambda p, A=None: 0.0

    def client_step(self, model: nn.Module, current_theta: dict) -> dict:
        generator = DataLoader(TensorDataset(self.X, self.Y, self.A), batch_size=self.batchsize,
                               shuffle=True, num_workers=0)
        model.load_state_dict(current_theta)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.stepsize)
        bce = nn.BCELoss()
        for _ in range(self.epochs):
            for x, y, a in generator:
                optimizer.zero_grad()
                prediction = model(x)
                accloss = bce(prediction.flatten(), y)
                fairloss = self.current_C(prediction[a == 0], A=0) - self.current_C(prediction[a == 1], A=1)
                loss = accloss + 2 * self.lambda_ * fairloss
                loss.backward()
                optimizer.step()
        self.stepsize = 0.99 * self.stepsize
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    def sample_C_update(self, num_points: list, model: nn.Module, current_theta: dict):
        with torch.no_grad():
            model.load_state_dict(current_theta)
            point_idxs_A1 = np.random.choice(int(self.A.sum().item()),
                                              size=int(self.alphak1 * num_points[1]), replace=False)
            point_idxs_A0 = np.random.choice(int((1 - self.A).sum().item()),
                                              size=int(self.alphak0 * num_points[0]), replace=False)
            p1 = model(self.X[self.A == 1][point_idxs_A1])
            p0 = model(self.X[self.A == 0][point_idxs_A0])
            return p0, p1

    def get_weight(self) -> int:
        return len(self.Y)

    def get_Pka(self, a: int) -> float:
        return float((self.A == a).to(torch.float32).mean().item())

    def set_alphaka(self, Pa0: float) -> None:
        self.alphak0 = self.get_Pka(0) / Pa0
        self.alphak1 = self.get_Pka(1) / (1 - Pa0)

    def set_N(self, N: int) -> None:
        self.N = N

    def set_C(self, Y0: "_YSet", Y1: "_YSet") -> None:
        N = self.N

        def current_C(p: torch.Tensor, A=None):
            if len(p) == 0:
                return 0.0
            if A is None:
                return _distance_kernel(p, Y0.values).mean() - _distance_kernel(p, Y1.values).mean()
            if A == 1:
                return (_distance_kernel(p, Y0.values).mean()
                        - _distance_kernel(p, Y1.values).mean() * (N / (N - 1)))
            if A == 0:
                return (_distance_kernel(p, Y0.values).mean() * (N / (N - 1))
                        - _distance_kernel(p, Y1.values).mean())
            return 0.0

        self.current_C = current_C


class GlobalGroupSP:
    family, variant = "P1", "GlobalGroup-SP"
    harmonised_cfg = dict(rounds=50, local_epochs=1, lr=0.005, batch=32, optimizer="sgd",
                          loss="BCE + lambda*MMD tracking (lambda=0.2)", NY=100, mu=1.0,
                          lambda_fair=0.2)

    def __init__(self, clients: list[dict], val: dict, seed: int, cfg: dict):
        seed_all(seed)
        self.cfg = cfg
        d = clients[0]["X"].shape[1]
        self.K = len(clients)
        self.NY = cfg.get("NY", 100)
        self.lambda_ = cfg.get("lambda_fair", 0.2)
        self.mu = cfg.get("mu", 1.0)

        # RNG draws #1..#K: one throwaway model per client, IN CLIENT ORDER.
        self._clients: list[dict] = []
        for c in clients:
            X = torch.tensor(np.asarray(c["X"]), dtype=torch.float32)
            Y = torch.tensor(np.asarray(c["y"]), dtype=torch.float32)
            A = torch.tensor(np.asarray(c["a"]), dtype=torch.float32)
            model = FedAvgP1Net(d)  # RNG draw, one per client, in order
            cl = _ClientSP(X, Y, A, stepsize=cfg["lr"], batchsize=32, epochs=cfg["local_epochs"],
                          lambda_=self.lambda_)
            self._clients.append(dict(cl=cl, model=model))
        # RNG draw #K+1: the global model, AFTER all client throwaways.
        self.global_model = FedAvgP1Net(d)

        raw_weights = np.array([cc["cl"].get_weight() for cc in self._clients], dtype=np.float64)
        self.client_weights = raw_weights / raw_weights.sum()

        # sync_N(): called by the oracle adapter immediately after Server construction, before
        # fit() -- pure arithmetic, no RNG.
        N = sum(cc["cl"].get_weight() for cc in self._clients)
        for cc in self._clients:
            cc["cl"].set_N(N)

        self.val = val
        self.Y0: _YSet | None = None
        self.Y1: _YSet | None = None

    @property
    def model(self) -> nn.Module:
        return self.global_model

    def _sync_Pa(self) -> None:
        Pk0s = [cc["cl"].get_Pka(0) for cc in self._clients]
        Pa0 = float((np.array(Pk0s) * self.client_weights).sum())
        for cc in self._clients:
            cc["cl"].set_alphaka(Pa0)

    def _update_C(self) -> None:
        self.Y0.drop(self.mu)
        self.Y1.drop(self.mu)
        gw = {k: v.detach().clone() for k, v in self.global_model.state_dict().items()}
        updates0, updates1 = [], []
        for cc, w in zip(self._clients, self.client_weights):
            n = int(w * self.mu * self.NY)
            p0, p1 = cc["cl"].sample_C_update([n, n], cc["model"], gw)
            updates0.append(p0)
            updates1.append(p1)
        self.Y0.update(updates0)
        self.Y1.update(updates1)

    def fit(self, rounds: int, hook=lambda r: None) -> None:
        self._sync_Pa()
        self.Y0 = _YSet(self.NY)
        self.Y1 = _YSet(self.NY)
        for r in range(rounds):
            self._update_C()
            for cc in self._clients:
                cc["cl"].set_C(self.Y0, self.Y1)
            participating = np.random.choice(self.K, self.K, p=self.client_weights, replace=False)
            gw = {k: v.detach().clone() for k, v in self.global_model.state_dict().items()}
            reports = []
            for idx in participating:
                cc = self._clients[int(idx)]
                new_state = cc["cl"].client_step(cc["model"], gw)
                reports.append(dict(weights=new_state, samples=cc["cl"].get_weight()))
            new_weights = aggregate_weighted_by_samples(reports)
            self.global_model.load_state_dict(new_weights)
            hook(r + 1)

    def proba(self, X):
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


# ================================================================================================
# GlobalGroup-EO
# ================================================================================================
class _ClientEO:
    """repo:p1 Global_Group_Eodd/clients.py::Client. `A` here is the JOINT group `s*2+y`
    (confirmed by reading Global_Group_Eodd/main.py's own `client_datasets` construction), not
    the raw sensitive attribute -- see the module-level audit finding for what that implies."""

    def __init__(self, X: torch.Tensor, Y: torch.Tensor, A_joint: torch.Tensor, *, stepsize: float,
                 batchsize: int, epochs: int, lambda_: float):
        self.X, self.Y, self.A = X, Y, A_joint
        self.stepsize = stepsize
        self.batchsize = batchsize
        self.epochs = epochs
        self.lambda_ = lambda_
        self.alpha_00 = None
        self.alpha_10 = None
        self.alpha_01 = None
        self.alpha_11 = None
        self.N = None
        self.current_C = lambda p, A, Y: 0.0

    def client_step(self, model: nn.Module, current_theta: dict) -> dict:
        generator = DataLoader(TensorDataset(self.X, self.Y, self.A), batch_size=self.batchsize,
                               shuffle=True, num_workers=0)
        model.load_state_dict(current_theta)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.stepsize)
        bce = nn.BCELoss()
        for _ in range(self.epochs):
            for x, y, a in generator:
                optimizer.zero_grad()
                prediction = model(x)
                accloss = bce(prediction.flatten(), y)
                fair_y0 = (self.current_C(prediction[(a == 0) & (y == 0)], A=0, Y=0)
                           - self.current_C(prediction[(a == 1) & (y == 0)], A=1, Y=0))
                fair_y1 = (self.current_C(prediction[(a == 0) & (y == 1)], A=0, Y=1)
                           - self.current_C(prediction[(a == 1) & (y == 1)], A=1, Y=1))
                loss = accloss + 2 * self.lambda_ * (fair_y0 + fair_y1)
                loss.backward()
                optimizer.step()
        self.stepsize = 0.99 * self.stepsize
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    def sample_C_update(self, num_points: list, model: nn.Module, current_theta: dict):
        with torch.no_grad():
            model.load_state_dict(current_theta)

            def get_preds(a_target: int, y_target: int, alpha, n_target: int):
                mask = (self.A == a_target) & (self.Y == y_target)
                count = int(mask.sum().item())
                if count == 0:
                    return torch.tensor([])  # NO RNG draw -- see module-level audit finding
                size = int(alpha * n_target)
                idxs = np.random.choice(count, size=max(1, size), replace=(size > count))
                return model(self.X[mask][idxs])

            p00 = get_preds(0, 0, self.alpha_00, num_points[0])
            p10 = get_preds(1, 0, self.alpha_10, num_points[1])
            p01 = get_preds(0, 1, self.alpha_01, num_points[2])
            p11 = get_preds(1, 1, self.alpha_11, num_points[3])
            return p00, p10, p01, p11

    def get_weight(self) -> int:
        return len(self.Y)

    def get_Pkay(self, a: int, y: int) -> float:
        count = ((self.A == a) & (self.Y == y)).sum().to(torch.float32)
        return float((count / len(self.Y)).item())

    def set_alpha_kay(self, Pa00: float, Pa10: float, Pa01: float, Pa11: float) -> None:
        def safe_div(n, d):
            return n / d if d > 0 else 0.0
        self.alpha_00 = safe_div(self.get_Pkay(0, 0), Pa00)
        self.alpha_10 = safe_div(self.get_Pkay(1, 0), Pa10)
        self.alpha_01 = safe_div(self.get_Pkay(0, 1), Pa01)
        self.alpha_11 = safe_div(self.get_Pkay(1, 1), Pa11)

    def set_N(self, N: int) -> None:
        self.N = N

    def set_C(self, Y_00: "_YSet", Y_10: "_YSet", Y_01: "_YSet", Y_11: "_YSet") -> None:
        N = self.N

        def current_C(p: torch.Tensor, A: int, Y: int):
            if len(p) == 0:
                return 0.0
            if Y == 0:
                target_0, target_1 = Y_00.values, Y_10.values
            elif Y == 1:
                target_0, target_1 = Y_01.values, Y_11.values
            else:
                return 0.0
            if A == 0:
                return _distance_kernel(p, target_0).mean() * (N / (N - 1)) - _distance_kernel(p, target_1).mean()
            if A == 1:
                return _distance_kernel(p, target_0).mean() - _distance_kernel(p, target_1).mean() * (N / (N - 1))
            return 0.0

        self.current_C = current_C


class GlobalGroupEO:
    family, variant = "P1", "GlobalGroup-EO"
    harmonised_cfg = dict(rounds=50, local_epochs=1, lr=0.005, batch=32, optimizer="sgd",
                          loss="BCE + lambda*MMD tracking, conditioned on Y (lambda=0.2)",
                          NY=100, mu=1.0, lambda_fair=0.2)

    def __init__(self, clients: list[dict], val: dict, seed: int, cfg: dict):
        seed_all(seed)
        self.cfg = cfg
        d = clients[0]["X"].shape[1]
        self.K = len(clients)
        self.NY = cfg.get("NY", 100)
        self.lambda_ = cfg.get("lambda_fair", 0.2)
        self.mu = cfg.get("mu", 1.0)

        self._clients: list[dict] = []
        for c in clients:
            X = torch.tensor(np.asarray(c["X"]), dtype=torch.float32)
            Y = torch.tensor(np.asarray(c["y"]), dtype=torch.float32)
            s_raw = torch.tensor(np.asarray(c["a"]), dtype=torch.float32)
            a_joint = s_raw * 2 + Y  # repo:p1 Global_Group_Eodd/main.py's own group encoding
            model = FedAvgP1Net(d)  # RNG draw, one per client, in order
            cl = _ClientEO(X, Y, a_joint, stepsize=cfg["lr"], batchsize=32, epochs=cfg["local_epochs"],
                           lambda_=self.lambda_)
            self._clients.append(dict(cl=cl, model=model))
        self.global_model = FedAvgP1Net(d)  # RNG draw #K+1, after all client throwaways

        raw_weights = np.array([cc["cl"].get_weight() for cc in self._clients], dtype=np.float64)
        self.client_weights = raw_weights / raw_weights.sum()

        N = sum(cc["cl"].get_weight() for cc in self._clients)
        for cc in self._clients:
            cc["cl"].set_N(N)

        self.val = val
        self.Y00: _YSet | None = None
        self.Y10: _YSet | None = None
        self.Y01: _YSet | None = None
        self.Y11: _YSet | None = None

    @property
    def model(self) -> nn.Module:
        return self.global_model

    def _sync_Pay(self) -> None:
        Pk00 = [cc["cl"].get_Pkay(0, 0) for cc in self._clients]
        Pk10 = [cc["cl"].get_Pkay(1, 0) for cc in self._clients]
        Pk01 = [cc["cl"].get_Pkay(0, 1) for cc in self._clients]
        Pk11 = [cc["cl"].get_Pkay(1, 1) for cc in self._clients]
        Pa00 = float((np.array(Pk00) * self.client_weights).sum())
        Pa10 = float((np.array(Pk10) * self.client_weights).sum())
        Pa01 = float((np.array(Pk01) * self.client_weights).sum())
        Pa11 = float((np.array(Pk11) * self.client_weights).sum())
        for cc in self._clients:
            cc["cl"].set_alpha_kay(Pa00, Pa10, Pa01, Pa11)

    def _update_C(self) -> None:
        self.Y00.drop(self.mu)
        self.Y10.drop(self.mu)
        self.Y01.drop(self.mu)
        self.Y11.drop(self.mu)
        gw = {k: v.detach().clone() for k, v in self.global_model.state_dict().items()}
        u00, u10, u01, u11 = [], [], [], []
        for cc, w in zip(self._clients, self.client_weights):
            n = int(w * self.mu * self.NY)
            p00, p10, p01, p11 = cc["cl"].sample_C_update([n, n, n, n], cc["model"], gw)
            u00.append(p00)
            u10.append(p10)
            u01.append(p01)
            u11.append(p11)
        self.Y00.update(u00)
        self.Y10.update(u10)
        self.Y01.update(u01)
        self.Y11.update(u11)

    def fit(self, rounds: int, hook=lambda r: None) -> None:
        self._sync_Pay()
        self.Y00 = _YSet(self.NY)
        self.Y10 = _YSet(self.NY)
        self.Y01 = _YSet(self.NY)
        self.Y11 = _YSet(self.NY)
        for r in range(rounds):
            self._update_C()
            for cc in self._clients:
                cc["cl"].set_C(self.Y00, self.Y10, self.Y01, self.Y11)
            participating = np.random.choice(self.K, self.K, p=self.client_weights, replace=False)
            gw = {k: v.detach().clone() for k, v in self.global_model.state_dict().items()}
            reports = []
            for idx in participating:
                cc = self._clients[int(idx)]
                new_state = cc["cl"].client_step(cc["model"], gw)
                reports.append(dict(weights=new_state, samples=cc["cl"].get_weight()))
            new_weights = aggregate_weighted_by_samples(reports)
            self.global_model.load_state_dict(new_weights)
            hook(r + 1)

    def proba(self, X):
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
