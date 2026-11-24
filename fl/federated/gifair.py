"""GIFAIR-FL, re-derived from reading repo:p2 code/GIFAIR-FL/{models,update,utils,
federated_main}.py in full, cross-checked against the oracle adapter
(`reference/oracle_fairbench/fairbench/algos/p2.py::GIFAIRP2`, which imports the ORIGINAL
`LocalUpdate`/`MLP`/`average_weights` directly -- confirming exactly which lines of
`federated_main.py` matter and which full-partition override it applies). No import from
reference/paper2_repo.

GIFAIR-FL's "individual fairness via group ranking" mechanism, read carefully, turns out to be
nothing more exotic than a per-client LEARNING-RATE SCALE: every round, clients are ranked by
their OWN previous-round training loss (highest loss first), each rank gets a fixed
`r_k` value from a symmetric sequence `[K-1, K-3, ..., -(K-1)]`, and that client's learning rate
for this round is `base_lr * (1 + (lambda_reg / p_k) * r_k)` where `p_k` is the client's
size-weighted sampling probability and `lambda_reg` is a fixed constant derived once from the
`p` distribution. A high-loss client (rank 0, most positive `r_k`) trains with an INFLATED
learning rate; a low-loss client (most negative `r_k`) trains with a DEFLATED one -- nudging
every client's loss towards the group average over rounds. There is no extra loss TERM (unlike
TrustFed/FairTrade's constraint losses) -- the loss function itself is plain `NLLLoss` on a
2-class `log_softmax` model; only the optimizer's `lr` argument changes per client per round.

Model: `GIFAIRMLP` (`fl/models.py`) -- Linear(d,64) -> Dropout(0.5) -> ReLU -> Linear(64,2) ->
log_softmax. Aggregation: `average_weights_unweighted` (repo:p2 `GIFAIR-FL/utils.py`, the SAME
function already re-derived for FedAvg-P2 in Phase 3).

RNG note (Section 5 L1 target): FOUR distinct global-RNG-consuming operations, replicated in
the EXACT order and with the EXACT arguments repo:p2 uses:
  1. Model construction (`GIFAIRP2.__init__`): ONE `MLP(...)` construction -- no per-client
     throwaway models (unlike FedAvg-P1/TrustFed/GlobalGroup), matching FedAvg-P2's own
     "no per-client throwaway" pattern, since local models come from `copy.deepcopy` every
     round, not a fresh constructor call.
  2. `self.initial_loss = random.sample(range(K), K)` -- ONE draw from the PYTHON `random`
     module's global generator (not `numpy.random`!) -- a random permutation used as a
     substitute initial "loss ranking" for round 1 only (every client's `initial_loss` entry
     gets overwritten with its own real training loss after round 1, since harmonised `C=1.0`
     means every client is always selected).
  3. `self.users_pool = np.random.choice(range(K), K, replace=False); self.users_pool.sort()`
     -- ONE numpy-RNG draw whose RESULT is immediately discarded by the following `.sort()`
     (`users_pool` ends up as plain `[0, ..., K-1]` regardless of what was drawn) -- kept
     anyway, because the draw itself still has to happen for the numpy RNG state to line up
     with the original code's, exactly the same "pointless but RNG-relevant" pattern as
     FedAvg-P1's per-client throwaway models.
  4. Every round, ONE `np.random.choice(range(K), m, replace=False, p=self.p)` draw (`m=K` at
     harmonised `C=1.0`, size-weighted by client sample count), THEN one `DataLoader(...,
     shuffle=True)` draw per selected client per local epoch, consumed in the order that choice
     call returned -- not client-index order. `GIFAIRMLP`'s `Dropout(p=0.5)` ALSO draws from
     the global torch RNG once per training-mode forward pass (once per batch, since
     `local_epochs=1` means exactly one batch loop per client per round) -- this interleaves
     with the DataLoader draws (DataLoader draws once per client at loader-iteration start,
     then the model's dropout draws once per batch within that iteration), so both must be
     consumed via the IDENTICAL sequence of PyTorch API calls, not simulated separately.

Everything else (loss-ranking sort, `r_k`/`lambda_reg`/`p` bookkeeping, the per-client
learning-rate formula) is pure deterministic arithmetic with no RNG involvement.
"""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fl.federated.common import average_weights_unweighted, seed_all
from fl.models import GIFAIRMLP


def _gifair_local_update(model: nn.Module, X: torch.Tensor, y: torch.Tensor, *, epochs: int,
                          lr: float, batch_size: int, optimizer_name: str) -> tuple[dict, float]:
    """repo:p2 GIFAIR-FL/update.py::LocalUpdate.update_weights, under the oracle adapter's
    full-partition `train_val_test` override (the client trains on its WHOLE partition, not an
    80/10/10 split -- `fairbench/algos/p2.py::_p2`'s `FullGIF` class)."""
    model.train()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        raise ValueError(f"unsupported optimizer {optimizer_name!r}")
    criterion = nn.NLLLoss()
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    epoch_losses = []
    for _ in range(epochs):
        batch_losses = []
        for batch_X, batch_y in loader:
            model.zero_grad()
            log_probs = model(batch_X)
            loss = criterion(log_probs, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_losses.append(sum(batch_losses) / len(batch_losses))
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return state, sum(epoch_losses) / len(epoch_losses)


class GIFAIRFL:
    family, variant = "P2", "GIFAIR-FL"
    harmonised_cfg = dict(rounds=50, local_epochs=1, lr=0.005, batch=128, optimizer="adam", C=1.0,
                          loss="NLL (2-class log-softmax, dropout)",
                          aggregation="unweighted mean (P2 code)",
                          client_sampling="weighted by size (all clients at C=1)")

    def __init__(self, clients: list[dict], val: dict, seed: int, cfg: dict):
        seed_all(seed)
        self.cfg = cfg
        self.K = K = len(clients)
        d = clients[0]["X"].shape[1]

        self.global_model = GIFAIRMLP(d, hidden_dim=64, dim_out=2)  # RNG draw #1

        self._client_X = [torch.tensor(c["X"], dtype=torch.float32) for c in clients]
        self._client_y = [torch.tensor(c["y"], dtype=torch.long) for c in clients]
        self.val = val

        # repo:p2 GIFAIR-FL/federated_main.py L74-112, re-expressed exactly (incl. the
        # pointless-but-RNG-relevant users_pool draw -- see module docstring).
        self.initial_loss = random.sample(range(K), K)  # RNG draw #2 (python `random`, not numpy)
        self.users_pool = np.random.choice(range(K), K, replace=False)  # RNG draw #3 (numpy)
        self.users_pool.sort()
        self.r_k_values = [K - (2 * (i + 1) - 1) for i in range(K)]
        sizes = [len(c["y"]) for c in clients]
        self.p = [s / sum(sizes) for s in sizes]
        self.lambda_reg = min(x / (K - 1) for x in self.p) / 2

    @property
    def model(self) -> nn.Module:
        return self.global_model

    def _round(self, r: int) -> None:
        K = self.K
        loss_dic = {self.users_pool[i]: self.initial_loss[i] for i in range(len(self.users_pool))}
        loss_dic = dict(sorted(loss_dic.items(), key=lambda it: it[1], reverse=True))
        r_k_dic = {list(loss_dic.keys())[i]: self.r_k_values[i] for i in range(len(loss_dic))}

        m = max(int(self.cfg["C"] * K), 1)
        sel = np.random.choice(range(K), m, replace=False, p=self.p)  # RNG draw, once per round

        weight_dicts = []
        for idx in sel:
            idx = int(idx)
            local_model = copy.deepcopy(self.global_model)
            r_k = r_k_dic[idx]
            p_idx = self.p[idx]
            lr = self.cfg["lr"] * (1 + (self.lambda_reg / p_idx) * r_k)
            state, loss = _gifair_local_update(
                local_model, self._client_X[idx], self._client_y[idx],
                epochs=self.cfg["local_epochs"], lr=lr, batch_size=self.cfg["batch"],
                optimizer_name=self.cfg["optimizer"],
            )
            weight_dicts.append(state)
            self.initial_loss[idx] = loss
        self.global_model.load_state_dict(average_weights_unweighted(weight_dicts))

    def fit(self, rounds: int, hook=lambda r: None) -> None:
        for r in range(rounds):
            self._round(r)
            hook(r + 1)

    def proba(self, X):
        was = self.global_model.training
        self.global_model.eval()
        with torch.no_grad():
            log_probs = self.global_model(torch.as_tensor(np.asarray(X), dtype=torch.float32))
            out = torch.exp(log_probs)[:, 1]  # P(y=1), repo:p2 adapter's own `_forward`
        self.global_model.train(was)
        return out.reshape(-1).detach().cpu().numpy().astype(np.float64)

    def get_state(self) -> dict:
        return {k: v.detach().clone() for k, v in self.global_model.state_dict().items()}

    def set_state(self, state: dict) -> None:
        self.global_model.load_state_dict(state)
