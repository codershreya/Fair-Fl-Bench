"""FedAvg-P2, re-derived from reading repo:p2 code/FedAvg/fedavg_federated.py (LocalUpdate,
TabularMLP, run()) and repo:p2 code/GIFAIR-FL/utils.py::average_weights (no import from
reference/paper2_repo). Differs from FedAvg-P1 in three ways that matter for L1 equivalence:

  1. Model: TabularMLP (Linear-ReLU-Dropout(0.2)-Linear, raw logits) + BCEWithLogitsLoss,
     instead of P1's Sigmoid-output + BCELoss -- Dropout means every forward pass in training
     mode draws from the global torch RNG too, not just the once-per-epoch DataLoader shuffle.
  2. No per-client model is pre-built at construction time (repo:p2 builds a fresh
     `copy.deepcopy(global_model)` INSIDE every round, for every selected client -- deepcopy
     copies values, consumes no RNG) -- so FedAvgP2's RNG state entering round 1 is just
     "one global-model construction", not "one global model + K throwaway client models" as
     in FedAvg-P1.
  3. Client order per round is NOT simply 0..K-1: even at full participation (C=1.0, so every
     client trains every round), repo:p2 draws `np.random.choice(range(K), K, replace=False)`
     every round -- a full random permutation of the client order -- and the (unweighted-mean)
     aggregation sums in THAT order. We must call the identical numpy API (not
     `np.random.permutation`, which consumes the numpy RNG differently) to stay bit-exact.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from fl.federated.client import local_sgd_update
from fl.federated.common import average_weights_unweighted, seed_all
from fl.models import TabularMLP


class FedAvgP2:
    family, variant = "P2", "FedAvg-P2"
    harmonised_cfg = dict(rounds=50, local_epochs=1, lr=0.005, batch=128, optimizer="adam", C=1.0,
                          loss="BCE-with-logits", aggregation="unweighted mean (P2 code)")
    WEIGHT_DECAY = 1e-4  # hard-coded in repo:p2 LocalUpdate._make_optimizer for optimizer="adam"

    def __init__(self, clients: list[dict], val: dict, seed: int, cfg: dict):
        seed_all(seed)
        self.cfg = cfg
        self.K = len(clients)
        d = clients[0]["X"].shape[1]

        # ONE model construction (repo:p2 FedAvgP2.__init__ builds self._model and nothing
        # else that touches torch RNG); no per-client throwaway models, unlike FedAvg-P1.
        self.global_model = TabularMLP(d, hidden_dim=64, dim_out=1)

        self._client_X = [torch.tensor(c["X"], dtype=torch.float32) for c in clients]
        self._client_y = [torch.tensor(c["y"], dtype=torch.float32) for c in clients]
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.val = val

    @property
    def model(self) -> nn.Module:
        return self.global_model

    def _round(self, r: int) -> None:
        m = max(int(self.cfg["C"] * self.K), 1)
        sel = np.random.choice(range(self.K), m, replace=False)  # repo:p2 run()/FedAvgP2._round
        weight_dicts = []
        for u in sel:
            local_model = copy.deepcopy(self.global_model)
            state, _loss = local_sgd_update(
                local_model, self._client_X[u], self._client_y[u],
                epochs=self.cfg["local_epochs"], lr=self.cfg["lr"], batch_size=self.cfg["batch"],
                loss_fn=self.loss_fn, weight_decay=self.WEIGHT_DECAY, optimizer_name="adam",
            )
            weight_dicts.append(state)
        self.global_model.load_state_dict(average_weights_unweighted(weight_dicts))

    def fit(self, rounds: int, hook=lambda r: None) -> None:
        for r in range(rounds):
            self._round(r)
            hook(r + 1)

    def proba(self, X) -> "np.ndarray":
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
