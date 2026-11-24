"""FedAvg-P1, re-derived from reading repo:p1 FedAvg/{FedAvgServer,FedAvgClient,FedAvgLossStrategies}.py
(no import from reference/paper1_repo). Plain FedAvg: Adam locally, BCELoss on a sigmoid output,
weighted-by-sample-count aggregation, full client participation every round (C=1.0, so no
client-selection randomness).

Construction-order note (Section 5 L1 target): the original `FedAvgServer.__init__` builds the
global model, and then each `FedAvgClient.__init__` builds its OWN local model (immediately
thrown away -- every round starts by overwriting it with the current global weights via
`set_parameters`). Both calls draw from the global `torch` RNG (`nn.Linear`'s Kaiming/uniform
init). We replicate the exact same sequence of model constructions here, in the exact same
order, purely so the global RNG is in the same state entering round 1 as it is in the oracle --
the throwaway per-client models themselves have zero effect on the algorithm's behaviour.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fl.federated.client import local_sgd_update
from fl.federated.common import aggregate_weighted_by_samples, seed_all
from fl.models import FedAvgP1Net


class FedAvgP1:
    family, variant = "P1", "FedAvg-P1"
    harmonised_cfg = dict(rounds=50, local_epochs=1, lr=0.005, batch=128, optimizer="adam", loss="BCE", C=1.0)

    def __init__(self, clients: list[dict], val: dict, seed: int, cfg: dict):
        seed_all(seed)
        self.cfg = cfg
        d = clients[0]["X"].shape[1]
        self.K = len(clients)

        # 1) global model (repo:p1 FedAvgServer.__init__: self.global_model = simpleModel(d))
        self.global_model = FedAvgP1Net(d)

        # 2) one persistent local model + one persistent DataLoader PER CLIENT, built in client
        #    order (repo:p1 FedAvgClient.__init__), before any training -- see module docstring.
        self._client_models = []
        self._client_X = []
        self._client_y = []
        self._client_n = []
        for c in clients:
            self._client_models.append(FedAvgP1Net(d))  # throwaway init; overwritten every round
            self._client_X.append(torch.tensor(c["X"], dtype=torch.float32))
            self._client_y.append(torch.tensor(c["y"], dtype=torch.float32))
            self._client_n.append(len(c["y"]))
        # repo:p1 FedAvgClient.__init__ also builds `self.loader = DataLoader(..., shuffle=True)`
        # once per client; that construction itself draws no RNG (only *iterating* it does, see
        # fl/federated/client.py), so building it lazily inside the round loop is equivalent.

        self.loss_fn = nn.BCELoss()
        self.val = val

    @property
    def model(self) -> nn.Module:
        return self.global_model

    def _round(self, r: int) -> None:
        gw = self.global_model.state_dict()
        reports = []
        for i in range(self.K):
            self._client_models[i].load_state_dict(gw)  # repo:p1 FedAvgClient.set_parameters
            state, _loss = local_sgd_update(
                self._client_models[i], self._client_X[i], self._client_y[i],
                epochs=self.cfg["local_epochs"], lr=self.cfg["lr"], batch_size=self.cfg["batch"],
                loss_fn=self.loss_fn, weight_decay=0.0, optimizer_name="adam",
            )
            reports.append(dict(weights=state, samples=self._client_n[i]))
        new_weights = aggregate_weighted_by_samples(reports)
        self.global_model.load_state_dict(new_weights)

    def fit(self, rounds: int, hook=lambda r: None) -> None:
        for r in range(rounds):
            self._round(r)
            hook(r + 1)

    def proba(self, X) -> "torch.Tensor":
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
