"""From-scratch model architectures (Section 2's per-paper FedAvg baselines).

Each class below is independently re-derived from reading the ORIGINAL repository's model
definition (cited in its docstring) -- nothing here imports from `reference/paper1_repo` or
`reference/paper2_repo`. Layer names differ from the originals (that doesn't matter: a
`torch.nn.Linear`'s random initialisation depends only on the ORDER its constructor runs in and
its shape, never on the Python attribute name it is assigned to -- see
`reports/phase3_report.md` for why this lets Phase-3 chase bit-exact (L1) equivalence instead of
settling for L2/L3 on the two FedAvg variants).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FedAvgP1Net(nn.Module):
    """Paper-1's own FedAvg model (repo:p1 models.py::simpleModel): Linear(d,64) -> ReLU ->
    Linear(64,1) -> Sigmoid. Outputs P(y=1|x) directly (no separate logits head); trained with
    plain BCELoss in repo:p1 FedAvg/FedAvgClient.py."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.out = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        return self.out(self.fc2(x))


class FedMinMaxNet(nn.Module):
    """Paper-1's FedMinMax/EOFedMinMax model (repo:p1 models.py::modelFedMinMax):
    Linear(d,64) -> ReLU -> Linear(64,1). Returns raw LOGITS (no activation) -- unlike
    `FedAvgP1Net`, which uses the same two-layer shape but ends in Sigmoid. Trained with
    BCEWithLogitsLoss in both repo:p1 OrigFedMinMax/ and EOFedMinMax/."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class GIFAIRMLP(nn.Module):
    """Paper-2's GIFAIR-FL model (repo:p2 code/GIFAIR-FL/models.py::MLP): Linear(d,64) ->
    Dropout(p=0.5) -> ReLU -> Linear(64,2) -> log_softmax(dim=1). Note the layer ORDER
    (Dropout BEFORE ReLU -- unlike `TabularMLP`'s Dropout-after-ReLU) and the 2-class,
    log-probability output (trained with NLLLoss, not BCE-with-logits like every other model in
    this codebase). Dropout's default p=0.5 means every TRAINING-mode forward pass draws from
    the global torch RNG (a Bernoulli mask), not just the once-per-epoch DataLoader shuffle."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dim_out: int = 2):
        super().__init__()
        self.layer_input = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout()  # p=0.5, repo:p2's own default (not overridden)
        self.relu = nn.ReLU()
        self.layer_hidden = nn.Linear(hidden_dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return nn.functional.log_softmax(x, dim=1)


class GFLTabularNet(nn.Module):
    """Paper-2's GFL model (repo:p2 code/Enforcing_Group_Fairness_in_Privacy_Preserving/
    main.py::TabularNet): a per-categorical-column `nn.Embedding`, concatenated with the
    numeric block, fed through Linear(in_dim,hidden)->ReLU->Linear(hidden,hidden)->ReLU->
    Linear(hidden,1) [logits]. Unlike every other model in this codebase, forward() takes TWO
    tensors (`x_num`, `x_cat`) rather than one -- `x_cat` holds INTEGER category codes (one
    column per categorical feature), not one-hot columns. Embeddings are constructed BEFORE the
    `net` Sequential, in `cat_cardinalities` order -- that construction order matters for RNG
    just as much as `net`'s three Linear layers' order."""

    def __init__(self, d_num: int, cat_cardinalities: list[int], emb_dim: int = 8, hidden: int = 128):
        super().__init__()
        self.d_num = d_num
        self.cat_cardinalities = cat_cardinalities

        self.embeddings = nn.ModuleList()
        for card in cat_cardinalities:
            card = max(int(card), 2)
            self.embeddings.append(nn.Embedding(card, emb_dim))

        in_dim = d_num + emb_dim * len(cat_cardinalities)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        embs = []
        if x_cat.numel() > 0 and len(self.embeddings) > 0:
            for j, emb in enumerate(self.embeddings):
                idx = x_cat[:, j]
                if idx.dtype != torch.long:
                    idx = idx.long()
                idx = idx.clamp(min=0, max=int(emb.num_embeddings) - 1)
                embs.append(emb(idx))
        x = torch.cat([x_num] + embs, dim=1) if embs else x_num
        return self.net(x).squeeze(1)


class TabularMLP(nn.Module):
    """Paper-2's own tabular model (repo:p2 code/FedAvg/fedavg_federated.py::TabularMLP):
    Linear(d,64) -> ReLU -> Dropout(p=0.2) -> Linear(64,dim_out). Returns raw LOGITS (no
    activation); trained with BCEWithLogitsLoss in repo:p2's LocalUpdate.update_weights.
    dim_out=1 for FedAvg-P2/FairTrade-SP/-ATE (this phase only builds FedAvg-P2)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dim_out: int = 1):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)
        self.fc_out = nn.Linear(hidden_dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc_in(x))
        x = self.drop(x)
        return self.fc_out(x)
