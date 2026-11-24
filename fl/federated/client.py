"""Generic local-training step shared by FedAvg-P1 and FedAvg-P2 (and, later, every other
harmonised variant that does plain local SGD/Adam with no extra fairness mechanism).
Independently re-derived from reading repo:p1 FedAvg/FedAvgClient.py::train and
repo:p2 code/FedAvg/fedavg_federated.py::LocalUpdate.update_weights -- both repos follow the
identical shape (fresh optimizer every round, one DataLoader pass per local epoch, shuffled
batches), differing only in optimizer/loss/weight_decay, which the caller supplies.

RNG note (ties into Section 5's L1 target): constructing the DataLoader here does NOT draw from
the global RNG; the draw happens lazily, once per `for batch in loader:` loop entry (one epoch),
when PyTorch's RandomSampler seeds a fresh *local* generator from the global one and then shuffles
with that local generator. So as long as this function is called the same number of times, in the
same order, with the same `epochs`/`batch_size`, the global-RNG draws line up with the original
code's, regardless of anything else going on around it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def local_sgd_update(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    loss_fn: nn.Module,
    weight_decay: float = 0.0,
    optimizer_name: str = "adam",
) -> tuple[dict, float]:
    """Trains `model` in place for `epochs` local passes; returns (state_dict, mean_epoch_loss).
    `model` must already hold the weights to start from (the caller is responsible for loading
    the current global weights into it before calling this, exactly as both original repos do)."""
    model.train()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5, weight_decay=weight_decay)
    else:
        raise ValueError(f"unsupported optimizer {optimizer_name!r}")

    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    epoch_losses = []
    for _ in range(epochs):
        batch_losses = []
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_X)
            loss = loss_fn(out.view(-1), batch_y.view(-1))
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_losses.append(sum(batch_losses) / len(batch_losses))
    return (
        {k: v.detach().clone() for k, v in model.state_dict().items()},
        sum(epoch_losses) / len(epoch_losses),
    )
