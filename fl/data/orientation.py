"""Group orientation and partition-coordinate derivation (Section 3 / Section 9 wording).

Orientation rule: A=1 is always the group with the HIGHER positive rate in the TRAIN labels,
frozen once per (dataset, seed) and never recomputed from validation or test labels. This
makes signed SP, EO and DI comparable across datasets (Paper 1's own sign convention was not:
see reference/analysis/AUDIT_AND_PROTOCOL.md finding D13).

Partition-variable coordinates (used only by fl.partition.attribute): discrete levels are
ordered by ascending TRAIN base rate; the continuous case (CaC) uses a tie-broken rank quantile.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def orient_binary_attribute(y_train: np.ndarray, a_raw1_train: np.ndarray) -> bool:
    """Returns True iff the raw-encoded group 1 is actually the LOWER base-rate group (i.e. the
    final A=1 must be the flip of a_raw1)."""
    r1 = float(y_train[a_raw1_train == 1].mean()) if (a_raw1_train == 1).any() else float("nan")
    r0 = float(y_train[a_raw1_train == 0].mean()) if (a_raw1_train == 0).any() else float("nan")
    return r0 > r1, r1, r0


def partition_coordinate_inputs(
    *, p_levels_train: Optional[np.ndarray], a_kind: str,
    a_raw_train: np.ndarray, a_oriented_train: np.ndarray, y_train: np.ndarray,
):
    """Returns (kind, values, order) describing the TRAIN-row partition variable P.

    kind  : "categorical" | "continuous"
    values: the raw per-row coordinate input (fed to fl.partition.attribute.coordinates)
    order : ordered list of levels (ascending TRAIN base rate) for "categorical"; None for
            "continuous".
    """
    if p_levels_train is not None:  # Bank: marital has 3 levels, P != A
        levels = sorted(np.unique(p_levels_train).tolist())
        rates = {lvl: float(y_train[p_levels_train == lvl].mean()) for lvl in levels}
        order = sorted(levels, key=lambda lvl: (rates[lvl], lvl))
        return "categorical", p_levels_train, order

    if a_kind == "median":  # CaC: racepctblack is continuous
        return "continuous", a_raw_train.astype(float), None

    # Every other dataset: P == A (already oriented so level 0 = lower TRAIN base-rate group)
    return "categorical", a_oriented_train, [0, 1]
