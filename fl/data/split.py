"""Global stratified 70/15/15 split with train-only threshold fitting and group orientation
(Section 3: "Split" and "Sensitive attribute roles").

One split per (dataset, seed): a single global validation set and a single global test set are
shared by every client configuration (IID-3/5/10, attribute-3/5) for that seed, as required by
Section 3 ("One global val and one global test set (not partitioned)").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from fl import config as C
from fl.data.loaders import RawDataset
from fl.data.orientation import orient_binary_attribute


@dataclass
class SplitData:
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    y: np.ndarray    # final binary target, every row (threshold fit on TRAIN only for CaC)
    a: np.ndarray    # final ORIENTED sensitive attribute, every row (1 = higher TRAIN base rate)
    info: dict


def make_split(raw: RawDataset, seed: int) -> SplitData:
    n = len(raw.y_raw)
    idx = np.arange(n)

    # Global stratified 70/15/15 on (A, Y) -- CaC uses full-data median bins as a *sampling
    # device only*; no model, threshold or metric below is fit on those full-data medians.
    idx_train, idx_tmp = train_test_split(
        idx, test_size=C.VAL_FRAC + C.TEST_FRAC, stratify=raw.strata, random_state=seed,
    )
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=C.TEST_FRAC / (C.VAL_FRAC + C.TEST_FRAC),
        stratify=raw.strata[idx_tmp], random_state=seed,
    )

    info = dict(seed=seed, n=n, n_train=len(idx_train), n_val=len(idx_val), n_test=len(idx_test))

    if raw.y_kind == "median":
        thr_y = float(np.median(raw.y_raw[idx_train]))
        y = (raw.y_raw > thr_y).astype(int)
        info["y_threshold_train_median"] = thr_y
    else:
        y = raw.y_raw.astype(int)

    if raw.a_kind == "median":
        thr_a = float(np.median(raw.a_raw[idx_train]))
        a_raw1 = (raw.a_raw > thr_a).astype(int)
        info["a_threshold_train_median"] = thr_a
    else:
        a_raw1 = raw.a_raw.astype(int)

    flipped, r1, r0 = orient_binary_attribute(y[idx_train], a_raw1[idx_train])
    a = (1 - a_raw1) if flipped else a_raw1

    spec = C.DATASETS.get(raw.name)
    a1_raw1_name = spec["a_raw1_name"] if spec else "raw group 1"
    a1_raw0_name = spec["a_raw0_name"] if spec else "raw group 0"
    info.update(
        train_base_rate_raw1=r1, train_base_rate_raw0=r0, orientation_flipped=bool(flipped),
        A1_group=a1_raw0_name if flipped else a1_raw1_name,
        A0_group=a1_raw1_name if flipped else a1_raw0_name,
        train_base_rate_A1=float(y[idx_train][a[idx_train] == 1].mean()),
        train_base_rate_A0=float(y[idx_train][a[idx_train] == 0].mean()),
    )
    return SplitData(idx_train, idx_val, idx_test, y, a, info)
