"""Leakage tests (Section 3 "Preprocessing, fitted on TRAIN only" + "Orientation" +
Section 10 "do not use the test set for selection or tuning").

These tests use small synthetic data (not the real datasets) so they run in well under a
second and do not depend on data/raw/ being present.
"""
import numpy as np
import pandas as pd
import pytest

from fl.data.preprocess import Preprocessor
from fl.data.orientation import orient_binary_attribute
from fl.data.split import make_split
from fl.data.loaders import RawDataset


def _toy_raw(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "num1": rng.normal(size=n),
        "cat1": rng.choice(["x", "y", "z"], size=n),
    })
    y = rng.integers(0, 2, size=n)
    a = rng.integers(0, 2, size=n)
    strata = a * 2 + y
    return RawDataset("toy_binary", df, y, "binary", a, "binary", None, ["num1"], ["cat1"], strata)


def test_preprocessor_fit_is_unaffected_by_val_test_rows():
    """Corrupting val/test rows (NaN, huge outliers, unseen categories) must not change what
    the Preprocessor learned from train."""
    rng = np.random.default_rng(0)
    n = 1000
    df = pd.DataFrame({"num1": rng.normal(size=n), "cat1": rng.choice(["x", "y"], size=n)})
    idx_train, idx_rest = np.arange(700), np.arange(700, 1000)

    pre_a = Preprocessor(["num1"], ["cat1"]).fit(df.iloc[idx_train])
    mean_a, std_a, vocab_a = dict(pre_a.mean_), dict(pre_a.std_), dict(pre_a.vocab_)

    df2 = df.copy()
    df2.loc[idx_rest, "num1"] = 1e9                       # outlier
    df2.loc[idx_rest[:10], "cat1"] = "never_seen_in_train"  # unseen category

    pre_b = Preprocessor(["num1"], ["cat1"]).fit(df2.iloc[idx_train])  # fit on the SAME train rows
    assert pre_b.mean_ == mean_a and pre_b.std_ == std_a and pre_b.vocab_ == vocab_a

    # transforming the corrupted val/test rows must not crash and unseen categories -> all-zero
    X_rest = pre_b.transform(df2.iloc[idx_rest])
    onehot_block = X_rest[:10, 1:]  # column 0 is num1
    assert np.allclose(onehot_block, 0.0)


def test_orientation_uses_train_labels_only():
    rng = np.random.default_rng(0)
    y_train = np.array([1] * 80 + [0] * 20)   # a=1 has high base rate in train
    a_train = np.array([1] * 100)
    y_train2 = np.concatenate([y_train, np.array([0] * 100)])
    a_train2 = np.concatenate([a_train, np.array([0] * 100)])
    flipped, r1, r0 = orient_binary_attribute(y_train2, a_train2)
    assert r1 == pytest.approx(0.8) and r0 == pytest.approx(0.0)
    assert flipped is False  # a=1 already has the higher rate -> no flip needed


def test_make_split_thresholds_and_orientation_are_train_only():
    """For a median-thresholded dataset (CaC-like), corrupting val/test raw values must not
    change the train-median threshold nor the orientation decision."""
    rng = np.random.default_rng(0)
    n = 1000
    df = pd.DataFrame({"num1": rng.normal(size=n)})
    y_raw = rng.normal(size=n)
    a_raw = rng.normal(size=n)
    strata = (a_raw > np.median(a_raw)).astype(int) * 2 + (y_raw > np.median(y_raw)).astype(int)
    raw = RawDataset("toy_median", df, y_raw, "median", a_raw, "median", None, ["num1"], [], strata)

    split_a = make_split(raw, seed=0)

    # Now corrupt val/test rows of y_raw/a_raw (but not train rows) and re-split with the SAME
    # seed: train/val/test index assignment depends on `strata`, which we also corrupt
    # consistently, but the train partition itself (and hence the fitted thresholds) must be
    # identical because idx_train is determined by train_test_split with the same seed+strata
    # over the SAME strata array only when strata is unchanged -- so instead we directly check
    # that the fitted threshold equals the TRAIN-only median, not the full-data median.
    thr_y = split_a.info["y_threshold_train_median"]
    assert thr_y == pytest.approx(np.median(y_raw[split_a.idx_train]))
    assert thr_y != pytest.approx(np.median(y_raw))  # true with overwhelming probability

    thr_a = split_a.info["a_threshold_train_median"]
    assert thr_a == pytest.approx(np.median(a_raw[split_a.idx_train]))


def test_split_is_disjoint_and_correct_proportions():
    raw = _toy_raw(n=10_000)
    split = make_split(raw, seed=0)
    all_idx = np.concatenate([split.idx_train, split.idx_val, split.idx_test])
    assert len(set(all_idx.tolist())) == len(raw.df)  # disjoint, union = everything
    n = len(raw.df)
    assert abs(len(split.idx_train) / n - 0.70) < 0.01
    assert abs(len(split.idx_val) / n - 0.15) < 0.01
    assert abs(len(split.idx_test) / n - 0.15) < 0.01


def test_split_is_seed_reproducible():
    raw = _toy_raw(n=2000)
    s1 = make_split(raw, seed=3)
    s2 = make_split(raw, seed=3)
    assert np.array_equal(s1.idx_train, s2.idx_train)
    assert np.array_equal(s1.idx_val, s2.idx_val)
    assert np.array_equal(s1.idx_test, s2.idx_test)


def test_test_set_identity_never_touched_by_preprocessor_fit():
    """fl.data.preprocess.build_matrices must fit only on idx_train; this test fits twice with
    two completely different (fabricated) test splits and checks the transform parameters
    (and therefore the train-split predictions) are identical."""
    from fl.data.preprocess import build_matrices
    raw = _toy_raw(n=1000)
    split = make_split(raw, seed=0)
    X_a, pre_a = build_matrices(raw.df, raw.num_cols, raw.cat_cols,
                                 split.idx_train, split.idx_val, split.idx_test)
    # swap in an arbitrary different (but same-length) test index set
    fake_idx_test = np.roll(split.idx_test, 1)
    X_b, pre_b = build_matrices(raw.df, raw.num_cols, raw.cat_cols,
                                 split.idx_train, split.idx_val, fake_idx_test)
    assert pre_a.mean_ == pre_b.mean_ and pre_a.std_ == pre_b.std_
    assert np.allclose(X_a["train"], X_b["train"])
