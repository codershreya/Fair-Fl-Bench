"""From-scratch dataset loaders for the unified Fair-FL benchmark (Section 3 of the spec).

Each loader reads ONE raw file (or set of files) from data/raw/, builds:
  * df        - feature columns ONLY (target and sensitive attribute removed)
  * y_raw     - raw target: binary 0/1 array, or a continuous array for CaC (binarised later,
                at the TRAIN median, by fl.data.split.make_split)
  * a_raw     - raw sensitive attribute: binary 0/1 array, or a continuous array for CaC
  * p_levels  - ordered discrete partition-variable levels when the partition variable has more
                than 2 categories (Bank's `marital`); None for every other dataset (there P == A)
  * num_cols / cat_cols - which feature columns are numeric vs categorical (fed to Preprocessor)
  * strata    - a label used ONLY to stratify the global train/val/test split (never used to fit
                any model, threshold, encoder or metric)

No code here imports anything from reference/paper1_repo or reference/paper2_repo: every
decision is re-derived from reading that original source (repo:path::function is cited in each
loader's docstring) and re-implemented independently.

Sensitive-attribute rule (never violated): the sensitive attribute is removed from `df` by every
loader before `df` is returned, so it can never leak into the model input X.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from fl import config as C


@dataclass
class RawDataset:
    name: str
    df: pd.DataFrame
    y_raw: np.ndarray
    y_kind: str                      # "binary" | "median"
    a_raw: np.ndarray
    a_kind: str                      # "binary" | "median"
    p_levels: Optional[np.ndarray]   # non-None only for Bank (3 marital levels)
    num_cols: list
    cat_cols: list
    strata: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.df)
        assert len(self.y_raw) == n and len(self.a_raw) == n and len(self.strata) == n
        assert set(self.num_cols) | set(self.cat_cols) == set(self.df.columns)
        assert set(self.num_cols) & set(self.cat_cols) == set()
        if self.name in C.DATASETS:
            sensitive_col = C.DATASETS[self.name]["sensitive"]
            assert sensitive_col not in (self.num_cols + self.cat_cols), (
                f"sensitive attribute leaked into features of {self.name}: {sensitive_col}"
            )


def _strip_object_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Whitespace in UCI text fields (' Male', ' <=50K', ...) is stripped everywhere."""
    for c in df.columns:
        if df[c].dtype.kind not in "iufb":
            df[c] = df[c].astype(str).str.strip()
    return df


# ---------------------------------------------------------------------------------------------
# Adult  (repo:p2 datasets/adult/adult.csv == UCI adult.data + adult.test combined;
#         repo:p1 DatasetLoader/load_adult_data.py::load_adult for the column/target convention)
# ---------------------------------------------------------------------------------------------
def load_adult(*, root=None, expected_n=None) -> RawDataset:
    path = (root or C.DATA_RAW) / "adult.csv"
    df = _strip_object_cols(pd.read_csv(path))
    n = len(df)
    expected_n = C.EXPECTED_ROW_COUNTS["adult"] if expected_n is None else expected_n
    if expected_n is not False:
        assert n == expected_n, f"adult: expected {expected_n} rows, got {n}"

    n_exact_dups = int(df.duplicated().sum())

    income = df.pop("income").str.rstrip(".")  # adult.test labels carry a trailing '.'
    assert set(income.unique()) == {"<=50K", ">50K"}, f"unexpected income labels: {income.unique()}"
    y = (income == ">50K").to_numpy(dtype=int)

    a = (df.pop("sex") == "Male").to_numpy(dtype=int)

    num_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    cat_cols = [c for c in df.columns if c not in num_cols]

    strata = a * 2 + y
    return RawDataset(
        "adult", df, y, "binary", a, "binary", None, num_cols, cat_cols, strata,
        meta=dict(n_exact_duplicate_rows_kept=n_exact_dups,
                  n_question_mark_cells=int((df == "?").to_numpy().sum())),
    )


# ---------------------------------------------------------------------------------------------
# Bank  (repo:p2 datasets/bank/bank.csv == UCI bank-full.csv, 45,211 rows;
#         repo:p1 DatasetLoader/load_bank_data.py for the categorical/numeric column split)
# ---------------------------------------------------------------------------------------------
def load_bank(*, root=None, expected_n=None) -> RawDataset:
    path = (root or C.DATA_RAW) / "bank.csv"
    df = _strip_object_cols(pd.read_csv(path))
    n = len(df)
    expected_n = C.EXPECTED_ROW_COUNTS["bank"] if expected_n is None else expected_n
    if expected_n is not False:
        assert n == expected_n, f"bank: expected {expected_n} rows, got {n}"

    y = (df.pop("y") == "yes").to_numpy(dtype=int)

    marital = df.pop("marital").to_numpy()
    assert set(np.unique(marital)) == {"married", "single", "divorced"}
    a = (marital == "married").astype(int)  # raw1 = married; may be re-oriented later

    # bank.csv's own column is literally named "day_of_week" but holds the day-of-month (1-31),
    # not a weekday -- documented in DEVIATIONS.md, kept as-is (both papers use it this way).
    num_cols = ["age", "balance", "day_of_week", "duration", "campaign", "pdays", "previous"]
    cat_cols = [c for c in df.columns if c not in num_cols]

    strata = a * 2 + y
    return RawDataset(
        "bank", df, y, "binary", a, "binary", marital, num_cols, cat_cols, strata,
        meta=dict(note="'day_of_week' column is day-of-month (1-31), not a weekday; "
                       "'duration' kept as a feature as in both original papers (documented caveat: "
                       "not known before a call completes)."),
    )


# ---------------------------------------------------------------------------------------------
# CaC (Communities & Crime)  (repo:p1 Datasets/communities.csv, 1,994x128;
#         repo:p1 DatasetLoader/load_cac_data.py::DROP_COLS for the Komiyama et al. 2018 drop list)
# ---------------------------------------------------------------------------------------------
def load_cac(*, root=None, expected_n=None) -> RawDataset:
    path = (root or C.DATA_RAW) / "communities.csv"
    df = pd.read_csv(path).replace("?", np.nan)
    n = len(df)
    expected_n = C.EXPECTED_ROW_COUNTS["cac"] if expected_n is None else expected_n
    if expected_n is not False:
        assert n == expected_n, f"cac: expected {expected_n} rows, got {n}"
    assert df.shape[1] == 128, f"cac: expected 128 raw columns, got {df.shape[1]}"

    df = df.drop(columns=[c for c in C.CAC_DROP_COLS if c in df.columns])

    y_raw = df.pop("ViolentCrimesPerPop").astype(float).to_numpy()
    a_raw = df.pop("racepctblack").astype(float).to_numpy()
    state = df.pop("state")  # meta only: the original papers' own (non-affinity) partition key

    num_cols = list(df.columns)  # everything left (incl. "fold") is numeric; 128-27-3 = 98 cols
    assert len(num_cols) == 98, f"cac: expected 98 feature columns after drops, got {len(num_cols)}"
    df = df.astype(float)

    # Stratification proxy ONLY: full-data medians, used solely to stratify the global split.
    # The actual Y/A binarisation thresholds are fit on TRAIN only, in fl.data.split.make_split.
    strata = (a_raw > np.median(a_raw)).astype(int) * 2 + (y_raw > np.median(y_raw)).astype(int)

    return RawDataset(
        "cac", df, y_raw, "median", a_raw, "median", None, num_cols, [], strata,
        meta=dict(state=state.to_numpy(),
                  note="positive class = ABOVE TRAIN-median crime rate (an adverse outcome), "
                       "as in both original papers."),
    )


# ---------------------------------------------------------------------------------------------
# KDD Census-Income  (repo:p1 Datasets/kdd.zip, 299,285 rows -- NOT Paper-2's 199,523-row file;
#         repo:p1 DatasetLoader/load_kdd_data.py for the categorical/numeric column split)
# ---------------------------------------------------------------------------------------------
def load_kdd(*, root=None, expected_n=None) -> RawDataset:
    path = (root or C.DATA_RAW) / "kdd.csv"
    df = _strip_object_cols(pd.read_csv(path))
    n = len(df)
    expected_n = C.EXPECTED_ROW_COUNTS["kdd"] if expected_n is None else expected_n
    if expected_n is not False:
        assert n == expected_n, f"kdd: expected {expected_n} rows, got {n}"

    y = df.pop("class").astype(int).to_numpy()
    assert set(np.unique(y)) == {0, 1}

    a = (df.pop("sex") == "Male").to_numpy(dtype=int)

    num_cols = ["age", "wage-per-hour", "capital-gains", "capital-losses",
                "dividends-from-stocks", "num-persons-worked-for-employer", "weeks-worked-in-year"]
    cat_cols = [c for c in df.columns if c not in num_cols]
    for c in cat_cols:
        df[c] = df[c].replace({"nan": "missing"})

    strata = a * 2 + y
    return RawDataset(
        "kdd", df, y, "binary", a, "binary", None, num_cols, cat_cols, strata,
        meta=dict(note="Paper-2's own kdd file has only 199,523 rows; the unified benchmark "
                       "uses Paper-1's 299,285-row file (train+test concatenated)."),
    )


# ---------------------------------------------------------------------------------------------
# ACS Income  (repo:p1 Datasets/acs_dataset/*.csv, 50 state files via folktables ACSIncome;
#         repo:p1 DatasetLoader/load_acs_data.py for the categorical/numeric column split)
# ---------------------------------------------------------------------------------------------
def load_acs(*, root=None, expected_n=None) -> RawDataset:
    acs_root = (root or C.DATA_RAW) / "acs"
    parts = []
    for state in C.ACS_STATES:
        d = pd.read_csv(acs_root / f"{state}_data.csv",
                         dtype=np.float32, true_values=["True"], false_values=["False"])
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    n = len(df)
    expected_n = C.EXPECTED_ROW_COUNTS["acs"] if expected_n is None else expected_n
    if expected_n is not False:
        assert n == expected_n, f"acs: expected {expected_n} rows, got {n}"

    y = df.pop("PINCP").astype(int).to_numpy()
    assert set(np.unique(y)) == {0, 1}

    sex = df.pop("SEX").to_numpy()
    assert set(np.unique(sex)) == {1.0, 2.0}, f"unexpected SEX codes: {np.unique(sex)}"
    a = (sex == 1.0).astype(int)  # folktables coding: 1 = male, 2 = female

    num_cols = ["AGEP", "WKHP"]
    cat_cols = [c for c in df.columns if c not in num_cols]  # COW, SCHL, MAR, OCCP, POBP, RELP, RAC1P

    strata = a * 2 + y
    return RawDataset(
        "acs", df, y, "binary", a, "binary", None, num_cols, cat_cols, strata,
        meta=dict(note="RAC1P kept multi-level (not binarised, unlike Paper 1's own loader); "
                       "high-cardinality categoricals (OCCP, POBP) are integer-coded, not one-hot, "
                       "by fl.data.preprocess.Preprocessor (> MAX_ONEHOT_LEVELS train levels)."),
    )


LOADERS = dict(adult=load_adult, bank=load_bank, cac=load_cac, kdd=load_kdd, acs=load_acs)


def load_raw(name: str, *, root=None, expected_n=None) -> RawDataset:
    """`root` overrides the directory raw files are read from (default fl.config.DATA_RAW);
    `expected_n` overrides the hard-coded row-count assertion (pass False to skip it entirely).
    Both exist for scripts/subsample.py (Section 5.1), which runs the SAME loaders against
    smaller, stratified subsample files living under a separate root -- never against data/raw/
    itself, and subsample outputs are always stamped `subsample=true` by the caller."""
    if name not in LOADERS:
        raise ValueError(f"unknown dataset {name!r}; choices are {sorted(LOADERS)}")
    return LOADERS[name](root=root, expected_n=expected_n)
