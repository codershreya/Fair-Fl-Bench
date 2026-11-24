"""Train-only preprocessing (Section 3: "Preprocessing, fitted on TRAIN only").

Numeric    -> train-median impute, then standardise (train mean/std).
Categorical-> one-hot if <= MAX_ONEHOT_LEVELS train levels (unseen category at transform time
              -> all-zero row); otherwise integer-code + standardise (unseen -> len(vocab), i.e.
              one code beyond every known level).

`transform_codes` is the GFL-native encoding: standardised numeric block + RAW (non-standardised)
integer category codes, used only by the GFL algorithm (Section 2's GFL bullet: "integer-coded
categoricals with embeddings").
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from fl import config as C


class Preprocessor:
    def __init__(self, num_cols: list, cat_cols: list, max_onehot: int = C.MAX_ONEHOT_LEVELS):
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.max_onehot = max_onehot
        self._fitted = False

    def fit(self, df_train: pd.DataFrame) -> "Preprocessor":
        self.median_ = {}
        self.mean_ = {}
        self.std_ = {}
        for c in self.num_cols:
            v = pd.to_numeric(df_train[c], errors="coerce").astype(float)
            med = float(v.median())
            self.median_[c] = med
            v = v.fillna(med)
            self.mean_[c] = float(v.mean())
            sd = float(v.std(ddof=0))
            self.std_[c] = sd if sd > 0 else 1.0

        self.vocab_ = {}
        self.encoding_ = {}       # "onehot" | "code"
        self.code_mean_ = {}
        self.code_std_ = {}
        for c in self.cat_cols:
            vocab = sorted(pd.unique(df_train[c].dropna()))
            self.vocab_[c] = vocab
            if len(vocab) <= self.max_onehot:
                self.encoding_[c] = "onehot"
            else:
                self.encoding_[c] = "code"
                codes = pd.Categorical(df_train[c], categories=vocab).codes.astype(float)
                codes = np.where(codes >= 0, codes, len(vocab))  # unseen (NaN) -> extra code
                self.code_mean_[c] = float(codes.mean())
                sd = float(codes.std())
                self.code_std_[c] = sd if sd > 0 else 1.0

        feature_names = list(self.num_cols)
        for c in self.cat_cols:
            if self.encoding_[c] == "onehot":
                feature_names += [f"{c}={level}" for level in self.vocab_[c]]
            else:
                feature_names.append(f"{c}#code")
        self.feature_names_ = feature_names
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self._fitted, "Preprocessor.fit must be called before transform"
        n = len(df)
        blocks = []
        for c in self.num_cols:
            v = pd.to_numeric(df[c], errors="coerce").astype(float).fillna(self.median_[c]).to_numpy()
            blocks.append(((v - self.mean_[c]) / self.std_[c]).astype(np.float32)[:, None])
        for c in self.cat_cols:
            codes = pd.Categorical(df[c], categories=self.vocab_[c]).codes
            if self.encoding_[c] == "onehot":
                m = np.zeros((n, len(self.vocab_[c])), dtype=np.float32)
                seen = codes >= 0
                m[np.nonzero(seen)[0], codes[seen]] = 1.0  # unseen category -> all-zero row
                blocks.append(m)
            else:
                v = np.where(codes >= 0, codes, len(self.vocab_[c])).astype(float)
                blocks.append(((v - self.code_mean_[c]) / self.code_std_[c]).astype(np.float32)[:, None])
        return np.concatenate(blocks, axis=1) if blocks else np.zeros((n, 0), dtype=np.float32)

    @property
    def n_features(self) -> int:
        return len(self.feature_names_)


def transform_codes(pre: Preprocessor, df: pd.DataFrame):
    """GFL-native input: standardised numeric block + RAW integer category codes
    (unseen -> len(vocab)). Returns (X, d_num, cardinalities) where `cardinalities[i]` is the
    number of embedding rows needed for categorical column i (len(vocab) + 1, the +1 for the
    unseen/overflow code)."""
    if pre.num_cols:
        num = pre.transform(df)[:, : len(pre.num_cols)]
    else:
        num = np.zeros((len(df), 0), dtype=np.float32)

    cat_blocks = []
    cardinalities = []
    for c in pre.cat_cols:
        codes = pd.Categorical(df[c], categories=pre.vocab_[c]).codes
        v = np.where(codes >= 0, codes, len(pre.vocab_[c])).astype(np.float32)
        cat_blocks.append(v[:, None])
        cardinalities.append(len(pre.vocab_[c]) + 1)

    X = np.concatenate([num] + cat_blocks, axis=1) if cat_blocks else num
    return X.astype(np.float32), num.shape[1], cardinalities


def build_matrices(df: pd.DataFrame, num_cols: list, cat_cols: list,
                    idx_train: np.ndarray, idx_val: np.ndarray, idx_test: np.ndarray):
    """Convenience: fit on TRAIN rows, transform all three splits."""
    pre = Preprocessor(num_cols, cat_cols).fit(df.iloc[idx_train])
    X = {
        "train": pre.transform(df.iloc[idx_train]),
        "val": pre.transform(df.iloc[idx_val]),
        "test": pre.transform(df.iloc[idx_test]),
    }
    return X, pre
