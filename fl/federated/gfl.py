"""GFL / GFL-GBCFSS-EO / GFL-GBCFSS-DP, re-derived from reading repo:p2
code/Enforcing_Group_Fairness_in_Privacy_Preserving/{main.py, clients/statistics.py,
server/{server,agdg,lsh_cadc,gbcfss,fairness,aggregation}.py} in full (no import from
reference/paper2_repo). By far the most structurally different algorithm in this codebase:
it uses a GFL-NATIVE data representation (`input_kind="codes"` -- standardised numeric block
plus INTEGER categorical codes, not one-hot) and a server-side synthetic-dataset-generation
pipeline instead of a plain weighted/unweighted average of client updates.

============================================================================================
Per-round orchestration (`GFLServer.run_round`, server/server.py), in the EXACT order the
original executes it:
  1. AGDG (server/agdg.py)       -- aggregate per-client per-group {num,mu,sigma} (pure
                                     arithmetic: weighted mean/cov via Python `sum()`), then
                                     sample synthetic NUMERIC data per group via
                                     `np.random.multivariate_normal(mean=mu, cov=sigma,
                                     size=num)` -- ONE draw per group, consuming the GLOBAL
                                     numpy legacy RNG, iterated in `global_stats.items()`
                                     order (a Python `set()` union of group keys -- for the
                                     binary groups {0,1} this reliably iterates 0 then 1 in
                                     CPython, since hash(0)=0 and hash(1)=1 land in ascending
                                     hash-table slots for a tiny set).
  2. LSH-CADC (server/lsh_cadc.py) -- reconstructs categorical attributes via random-
                                     hyperplane hashing. `rng = np.random.default_rng(seed)`
                                     with `seed=42` -- the UN-OVERRIDDEN default (server.py's
                                     call site never passes `seed=`) -- so this `Generator` is
                                     COMPLETELY ISOLATED from the tracked global legacy numpy
                                     RNG: it never advances `np.random`'s own state, and its
                                     own output is identical across all three benchmark seeds
                                     for a given dataset/partition (a property of the original
                                     algorithm, not a bug). Bucket lookup + Mahalanobis nearest-
                                     neighbour pick is fully deterministic given `W` and the
                                     real auxiliary data.
  3. Build the full synthetic feature matrix (numeric + imputed categorical); pure arithmetic.
  4. Pseudo-label synthetic data with the CURRENT global model's own predictions
     (`label_strategy="pseudo"` always, per the oracle adapter's call -- the alternative
     `"sample"` strategy, which WOULD consume `np.random.rand`, is never invoked here).
  5. Score synthetic data with the current global model (`model.eval()`, no RNG -- this model
     has no Dropout).
  6. Optional GBCFSS (server/gbcfss.py) -- a FULLY DETERMINISTIC greedy subset selection (no
     RNG at all) based on per-sample cleanliness (percentile-normalised inverse log-loss) and a
     fairness-contribution heuristic that differs between "eo" and "dp" `fairness_mode`, with a
     group-balance-constrained two-pass greedy loop (`gbcfss_group_balance=0.2` is NOT None for
     either GBCFSS variant, so this constrained path always runs for them) -- an interleaved
     pass that can defer some indices, followed by a "fill remaining" catch-all pass appending
     deferred items in their original score-sorted relative order. `budget=5000` exceeds every
     dataset's per-round synthetic-data size in this benchmark (`N` = sum of per-group `num`,
     which only ever reaches a few hundred here), so EVERY row is eventually selected, just in a
     different (GBCFSS-determined) row order than the natural synthetic-data order.
  7. Train the global model on the (possibly GBCFSS-reordered) synthetic data for ONE epoch
     (`train_on_dataset`'s default): a fresh `torch.optim.Adam`, fresh `DataLoader(shuffle=True)`
     -- ONE more global TORCH RNG draw per round -- `BCEWithLogitsLoss`.
  8. Compute global EOD/DPD fairness metrics on synthetic data (pure arithmetic).
  9. Compute a per-SELECTED-CLIENT fairness-bias scalar Fi: clone the global model, load that
     client's OWN state dict, score it on the SAME synthetic data (deterministic, no RNG).
 10. AFW alphas: `alpha_i = max(0,Fi) / sum(max(0,Fi))`, falling back to uniform weights if the
     total is ~0 (pure arithmetic).
 11. `aggregate_state_dicts`: converts every tensor to NUMPY, accumulates `s = w*a if s is None
     else s+w*a` per parameter key in `client_ids = list(client_state_dicts.keys())` order
     (dict-insertion order == this round's client-selection order), then converts BACK to torch
     via `torch.tensor(v, dtype=first_val.dtype, device=first_val.device)` using the FIRST
     key's dtype for every key (harmless here: every TabularNet parameter is float32 anyway,
     but the mechanism is replicated literally, not approximated with pure-PyTorch tensor ops,
     to avoid any float32/float64 numpy/torch promotion-rule mismatch).

Per-round RNG draws against the two TRACKED global streams, in order:
  numpy legacy:  1 draw for `np.random.choice(range(K), m, replace=False)` (client selection,
                 unweighted, identical shape to FedAvg-P2/FairTrade), then 2 draws for AGDG's
                 `multivariate_normal` (group 0, then group 1).
  torch default: 1 draw per SELECTED client (in `sel` iteration order) for that client's local
                 `DataLoader(shuffle=True)` (local_epochs=1 harmonised), then 1 more draw for the
                 server-side synthetic-data `DataLoader(shuffle=True)` (also 1 epoch, the
                 `train_on_dataset` default) -- AFTER all client draws, since step 7 runs after
                 every selected client has already been locally trained in `_round`.
  isolated (np.random.default_rng(42) inside LSH_CADC.fit): never touches either tracked
                 stream; genuinely out-of-band.

Model (`fl/models.py::GFLTabularNet`, matching repo:p2 main.py::TabularNet): a per-categorical-
column `nn.Embedding` (built FIRST, in column order) feeding into a 3-layer
Linear(in_dim,hidden)->ReLU->Linear(hidden,hidden)->ReLU->Linear(hidden,1) (built SECOND, logits
output). `in_dim = d_num + emb_dim*len(cat_cardinalities)`. Constructed ONCE in `__init__`
(no per-client throwaway models -- local models come from `clone()`, i.e. `copy.deepcopy`).

Data representation: `val`/each client dict's `X` is the GFL-native "codes" matrix (numeric
block followed by integer category codes, columns `[:d_num]`/`[d_num:]`) -- NOT the standard
one-hot matrix every other algorithm in this codebase uses. `val["meta"] = {"d_num":...,
"cat_cards":[...]}` carries the split needed to reconstruct `(X_num, X_cat)`.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fl.federated.common import seed_all
from fl.models import GFLTabularNet


# ================================================================================================
# SIC -- Statistical Information Collection (repo:p2 clients/statistics.py::ClientStatistics)
# ================================================================================================

def _client_statistics(X_num: np.ndarray, g: np.ndarray) -> dict[int, dict]:
    """Per-group {num, mu, sigma} (pure arithmetic; `y` is accepted by the original for
    extension but never used -- omitted here). `sigma` gets a 1e-6 diagonal-jitter ridge."""
    stats: dict[int, dict] = {}
    for group in np.unique(g):
        idx = g == group
        Xg = X_num[idx]
        if Xg.shape[0] == 0:
            continue
        num = Xg.shape[0]
        mu = Xg.mean(axis=0)
        sigma = np.cov(Xg, rowvar=False)
        sigma = sigma + 1e-6 * np.eye(sigma.shape[0])
        stats[int(group)] = {"num": int(num), "mu": mu, "sigma": sigma}
    return stats


# ================================================================================================
# AGDG -- Approximate Global Dataset Generation (repo:p2 server/agdg.py)
# ================================================================================================

def _agdg_aggregate_statistics(client_stats: list[dict[int, dict]]) -> dict[int, dict]:
    """Weighted mean/covariance across clients per group (pure arithmetic; a SECOND 1e-6
    diagonal-jitter ridge is added here, on top of the one already baked into each client's own
    `sigma` -- both additions are replicated, not deduplicated, matching the original)."""
    global_stats: dict[int, dict] = {}
    groups: set[int] = set()
    for cs in client_stats:
        groups.update(cs.keys())
    for g in groups:
        total_num = sum(cs[g]["num"] for cs in client_stats if g in cs)
        mu = sum((cs[g]["num"] / total_num) * cs[g]["mu"] for cs in client_stats if g in cs)
        sigma = sum((cs[g]["num"] / total_num) * cs[g]["sigma"] for cs in client_stats if g in cs)
        sigma = sigma + 1e-6 * np.eye(sigma.shape[0])
        global_stats[g] = {"num": int(total_num), "mu": mu, "sigma": sigma}
    return global_stats


def _agdg_generate_synthetic_numeric(global_stats: dict[int, dict]) -> dict[int, dict]:
    """ONE `np.random.multivariate_normal` draw per group (GLOBAL numpy legacy RNG), iterated
    in `global_stats.items()` order -- ascending for binary {0,1} groups in CPython."""
    synth: dict[int, dict] = {}
    for g, s in global_stats.items():
        num, mu, sigma = s["num"], s["mu"], s["sigma"]
        Xg = np.random.multivariate_normal(mean=mu, cov=sigma, size=num)
        synth[g] = {"X_num": Xg, "g": np.full(num, g)}
    return synth


def _agdg_run(client_stats: list[dict[int, dict]]) -> tuple[dict[int, dict], dict[int, dict]]:
    global_stats = _agdg_aggregate_statistics(client_stats)
    synth_data = _agdg_generate_synthetic_numeric(global_stats)
    return synth_data, global_stats


# ================================================================================================
# LSH-CADC -- categorical attribute completion (repo:p2 server/lsh_cadc.py)
# ================================================================================================

def _safe_pinv_cov(cov: np.ndarray | None, d: int) -> np.ndarray:
    if cov is None:
        return np.eye(d, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] != d:
        return np.eye(d, dtype=float)
    cov = cov + 1e-6 * np.eye(d, dtype=float)
    try:
        return np.linalg.pinv(cov)
    except Exception:
        return np.eye(d, dtype=float)


class LSHCadc:
    """repo:p2 server/lsh_cadc.py::LSH_CADC. `fit()` uses an ISOLATED
    `np.random.default_rng(seed)` generator (seed=42 default, never overridden by the server's
    own call site) -- this never touches the tracked global numpy legacy RNG."""

    def __init__(self, k: int = 7):
        self.k = k
        self._W = None
        self._X_num = None
        self._X_cat = None
        self._g = None
        self._cov_inv: dict[int, np.ndarray] | None = None
        self._buckets: dict[int, dict[int, np.ndarray]] | None = None

    def fit(self, X_num: np.ndarray, X_cat: np.ndarray, g: np.ndarray,
            cov_matrices: dict[int, np.ndarray], seed: int = 42) -> "LSHCadc":
        X_num = np.asarray(X_num, dtype=float)
        if X_num.ndim == 1:
            X_num = X_num.reshape(-1, 1)
        X_cat_arr = np.asarray(X_cat)
        if X_cat_arr.size:
            X_cat = np.asarray(X_cat, dtype=int)
            if X_cat.ndim == 1:
                X_cat = X_cat.reshape(-1, 1)
        else:
            X_cat = np.empty((X_num.shape[0], 0), dtype=int)
        g = np.asarray(g).reshape(-1)

        N, d = X_num.shape
        cov_norm = {int(kk): np.asarray(vv, dtype=float) for kk, vv in cov_matrices.items()}

        groups_present = sorted(set(int(x) for x in np.unique(g)))
        cov_inv: dict[int, np.ndarray] = {}
        for grp in groups_present:
            cov = cov_norm.get(grp, None)
            if cov is None and len(cov_norm) > 0:
                cov = next(iter(cov_norm.values()))
            cov_inv[grp] = _safe_pinv_cov(cov, d)
        if not cov_inv:
            cov_inv = {0: np.eye(d, dtype=float), 1: np.eye(d, dtype=float)}

        rng = np.random.default_rng(seed)  # ISOLATED -- no interaction w/ global numpy state
        W = rng.normal(size=(self.k, d)).astype(float)

        def hash_vec(x: np.ndarray) -> int:
            bits = (W @ x) >= 0.0
            h = 0
            for b in bits.astype(int):
                h = (h << 1) | int(b)
            return int(h)

        buckets: dict[int, dict[int, list[int]]] = {}
        for i in range(N):
            grp = int(g[i])
            h = hash_vec(X_num[i])
            buckets.setdefault(grp, {}).setdefault(h, []).append(i)
        buckets_np = {grp: {h: np.asarray(idxs, dtype=int) for h, idxs in bd.items()}
                      for grp, bd in buckets.items()}

        self._W, self._X_num, self._X_cat, self._g = W, X_num, X_cat, g
        self._cov_inv, self._buckets = cov_inv, buckets_np
        return self

    def _hash_vec(self, x: np.ndarray) -> int:
        bits = (self._W @ x) >= 0.0
        h = 0
        for b in bits.astype(int):
            h = (h << 1) | int(b)
        return int(h)

    @staticmethod
    def _mahalanobis(x: np.ndarray, Xcand: np.ndarray, VI: np.ndarray) -> np.ndarray:
        diff = Xcand - x.reshape(1, -1)
        return np.einsum("ij,jk,ik->i", diff, VI, diff)

    def complete(self, synth_numeric_by_group: dict[int, np.ndarray],
                 synth_groups_by_group: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        X_num_real, X_cat_real, g_real = self._X_num, self._X_cat, self._g
        buckets, cov_inv = self._buckets, self._cov_inv
        d_cat = X_cat_real.shape[1]
        out: dict[int, np.ndarray] = {}

        for grp_key, Xn_s in synth_numeric_by_group.items():
            grp = int(grp_key)
            Xn_s = np.asarray(Xn_s, dtype=float)
            if Xn_s.ndim == 1:
                Xn_s = Xn_s.reshape(-1, 1)
            if Xn_s.shape[0] == 0:
                out[grp] = np.empty((0, d_cat), dtype=int)
                continue

            VI = cov_inv.get(grp, next(iter(cov_inv.values())))
            Xc_s = np.empty((Xn_s.shape[0], d_cat), dtype=int)

            same_group_idxs = np.where(g_real.astype(int) == grp)[0]
            if same_group_idxs.size == 0:
                same_group_idxs = np.arange(X_num_real.shape[0])

            for i in range(Xn_s.shape[0]):
                x = Xn_s[i]
                h = self._hash_vec(x)
                cand_idxs = None
                if grp in buckets and h in buckets[grp]:
                    cand_idxs = buckets[grp][h]
                if cand_idxs is None or cand_idxs.size == 0:
                    cand_idxs = same_group_idxs
                Xcand = X_num_real[cand_idxs]
                dists = self._mahalanobis(x, Xcand, VI)
                j = int(np.argmin(dists))
                best_idx = int(cand_idxs[j])
                if d_cat > 0:
                    Xc_s[i] = X_cat_real[best_idx]
            out[grp] = Xc_s
        return out


# ================================================================================================
# GBCFSS -- Greedy-Based Clean and Fair Sample Selection (repo:p2 server/gbcfss.py)
# Fully deterministic -- no RNG anywhere in this section.
# ================================================================================================

def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, 1.0 - eps))


def _binary_log_loss(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    y_true = y_true.astype(np.float64)
    y_score = y_score.astype(np.float64)
    return -(y_true * _safe_log(y_score) + (1.0 - y_true) * _safe_log(1.0 - y_score))


def _fairness_contribution_scores(y_true: np.ndarray, g: np.ndarray, mode: str) -> np.ndarray:
    y_true = y_true.reshape(-1)
    g = g.reshape(-1)
    s = np.full_like(y_true, 0.5, dtype=np.float64)

    if mode == "eo":
        pos = y_true == 1
        if not np.any(pos):
            return s
        n0 = np.sum((g == 0) & pos)
        n1 = np.sum((g == 1) & pos)
        disadvantaged = 0 if n0 < n1 else 1
        s[pos & (g == disadvantaged)] = 1.0
        s[pos & (g != disadvantaged)] = 0.0
        return s

    if mode == "dp":
        n0 = np.sum(g == 0)
        n1 = np.sum(g == 1)
        disadvantaged = 0 if n0 < n1 else 1
        s[(g == disadvantaged)] = 1.0
        s[(g != disadvantaged)] = 0.0
        return s

    raise ValueError(f"unknown GBCFSS fairness_mode {mode!r}")


def _greedy_select_gbcfss(y_true: np.ndarray, g: np.ndarray, y_score: np.ndarray, *,
                           budget: int, clean_w: float, fair_w: float,
                           group_balance: float | None, fairness_mode: str) -> np.ndarray:
    y_true = y_true.reshape(-1)
    g = g.reshape(-1)
    y_score = y_score.reshape(-1)
    N = y_true.shape[0]
    budget = min(budget, N)

    loss = _binary_log_loss(y_true, y_score)
    lo, hi = np.percentile(loss, [5, 95])
    denom = max(hi - lo, 1e-12)
    clean = 1.0 - np.clip((loss - lo) / denom, 0.0, 1.0)

    fair = _fairness_contribution_scores(y_true, g, mode=fairness_mode)
    score = clean_w * clean + fair_w * fair
    order = np.argsort(-score)

    if group_balance is None:
        return np.array(order[:budget].tolist(), dtype=int)

    min_per_group = int(np.floor(group_balance * budget))
    count = {0: 0, 1: 0}
    selected: list[int] = []

    for idx in order:
        if len(selected) >= budget:
            break
        gi = int(g[idx])
        if count[gi] < min_per_group:
            selected.append(int(idx))
            count[gi] += 1
            continue
        if count[0] >= min_per_group and count[1] >= min_per_group:
            selected.append(int(idx))
            count[gi] += 1
            continue
        other = 1 - gi
        if count[other] < min_per_group:
            continue
        selected.append(int(idx))
        count[gi] += 1

    if len(selected) < budget:
        sel_set = set(selected)
        remaining = [int(i) for i in order if int(i) not in sel_set]
        selected.extend(remaining[: budget - len(selected)])

    return np.array(selected, dtype=int)


# ================================================================================================
# Fairness metrics (repo:p2 server/fairness.py) -- pure arithmetic.
# ================================================================================================

def _safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(x.mean())


def _metrics_from_scores(y_score: np.ndarray, y_true: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    """Returns (eod, dpd)."""
    y_score = np.asarray(y_score).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    g = np.asarray(g).reshape(-1)

    mask_g0, mask_g1 = g == 0, g == 1
    dpd = abs(_safe_mean(y_score[mask_g1]) - _safe_mean(y_score[mask_g0]))

    pos = y_true == 1
    eod = abs(_safe_mean(y_score[mask_g1 & pos]) - _safe_mean(y_score[mask_g0 & pos]))
    return eod, dpd


def _client_fairness_bias(eod: float, dpd: float, mode: str) -> float:
    if mode == "sum":
        return eod + dpd
    if mode == "max":
        return max(eod, dpd)
    raise ValueError(f"unknown bias_scalar_mode {mode!r}")


# ================================================================================================
# AFW aggregation (repo:p2 server/aggregation.py) -- numpy round-trip, replicated literally.
# ================================================================================================

def _compute_alphas(fairness_bias: dict[int, float], eps: float = 1e-12) -> dict[int, float]:
    total = float(sum(max(0.0, v) for v in fairness_bias.values()))
    if total <= eps:
        n = len(fairness_bias)
        return {cid: 1.0 / n for cid in fairness_bias}
    return {cid: float(max(0.0, Fi) / total) for cid, Fi in fairness_bias.items()}


def _aggregate_state_dicts(client_state_dicts: dict[int, dict], alphas: dict[int, float]) -> dict:
    client_ids = list(client_state_dicts.keys())
    keys = list(next(iter(client_state_dicts.values())).keys())

    agg_np: dict[str, np.ndarray] = {}
    for k in keys:
        s = None
        for cid in client_ids:
            w = client_state_dicts[cid][k].detach().cpu().numpy()
            a = alphas[cid]
            s = w * a if s is None else s + w * a
        agg_np[k] = s

    first_val = next(iter(client_state_dicts.values()))[keys[0]]
    return {k: torch.tensor(v, dtype=first_val.dtype, device=first_val.device) for k, v in agg_np.items()}


# ================================================================================================
# TorchModelWrapper-equivalent (repo:p2 main.py::TorchModelWrapper) + local client training.
# ================================================================================================

class _Wrapper:
    """Minimal port of repo:p2's `TorchModelWrapper`: `clone`/`state_dict`/`load_state_dict`/
    `predict_score`/`train_on_dataset`."""

    def __init__(self, model: nn.Module, d_num: int, lr: float, batch_size: int):
        self.model = model
        self.d_num = d_num
        self.lr = lr
        self.batch_size = batch_size

    def clone(self) -> "_Wrapper":
        return _Wrapper(copy.deepcopy(self.model), self.d_num, self.lr, self.batch_size)

    def state_dict(self) -> dict:
        return self.model.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.model.load_state_dict(sd)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X = np.asarray(X)
        X_num = torch.tensor(X[:, :self.d_num], dtype=torch.float32)
        X_cat_np = X[:, self.d_num:]
        X_cat = (torch.empty((X.shape[0], 0), dtype=torch.long) if X_cat_np.size == 0
                 else torch.tensor(X_cat_np, dtype=torch.long))
        with torch.no_grad():
            logits = self.model(X_num, X_cat)
            probs = torch.sigmoid(logits)
        return probs.detach().cpu().numpy().reshape(-1)

    def train_on_dataset(self, X: np.ndarray, y: np.ndarray, epochs: int = 1) -> None:
        self.model.train()
        X = np.asarray(X)
        y = np.asarray(y).astype(np.float32)
        X_num = torch.tensor(X[:, :self.d_num], dtype=torch.float32)
        X_cat_np = X[:, self.d_num:]
        X_cat = (torch.empty((X.shape[0], 0), dtype=torch.long) if X_cat_np.size == 0
                 else torch.tensor(X_cat_np, dtype=torch.long))
        y_t = torch.tensor(y, dtype=torch.float32)

        loader = DataLoader(TensorDataset(X_num, X_cat, y_t), batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(epochs):
            for xb_num, xb_cat, yb in loader:
                opt.zero_grad()
                logits = self.model(xb_num, xb_cat)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()


def _local_train_one_client(global_wrapper: _Wrapper, X_num: np.ndarray, X_cat: np.ndarray,
                             y: np.ndarray, *, local_epochs: int, lr: float, batch_size: int) -> dict:
    local = global_wrapper.clone()
    local.lr = lr
    local.batch_size = batch_size
    X_full = np.hstack([X_num, X_cat])
    local.train_on_dataset(X_full, y, epochs=local_epochs)
    return local.state_dict()


# ================================================================================================
# GFLServer.run_round -- the full 11-step orchestration (repo:p2 server/server.py).
# ================================================================================================

def _server_run_round(wrapper: _Wrapper, *, client_state_dicts: dict[int, dict],
                       client_stats: list[dict[int, dict]], real_X_num: np.ndarray,
                       real_X_cat: np.ndarray, real_g: np.ndarray, lsh_k: int,
                       use_gbcfss: bool, gbcfss_budget: int, gbcfss_clean_w: float,
                       gbcfss_fair_w: float, gbcfss_group_balance: float | None,
                       fairness_mode: str, bias_scalar_mode: str) -> None:
    # 1) AGDG
    synth_data, global_stats = _agdg_run(client_stats)

    # 2) LSH-CADC
    covs = {g: global_stats[g]["sigma"] for g in global_stats}
    lsh = LSHCadc(k=lsh_k)
    lsh.fit(real_X_num, real_X_cat, real_g, covs)
    synth_numeric = {g: synth_data[g]["X_num"] for g in synth_data}
    synth_groups = {g: synth_data[g]["g"] for g in synth_data}
    synth_cat = lsh.complete(synth_numeric, synth_groups)

    # 3) Build full synthetic feature matrix
    Xn_list, gg_list = [], []
    for g, d in synth_data.items():
        Xn_list.append(d["X_num"])
        gg_list.append(d["g"])
    X_num_synth = np.vstack(Xn_list)
    g_synth = np.concatenate(gg_list)
    X_cat_synth = (np.vstack([synth_cat[g] for g in synth_cat]) if real_X_cat.shape[1] > 0
                   else np.empty((X_num_synth.shape[0], 0)))
    X_synth = np.hstack([X_num_synth, X_cat_synth])

    # 4) Pseudo-label with current global model
    y_score = wrapper.predict_score(X_synth)
    y_synth = (y_score >= 0.5).astype(int)

    # 5) Score synthetic dataset with current global model
    y_score_synth = wrapper.predict_score(X_synth)

    # 6) Optional GBCFSS
    if use_gbcfss:
        sel_idx = _greedy_select_gbcfss(
            y_true=y_synth, g=g_synth, y_score=y_score_synth,
            budget=gbcfss_budget, clean_w=gbcfss_clean_w, fair_w=gbcfss_fair_w,
            group_balance=gbcfss_group_balance, fairness_mode=fairness_mode,
        )
        X_train, y_train = X_synth[sel_idx], y_synth[sel_idx]
    else:
        X_train, y_train = X_synth, y_synth

    # 7) Server-side training on synthetic data (1 epoch default)
    wrapper.train_on_dataset(X_train, y_train)

    # 8) Global fairness metrics (post-training)
    y_score_after = wrapper.predict_score(X_synth)
    _eod, _dpd = _metrics_from_scores(y_score_after, y_synth, g_synth)

    # 9) Per-client fairness bias Fi
    fairness_bias: dict[int, float] = {}
    for cid, w in client_state_dicts.items():
        m = wrapper.clone()
        m.load_state_dict(w)
        y_score_client = m.predict_score(X_synth)
        eod_c, dpd_c = _metrics_from_scores(y_score_client, y_synth, g_synth)
        fairness_bias[cid] = _client_fairness_bias(eod_c, dpd_c, mode=bias_scalar_mode)

    # 10) AFW alphas
    alphas = _compute_alphas(fairness_bias)

    # 11) Aggregate
    new_state = _aggregate_state_dicts(client_state_dicts, alphas)
    wrapper.load_state_dict(new_state)


# ================================================================================================
# Top-level algorithm classes.
# ================================================================================================

class GFL:
    family, variant = "P2", "GFL"
    input_kind = "codes"  # flags the equivalence harness to use transform_codes, not one-hot
    use_gbcfss, fairness_mode = False, "eo"
    harmonised_cfg = dict(rounds=50, local_epochs=1, batch=256, lr=1e-3, optimizer="adam", C=1.0,
                          hidden=128, emb_dim=8, global_lr=1e-3, global_bs=256, aux_source="val",
                          use_gbcfss=False, fairness_mode="eo")
    GBCFSS_BUDGET = 5000
    GBCFSS_GROUP_BALANCE = 0.2
    LSH_K = 7
    BIAS_SCALAR_MODE = "sum"

    def __init__(self, clients: list[dict], val: dict, seed: int, cfg: dict):
        seed_all(seed)
        self.cfg = cfg
        self.clients = clients
        self.K = len(clients)

        meta = val["meta"]
        self.d_num = int(meta["d_num"])
        cat_cards = meta["cat_cards"]

        model = GFLTabularNet(d_num=self.d_num, cat_cardinalities=cat_cards,
                               emb_dim=cfg["emb_dim"], hidden=cfg["hidden"])
        self.wrapper = _Wrapper(model, self.d_num, lr=cfg["global_lr"], batch_size=cfg["global_bs"])

        if cfg.get("aux_source", "val") == "val":
            aux = val
        else:
            aux = dict(
                X=np.concatenate([c["X"] for c in clients]),
                y=np.concatenate([c["y"] for c in clients]),
                a=np.concatenate([c["a"] for c in clients]),
            )
        d = self.d_num
        self.aux = (aux["X"][:, :d], aux["X"][:, d:], aux["a"].astype(int), aux["y"].astype(int))

    @property
    def model(self) -> nn.Module:
        return self.wrapper.model

    def _round(self, r: int) -> None:
        cfg, d = self.cfg, self.d_num
        m = max(int(cfg["C"] * self.K), 1)
        sel = np.random.choice(np.arange(self.K), m, replace=False)

        sds: dict[int, dict] = {}
        stats: list[dict[int, dict]] = []
        for cid in sel:
            cid = int(cid)
            c = self.clients[cid]
            sds[cid] = _local_train_one_client(
                self.wrapper, c["X"][:, :d], c["X"][:, d:], c["y"],
                local_epochs=cfg["local_epochs"], lr=cfg["lr"], batch_size=cfg["batch"],
            )
            stats.append(_client_statistics(c["X"][:, :d], c["a"].astype(int)))

        _server_run_round(
            self.wrapper, client_state_dicts=sds, client_stats=stats,
            real_X_num=self.aux[0], real_X_cat=self.aux[1], real_g=self.aux[2],
            lsh_k=self.LSH_K, use_gbcfss=cfg.get("use_gbcfss", self.use_gbcfss),
            gbcfss_budget=self.GBCFSS_BUDGET, gbcfss_clean_w=1.0, gbcfss_fair_w=1.0,
            gbcfss_group_balance=self.GBCFSS_GROUP_BALANCE,
            fairness_mode=cfg.get("fairness_mode", self.fairness_mode),
            bias_scalar_mode=self.BIAS_SCALAR_MODE,
        )

    def fit(self, rounds: int, hook=lambda r: None) -> None:
        for r in range(rounds):
            self._round(r)
            hook(r + 1)

    def proba(self, X):
        return self.wrapper.predict_score(np.asarray(X, dtype=np.float32)).astype(np.float64)

    def get_state(self) -> dict:
        return {k: v.detach().clone() for k, v in self.wrapper.state_dict().items()}

    def set_state(self, state: dict) -> None:
        self.wrapper.load_state_dict(state)


class GFLGBCFSSEO(GFL):
    variant = "GFL-GBCFSS-EO"
    use_gbcfss, fairness_mode = True, "eo"
    harmonised_cfg = dict(GFL.harmonised_cfg, use_gbcfss=True, fairness_mode="eo")


class GFLGBCFSSDP(GFL):
    variant = "GFL-GBCFSS-DP"
    use_gbcfss, fairness_mode = True, "dp"
    harmonised_cfg = dict(GFL.harmonised_cfg, use_gbcfss=True, fairness_mode="dp")
