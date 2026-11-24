"""Phase 7: the actual benchmark driver -- runs ONE from-scratch algorithm on ONE (dataset,
client-distribution, K, seed) combination, using ONLY this project's own code
(`fl.data`/`fl.partition`/`fl.algorithms`). Unlike `scripts/check_equivalence.py` (Phases 1-6),
this module never imports anything from `/home/claude/work` (the oracle) -- the oracle's job was
to validate every algorithm is correct; this module is what actually produces the benchmark's
results, on full (non-subsampled) data, and is what gets shipped to the cluster.

Every result row records enough provenance to never be ambiguous about what produced it
(Pascal guide section 6: "Every result row must record the code commit ... and machine_id"):
`commit` (from a COMMIT.txt placed next to this package when shipped; None if there isn't one),
`code_hash` (a sha256 over every tracked .py file's contents, always computable, so a result is
traceable even without a COMMIT.txt), `machine_id` (hostname), and a wall-clock timestamp.

Failures are recorded, never hidden or silently skipped (status="Failed" + traceback), matching
the oracle's own `run_experiment`'s own error-handling convention.
"""
from __future__ import annotations

import hashlib
import json
import socket
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fl import config as C
from fl.algorithms import REGISTRY
from fl.algorithms.runner import run_from_scratch
from fl.data.loaders import load_raw
from fl.data.orientation import partition_coordinate_inputs
from fl.data.preprocess import Preprocessor, transform_codes
from fl.data.split import make_split
from fl.partition.attribute import partition_stats, partition_with_guard
from fl.partition.iid import iid_partition

METRIC_KEYS = ["accuracy", "balanced_accuracy", "sp_signed", "sp_abs", "eo", "eop", "di"]
_CACHE: dict = {}  # (dataset, seed) -> (raw, sp) -- load_raw/make_split are deterministic, reused
                    # across every (dist, K, algo) combo sharing the same (dataset, seed)


def exp_id(dataset: str, dist: str, K: int, algo: str, tier: str, seed: int) -> str:
    return f"{dataset}__{dist}{K}__{algo}__{tier}__s{seed}"


def _code_hash() -> str:
    """sha256 over every tracked .py file under fl/ and scripts/, sorted by path -- a stable
    provenance fingerprint that needs no git repo and can never silently drift unnoticed."""
    root = C.PROJECT_ROOT
    h = hashlib.sha256()
    paths = sorted(p for p in (root / "fl").rglob("*.py")) + sorted((root / "scripts").rglob("*.py"))
    for p in paths:
        if "__pycache__" in p.parts:
            continue
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _provenance() -> dict:
    commit_file = C.PROJECT_ROOT / "COMMIT.txt"
    commit = commit_file.read_text().strip() if commit_file.exists() else None
    return dict(commit=commit, code_hash=_code_hash(), machine_id=socket.gethostname(),
                timestamp=datetime.now(timezone.utc).isoformat())


def _prepared(dataset: str, seed: int):
    key = (dataset, seed)
    if key not in _CACHE:
        _CACHE.clear()  # one (dataset, seed) resident at a time -- ACS alone is ~1.4 GB in memory
        raw = load_raw(dataset)
        sp = make_split(raw, seed)
        _CACHE[key] = (raw, sp)
    return _CACHE[key]


def build_partition(raw, sp, dist: str, K: int, seed: int) -> tuple[list, dict]:
    tr = sp.idx_train
    y_tr, a_tr = sp.y[tr], sp.a[tr]
    if dist == "iid":
        parts = iid_partition(len(tr), K, seed)
        stats = partition_stats(parts, y_tr, a_tr)
        info = dict(method="iid", attempts=1, feasible=True)
    elif dist == "attr":
        p_levels_tr = raw.p_levels[tr] if raw.p_levels is not None else None
        kind, values, order = partition_coordinate_inputs(
            p_levels_train=p_levels_tr, a_kind=raw.a_kind,
            a_raw_train=raw.a_raw[tr], a_oriented_train=a_tr, y_train=y_tr,
        )
        parts, stats, attempts, feasible = partition_with_guard(
            kind, values, order, y_tr, a_tr, K, C.AFFINITY_EPS, seed)
        info = dict(method="attribute_affinity", eps=C.AFFINITY_EPS, attempts=attempts,
                    feasible=bool(feasible), kind=kind,
                    level_order=None if order is None else [str(o) for o in order])
    else:
        raise ValueError(f"unknown dist {dist!r}; choices are 'iid'/'attr'")
    info.update(client_sizes=[int(s) for s in stats["sizes"]],
                a1_share=[float(s) for s in stats["a_share"]],
                pos_rate=[float(s) for s in stats["pos_rate"]],
                cells_a0y0_a0y1_a1y0_a1y1=stats["cells"].tolist(), tv_mean=stats["tv_mean"])
    return parts, info


def build_inputs(raw, sp, input_kind: str):
    """Fits one Preprocessor on TRAIN rows, transforms train/val/test. `input_kind="codes"`
    (GFL family only) additionally returns `meta={"d_num":..., "cat_cards":...}`; every other
    algorithm gets `meta=None` and the standard one-hot matrices."""
    pre = Preprocessor(raw.num_cols, raw.cat_cols).fit(raw.df.iloc[sp.idx_train])
    splits = (("train", sp.idx_train), ("val", sp.idx_val), ("test", sp.idx_test))
    if input_kind == "codes":
        X, meta = {}, None
        for name, ix in splits:
            X[name], dn, cards = transform_codes(pre, raw.df.iloc[ix])
            meta = dict(d_num=dn, cat_cards=cards)
        return X, meta, pre
    X = {name: pre.transform(raw.df.iloc[ix]) for name, ix in splits}
    return X, None, pre


def run_one(dataset: str, dist: str, K: int, algo: str, seed: int, *, tier: str = "harmonised",
            rounds: int | None = None, out_dir: Path | str = None, save: bool = True) -> dict:
    """Runs exactly one (dataset, dist, K, algo, seed) combination end to end and writes
    `<out_dir>/<dataset>/<exp_id>.json` (+ `.npz` with test predictions) if `save`. If that file
    already exists and is a completed run, returns it immediately without recomputing anything
    (resume-by-file-existence -- the Pascal guide's own "resubmitting must rerun nothing" rule).
    """
    out = Path(out_dir) if out_dir is not None else C.RESULTS_RUNS
    out_ds = out / dataset
    eid = exp_id(dataset, dist, K, algo, tier, seed)
    json_path = out_ds / f"{eid}.json"
    if save and json_path.exists():
        existing = json.loads(json_path.read_text())
        if existing.get("status") == "Completed":
            return existing

    res = dict(exp_id=eid, dataset=dataset, distribution=dist, K=K, algorithm=algo, tier=tier,
               seed=seed, status="Running", **_provenance())
    t0 = time.time()
    try:
        cls = REGISTRY[algo]
        raw, sp = _prepared(dataset, seed)
        parts, pinfo = build_partition(raw, sp, dist, K, seed)
        X, meta, _pre = build_inputs(raw, sp, getattr(cls, "input_kind", "dense"))

        tr = sp.idx_train
        Xtr, ytr, atr = X["train"], sp.y[tr], sp.a[tr]
        clients = [dict(X=Xtr[p], y=ytr[p], a=atr[p]) for p in parts]
        val = dict(X=X["val"], y=sp.y[sp.idx_val], a=sp.a[sp.idx_val], meta=meta)
        test = dict(X=X["test"], y=sp.y[sp.idx_test], a=sp.a[sp.idx_test])

        cfg = dict(cls.harmonised_cfg if tier == "harmonised" else cls.native_cfg)
        if rounds is not None:
            cfg["rounds"] = rounds

        out_mine = run_from_scratch(algo, clients, val, test, seed, cfg)
        res.update(status="Completed", config=cfg, partition=pinfo, split=sp.info,
                   results=out_mine["results"], val_history=out_mine["val_history"],
                   runtime_s=round(time.time() - t0, 1))
        if save:
            out_ds.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(res, indent=1))
            np.savez_compressed(out_ds / f"{eid}.npz",
                                **{f"test_pred_{k}": v for k, v in out_mine["preds"].items()})
    except Exception:  # noqa: BLE001 -- failures are recorded, never hidden (Section-5 honesty rule)
        res.update(status="Failed", error=traceback.format_exc(), runtime_s=round(time.time() - t0, 1))
        if save:
            out_ds.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(res, indent=1))
    return res
