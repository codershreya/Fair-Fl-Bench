"""Phase 7 -> Phase 8 bridge: consolidate the per-combo result JSONs written by
`fl.experiment.run_one` (results/runs/<dataset>/<exp_id>.json) into ONE master table.

This is the only place that decides which metric snapshot represents a combo's result: per the
project owner's explicit choice (2026-11-12), every row uses `results.last_round` (round
HARMONISED_ROUNDS=50 for every algorithm) rather than `results.best_val_acc`, so every row is
directly comparable across algorithms on an equal-rounds basis. `best_val_acc`'s own round number
is kept as a reference column (`best_val_round`) so a reader can see how much early-stopping would
have changed things, without making it the primary snapshot.

Nothing here recomputes metrics or touches the oracle -- it only reads what `run_one` already
wrote and reshapes it. A row is only included if status=="Completed"; a non-Completed row is
reported separately (never silently dropped) so a real failure can't vanish into a clean-looking
spreadsheet.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fl import config as C

# Columns pulled straight from results.last_round (the chosen primary snapshot).
_LAST_ROUND_FIELDS = [
    "accuracy", "balanced_accuracy", "sp_signed", "sp_abs", "eo", "eop", "di",
    "tpr_a1", "tpr_a0", "fpr_a1", "fpr_a0", "ppr_a1", "ppr_a0",
    "n_a1", "n_a0", "positive_rate_pred", "n", "round",
    "valid_predictions", "n_unique_pred", "val_accuracy_of_this_model",
]


def _flatten_one(rec: dict) -> dict:
    """One result JSON's parsed dict -> one flat row. Assumes status == 'Completed'."""
    cfg = rec.get("config", {})
    part = rec.get("partition", {})
    split = rec.get("split", {})
    lr = rec["results"]["last_round"]
    bv = rec["results"].get("best_val_acc", {})

    client_sizes = part.get("client_sizes") or []
    a1_share = part.get("a1_share") or []
    pos_rate = part.get("pos_rate") or []

    row = dict(
        exp_id=rec["exp_id"], dataset=rec["dataset"], distribution=rec["distribution"],
        K=rec["K"], algorithm=rec["algorithm"], tier=rec.get("tier"), seed=rec["seed"],
        status=rec["status"],
        commit=rec.get("commit"), code_hash=rec.get("code_hash"), machine_id=rec.get("machine_id"),
        timestamp=rec.get("timestamp"), runtime_s=rec.get("runtime_s"),
        rounds_cfg=cfg.get("rounds"), local_epochs=cfg.get("local_epochs"), lr_cfg=cfg.get("lr"),
        batch=cfg.get("batch"), optimizer=cfg.get("optimizer"), loss=cfg.get("loss"), C=cfg.get("C"),
        partition_method=part.get("method"), partition_feasible=part.get("feasible"),
        n_clients=len(client_sizes), client_size_min=min(client_sizes) if client_sizes else None,
        client_size_max=max(client_sizes) if client_sizes else None,
        a1_share_mean=(sum(a1_share) / len(a1_share)) if a1_share else None,
        pos_rate_mean=(sum(pos_rate) / len(pos_rate)) if pos_rate else None,
        tv_mean=part.get("tv_mean"),
        split_n=split.get("n"), n_train=split.get("n_train"), n_val=split.get("n_val"),
        n_test=split.get("n_test"), orientation_flipped=split.get("orientation_flipped"),
        a1_group=split.get("A1_group"), a0_group=split.get("A0_group"),
        train_base_rate_a1=split.get("train_base_rate_A1"), train_base_rate_a0=split.get("train_base_rate_A0"),
        best_val_round=bv.get("round"), best_val_accuracy=bv.get("accuracy"),
    )
    for f in _LAST_ROUND_FIELDS:
        row[f] = lr.get(f)
    return row


def load_all_results(out_dir: str | Path | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Read every *.json under out_dir (default RESULTS_RUNS), split into
    (completed_rows_df, non_completed_records). Never raises on a bad/missing field in one file --
    a parse failure for a single file is reported as a non-completed record (reason="parse_error"),
    the rest of the merge still proceeds, and nothing is silently skipped."""
    base = Path(out_dir) if out_dir else C.RESULTS_RUNS
    rows: list[dict] = []
    problems: list[dict] = []
    for path in sorted(base.rglob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except Exception as e:
            problems.append(dict(path=str(path), reason="parse_error", error=str(e)))
            continue
        status = rec.get("status")
        if status != "Completed":
            problems.append(dict(path=str(path), reason=f"status={status!r}",
                                  exp_id=rec.get("exp_id"), error=rec.get("error")))
            continue
        try:
            rows.append(_flatten_one(rec))
        except Exception as e:
            problems.append(dict(path=str(path), reason="flatten_error",
                                  exp_id=rec.get("exp_id"), error=str(e)))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["dataset", "distribution", "K", "algorithm", "seed"]).reset_index(drop=True)
    return df, problems


def expected_combo_count() -> int:
    """The grid size run_benchmark.py's sweep actually targets (mirrors its own combo-generation
    logic, including EXCLUDED_COMBOS) -- used to sanity-check load_all_results' row count against
    what SHOULD exist, never the other way around."""
    import sys
    sys.path.insert(0, str(C.PROJECT_ROOT / "scripts"))
    import run_benchmark as rb  # noqa: E402

    n = 0
    for ds in rb.ALL_DATASETS:
        for dist, k in rb.ALL_SETTINGS:
            for algo in rb.ALL_ALGOS:
                if (ds, algo) in rb.EXCLUDED_COMBOS:
                    continue
                n += len(rb.ALL_SEEDS)
    return n
