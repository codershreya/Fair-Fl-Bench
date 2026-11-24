"""Tests for fl/merge.py (Phase 7 -> Phase 8 bridge). Builds its own tiny fake result JSONs
rather than depending on real cluster output, so this suite runs anywhere."""
from __future__ import annotations

import json
from pathlib import Path

from fl.merge import load_all_results


def _write_fake_result(out_dir: Path, dataset: str, algo: str, seed: int, status: str = "Completed") -> None:
    exp_id = f"{dataset}__iid3__{algo}__harmonised__s{seed}"
    rec = dict(
        exp_id=exp_id, dataset=dataset, distribution="iid", K=3, algorithm=algo, tier="harmonised",
        seed=seed, status=status, commit="fake-commit", code_hash="ab" * 32, machine_id="testhost",
        timestamp="2026-01-01T00:00:00+00:00", runtime_s=1.23,
        config=dict(rounds=50, local_epochs=1, lr=0.005, batch=128, optimizer="adam", loss="BCE", C=1.0),
        partition=dict(method="iid", feasible=True, client_sizes=[10, 10, 10],
                        a1_share=[0.5, 0.5, 0.5], pos_rate=[0.4, 0.4, 0.4], tv_mean=0.01),
        split=dict(n=100, n_train=70, n_val=15, n_test=15, orientation_flipped=False,
                    A1_group="x", A0_group="y", train_base_rate_A1=0.5, train_base_rate_A0=0.3),
        results=dict(
            last_round=dict(accuracy=0.8, balanced_accuracy=0.79, sp_signed=0.1, sp_abs=0.1,
                             eo=0.05, eop=0.05, di=0.9, tpr_a1=0.8, tpr_a0=0.7, fpr_a1=0.1,
                             fpr_a0=0.1, ppr_a1=0.5, ppr_a0=0.4, n_a1=7, n_a0=8,
                             positive_rate_pred=0.45, n=15, round=50, valid_predictions=True,
                             n_unique_pred=2, val_accuracy_of_this_model=0.81),
            best_val_acc=dict(round=42, accuracy=0.82),
        ),
        val_history=[],
    )
    if status != "Completed":
        rec["error"] = "SomeError: fake failure for testing"
    d = out_dir / dataset
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{exp_id}.json").write_text(json.dumps(rec))


def test_load_all_results_happy_path(tmp_path):
    _write_fake_result(tmp_path, "cac", "FedAvg-P1", 0)
    _write_fake_result(tmp_path, "cac", "FedAvg-P1", 1)
    df, problems = load_all_results(tmp_path)
    assert len(df) == 2
    assert problems == []
    assert set(df["exp_id"]) == {"cac__iid3__FedAvg-P1__harmonised__s0", "cac__iid3__FedAvg-P1__harmonised__s1"}
    # the chosen primary snapshot is last_round, not best_val_acc
    assert (df["accuracy"] == 0.8).all()
    assert (df["best_val_round"] == 42).all()  # kept only as a reference column
    assert (df["best_val_accuracy"] == 0.82).all()


def test_load_all_results_never_silently_drops_a_failure(tmp_path):
    _write_fake_result(tmp_path, "cac", "FedAvg-P1", 0, status="Completed")
    _write_fake_result(tmp_path, "cac", "FedAvg-P1", 1, status="Failed")
    df, problems = load_all_results(tmp_path)
    assert len(df) == 1
    assert len(problems) == 1
    assert problems[0]["reason"] == "status='Failed'"
    assert "SomeError" in problems[0]["error"]


def test_load_all_results_handles_unparseable_json(tmp_path):
    d = tmp_path / "cac"
    d.mkdir()
    (d / "broken.json").write_text("{not valid json")
    _write_fake_result(tmp_path, "cac", "FedAvg-P1", 0)
    df, problems = load_all_results(tmp_path)
    assert len(df) == 1
    assert len(problems) == 1
    assert problems[0]["reason"] == "parse_error"


def test_load_all_results_empty_dir(tmp_path):
    df, problems = load_all_results(tmp_path)
    assert len(df) == 0
    assert problems == []
