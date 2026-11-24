"""Unit tests for Phase 7's benchmark driver (fl/experiment.py + scripts/run_benchmark.py).
Unlike every prior phase's tests, these never touch the oracle (`/home/claude/work`) --
Phase 7 is OUR OWN pipeline end to end, which is exactly what these tests check: that
`run_one` wires `fl.data`/`fl.partition`/`fl.algorithms` together correctly, resumes without
recomputing, records honest provenance, and never hides a failure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fl.experiment import _code_hash, build_inputs, build_partition, exp_id, run_one  # noqa: E402
from fl.data.loaders import load_raw  # noqa: E402
from fl.data.split import make_split  # noqa: E402


def test_exp_id_format():
    assert exp_id("cac", "iid", 3, "FedAvg-P1", "harmonised", 0) == "cac__iid3__FedAvg-P1__harmonised__s0"


def test_code_hash_is_deterministic_and_hex():
    h1 = _code_hash()
    h2 = _code_hash()
    assert h1 == h2
    assert len(h1) == 64
    int(h1, 16)  # valid hex


def test_build_partition_iid_covers_every_train_row_disjointly():
    raw = load_raw("cac")
    sp = make_split(raw, seed=0)
    parts, info = build_partition(raw, sp, "iid", 3, seed=0)
    assert info["method"] == "iid" and info["feasible"] is True
    all_idx = sorted(i for p in parts for i in p)
    assert all_idx == list(range(len(sp.idx_train)))  # covers every train-row POSITION exactly once
    assert sum(info["client_sizes"]) == len(sp.idx_train)


def test_build_partition_attr_runs_feasibility_guard():
    raw = load_raw("cac")
    sp = make_split(raw, seed=0)
    parts, info = build_partition(raw, sp, "attr", 3, seed=0)
    assert info["method"] == "attribute_affinity"
    assert info["kind"] == "continuous"  # CaC's partition variable (racepctblack) is continuous
    assert sum(info["client_sizes"]) == len(sp.idx_train)
    assert len(info["cells_a0y0_a0y1_a1y0_a1y1"]) == 3  # one row per client


def test_build_inputs_dense_vs_codes_shapes_differ_for_gfl_family():
    raw = load_raw("cac")
    sp = make_split(raw, seed=0)
    X_dense, meta_dense, _ = build_inputs(raw, sp, "dense")
    X_codes, meta_codes, _ = build_inputs(raw, sp, "codes")
    assert meta_dense is None
    assert meta_codes is not None and "d_num" in meta_codes and "cat_cards" in meta_codes
    # codes representation has exactly d_num + len(cat_cols) columns (one int code per
    # categorical column), almost certainly narrower than the one-hot "dense" representation.
    assert X_codes["train"].shape[1] == meta_codes["d_num"] + len(meta_codes["cat_cards"])
    assert X_codes["train"].shape[0] == X_dense["train"].shape[0]


def test_run_one_end_to_end_light_algo(tmp_path):
    r = run_one("cac", "iid", 3, "FedAvg-P1", 0, rounds=2, out_dir=tmp_path)
    assert r["status"] == "Completed"
    assert set(r["results"]) == {"last_round", "best_val_acc"}
    for field in ("commit", "code_hash", "machine_id", "timestamp"):
        assert field in r
    assert (tmp_path / "cac" / f"{r['exp_id']}.json").exists()
    assert (tmp_path / "cac" / f"{r['exp_id']}.npz").exists()


def test_run_one_end_to_end_gfl_family_codes_path(tmp_path):
    r = run_one("cac", "attr", 3, "GFL-GBCFSS-DP", 0, rounds=2, out_dir=tmp_path)
    assert r["status"] == "Completed"
    assert set(r["results"]) == {"last_round", "best_val_acc"}


def test_run_one_resumes_without_recomputing(tmp_path):
    r1 = run_one("cac", "iid", 3, "FedAvg-P1", 0, rounds=2, out_dir=tmp_path)
    path = tmp_path / "cac" / f"{r1['exp_id']}.json"
    mtime_before = path.stat().st_mtime
    r2 = run_one("cac", "iid", 3, "FedAvg-P1", 0, rounds=2, out_dir=tmp_path)
    assert r2 == r1  # identical dict, not recomputed
    assert path.stat().st_mtime == mtime_before  # file was never rewritten


def test_run_one_records_failure_not_silently(tmp_path, monkeypatch):
    # Force a failure deep inside run_one (an algo not in REGISTRY) and confirm it's recorded,
    # not raised or silently dropped -- "never fabricate results" extends to never losing one.
    from fl.algorithms import REGISTRY
    bad_algo = "NotARealAlgorithm"
    assert bad_algo not in REGISTRY
    r = run_one("cac", "iid", 3, bad_algo, 0, rounds=2, out_dir=tmp_path)
    assert r["status"] == "Failed"
    assert "error" in r and "KeyError" in r["error"]
    saved = json.loads((tmp_path / "cac" / f"{r['exp_id']}.json").read_text())
    assert saved["status"] == "Failed"


def test_run_one_save_false_does_not_write_files(tmp_path):
    r = run_one("cac", "iid", 3, "FedAvg-P1", 0, rounds=2, out_dir=tmp_path, save=False)
    assert r["status"] == "Completed"
    assert not (tmp_path / "cac").exists()


def test_cli_settings_parser():
    import run_benchmark
    assert run_benchmark._settings("iid3,iid5,attr3") == [("iid", 3), ("iid", 5), ("attr", 3)]
    assert run_benchmark._settings(None) == run_benchmark.ALL_SETTINGS


def test_cli_csv_parser_rejects_unknown_values():
    import run_benchmark
    with pytest.raises(ValueError):
        run_benchmark._csv("NotADataset", run_benchmark.ALL_DATASETS)
    assert run_benchmark._csv("cac,adult", run_benchmark.ALL_DATASETS) == ["cac", "adult"]


def test_sweep_excludes_gfl_family_on_acs(tmp_path, capsys):
    # DEVIATIONS.md D11: GFL/GFL-GBCFSS-EO/GFL-GBCFSS-DP on ACS are explicitly excluded from the
    # sweep (prohibitive LSH-CADC completion cost at that scale) -- confirm the exclusion is
    # actually enforced in combo generation, not just documented.
    import argparse
    import run_benchmark

    args = argparse.Namespace(
        datasets="acs", algos="GFL,GFL-GBCFSS-EO,GFL-GBCFSS-DP,FedAvg-P1",
        seeds="0", settings="iid3", rounds=None, workers=1, limit=None,
        out_dir=str(tmp_path), dry_run=True, allow_excluded=False,
    )
    run_benchmark.cmd_sweep(args)
    out = capsys.readouterr().out
    assert "skipped 3" in out
    # the dry-run listing (not the NOTE line) must never mention an excluded algo on acs
    listing = out.split("to run,")[-1]
    assert "GFL-GBCFSS-EO" not in listing.replace("NOTE:", "")
    assert "GFL-GBCFSS-DP" not in listing.replace("NOTE:", "")
    assert " GFL " not in listing  # plain GFL also excluded (not just the GBCFSS variants)
    assert "FedAvg-P1" in listing  # unaffected algo on the same dataset still runs


def test_sweep_allow_excluded_overrides_the_acs_exclusion_deliberately(tmp_path, capsys):
    # --allow-excluded (added 2026-11-14 for the overnight GFL-family x acs catch-up run) must be
    # OFF by default (the exclusion still applies with no flag) and, when explicitly passed, must
    # actually include the excluded cells rather than silently doing nothing.
    import argparse
    import run_benchmark

    args_default = argparse.Namespace(
        datasets="acs", algos="GFL,FedAvg-P1", seeds="0", settings="iid3", rounds=None,
        workers=1, limit=None, out_dir=str(tmp_path), dry_run=True, allow_excluded=False,
    )
    run_benchmark.cmd_sweep(args_default)
    out = capsys.readouterr().out
    assert "skipped 1" in out
    assert "1 combos requested" in out  # only FedAvg-P1 survives

    args_override = argparse.Namespace(
        datasets="acs", algos="GFL,FedAvg-P1", seeds="0", settings="iid3", rounds=None,
        workers=1, limit=None, out_dir=str(tmp_path), dry_run=True, allow_excluded=True,
    )
    run_benchmark.cmd_sweep(args_override)
    out = capsys.readouterr().out
    assert "WARNING" in out and "on purpose" in out
    assert "2 combos requested" in out  # both GFL and FedAvg-P1 now present
    assert "GFL" in out.split("to run,")[-1]
