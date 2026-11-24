"""Phase 7 CLI driver -- the script that actually ships to the cluster. Three subcommands:

  run     one (dataset, dist, K, algo, seed) combination -- the unit a SLURM array task runs.
  sweep   many combinations, resumable, parallelised across worker PROCESSES (not threads -- each
          worker calls torch.set_num_threads(1) AND runs with BLAS pinned to 1 thread, see below,
          so N workers on N cores don't oversubscribe each other; matches the Pascal guide's own
          "worker scaling" pattern, section 4).
  measure times ONE combination twice (rounds=1, then rounds=N) in the SAME process, so the
          second call hits fl.experiment's (dataset, seed) cache and measures pure per-round
          training cost with data-loading/preprocessing/partitioning cost already paid -- never
          guesses timing from another project's numbers (Pascal guide section 4's explicit
          warning), always measures THIS codebase's THIS algorithm on REAL full data first.

Usage:
  python3 scripts/run_benchmark.py run --dataset cac --algo FedAvg-P1 --dist iid --k 3 --seed 0
  python3 scripts/run_benchmark.py measure --dataset acs --algo GFL-GBCFSS-EO --dist iid --k 5 --seed 0 --rounds 50
  python3 scripts/run_benchmark.py sweep --workers 12 --limit 20
  python3 scripts/run_benchmark.py sweep --datasets cac,adult --algos FedAvg-P1,GFL --workers 4 --dry-run
"""
from __future__ import annotations

import os

# MUST run before numpy is imported anywhere (including transitively, via the `fl` imports right
# below) -- OpenBLAS/MKL read these once, at first use, not dynamically. `torch.set_num_threads(1)`
# (in `_worker_init`, for sweep workers) only pins PyTorch's own thread pool; the GFL family's
# LSH-CADC completion step (fl/federated/gfl.py, DEVIATIONS.md D10) runs its bottleneck matmul
# through plain NUMPY, which has its own, separately-controlled BLAS thread pool. Verified on
# Pascal directly: a fresh mamba env has NEITHER OMP_NUM_THREADS NOR OPENBLAS_NUM_THREADS set, so
# BLAS was free to auto-parallelize -- harmless for a single `run`/`measure` invocation, but had
# this gone unnoticed into `sweep`, N worker PROCESSES each ALSO spawning multiple BLAS threads on
# the same allocated CPUs is exactly the oversubscription the Pascal guide warns against (section
# 4), and would have made a job with MORE allocated CPUs potentially run slower, not faster, from
# thread contention. `setdefault` so an operator can still deliberately override via their own
# shell (e.g. to get a one-off multi-threaded number while debugging a single `measure` call).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fl import config as C  # noqa: E402
from fl.algorithms import REGISTRY  # noqa: E402
from fl.experiment import run_one  # noqa: E402

ALL_DATASETS = list(C.DATASETS)
ALL_ALGOS = list(REGISTRY)
ALL_SEEDS = list(C.SEEDS)
ALL_SETTINGS = [("iid", k) for k in C.IID_K_VALUES] + [("attr", k) for k in C.ATTR_K_VALUES]

# DEVIATIONS.md D10/D11: the whole GFL family (GFL, GFL-GBCFSS-EO, GFL-GBCFSS-DP) shares an
# AGDG+LSH-CADC completion step that runs unconditionally every round and scales ~O(N^2/128) in
# the real per-group training-set size (LSH's bucket count is fixed at 2^k=128). Verified on real
# data: ~100s/round on KDD even after vectorizing + BLAS-reformulating that step; ACS's ~5.5x more
# rows would cost roughly 5.5^2~30x more per round (on the order of an hour+ per round), making a
# real 50-round x 25 (setting,seed) grid prohibitive for all three algorithms on this one dataset.
# Decided with the project owner (2026-11-11) to exclude these 3 cells rather than accept a
# materially riskier numerical shortcut or burn disproportionate cluster time on them.
EXCLUDED_COMBOS = {("acs", "GFL"), ("acs", "GFL-GBCFSS-EO"), ("acs", "GFL-GBCFSS-DP")}


def _csv(s: str | None, choices: list) -> list:
    if not s:
        return list(choices)
    vals = [v.strip() for v in s.split(",") if v.strip()]
    bad = [v for v in vals if v not in choices]
    if bad:
        raise ValueError(f"unknown value(s) {bad}; choices are {choices}")
    return vals


def _settings(s: str | None) -> list:
    """`--settings iid3,iid5,attr3` -> [("iid",3), ("iid",5), ("attr",3)]."""
    if not s:
        return list(ALL_SETTINGS)
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        dist = "attr" if tok.startswith("attr") else "iid"
        k = int(tok[len(dist):])
        if (dist, k) not in ALL_SETTINGS:
            raise ValueError(f"unknown setting {tok!r}; choices are {ALL_SETTINGS}")
        out.append((dist, k))
    return out


def _worker_init() -> None:
    import torch
    torch.set_num_threads(1)  # one thread per WORKER PROCESS -- the pool provides the parallelism


def _worker(kwargs: dict) -> dict:
    return run_one(**kwargs)


def cmd_run(args: argparse.Namespace) -> None:
    r = run_one(args.dataset, args.dist, args.k, args.algo, args.seed,
                rounds=args.rounds, out_dir=args.out_dir)
    print(json.dumps({k: v for k, v in r.items() if k not in ("val_history",)}, indent=1, default=str))


def cmd_measure(args: argparse.Namespace) -> None:
    out_dir = "/tmp/fairbench_measure_scratch"  # never touches the real results store
    kwargs = dict(dataset=args.dataset, dist=args.dist, K=args.k, algo=args.algo, seed=args.seed,
                  out_dir=out_dir, save=False)

    t0 = time.time()
    r1 = run_one(**kwargs, rounds=1)
    cold_s = time.time() - t0
    if r1["status"] != "Completed":
        print(f"FAILED at rounds=1: {r1.get('error')}")
        return

    t0 = time.time()
    rN = run_one(**kwargs, rounds=args.rounds)
    warm_s = time.time() - t0
    if rN["status"] != "Completed":
        print(f"FAILED at rounds={args.rounds}: {rN.get('error')}")
        return

    per_round_s = warm_s / args.rounds
    prep_s = max(cold_s - per_round_s, 0.0)
    print(f"dataset={args.dataset} dist={args.dist} k={args.k} algo={args.algo} seed={args.seed}")
    print(f"  data load+split+partition+preprocess (one-time, cached per dataset/seed): ~{prep_s:.2f}s")
    print(f"  per-round training cost (this algo/K/dataset):                            ~{per_round_s:.3f}s")
    print(f"  extrapolated single run at rounds={args.rounds}: {prep_s + per_round_s * args.rounds:.1f}s"
          f" ({(prep_s + per_round_s * args.rounds) / 60:.1f} min)")
    n_combos_this_dataset = len(ALL_SETTINGS) * len(ALL_SEEDS)  # 5 settings x 5 seeds, one algo
    print(f"  all {n_combos_this_dataset} (setting,seed) combos for just THIS algo on THIS "
          f"dataset, single worker, rounds={args.rounds}: "
          f"~{(prep_s * len(ALL_SEEDS) + per_round_s * args.rounds * n_combos_this_dataset) / 60:.1f} min"
          f" (prep is only paid once per seed, re-amortised across the 5 settings)")


def cmd_sweep(args: argparse.Namespace) -> None:
    datasets = _csv(args.datasets, ALL_DATASETS)
    algos = _csv(args.algos, ALL_ALGOS)
    seeds = [int(s) for s in _csv(args.seeds, [str(s) for s in ALL_SEEDS])]
    settings = _settings(args.settings)

    if args.allow_excluded:
        print(f"WARNING: --allow-excluded set -- running {sorted(EXCLUDED_COMBOS)} on purpose "
              f"(DEVIATIONS.md D11 override, deliberate, not a default-path run).")

    combos = []
    n_excluded = 0
    for ds in datasets:
        for dist, k in settings:
            for algo in algos:
                if (ds, algo) in EXCLUDED_COMBOS and not args.allow_excluded:
                    n_excluded += 1
                    continue
                for seed in seeds:
                    combos.append(dict(dataset=ds, dist=dist, K=k, algo=algo, seed=seed,
                                       rounds=args.rounds, out_dir=args.out_dir))
    if n_excluded:
        print(f"NOTE: skipped {n_excluded} (dataset,setting,algo) cells entirely -- "
              f"{sorted(EXCLUDED_COMBOS)} (see DEVIATIONS.md D11)")
    if args.limit:
        combos = combos[: args.limit]

    out_dir = Path(args.out_dir) if args.out_dir else C.RESULTS_RUNS
    pending = []
    for kw in combos:
        eid_path = out_dir / kw["dataset"] / f"{kw['dataset']}__{kw['dist']}{kw['K']}__{kw['algo']}__harmonised__s{kw['seed']}.json"
        if eid_path.exists():
            try:
                if json.loads(eid_path.read_text()).get("status") == "Completed":
                    continue
            except Exception:
                pass
        pending.append(kw)

    print(f"{len(combos)} combos requested, {len(combos) - len(pending)} already completed "
          f"(resume), {len(pending)} to run, {args.workers} worker(s)")
    if args.dry_run or not pending:
        for kw in pending[:20]:
            print(" ", kw["dataset"], kw["dist"], kw["K"], kw["algo"], kw["seed"])
        if len(pending) > 20:
            print(f"  ... and {len(pending) - 20} more")
        return

    t0 = time.time()
    n_ok = n_fail = 0
    ctx = mp.get_context("spawn")
    if args.workers <= 1:
        _worker_init()
        for kw in pending:
            r = _worker(kw)
            n_ok += r["status"] == "Completed"
            n_fail += r["status"] != "Completed"
            print(f"[{n_ok + n_fail}/{len(pending)}] {r['exp_id']} -> {r['status']} "
                  f"({r.get('runtime_s', '?')}s)", flush=True)
    else:
        with ctx.Pool(args.workers, initializer=_worker_init) as pool:
            for i, r in enumerate(pool.imap_unordered(_worker, pending), 1):
                n_ok += r["status"] == "Completed"
                n_fail += r["status"] != "Completed"
                print(f"[{i}/{len(pending)}] {r['exp_id']} -> {r['status']} "
                      f"({r.get('runtime_s', '?')}s)", flush=True)

    dt = time.time() - t0
    print(f"\n{n_ok}/{len(pending)} completed, {n_fail} failed, {dt:.1f}s total "
          f"({dt / max(len(pending), 1):.1f}s/combo average, {args.workers} workers)")
    if n_fail:
        print("NOTE: failed combos have status='Failed' + a traceback saved in their .json -- "
              "nothing was silently skipped or fabricated.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run exactly one combination")
    p_run.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    p_run.add_argument("--algo", required=True, choices=ALL_ALGOS)
    p_run.add_argument("--dist", required=True, choices=["iid", "attr"])
    p_run.add_argument("--k", type=int, required=True)
    p_run.add_argument("--seed", type=int, required=True)
    p_run.add_argument("--rounds", type=int, default=None, help="override harmonised_cfg's rounds")
    p_run.add_argument("--out-dir", default=None)
    p_run.set_defaults(func=cmd_run)

    p_meas = sub.add_parser("measure", help="time ONE combination on real data before sizing cluster jobs")
    p_meas.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    p_meas.add_argument("--algo", required=True, choices=ALL_ALGOS)
    p_meas.add_argument("--dist", required=True, choices=["iid", "attr"])
    p_meas.add_argument("--k", type=int, required=True)
    p_meas.add_argument("--seed", type=int, default=0)
    p_meas.add_argument("--rounds", type=int, default=50, help="the real harmonised round count to extrapolate to")
    p_meas.set_defaults(func=cmd_measure)

    p_sweep = sub.add_parser("sweep", help="run many combinations, resumable, multi-process")
    p_sweep.add_argument("--datasets", default=None, help="comma-separated, default all 5")
    p_sweep.add_argument("--algos", default=None, help="comma-separated, default all 14")
    p_sweep.add_argument("--seeds", default=None, help="comma-separated, default 0,1,2,3,4")
    p_sweep.add_argument("--settings", default=None, help="comma-separated e.g. iid3,iid5,attr3, default all 5")
    p_sweep.add_argument("--rounds", type=int, default=None, help="override harmonised_cfg's rounds")
    p_sweep.add_argument("--workers", type=int, default=1)
    p_sweep.add_argument("--limit", type=int, default=None, help="cap the combo list (debugging)")
    p_sweep.add_argument("--out-dir", default=None)
    p_sweep.add_argument("--dry-run", action="store_true", help="list what would run, run nothing")
    p_sweep.add_argument("--allow-excluded", action="store_true",
                          help="deliberately include EXCLUDED_COMBOS cells (DEVIATIONS.md D11) -- "
                               "off by default on purpose; this is an explicit override, not a "
                               "silent bypass")
    p_sweep.set_defaults(func=cmd_sweep)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
