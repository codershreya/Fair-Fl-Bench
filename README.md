# Fair-FL Unified Benchmark

A from-scratch, single-codebase benchmark that re-implements **14 Fair Federated Learning
algorithm variants** drawn from two independent university course projects, runs them all under
one shared client/split/metric protocol, and reports them side by side on **5 datasets**.

No code here imports from either original project's repository: every algorithm and every
data-handling rule was independently re-derived by reading that source, then validated
side-by-side against it before any large-scale run (that validation step, and the two
originals it depends on, are not part of this slimmed-down repo — see "What's not here" below).
`DEVIATIONS.md` is the full, evidence-backed log of every place this re-implementation's
behaviour had to be pinned down, which directly affects how to read the results.

This repo is trimmed to exactly what's needed to **set it up and reproduce the experiment on an
ordinary laptop** — no cluster, no scheduler. If you want anything else (the original two
projects, the validation harness, a different compute setup), you're welcome to add it yourself.

## Status

**Complete: 1,750 / 1,750 cells filled.** 5 datasets x 14 algorithms x 5 client settings x 5
seeds, zero missing experiment cells. See `results/summary/master_results.csv` for the full table
and `DEVIATIONS.md` for the phase-by-phase build log (partitioning, metrics, per-algorithm
harmonisation choices).

**Analysis:** exploratory analysis and visualizations of the results are in this Colab notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19dEIPiXVaPy3LRnDN9qxln9bcFiSzSr3?usp=sharing)
A final Excel workbook summarising everything is still in progress.

## The two source projects

| | Algorithms in this benchmark |
|---|---|
| **Paper 1** (Arnous, Schrader, Olschewski) | `FedAvg-P1`, `FedMinMax`, `EOFedMinMax`, `GlobalGroup-SP`, `GlobalGroup-EO`, `TrustFed-SP`, `TrustFed-EO` |
| **Paper 2** (Thern, Rehn) | `FedAvg-P2`, `GIFAIR-FL`, `FairTrade-SP`, `FairTrade-ATE`* , `GFL`, `GFL-GBCFSS-EO`, `GFL-GBCFSS-DP` |

*`FairTrade-ATE` is an extra variant added by this benchmark, not present in the original project.*

The two original repositories are classmates' own coursework and are not included here; they're
cited only as the documented provenance for each algorithm (`DEVIATIONS.md` cites the exact
file/function each implementation was derived from).

## Datasets

Adult, Bank Marketing, Communities & Crime ("CaC"), Census-Income KDD, and ACS (American
Community Survey, per-US-state files) — all standard public fairness-ML benchmark datasets.
Not included in this repo (see Setup); put your own copies under `data/raw/`.

## Layout

```
fl/                 the unified codebase
  config.py            shared protocol constants (seeds, client fraction, splits, ...)
  data/                loaders, preprocessing, train/val/test split (fit on TRAIN only)
  partition/           IID and attribute-based client partitioning
  metrics.py           fairness/accuracy metrics (undefined -> NaN, never 0)
  models.py            shared model architectures
  federated/           one module per algorithm: client/server training loop
  algorithms/          thin per-algorithm adapters used by the runner
  experiment.py         one experiment = dataset x setting x algorithm x seed, resumable
  merge.py             flattens every results/runs/*.json into one summary table
scripts/
  run_benchmark.py       main CLI: run one combo, time one combo, or sweep many (resumable)
  merge_results.py       builds results/summary/master_results.csv from results/runs/
tests/               pytest suite (metrics, partitioning, no-leakage, merge, per-phase)
results/
  runs/                  one JSON per experiment (gitignored: regenerate by re-running)
  summary/master_results.csv   the full 1,750-row flattened table (tracked)
environment.yml      conda/micromamba environment spec
DEVIATIONS.md        every deviation from the original code / harmonisation choice, with evidence
```

### What's not here

A full history of this project (phase-by-phase write-ups, the oracle-equivalence validation
harness and its evidence, the two original repos, the course PDFs, and the SLURM scripts used to
run the full sweep on a university cluster) exists but is deliberately left out of this repo,
since none of it is needed to set up and reproduce the experiment on a laptop. If you want any of
that, you'll need to recreate it yourself (e.g. write your own SLURM/cluster submission script
around `scripts/run_benchmark.py` — the CLI itself has no cluster dependency, it's a plain
Python process).

## Setup

```bash
conda env create -f environment.yml   # or: micromamba create -f environment.yml
conda activate fair-fl-benchmark      # or: micromamba activate fair-fl-benchmark
```
Requires Python 3.10. The datasets are not included in this repository (several hundred MB) —
place your own copies under `data/raw/` (`adult.csv`, `bank.csv`, `communities.csv`, `kdd.csv`,
and `acs/<state>_data.csv` per US state) before running anything.

## Running the tests

```bash
python3 -m pytest tests/ -v
```

## Reproducing a single result

```bash
python3 scripts/run_benchmark.py run \
  --dataset adult --algo FedAvg-P1 --dist iid --k 3 --seed 0
```
(`run_benchmark.py run --help` lists every option; `--dataset`/`--algo` accept any of the 5
datasets / 14 algorithms, `--dist` is `iid` or `attr`, `--k` is the client count.)

## Reproducing the full benchmark (or any slice of it)

```bash
python3 scripts/run_benchmark.py sweep \
  --datasets adult,bank,cac,kdd,acs \
  --algos FedAvg-P1,FedMinMax,EOFedMinMax,GlobalGroup-SP,GlobalGroup-EO,TrustFed-SP,TrustFed-EO,FedAvg-P2,GIFAIR-FL,FairTrade-SP,FairTrade-ATE,GFL,GFL-GBCFSS-EO,GFL-GBCFSS-DP \
  --settings iid10,iid5,iid3,attr5,attr3 \
  --seeds 0,1,2,3,4 \
  --workers 4
```
Every flag defaults to "all" if omitted, so a bare `sweep` with no flags reproduces the entire
1,750-cell grid (this took on the order of weeks of wall-clock time on a 12-core cluster node for
the largest dataset — expect it to take a while on a laptop; use `--datasets`/`--algos`/`--settings`
to run a smaller slice, and `--workers` to match your machine's CPU core count). The sweep is
**resumable**: each combination is one JSON file under `results/runs/`, and a combo already marked
`"status": "Completed"` is never recomputed, so you can safely stop and restart it. Add `--dry-run`
to preview the combo list without running anything, or `--limit N` to cap it for a quick smoke
test.

## Rebuilding the summary table

```bash
python3 scripts/merge_results.py
```
Flattens every file in `results/runs/` into `results/summary/master_results.csv`. Three of its 60
columns can be legitimately missing per-row (`DEVIATIONS.md` has the full breakdown): `loss` and
`C` are null for specific algorithms whose own config dict never recorded that field (a documented
code-level fact, not incomplete logging — the real value still exists in that algorithm's training
code); `di` (Disparate Impact) is `NaN`, never `0`, whenever its denominator is mathematically
undefined for that run. `pandas`' default `.mean()` (skips `NaN`) is the correct way to aggregate
these columns; `.fillna(0)` must never be applied to them.

## Principles (do not deviate without recording it in `DEVIATIONS.md`)
* No code here imports from either original project's repository: every algorithm and every
  data-handling rule is independently re-implemented from reading that source.
* The sensitive attribute is removed from the model input `X` by the data loaders themselves.
* Every data-dependent quantity (medians, encoders, scalers, orientation) is fit on the TRAIN
  split only; the test set is never used for any kind of selection.
* Metrics return `NaN`, never `0`, for undefined quantities.

## License

MIT — see `LICENSE`. This covers this project's own code only; the datasets keep their own
original licenses/terms, and the two original student repositories cited above are not included.
