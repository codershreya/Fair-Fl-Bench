# DEVIATIONS.md

Every deviation from the original Paper-1 / Paper-2 code, and every harmonisation choice that
changes behaviour, is recorded here with: what, why, evidence, effect. Phase-tagged as it grows.
See also reference/analysis/AUDIT_AND_PROTOCOL.md and INTEGRATION_REPORT.md (earlier, exploratory
audit of the same two repos; informative but superseded by this file and by the code itself
wherever the two disagree).

## Phase 1 (Core: loaders, split, preprocessing, orientation, partitioner, metrics)

### D1 -- Sensitive attribute excluded from model input X
Both original papers differ here (Paper 1 keeps sex/marital/etc. in X; Paper 2 drops it). The
spec requires exclusion for every algorithm. `fl/data/loaders.py` pops the sensitive column out
of `df` in every loader, and `RawDataset.__post_init__` asserts it is not among `num_cols` /
`cat_cols` for the 5 registered datasets. Effect: algorithms cannot use the sensitive attribute
as an input feature, only as metadata for fairness-aware training and for evaluation.

### D2 -- Dataset source files
Adult: Paper-2's `datasets/adult/adult.csv` (48,842 rows = UCI adult.data + adult.test combined,
labels normalised by stripping the trailing "." that adult.test's labels carry). 29 exact
duplicate rows are kept (verified: `n_exact_duplicate_rows_kept == 29`, matching the earlier
audit). Bank: Paper-2's `datasets/bank/bank.csv` (45,211 rows, UCI bank-full); Paper-1's own
`bank-full.csv` is actually a reduced 40,004-row file (divorced clients removed) and was
rejected. CaC: Paper-1's `Datasets/communities.csv` (1,994x128). KDD: Paper-1's `kdd.zip`
(299,285 rows); Paper-2's own KDD file has only 199,523 rows and was rejected. ACS: Paper-1's
`Datasets/acs_dataset/*.csv` (50 states, folktables ACSIncome, 1,655,429 rows); Paper 2 has no
ACS code at all. All five row counts are asserted in `fl/data/loaders.py` and were verified
against the actual files (see Phase 1 report).

### D3 -- CaC: Komiyama drop list + "fold" kept as a feature
27 columns dropped per Paper-1's `DatasetLoader/load_cac_data.py::DROP_COLS` (Komiyama et al.
2018); independently re-verified against the raw communities.csv column list (128 total). The
drop list removes `community`, `communityname`, `county` and 24 policing-related columns with
many missing values / no predictive value, but does NOT drop `fold` (the original 10-fold CV
index) or `state` (kept as a feature in the original, here removed from `df` because the
unified partitioner replaces the original's state-based client assignment; `fold` IS kept as a
feature, matching the original and giving exactly 98 feature columns = 128 - 27 - {state, Y,
sensitive}, verified in code and at runtime).

### D4 -- Bank: `day_of_week` is actually day-of-month
`bank.csv`'s own column is literally named `day_of_week` but holds the day of the month
(1-31), not a weekday. Kept under that name (both original papers use it this way); documented
so nobody "fixes" it into an actual weekday later.

### D5 -- Bank: missing categorical values become the literal string category "nan"
`job`, `education`, `contact` and `poutcome` contain true missing values (empty cells) in
`bank.csv`. The shared whitespace-stripping helper (`_strip_object_cols`) converts every
non-numeric column to `str`, which turns a real `NaN` into the literal 3-character string
"nan" -- so it becomes its own one-hot category instead of being treated as "unseen" (i.e. an
all-zero row). Effect: Bank's encoded feature count is 48 (7 numeric + 41 one-hot/code dims),
not the "15 raw -> 44" figure quoted in the earlier, exploratory
`reference/analysis/AUDIT_AND_PROTOCOL.md` (Section 6) -- that number predates this detail and
is superseded. This is INTENTIONAL and matches `reference/oracle_fairbench` (the Phase 2
validation target), whose own `_strip_obj` helper has the identical behaviour; it is preserved
here for L1 equivalence with the oracle rather than "corrected", and is flagged here as a
borderline design choice (arguably "unseen" would be more defensible) rather than hidden.
KDD's loader independently already needed the same `"nan" -> "missing"` remap for its own
categorical columns (also true missing values, not a "?" marker), which is already an explicit
category in the KDD case (`fl/data/loaders.py::load_kdd`).

### D6 -- CaC stratification proxy uses full-data medians; actual thresholds use TRAIN medians only
The global 70/15/15 split is stratified on an (A,Y) proxy computed from FULL-DATA medians
(`fl/data/loaders.py::load_cac`, `strata`) purely so the split itself is representative; this
proxy is never used to fit anything. The actual Y threshold (`ViolentCrimesPerPop` > median)
and A threshold (`racepctblack` > median) used everywhere else are fit on the TRAIN split only,
per seed (`fl/data/split.py::make_split`). Verified by `tests/test_no_leakage.py`.

### D7 -- Group orientation is a convention, not a modelling choice
A=1 is defined as the group with the higher TRAIN positive rate, frozen per (dataset, seed) and
never recomputed from validation/test labels (`fl/data/orientation.py`). This makes signed SP /
EO / DI comparable across datasets; Paper 1's own signed-SP convention (`group 1` = the larger
label-encoded value) is not comparable across datasets (e.g. it makes "group 1" male for Adult
but female for ACS). Verified against real data: Bank orients to A=1 = "not married" (TRAIN
rate 14.08% vs 10.12% for married); CaC orients to A=1 = "> median" (72.90% vs 27.93%) -- both
match the independent audit's numbers exactly.

### D8 -- Metrics: NaN, never 0, for undefined quantities
`fl/metrics.py` returns NaN for empty-group/class rates and for DI when PPR(A=1) == 0, instead
of silently returning 0.0 as both original papers' own metric code does. Cross-checked against
both papers' own formulas (transcribed, not imported) on 8 random trials in
`tests/test_metrics.py`; they agree to 1e-9 whenever every (A,Y) cell is non-empty (the only
regime where the 0-vs-NaN convention cannot differ).

### D9 -- Partitioner: eps is a floor on the un-normalised weight, not the final probability
After row-normalisation, a client's sampling probability for an extreme coordinate can be
below `eps` (e.g. eps=0.20, K=5 gives a minimum normalised weight of 0.125, not 0.20); eps
guarantees strict positivity (no client is ever excluded from a group), not a literal `>= eps`
floor on the final probability. `tests/test_partition.py` checks `w > 0`, not `w >= eps`.

## Phase 7 (Full-scale benchmark driver, cluster migration)

### D10 -- GBCFSS's LSH-CADC completion: vectorized + BLAS-reformulated (performance only, not behaviour)
`LSHCadc.complete()` (`fl/federated/gfl.py`) originally used a plain Python `for` loop, one
iteration per synthetic row, each doing a single-query Mahalanobis nearest-neighbour search
(`_mahalanobis`: `diff = Xcand - x`; `einsum("ij,jk,ik->i", diff, VI, diff)`). This was never
exercised at full scale before Phase 7 (Phases 1-6's oracle-equivalence checks all ran on small,
subsampled data). At full scale, `GFL.harmonised_cfg`'s `C=1.0` means every round's AGDG-generated
synthetic dataset is roughly as large as the real per-group training set, and LSH-CADC's fixed
2^k=128 bucket count (k=7) means each bucket's candidate set grows linearly with dataset size --
so the per-bucket pairwise distance computation is quadratic in bucket size, making the whole step
effectively O(N^2/128). Profiled on real full KDD data: this ONE step cost ~141s of a ~152s
training round (`tests/test_phase7_gbcfss_perf.py`'s docstring has the full profiling story).

Two changes, in order:
1. Grouped every synthetic row sharing the same LSH hash bucket into ONE batched call instead of
   one Python-loop iteration per row (removes per-row Python/numpy dispatch overhead; verified
   bit-identical to the original per-row loop on 5 random scales including empty/singleton edge
   cases, plus explicit chunk-boundary tests -- see git history / this file's prior version for
   the einsum-based intermediate step). This alone does not change the O(N^2/128) complexity
   class, only its constant factor.
2. Replaced the generic triple-contraction `einsum("mcj,jk,mck->mc", diff, VI, diff)` with the
   algebraically-identical `(diff @ VI * diff).sum(-1)`, which uses a real BLAS matmul for the
   `diff @ VI` step (~1.5-2x faster in practice; confirmed on real KDD data: 151.97s -> 100.13s
   per training round, 141.2s -> 88.5s for this step alone). NOT guaranteed bit-identical to the
   original (floating-point reassociation: max observed absolute difference ~1e-12 on random
   data) -- a behaviour change in principle, verified to make NO actual difference in practice:
   `tests/test_phase7_gbcfss_perf.py` checks the fast vs. exact argmin choice directly against
   every one of this project's 5 real datasets, deliberately oversampling any row that has a real
   exact duplicate elsewhere in its group (the only realistic way a reassociation-level numerical
   difference could flip an argmin) -- including KDD and ACS, where 83% and 99.9% of real rows
   respectively turn out to have an exact duplicate (coarse discrete census-style numeric
   features), far from a rare case there. Zero mismatches found. Full oracle-equivalence suite
   (`pytest tests/`, 115 tests) re-run and still passes after this change.

### D11 -- The whole GFL family (GFL, GFL-GBCFSS-EO, GFL-GBCFSS-DP) excluded for the ACS dataset
`complete()` (AGDG + LSH-CADC) runs UNCONDITIONALLY in `_server_run_round`, regardless of
`use_gbcfss` -- GBCFSS only changes what happens AFTER it (whether training uses the full
synthetic dataset or a budget=5000-row selected subset), so D10's cost driver is not GBCFSS-
specific. Verified directly: plain `GFL` (no GBCFSS) costs 99.55s/round on real KDD data, versus
GFL-GBCFSS-EO's 100.13s/round -- effectively identical, confirming the bottleneck is shared by all
three GFL-family variants. Even after D10's fix, LSH-CADC completion's O(N^2/128) scaling means
ACS (~5.5x KDD's row count) would cost roughly 5.5^2 ~ 30x more per round than KDD's measured
~100s/round for this step alone -- on the order of an hour-plus per training round, making a real
50-round x 25 (setting,seed) combo grid prohibitive within the project's cluster time budget, for
all three algorithms, not just the two GBCFSS ones.

Decided (with the project owner, 2026-11-11) to exclude {GFL, GFL-GBCFSS-EO, GFL-GBCFSS-DP} x
{ACS} from the benchmark sweep entirely, rather than either accept a materially riskier numerical
shortcut (the fully algebraically-expanded quadratic form, which has real catastrophic-
cancellation risk) or burn disproportionate cluster time on three cells of the grid.
`scripts/run_benchmark.py`'s combo generation enforces this exclusion explicitly
(`EXCLUDED_COMBOS`), and the benchmark report must state this exclusion plainly rather than
silently omitting those rows. All three GFL-family algorithms still run normally on the other 4
datasets (adult, bank, cac, kdd).

**Update (2026-11-14):** the project owner decided to attempt these 75 excluded cells after all,
rather than leave them out permanently. `scripts/run_benchmark.py sweep` gained an explicit
`--allow-excluded` flag (off by default -- the exclusion above still applies with no flag) that
deliberately includes them when passed. The real per-round cost on ACS was still never measured
directly before this run started (D11's ~30x/hour-plus figure stayed an extrapolation); sized
generously instead (6 workers, 400G mem, the partition's 2-day max) and ordered cheapest-setting-
first (iid10, attr5, iid5, attr3, iid3 -- larger K = smaller per-client data = cheaper, since the
bottleneck scales with roughly the square of per-client size) so that whatever time is available
completes the most combos, relying on `run_one`'s existing resumability if the time limit is hit
before the full 75 finish. See `docs/CLUSTER_LOG.md` for the real outcome.

### D12 -- Phase 7->8 merge: `last_round`, not `best_val_acc`, is the master table's primary metric snapshot
Every result JSON written by `run_one` carries two metric snapshots: `results.last_round` (the
final round, round 50 for every combo under the harmonised tier) and `results.best_val_acc` (the
round with the highest validation accuracy for that specific run, which can be an earlier round
and differs per combo). `fl/merge.py` (the script that consolidates all 1675 per-combo JSONs into
`results/summary/master_results.csv` for the Phase 8 Excel deliverable) uses `last_round` as the
ONE row per combo in the master table, keeping `best_val_acc`'s round number and accuracy only as
two reference columns (`best_val_round`, `best_val_accuracy`), never as an alternative row.

Decided with the project owner (2026-11-12): `last_round` keeps every algorithm's reported numbers
on an equal-rounds footing (all trained for the same 50 rounds under the harmonised protocol),
which matters for a benchmark whose whole point is comparing algorithms against each other --
`best_val_acc` would let an algorithm look better by reporting whichever round happened to validate
best, at a different round per algorithm, undermining a like-for-like comparison.

**Update (2026-11-15) -- real result: D11's O(N^2) severity did not hold in practice for GFL on
acs.** Job 127078 ran GFL's full 25-combo grid (all 5 settings x 5 seeds) on acs for real, to
completion (0 failed, 46.46h total wall-clock, comfortably inside the partition's 2-day limit).
Individual combo times ranged 47,109s-72,376s (~13-20h), fairly uniform across ALL FIVE settings
-- critically, `iid3`/`attr3` (the two settings D11 predicted would be dramatically most
expensive, ~30x KDD's measured cost) were NOT outliers: `iid3` s1/s3 came in at 62,438s/62,181s,
right in the same range as `iid10`'s combos. The original D10/D11 estimate scaled only the
per-round LSH-CADC cost; it did not account for a substantial FIXED per-combo cost each worker
process pays once per (dataset, seed) -- most plausibly loading/splitting/partitioning/
preprocessing acs's full 1.65M-row raw data in each separate worker process (no cross-process
cache, unlike the oracle-equivalence tests which always ran on small, cached data). That fixed
cost appears to dominate over the K-dependent completion-step cost in the K=3..10 range tested
here, which is why cheaper and pricier settings came out close together instead of showing the
predicted quadratic spread. This does not retroactively make D11's original exclusion decision
wrong -- it was the right call given the information available at the time -- but it does mean
the real cost of attempting the excluded cells was lower than estimated. See
`docs/CLUSTER_LOG.md` (job 127078) for the full real numbers.

**Update (2026-11-21) -- all 75 originally-excluded cells now complete; GFL-family x acs catchup
plan finished.** `GFL-GBCFSS-EO`'s 25-combo grid finished across two jobs (127303: 21/25 before
hitting the 2-day limit, individual combo times 93,154s-97,358s (~26-27h) under full 12-worker
contention; 127731 resumed the remaining 4, finishing naturally in 17h55m, 63,751s-64,458s
(~17.7-17.9h) each -- faster, since only 4 of 12 workers were actually competing for the node).
`GFL-GBCFSS-DP`'s 25-combo grid finished in a single job (127880), 25/25, 0 failed, completing
naturally ~42 minutes before its own 2-day deadline; combo times tapered from ~64,000s down to
44,703s as workers finished and contention eased. Combined with D11's original 2026-11-15 GFL
update, this closes out the entire catchup plan: all three GFL-family algorithms now have a
full, real 25-combo grid on `acs`, same as every other algorithm/dataset pair. `results/summary/
master_results.csv` now has 1,750 rows (5 datasets x 14 algorithms x 5 settings x 5 seeds),
with zero cells missing -- the 75-cell exclusion that D11 introduced is, in practice, now fully
filled in (the `EXCLUDED_COMBOS` set and `--allow-excluded` override in `run_benchmark.py`
remain in the code as the honest historical record of why these cells needed a deliberate
override to run at all, not because any cell is still actually missing). See `docs/CLUSTER_LOG.md`
(jobs 127303, 127731, 127880) for the full real numbers.

## Reproduced from the (pre-existing, informative-only) audit
* Adult: 29 exact duplicate rows -- reproduced exactly.
* CaC: 98 feature columns after drops -- reproduced exactly.
* Bank orientation: A=1 "not married" at 14.08% vs 10.12% -- reproduced exactly (audit: 14.1%/10.1%).
* CaC orientation: A=1 "> median" at 72.90% vs 27.93% -- reproduced exactly (audit: 72.9%/27.9%).
* Partition feasibility at eps=0.20: 50/50 (dataset x K x seed) combinations feasible on the
  FIRST draw; CaC K=5 minimum client size 229, minimum (A,Y) cell 14 -- reproduced exactly.
