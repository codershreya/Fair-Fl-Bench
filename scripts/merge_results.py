"""Phase 7 -> Phase 8 bridge CLI: consolidate results/runs/**/*.json into one master table.

Usage:
  python3 scripts/merge_results.py --dry-run     # validate counts/completeness, write nothing
  python3 scripts/merge_results.py                # also write results/summary/master_results.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fl import config as C  # noqa: E402
from fl.merge import expected_combo_count, load_all_results  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=None, help="default: fl.config.RESULTS_RUNS")
    ap.add_argument("--out", default=None, help="default: fl.config.RESULTS_SUMMARY/master_results.csv")
    ap.add_argument("--dry-run", action="store_true", help="validate only, write nothing")
    args = ap.parse_args()

    df, problems = load_all_results(args.runs_dir)
    expected = expected_combo_count()

    print(f"expected combos (per run_benchmark.py's own grid + exclusions): {expected}")
    print(f"completed rows loaded: {len(df)}")
    print(f"non-completed / unreadable files: {len(problems)}")
    if problems:
        print("  -- these were NOT included in the merged table:")
        for p in problems[:20]:
            print(f"    {p}")
        if len(problems) > 20:
            print(f"    ... and {len(problems) - 20} more")

    if len(df):
        dup = df[df.duplicated(subset=["dataset", "distribution", "K", "algorithm", "seed"], keep=False)]
        if len(dup):
            print(f"WARNING: {len(dup)} rows share a (dataset,distribution,K,algorithm,seed) key "
                  f"with another row -- should be impossible (exp_id is supposed to be unique):")
            print(dup[["exp_id"]].to_string())

    if len(df) != expected:
        print(f"NOTE: loaded row count ({len(df)}) != expected grid size ({expected}). "
              f"This is informational, not fatal -- it can legitimately happen mid-sweep; "
              f"it should NOT happen once the sweep has fully finished with 0 failures.")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    out_path = Path(args.out) if args.out else (C.RESULTS_SUMMARY / "master_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nwrote {len(df)} rows x {len(df.columns)} columns -> {out_path}")


if __name__ == "__main__":
    main()
