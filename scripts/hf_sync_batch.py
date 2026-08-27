#!/usr/bin/env python
"""
Task 2.4 of the 2026-08-24 work order: the HF backup catch-up, list-then-batch.

`backup_to_hf.py --sync_all` uploads every candidate file one at a time, which means one commit per
file and a full re-walk of ~75 model checkpoints to find the 21 per-run log CSVs that are actually
missing. This does the same job the other way round: LIST what the repo already holds, DIFF against
what should be there, and upload only the gap -- in ONE batched commit per group.

It also covers what `--sync_all` does not: the analysis artifacts under
exploratory_data_analysis/results/tables/, which are what the writing actually reads. Plan 3.1
section 7 makes this the difference between "the VM is a convenience" and "the VM is a dependency".

The token is read from the file and never printed, echoed, or passed on a command line.

Dry run by default. Nothing leaves this machine without --push.

    python hf_sync_batch.py                  # dry run: print exactly what would be uploaded
    python hf_sync_batch.py --push
    python hf_sync_batch.py --push --include-models    # also fill any missing .pth / oof json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402

REPO_ID = "Lasbaskanenmas/befaestelsesdata-spatial-matrix"
TOKEN_FILE = Path(r"c:\thesis\hftoken_write.txt")
MATRIX_ROOT = C.SPATIAL_MATRIX
SPLIT_DIR = C.LOGS_AND_MODELS / "route_class_audit"

# Analysis artifacts the writing reads. Everything here is small.
TABLE_GLOBS = [
    ("cross_cell_summary.csv", "results/tables/"),
    ("route_cell_metrics.csv", "results/tables/"),
    ("route_cell_cms.npz", "results/tables/"),
    ("route_cell_provenance.json", "results/tables/"),
    ("performance_bounds.csv", "results/tables/"),
    ("performance_bounds.json", "results/tables/"),
    ("metric_breadth_by_cell.csv", "results/tables/"),
    ("metric_breadth_per_class.csv", "results/tables/"),
    ("metric_breadth_binary_collapse.csv", "results/tables/"),
    ("metric_breadth_provenance.json", "results/tables/"),
    ("boundary_iou.csv", "results/tables/"),
    ("boundary_iou_provenance.json", "results/tables/"),
    ("boundary_error_profile.csv", "results/tables/"),
    ("label_quality.json", "results/tables/"),
    ("high_ndsm_tiles.csv", "results/tables/"),
    ("high_ndsm_tiles.json", "results/tables/"),
    ("ndsm_clamped_stats.json", "results/tables/"),
    ("corrected_channel_constants.json", "results/tables/"),
    ("gate_arms_G_A_ortorgb_batch_stats.json", "results/tables/"),
]


def desired(include_models):
    """[(local Path, path_in_repo)] -- everything that should exist in the backup repo."""
    out = []
    for name in ["fold_assignment.csv", "route_class_audit.csv", "class_routes.json",
                 "fold_0_valid.txt", "fold_1_valid.txt", "fold_2_valid.txt"]:
        out.append((SPLIT_DIR / name, f"results/split/{name}"))

    for name, prefix in TABLE_GLOBS:
        out.append((C.TABLES / name, f"{prefix}{name}"))
    for sub in ("learning_curve", "part_b"):
        d = C.TABLES / sub
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    rel = p.relative_to(C.TABLES).as_posix()
                    out.append((p, f"results/tables/{rel}"))
    for p in sorted(C.FINDINGS.glob("*.md")):
        out.append((p, f"results/findings/{p.name}"))

    # per-run logs: the 21 missing CSVs live here
    for logf in sorted(list(MATRIX_ROOT.glob("*/*/logs/*_job_dictionary.json")) +
                       list(MATRIX_ROOT.glob("*/*/logs/*.csv"))):
        if logf.parents[2].name.startswith("_"):
            continue
        out.append((logf, f"results/logs/{logf.parents[1].name}/{logf.name}"))

    if include_models:
        for pth in sorted(MATRIX_ROOT.glob("*/*/models/*.pth")):
            job = pth.parents[1].name
            if pth.parents[2].name.startswith("_") or pth.stem != job:
                continue
            out.append((pth, f"models/{job}.pth"))
        for js in sorted(MATRIX_ROOT.glob("*/oof_*/pooled_oof_metrics.json")):
            if js.parents[1].name.startswith("_"):
                continue
            out.append((js, f"results/oof/{js.parent.name}.json"))

    return [(p, r) for p, r in out if p.is_file()]


def main():
    ap = argparse.ArgumentParser(description="list-then-batch HF backup catch-up")
    ap.add_argument("--push", action="store_true", help="actually upload (default: dry run)")
    ap.add_argument("--include-models", action="store_true",
                    help="also check the .pth checkpoints and pooled oof json")
    ap.add_argument("--refresh", action="store_true",
                    help="re-upload results/tables and results/findings even if present. "
                         "Those are regenerated as the analysis proceeds, so 'already in the repo' "
                         "does not mean 'current'. Models, logs and split files never change once "
                         "written and are left alone.")
    ap.add_argument("--repo_id", default=REPO_ID)
    args = ap.parse_args()

    if not TOKEN_FILE.is_file():
        sys.exit(f"no HF write token at {TOKEN_FILE}")
    from huggingface_hub import HfApi
    from huggingface_hub import CommitOperationAdd

    api = HfApi(token=TOKEN_FILE.read_text().strip())     # never printed
    try:
        have = set(api.list_repo_files(repo_id=args.repo_id, repo_type="model"))
    except Exception as exc:                              # noqa: BLE001
        sys.exit(f"could not list {args.repo_id}: {type(exc).__name__}: {exc}")
    print(f"repo {args.repo_id} currently holds {len(have)} files")

    want = desired(args.include_models)
    stale = ("results/tables/", "results/findings/") if args.refresh else ()
    missing = [(p, r) for p, r in want if r not in have or r.startswith(stale)]
    print(f"candidates on disk : {len(want)}")
    print(f"up to date         : {len(want) - len(missing)}")
    print(f"TO UPLOAD          : {len(missing)}"
          + ("  (missing, plus tables/findings refreshed)" if args.refresh else "  (missing only)"))

    by_group = {}
    for p, r in missing:
        by_group.setdefault(r.split("/")[1] if r.startswith("results/") else r.split("/")[0],
                            []).append((p, r))
    for g, items in sorted(by_group.items()):
        size = sum(p.stat().st_size for p, _r in items)
        print(f"  {g:<12} {len(items):>4} files, {size / 1e6:>9.1f} MB")
        for p, r in items[:6]:
            print(f"      {r}")
        if len(items) > 6:
            print(f"      ... and {len(items) - 6} more")

    if not missing:
        print("\nnothing to do -- the backup is complete")
        return
    if not args.push:
        print("\nDRY RUN -- pass --push to upload. Nothing left this machine.")
        return

    for g, items in sorted(by_group.items()):
        ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(p)) for p, r in items]
        api.create_commit(repo_id=args.repo_id, repo_type="model", operations=ops,
                          commit_message=f"catch-up sync: {g} ({len(ops)} files), 2026-08-24")
        print(f"  pushed {len(ops):>4} files in one commit  [{g}]")

    have2 = set(api.list_repo_files(repo_id=args.repo_id, repo_type="model"))
    still = [r for _p, r in want if r not in have2]
    print(f"\nrepo now holds {len(have2)} files; still missing: {len(still)}")
    if still:
        for r in still[:10]:
            print(f"  {r}")
        sys.exit("some files did not land -- re-run to retry")
    print("backup complete for every candidate checked")


if __name__ == "__main__":
    main()
