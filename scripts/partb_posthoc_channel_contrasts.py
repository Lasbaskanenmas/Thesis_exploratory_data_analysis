#!/usr/bin/env python
"""Two post-hoc paired whole-route pooled-difference bootstraps for the Chapter 5 channel table.

WHAT THIS IS. `convnext_upernet_6ch - convnext_upernet_rgb` and
`convnext_upernet_10ch - convnext_upernet_rgb` were tested in Family 2 (Wilcoxon on the median
route difference) but never given the pooled-difference bootstrap that the 21 rows of
`descriptive_contrast_paired_bootstrap.csv` carry. Those are different estimands -- a median of
per-route differences is not a difference of pooled metrics -- so the Chapter 5 table cannot borrow
the Wilcoxon number. This computes the missing quantity.

WHY IT IS A SEPARATE SCRIPT AND NOT AN EDIT TO THE DECLARATION. Adding the two pairs to
`descriptive_contrasts` in `2026-08-25_pre_declarations.md` would (a) edit a locked
pre-registration after seeing the results, which is the exact thing the lock exists to prevent, and
(b) make `partb_statistics.py --go` rewrite wilcoxon_by_family.csv, pooled_macro_iou_route_bootstrap
.csv and run_provenance.json in "w" mode. Both are unacceptable. So this driver imports the runner
and calls its own `bootstrap_paired_pooled_diff` with the declaration's own parameters. No estimator
is reimplemented here.

WHY CALLING IT IN ISOLATION IS LEGITIMATE. `partb_statistics` seeds every contrast identically --
`bootstrap_paired_pooled_diff(..., seed=D["seed"])` constructs a fresh `default_rng(20260825)` per
contrast -- so a contrast's result does not depend on how many contrasts ran before it. Computing
two in isolation is bit-identical to what a full re-run with the pairs appended would produce. The
reproduction check below is what proves that claim rather than asserting it: it recomputes an
existing row and compares against the value already on disk.

    python partb_posthoc_channel_contrasts.py            # verify + compute, write nothing
    python partb_posthoc_channel_contrasts.py --write     # also append the two rows
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C            # noqa: E402
import partb_statistics as PB     # noqa: E402

DECLARATION = C.EDA_ROOT / "2026-08-25_pre_declarations.md"
OUT_DIR = C.TABLES / "part_b"
CSV_PATH = OUT_DIR / "descriptive_contrast_paired_bootstrap.csv"
PROV_PATH = OUT_DIR / "run_provenance.json"
SYNC_DIR = Path(r"c:\thesis\Results\tables_new\part_b")

NEW_PAIRS = [("convnext_upernet_6ch", "convnext_upernet_rgb"),
             ("convnext_upernet_10ch", "convnext_upernet_rgb")]

# The row recomputed to prove the code path is identical. Values are read from the CSV on disk, not
# retyped, so the check cannot pass by my copying a number wrongly into this file.
CHECK_PAIR = ("convnext_upernet_rgb_ndsm", "convnext_upernet_rgb")

TEST_LABEL = ("NONE -- descriptive, post hoc addition 2026-09-12 for Chapter 5 Table, "
              "same estimator as the 21 declared contrasts")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="append the two rows and record provenance (default: compute only)")
    args = ap.parse_args()

    dec = PB.load_declaration(DECLARATION)          # enforces status == LOCKED
    D = PB.normalised(dec)
    print(f"declaration : {DECLARATION.name}  status={dec['status']}  "
          f"version={dec.get('declaration_version')}")
    print(f"bootstrap   : B={D['B']} seed={D['seed']} ci={D['ci']} "
          f"ignore_index={D['ignore_index']} report_only={D['report_only']}")

    z = np.load(C.TABLES / "route_cell_cms.npz", allow_pickle=False)
    idx = {str(c): i for i, c in enumerate(z["cells"])}
    for a, b in NEW_PAIRS + [CHECK_PAIR]:
        for cell in (a, b):
            if cell not in idx:
                sys.exit(f"no cached route matrices for {cell}. Refusing.")

    def run(a, b):
        return PB.bootstrap_paired_pooled_diff(
            z["cms"][idx[a]], z["cms"][idx[b]], B=D["B"], seed=D["seed"], ci=D["ci"],
            ignore_index=D["ignore_index"], report_only=D["report_only"])

    # ---------------------------------------------------------------- reproduction check, first
    existing = list(csv.DictReader(open(CSV_PATH, newline="", encoding="utf-8")))
    print(f"\nexisting CSV: {len(existing)} rows")
    ref = next((r for r in existing
                if r["cell_a"] == CHECK_PAIR[0] and r["cell_b"] == CHECK_PAIR[1]), None)
    if ref is None:
        sys.exit(f"check row {CHECK_PAIR[0]} - {CHECK_PAIR[1]} not found in {CSV_PATH.name}")

    print(f"\nreproduction check: {CHECK_PAIR[0]} - {CHECK_PAIR[1]}")
    got = run(*CHECK_PAIR)
    fields = ("macro_iou_a", "macro_iou_b", "difference", "ci_low", "ci_high")
    all_ok = True
    for f in fields:
        want, have = float(ref[f]), got[f]
        ok4 = round(want, 4) == round(have, 4)
        exact = want == have
        all_ok &= ok4
        print(f"    {f:<12} on disk {want:+.10f}   recomputed {have:+.10f}   "
              f"4dp {'PASS' if ok4 else 'FAIL'}   bitwise {'identical' if exact else 'differs'}")
    ez_ok = (ref["excludes_zero"] == "True") == got["excludes_zero"]
    all_ok &= ez_ok
    print(f"    {'excludes_zero':<12} on disk {ref['excludes_zero']:<6}      "
          f"recomputed {str(got['excludes_zero']):<6}      {'PASS' if ez_ok else 'FAIL'}")
    if not all_ok:
        sys.exit("\nREPRODUCTION CHECK FAILED -- the code path is not identical. "
                 "Refusing to compute or write the new rows.")
    print("    -> PASS. Code path, resampling, seed and CI method are identical.")

    # ---------------------------------------------------------------- the two new contrasts
    print("\nnew contrasts (cell_a minus cell_b):")
    new_rows = []
    for a, b in NEW_PAIRS:
        r = run(a, b)
        r.update({"cell_a": a, "cell_b": b, "test": TEST_LABEL})
        new_rows.append(r)
        print(f"    {a:<24} - {b:<22} diff {r['difference']:+.6f}  "
              f"CI [{r['ci_low']:+.6f}, {r['ci_high']:+.6f}]  "
              f"excludes_zero={r['excludes_zero']}")

    if not args.write:
        print("\nnothing written (pass --write to append).")
        return

    # ---------------------------------------------------------------- append, additively
    header = list(existing[0]) if existing else list(new_rows[0])
    for r in new_rows:
        missing = set(header) - set(r)
        extra = set(r) - set(header)
        if missing or extra:
            sys.exit(f"column mismatch against the existing CSV: missing={missing} extra={extra}")
    for a, b in NEW_PAIRS:
        if any(x["cell_a"] == a and x["cell_b"] == b for x in existing):
            sys.exit(f"row {a} - {b} already present. Refusing to duplicate.")

    before = sha256(CSV_PATH)
    out = C.assert_writes_are_local(CSV_PATH)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(existing)          # byte-for-byte the same 21 rows, re-emitted unchanged
        w.writerows(new_rows)
    after = list(csv.DictReader(open(CSV_PATH, newline="", encoding="utf-8")))
    assert len(after) == len(existing) + 2, f"expected {len(existing)+2} rows, got {len(after)}"
    for old, new in zip(existing, after):
        assert old == new, f"a pre-existing row changed: {old['cell_a']} - {old['cell_b']}"
    print(f"\nwrote {out}  ({len(after)} rows; {len(existing)} pre-existing rows verified unchanged)")
    print(f"    sha256 before {before[:16]}  after {sha256(CSV_PATH)[:16]}")

    prov = json.loads(PROV_PATH.read_text(encoding="utf-8"))
    prov.setdefault("posthoc_additions", []).append({
        "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "added_to": CSV_PATH.name,
        "rows_added": [{"cell_a": a, "cell_b": b} for a, b in NEW_PAIRS],
        "status": "POST HOC relative to the 2026-08-25 locked declaration. These two pairs are "
                  "NOT in `descriptive_contrasts`. They were tested in Family 2 (Wilcoxon on the "
                  "median route difference), which is a different estimand, and were never given a "
                  "pooled-difference bootstrap. Added for the Chapter 5 channel table. They support "
                  "no confirmatory claim and enter no Holm family.",
        "estimator": "partb_statistics.bootstrap_paired_pooled_diff -- the same function, "
                     "parameters and seed as the 21 declared contrasts",
        "B": int(D["B"]), "seed": int(D["seed"]), "ci": float(D["ci"]),
        "ignore_index": int(D["ignore_index"]), "report_only": list(D["report_only"]),
        "source_matrices": str(C.TABLES / "route_cell_cms.npz"),
        "evaluated_pixels_pooled": 12194633781,
        "n_routes": int(z["cms"].shape[1]),
        "declaration_unchanged": True,
        "declaration_sha256": sha256(DECLARATION),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "git_hash": "UNAVAILABLE -- git is not installed on this VM; script sha256 given instead",
        "reproduction_check": {
            "pair": list(CHECK_PAIR), "result": "PASS",
            "difference_on_disk": float(ref["difference"]), "difference_recomputed": got["difference"],
            "ci_low_on_disk": float(ref["ci_low"]), "ci_low_recomputed": got["ci_low"],
            "ci_high_on_disk": float(ref["ci_high"]), "ci_high_recomputed": got["ci_high"],
        },
        "results": [{"cell_a": r["cell_a"], "cell_b": r["cell_b"], "difference": r["difference"],
                     "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                     "excludes_zero": r["excludes_zero"]} for r in new_rows],
    })
    C.assert_writes_are_local(PROV_PATH).write_text(
        json.dumps(prov, indent=2, default=str), encoding="utf-8")
    print(f"wrote {PROV_PATH}")

    SYNC_DIR.mkdir(parents=True, exist_ok=True)
    for p in (CSV_PATH, PROV_PATH):
        shutil.copy2(p, SYNC_DIR / p.name)
        print(f"copied -> {SYNC_DIR / p.name}")


if __name__ == "__main__":
    main()
