#!/usr/bin/env python
"""
Great Plan 4, Steps 4.5 and 5 -- the production pick, exactly as declared in
2026-10-06_plan4_declarations.md (P2, P3, P4). Nothing in here is a choice; every choice is read from
the declaration's machine-readable block.

MODES
  --acceptance   machinery test, no lock needed because it reveals nothing new. Feeds the ALL-TILES
                 route matrices of rgb_ndsm and rgb (results/tables/route_cell_cms.npz) through the
                 same bootstrap call and must reproduce Part B's published pre-registered result
                 (+0.0039077, CI [-0.0050037, +0.0184344]) exactly.
  --build CELLS  build CLEAN per-route confusion matrices (the 18,033 uncontaminated tiles) for the
                 given cells and merge them into results/tables/plan4/clean_route_cms.npz. Reuses the
                 E6 worker (`eda_route_cell_metrics.worker`), restricted to clean tiles. Then
                 cross-checks: the routes must sum EXACTLY to each cell's clean matrix from the
                 subtraction method (contamination_decision/raw or plan4/clean_rescore). Two
                 independent paths, one answer. No statistic is computed.
  --dry-run      LOCKED required. Applies the rule to the existing challengers only. Writes
                 pick_dryrun_verdict.json. Not the pick.
  --pick         LOCKED required. Applies the rule to every declared candidate that is scored.
                 Runs ONCE (refuses if pick_verdict.json exists, per P4). Writes pick_verdict.json and
                 pick_table.csv.

    python plan4_pick_production.py --acceptance
    python plan4_pick_production.py --build convnext_upernet_rgb,convnext_upernet_rgb_ndsm
    python plan4_pick_production.py --dry-run
    python plan4_pick_production.py --pick
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402
import eda_route_cell_metrics as E6  # noqa: E402  (validated worker and directory resolution)
import partb_statistics as PB  # noqa: E402  (bootstrap_paired_pooled_diff)

DECLARATION = C.EDA_ROOT / "2026-10-06_plan4_declarations.md"
OUT = C.TABLES / "plan4"
CACHE = OUT / "clean_route_cms.npz"
ALL_TILES_CMS = C.TABLES / "route_cell_cms.npz"
POSTHOC = C.TABLES / "part_b" / "posthoc_exploratory_contrasts.csv"
CLEAN_SOURCES = [C.LOGS_AND_MODELS / "contamination_decision" / "raw", OUT / "clean_rescore"]


def log(msg=""):
    print(msg, flush=True)


def load_declaration(require_locked):
    text = DECLARATION.read_text(encoding="utf-8")
    m = re.search(r"```yaml\s*\n(.*?)\n```", text, re.S)
    if not m:
        sys.exit(f"no yaml block in {DECLARATION}")
    import yaml
    d = yaml.safe_load(m.group(1))
    if require_locked and d.get("status") != "LOCKED":
        sys.exit(f"declaration status is {d.get('status')!r}, not LOCKED. Lock it first "
                 f"(section 7 of {DECLARATION.name}). Nothing computed.")
    return d


def codes():
    return C.load_codes(str(C.CODES_TXT))


def macro(cm):
    return C.metrics_from_confusion(np.asarray(cm, dtype=np.int64), codes())["macro_iou"]


# ------------------------------------------------------------------------------------------------
def acceptance():
    log("=== acceptance: reproduce Part B's published rgb_ndsm vs rgb from ALL-TILES route CMs ===")
    z = np.load(ALL_TILES_CMS, allow_pickle=False)
    cells = list(z["cells"])
    a, b = "convnext_upernet_rgb_ndsm", "convnext_upernet_rgb"
    r = PB.bootstrap_paired_pooled_diff(z["cms"][cells.index(a)], z["cms"][cells.index(b)],
                                        B=10000, seed=20260825, ci=0.95, codes=codes())
    ref = next(row for row in csv.DictReader(open(POSTHOC, newline=""))
               if row["cell_a"] == a and row["cell_b"] == b)
    pairs = [("difference", r["difference"], float(ref["prereg_pooled_diff"])),
             ("ci_low", r["ci_low"], float(ref["prereg_ci_low"])),
             ("ci_high", r["ci_high"], float(ref["prereg_ci_high"]))]
    ok = True
    for name, got, want in pairs:
        same = abs(got - want) <= 1e-12
        ok &= same
        log(f"  {name:<11} got {got:+.10f}   published {want:+.10f}   {'EXACT' if same else 'DIFFERS'}")
    log("ACCEPTANCE PASS" if ok else "ACCEPTANCE FAIL -- do not use this machinery")
    return ok


# ------------------------------------------------------------------------------------------------
def clean_tasks():
    contam = {l.strip() for l in open(C.LOGS_AND_MODELS / "contamination_decision" / "raw"
                                      / "contaminated_tiles.txt") if l.strip()}
    df = C.load_tile_inventory()
    tasks = [(r.filename, int(r.fold)) for r in df.itertuples() if r.filename not in contam]
    return tasks, dict(zip(df["filename"], df["route"]))


def stored_clean_cm(cell):
    for d in CLEAN_SOURCES:
        p = d / f"rescore_{cell}.json"
        if p.is_file():
            return np.array(json.load(open(p))["cm_clean"], dtype=np.int64), p
    return None, None


def build(cells, procs):
    tasks, route_of = clean_tasks()
    log(f"=== build clean route CMs: {len(cells)} cells over {len(tasks):,} clean tiles ===")
    celldirs = [E6.cell_fold_dirs(c) for c in cells]
    routes = sorted(set(route_of.values()))
    ridx = {r: i for i, r in enumerate(routes)}
    n = C.NCLASS
    cms = np.zeros((len(cells), len(routes), n * n), dtype=np.int64)
    fails = []
    with Pool(processes=procs, initializer=E6._init, initargs=(celldirs,)) as pool:
        for i, res in enumerate(pool.imap_unordered(E6.worker, tasks, chunksize=4), 1):
            if res[0] != "OK":
                fails.append(res[1:])
                continue
            cms[:, ridx[route_of[res[1]]], :] += res[2]
            if i % 3000 == 0:
                log(f"  {i:,} / {len(tasks):,}")
    if fails:
        sys.exit(f"{len(fails)} tiles failed, first: {fails[:3]}")
    cms = cms.reshape(len(cells), len(routes), n, n)

    ok = True
    for ci, cell in enumerate(cells):                 # two independent paths must agree exactly
        ref, src = stored_clean_cm(cell)
        if ref is None:
            log(f"  {cell:<42} no subtraction-method clean CM to cross-check against")
            ok = False
            continue
        same = np.array_equal(cms[ci].sum(axis=0), ref)
        ok &= same
        log(f"  {cell:<42} sum of routes == subtraction CM: {'EXACT' if same else 'MISMATCH'}"
            f"   clean mIoU {macro(cms[ci].sum(axis=0)):.4f}")

    merged = {}
    if CACHE.is_file():
        z = np.load(CACHE, allow_pickle=False)
        assert list(z["routes"]) == routes, "route order changed"
        merged = {c: z["cms"][i] for i, c in enumerate(z["cells"])}
    merged.update({c: cms[i] for i, c in enumerate(cells)})
    names = sorted(merged)
    out = C.assert_writes_are_local(CACHE)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, cms=np.stack([merged[c] for c in names]), cells=np.array(names),
                        routes=np.array(routes), n_clean_tiles=np.array([len(tasks)]))
    log(f"cache now holds {len(names)} cells: {out}")
    return ok


# ------------------------------------------------------------------------------------------------
def judge(dec, challengers, out_name, once, cache=CACHE, out_dir=OUT):
    out_json = C.assert_writes_are_local(out_dir / out_name)
    if once and out_json.exists():
        sys.exit(f"{out_json.name} exists: the pick runs once (declaration P4). Nothing computed.")
    z = np.load(cache, allow_pickle=False)
    if int(z["n_clean_tiles"][0]) != int(dec["expected_clean_tiles"]):
        sys.exit(f"cache built on {int(z['n_clean_tiles'][0])} tiles, declaration expects "
                 f"{dec['expected_clean_tiles']}")
    have = {c: z["cms"][i] for i, c in enumerate(z["cells"])}
    inc = dec["incumbent"]
    if inc not in have:
        sys.exit(f"incumbent {inc} not in the clean cache; run --build first")

    rows, excluded = [], []
    for ch in challengers:
        if ch not in have:
            excluded.append(ch)
            continue
        r = PB.bootstrap_paired_pooled_diff(have[ch], have[inc], B=int(dec["B"]),
                                            seed=int(dec["seed"]), ci=float(dec["ci"]),
                                            codes=codes())
        qualifies = r["difference"] > 0 and r["ci_low"] > 0
        rows.append({"challenger": ch, "clean_macro_iou": r["macro_iou_a"],
                     "incumbent_clean_macro_iou": r["macro_iou_b"],
                     "difference": r["difference"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                     "input_products": int(dec["input_products"][ch]), "qualifies": qualifies,
                     "B": r["B"], "seed": r["seed"], "n_resamples_valid": r["n_resamples_valid"]})
        log(f"  {ch:<42} clean {r['macro_iou_a']:.4f}  diff {r['difference']:+.4f}  "
            f"CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  products {dec['input_products'][ch]}"
            f"  {'QUALIFIES' if qualifies else ''}")

    winners = sorted((r for r in rows if r["qualifies"]),
                     key=lambda r: (r["input_products"], -r["clean_macro_iou"]))
    winner = winners[0]["challenger"] if winners else inc
    # yaml.safe_load turns `declaration_version: 2026-10-06` into a datetime.date, which json cannot
    # serialise. str() it here, and default=str below catches any other yaml-typed value.
    verdict = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "declaration": str(DECLARATION), "declaration_version": str(dec["declaration_version"]),
               "incumbent": inc, "winner": winner, "incumbent_stays": winner == inc,
               "qualifying": [r["challenger"] for r in winners], "excluded_not_scored": excluded,
               "rule": dec["replace_if"], "tie_break": dec["tie_break"], "candidates": rows}
    out_json.write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    if rows:
        with open(out_json.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    log(f"\nVERDICT: {'incumbent stays' if winner == inc else 'REPLACED'} -> production = {winner}")
    if excluded:
        log(f"excluded, not scored at pick time: {excluded}")
    log(f"wrote {out_json}")
    return verdict


def selftest():
    """The rule on synthetic matrices with known answers. Touches no real score: the cache and the
    verdicts live in the gitignored cache/ folder and are deleted afterwards."""
    import shutil
    log("=== selftest: the declared rule on synthetic route matrices ===")
    tmp = C.CACHE / "plan4_pick_selftest"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    n_routes, n = 16, C.NCLASS
    rng = np.random.default_rng(0)

    def cell(acc):                       # routes where `acc` of each class's pixels are correct
        cms = np.zeros((n_routes, n, n), dtype=np.int64)
        for r in range(n_routes):
            for c in range(1, n):
                if c == 9:
                    continue
                tot = int(rng.integers(5_000, 20_000))
                ok = int(tot * acc)
                cms[r, c, c] = ok
                cms[r, c, (c % 10) + 1 if (c % 10) + 1 != 9 else 1] = tot - ok
        return cms

    base = cell(0.60)
    cells = {"inc": base, "better_1": cell(0.80), "same": base.copy(),
             "better_3": cell(0.85), "worse": cell(0.40)}
    names = sorted(cells)
    np.savez_compressed(tmp / "cache.npz", cms=np.stack([cells[c] for c in names]),
                        cells=np.array(names), routes=np.array([f"r{i}" for i in range(n_routes)]),
                        n_clean_tiles=np.array([18033]))
    # a real date, exactly as yaml.safe_load delivers the locked block's declaration_version --
    # the string used here originally is why the 2026-10-06 dry run's crash slipped past this test
    dec = {"declaration_version": dt.date(2026, 10, 6), "incumbent": "inc",
           "expected_clean_tiles": 18033,
           "B": 400, "seed": 7, "ci": 0.95, "replace_if": "diff_gt_zero_and_ci_excludes_zero",
           "tie_break": ["fewest_input_products", "higher_clean_macro_iou"],
           "input_products": {"better_1": 1, "same": 1, "better_3": 3, "worse": 1}}
    ok = True

    v = judge(dec, ["better_1", "same", "better_3", "worse"], "v1.json", once=True, cache=tmp / "cache.npz", out_dir=tmp)
    q = {r["challenger"]: r["qualifies"] for r in v["candidates"]}
    t1 = q == {"better_1": True, "same": False, "better_3": True, "worse": False}
    t2 = v["winner"] == "better_1"   # better_3 scores higher but needs 3 products: tie-break wins
    log(f"  qualification as expected: {t1}   fewest-products tie-break picks better_1: {t2}")
    ok &= t1 and t2

    v = judge(dec, ["same", "worse"], "v2.json", once=True, cache=tmp / "cache.npz", out_dir=tmp)
    t3 = v["winner"] == "inc" and v["incumbent_stays"]
    log(f"  no qualifier -> incumbent stays: {t3}")
    ok &= t3

    v = judge(dec, ["better_1", "not_scored"], "v3.json", once=True, cache=tmp / "cache.npz", out_dir=tmp)
    t4 = v["excluded_not_scored"] == ["not_scored"]
    log(f"  unscored cell excluded and reported: {t4}")
    ok &= t4

    try:
        judge(dec, ["better_1"], "v1.json", once=True, cache=tmp / "cache.npz", out_dir=tmp)
        t5 = False
    except SystemExit:
        t5 = True
    log(f"  second pick refused (runs once): {t5}")
    ok &= t5

    shutil.rmtree(tmp, ignore_errors=True)
    log("SELFTEST PASS" if ok else "SELFTEST FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--acceptance", action="store_true")
    g.add_argument("--selftest", action="store_true")
    g.add_argument("--build", metavar="CELLS")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--pick", action="store_true")
    ap.add_argument("--procs", type=int, default=min(16, os.cpu_count() or 8))
    args = ap.parse_args()

    if args.acceptance:
        sys.exit(0 if acceptance() else 1)
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.build:
        sys.exit(0 if build([c.strip() for c in args.build.split(",") if c.strip()], args.procs) else 1)
    dec = load_declaration(require_locked=True)
    if args.dry_run:
        log("=== DRY RUN on existing challengers (not the pick) ===")
        judge(dec, dec["existing_challengers"], "pick_dryrun_verdict.json", once=False)
    else:
        log("=== THE PICK (runs once) ===")
        judge(dec, dec["existing_challengers"] + dec["new_cells"], "pick_verdict.json", once=True)


if __name__ == "__main__":
    main()
