#!/usr/bin/env python
"""
WO-2 analysis: unseal, validate, and produce the D4 deliverables.

Runs only after both scoring passes are complete. Produces, per locked D4 item 7:
  - the incidence table, mode x cell, reported as "x of N assessable"
  - the self-consistency table from the 24 re-scored items
  - one exemplar panel per mode, chosen by D4's fixed rule, never hand-picked

Validation runs FIRST and hard-stops on any inconsistency, because a silently malformed sheet would
propagate into every number downstream. Checked: row counts, no blanks, modes in {0,1},
dominant_mode in the closed vocabulary, and the consistency rule that a dominant mode must itself be
marked present while `none` / `not_assessable` require all nine absent.

The `not_assessable` provision comes from the 2026-09-08 scoping amendment: those items leave the
denominator and the count is reported beside every proportion.

    python wo2_analyse.py
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402

PACK = C.EDA_ROOT / "results" / "qualitative_pack"
OUT = PACK / "analysis"
MODES = ["m1_speckle", "m2_boundary_bleed", "m3_spurious_patch", "m4_paved_swap",
         "m5_road_continuation", "m6_veg_occlusion", "m7_misregistration",
         "m8_large_object", "m9_other"]
SPECIAL = {"none", "not_assessable"}
CELL_SHORT = {"unet_resnet34_rgb": "resnet34 (production)", "convnext_upernet_rgb": "ConvNeXt (best)"}

# D4 item 8, fixed in advance. No consequence may be attached to a mode not listed here.
CONSEQUENCE = {
    "m1_speckle": "false imperviousness change signals; noise in parcel-level statistics",
    "m2_boundary_bleed": "systematic area misestimation at parcel scale",
    "m3_spurious_patch": "false imperviousness change signals; noise in parcel-level statistics",
    "m4_paved_swap": "wrong surface-type input to drainage coefficients",
    "m5_road_continuation": "connectivity and access-surface errors in municipal use",
    "m6_veg_occlusion": "connectivity and access-surface errors in municipal use",
    "m7_misregistration": "geometric distrust of the product",
    "m8_large_object": "wrong classification of the installations policy monitoring targets",
    "m9_other": "see per-item notes; no pre-declared consequence",
}


def read_sheet(path, id_col):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get(id_col) or "").strip()]
    return rows


def validate(rows, id_col, label, expect_n):
    problems = []
    if len(rows) != expect_n:
        problems.append(f"{label}: {len(rows)} rows, expected {expect_n}")
    for r in rows:
        i = r[id_col]
        vals = {}
        for m in MODES:
            v = (r.get(m) or "").strip()
            if v not in ("0", "1"):
                problems.append(f"{label} {i}: {m} = {v!r}, expected 0 or 1")
                v = "0"
            vals[m] = v == "1"
        dm = (r.get("dominant_mode") or "").strip()
        if dm not in set(MODES) | SPECIAL:
            problems.append(f"{label} {i}: dominant_mode = {dm!r} not in the closed vocabulary")
            continue
        if dm in SPECIAL:
            if any(vals.values()):
                problems.append(f"{label} {i}: dominant_mode = {dm} but modes present: "
                                f"{[m for m in MODES if vals[m]]}")
        elif not vals[dm]:
            problems.append(f"{label} {i}: dominant_mode = {dm} but that mode is marked 0")
    return problems


def kappa(a, b):
    """Cohen's kappa for two binary raters over paired items. None when undefined."""
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if abs(1 - pe) < 1e-12:
        return None, po, pe
    return (po - pe) / (1 - pe), po, pe


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p1 = read_sheet(PACK / "scoring_sheet.csv", "item_id")
    p2 = read_sheet(PACK / "scoring_sheet_pass2.csv", "item_id")

    print("=" * 92)
    print("VALIDATION")
    print("=" * 92)
    probs = validate(p1, "item_id", "pass1", 192) + validate(p2, "item_id", "pass2", 24)
    if probs:
        for x in probs[:40]:
            print(f"  PROBLEM  {x}")
        sys.exit(f"\n{len(probs)} problem(s) -- refusing to analyse a malformed sheet")
    print(f"  pass 1: {len(p1)} rows, all modes 0/1, dominant_mode in vocabulary, consistent")
    print(f"  pass 2: {len(p2)} rows, same")

    # ---------------- unseal ----------------
    print("\n" + "=" * 92)
    print("UNSEALING — scoring is complete, so the keys may now be opened")
    print("=" * 92)
    key1 = {r["item_id"]: r for r in read_sheet(PACK / "SEALED_item_key.csv", "item_id")}
    key2 = {r["pass2_id"]: r for r in read_sheet(PACK / "SEALED_item_key_pass2.csv", "pass2_id")}
    order = [r["filename"] for r in read_sheet(PACK / "sample_tiles.csv", "filename")]
    strat = {r["filename"]: r for r in read_sheet(PACK / "sample_tiles.csv", "filename")}
    cells = sorted({v["cell"] for v in key1.values()})
    print(f"  {len(key1)} pass-1 items over {len(order)} tiles and {len(cells)} cells")

    scored = {r["item_id"]: r for r in p1}
    for i in scored:
        if i not in key1:
            sys.exit(f"item {i} is not in the key")

    # ---------------- incidence ----------------
    by_cell = defaultdict(list)
    for i, r in scored.items():
        by_cell[key1[i]["cell"]].append((i, r))

    na = {c: sum(1 for _i, r in v if r["dominant_mode"].strip() == "not_assessable")
          for c, v in by_cell.items()}
    assessable = {c: len(v) - na[c] for c, v in by_cell.items()}

    print("\n" + "=" * 92)
    print("INCIDENCE — mode x cell, over assessable items only")
    print("=" * 92)
    for c in cells:
        print(f"  {CELL_SHORT.get(c, c):<24} {len(by_cell[c])} items, "
              f"{na[c]} not assessable, N = {assessable[c]}")

    inc_rows = []
    print(f"\n  {'mode':<24}" + "".join(f"{CELL_SHORT.get(c, c):>26}" for c in cells))
    for m in MODES:
        line = f"  {m:<24}"
        rec = {"mode": m, "consequence": CONSEQUENCE[m]}
        for c in cells:
            k = sum(1 for _i, r in by_cell[c]
                    if r[m].strip() == "1" and r["dominant_mode"].strip() != "not_assessable")
            n = assessable[c]
            line += f"{f'{k:>3} / {n}  ({k/n:5.1%})':>26}"
            rec[f"{c}_count"], rec[f"{c}_n"], rec[f"{c}_pct"] = k, n, k / n
        inc_rows.append(rec)
        print(line)

    print(f"\n  {'DOMINANT MODE':<24}" + "".join(f"{CELL_SHORT.get(c, c):>26}" for c in cells))
    dom_rows = []
    for m in MODES + ["none"]:
        line = f"  {m:<24}"
        rec = {"dominant_mode": m}
        for c in cells:
            k = sum(1 for _i, r in by_cell[c] if r["dominant_mode"].strip() == m)
            n = assessable[c]
            line += f"{f'{k:>3} / {n}  ({k/n:5.1%})':>26}"
            rec[f"{c}_count"], rec[f"{c}_n"] = k, n
        dom_rows.append(rec)
        print(line)

    with open(C.assert_writes_are_local(OUT / "incidence_by_mode.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(inc_rows[0]))
        w.writeheader(); w.writerows(inc_rows)
    with open(C.assert_writes_are_local(OUT / "dominant_mode_by_cell.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dom_rows[0]))
        w.writeheader(); w.writerows(dom_rows)

    # ---------------- self-consistency ----------------
    print("\n" + "=" * 92)
    print("SELF-CONSISTENCY — 24 re-scored items, reported as-is")
    print("=" * 92)
    pairs = []
    for r2 in p2:
        k = key2[r2["item_id"]]
        r1 = scored[k["item_id"]]
        pairs.append((r1, r2, k))
    print(f"  {len(pairs)} paired items over {len({k['tile'] for _a, _b, k in pairs})} tiles")

    sc_rows = []
    print(f"\n  {'mode':<24}{'pass1':>7}{'pass2':>7}{'agree':>8}{'kappa':>9}   reading")
    for m in MODES:
        a = [r1[m].strip() == "1" for r1, _r2, _k in pairs]
        b = [r2[m].strip() == "1" for _r1, r2, _k in pairs]
        kap, po, pe = kappa(a, b)
        note = ("undefined - no variation" if kap is None else
                "prevalence too low to be stable" if min(sum(a), sum(b)) < 3 or
                max(sum(a), sum(b)) > len(a) - 3 else "")
        sc_rows.append({"mode": m, "pass1_present": sum(a), "pass2_present": sum(b),
                        "n": len(a), "raw_agreement": po,
                        "cohens_kappa": "" if kap is None else kap,
                        "kappa_reportable": kap is not None and not note, "note": note})
        print(f"  {m:<24}{sum(a):>7}{sum(b):>7}{po:>8.1%}"
              f"{('  n/a' if kap is None else f'{kap:>9.3f}')}   {note}")

    dom_agree = sum(1 for r1, r2, _k in pairs
                    if r1["dominant_mode"].strip() == r2["dominant_mode"].strip()) / len(pairs)
    any_agree = sum(1 for r1, r2, _k in pairs
                    if (r1["dominant_mode"].strip() in SPECIAL) ==
                       (r2["dominant_mode"].strip() in SPECIAL)) / len(pairs)
    print(f"\n  dominant-mode exact agreement : {dom_agree:.1%}")
    print(f"  clean/not-clean agreement     : {any_agree:.1%}")
    with open(C.assert_writes_are_local(OUT / "self_consistency.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sc_rows[0]))
        w.writeheader(); w.writerows(sc_rows)

    # ---------------- exemplars, D4 item 7 fixed rule ----------------
    print("\n" + "=" * 92)
    print("EXEMPLARS — first tile in sampled order showing the mode in ConvNeXt, else production")
    print("=" * 92)
    item_of = defaultdict(dict)
    for i, r in scored.items():
        item_of[key1[i]["tile"]][key1[i]["cell"]] = (i, r)
    ex_dir = OUT / "exemplars"
    ex_dir.mkdir(exist_ok=True)
    ex_rows = []
    for m in MODES:
        pick = None
        for pref in ("convnext_upernet_rgb", "unet_resnet34_rgb"):
            for t in order:
                got = item_of.get(t, {}).get(pref)
                if got and got[1][m].strip() == "1" \
                        and got[1]["dominant_mode"].strip() != "not_assessable":
                    pick = (t, pref, got[0])
                    break
            if pick:
                break
        if not pick:
            print(f"  {m:<24} no item exhibits this mode")
            ex_rows.append({"mode": m, "tile": "", "cell": "", "item_id": "", "panel": ""})
            continue
        t, cell, iid = pick
        dst = ex_dir / f"exemplar_{m}.png"
        shutil.copyfile(PACK / "panels" / f"item_{iid}.png", dst)
        ex_rows.append({"mode": m, "tile": t, "cell": cell, "item_id": iid,
                        "fold": strat[t]["fold"], "stratum": strat[t]["stratum"],
                        "panel": dst.name})
        print(f"  {m:<24} item {iid}  {CELL_SHORT.get(cell, cell):<22} {t}")
    with open(C.assert_writes_are_local(OUT / "exemplars.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["mode", "tile", "cell", "item_id", "fold", "stratum",
                                           "panel"], extrasaction="ignore")
        w.writeheader(); w.writerows(ex_rows)

    # ---------------- emergent themes from the free-text notes ----------------
    print("\n" + "=" * 92)
    print("FREE-TEXT NOTES — post-hoc thematic coding, NOT declared modes")
    print("=" * 92)
    tags = Counter()
    tagged = defaultdict(list)
    for i, r in scored.items():
        note = (r.get("note") or "").strip()
        if not note:
            continue
        for tag in re.findall(r"\[([a-z]+)\]", note.lower()):
            tags[tag] += 1
            tagged[tag].append(i)
    n_notes = sum(1 for r in p1 if (r.get("note") or "").strip())
    print(f"  {n_notes} of 192 items carry a note")
    for tag, k in tags.most_common():
        print(f"    [{tag}]  {k:>3} items")
    with open(C.assert_writes_are_local(OUT / "note_themes.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", "count", "item_ids", "status"])
        for tag, k in tags.most_common():
            w.writerow([tag, k, " ".join(sorted(tagged[tag])),
                        "POST-HOC thematic coding of free text; not a declared mode"])

    prov = C.assert_writes_are_local(OUT / "analysis_provenance.json")
    prov.write_text(json.dumps({
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "declaration": "2026-08-25_pre_declarations.md, LOCKED",
        "amendments": ["2026-09-08_D4_amendment_mode_scoping_rule.md",
                       "2026-09-08_D4_section_4_6_drop_invoked_and_withdrawn.md"],
        "n_tiles": len(order), "n_pass1_items": len(p1), "n_pass2_items": len(p2),
        "cells": cells,
        "not_assessable": na, "assessable_n": assessable,
        "dominant_mode_agreement": dom_agree,
        "validation": "passed: row counts, 0/1 modes, closed vocabulary, dominance consistency",
        "note_themes_status": "post-hoc thematic coding of free-text notes; not declared modes",
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
