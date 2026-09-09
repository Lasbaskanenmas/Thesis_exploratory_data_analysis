#!/usr/bin/env python
"""
Part B, POST-HOC EXPLORATORY LAYER — significance tests on the declared descriptive contrasts.

=============================================================================================
THIS IS NOT CONFIRMATORY. Nothing computed here may be used to support a claim. Every output
row carries `status = POST-HOC EXPLORATORY`. The pre-registered layer is untouched: this
module never writes to `wilcoxon_by_family.csv` and asserts its hash is unchanged on exit.
=============================================================================================

WHAT IS AND IS NOT PRE-REGISTERED HERE, because the distinction is the whole point.

  PRE-REGISTERED  the contrast set. D1 fixed exactly twenty-one permitted descriptive contrasts
                  on 2026-08-25, before any of the cells existed, and stated that "no other
                  descriptive contrast is permitted". This module reads that list from the locked
                  declaration and cannot add to it. There is no cherry-picking of which contrasts
                  to test, because the menu was closed before the results arrived.

  PRE-REGISTERED  the per-route metric, the route rules, the tie rules, the seed, the bootstrap.
                  All inherited unchanged from the locked declaration.

  POST-HOC        the decision to compute significance tests on these contrasts at all. D1 says
                  they "stay descriptive with CIs, outside every family... never tested for
                  significance". That decision is being set aside, after the results were seen,
                  at the author's instruction on 2026-09-08.

WHY ONE FAMILY OF TWENTY-ONE AND NOT SEVERAL. Family structure chosen after seeing results is
exactly the failure mode this layer is trying not to commit. Any narrower grouping — by
intervention, by architecture — would be a data-informed choice about which tests get the
easier correction. A single family containing every declared contrast is the only structure
that cannot be accused of favourable selection, and it is the most punishing: Holm's sharpest
threshold is 0.05/21 = 0.00238. At n' = 16 the smallest attainable two-sided p is 3.05e-5, so
the threshold is reachable, but only by a clean or near-clean sweep of the routes.

Uncorrected p is reported beside the Holm-adjusted value so the cost of the correction is
visible rather than hidden, and the pre-registered bootstrap CI is joined onto every row so a
reader sees the estimation-based evidence that IS pre-registered next to the exploratory p.

    python partb_posthoc.py --declaration ..\\2026-08-25_pre_declarations.md --go
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402
import partb_statistics as PB  # noqa: E402

OUT_DIR = C.TABLES / "part_b"
CONFIRMATORY = OUT_DIR / "wilcoxon_by_family.csv"
STATUS = "POST-HOC EXPLORATORY -- not pre-registered, supports no claim"
SHORT = [("convnext_upernet", "cnx"), ("unet_resnet34", "rn34"),
         ("swin_upernet", "swin"), ("segformer_b1", "segf")]


def s(n):
    for a, b in SHORT:
        n = n.replace(a, b)
    return n


def main():
    ap = argparse.ArgumentParser(description="post-hoc exploratory tests on declared contrasts")
    ap.add_argument("--declaration", required=True)
    ap.add_argument("--go", action="store_true", help="required; this touches real scores")
    args = ap.parse_args()

    dec = PB.load_declaration(args.declaration)          # refuses unless LOCKED
    D = PB.normalised(dec)
    if not args.go:
        sys.exit("refusing to run without --go")

    before = hashlib.sha256(CONFIRMATORY.read_bytes()).hexdigest()

    print("=" * 92)
    print("POST-HOC EXPLORATORY LAYER — NOT CONFIRMATORY, SUPPORTS NO CLAIM")
    print("=" * 92)
    print("The 21 contrasts were pre-declared in D1 on 2026-08-25 and cannot be added to here.")
    print("What is post-hoc is testing them at all: D1 declared them descriptive with CIs only.")
    print("Authorised by the author 2026-09-08, after the results were seen.\n")

    scores, tiles = PB.load_route_scores(D["metric"])
    routes_all = sorted(tiles)
    routes_sens = [r for r in routes_all
                   if D["min_tiles"] is None or tiles[r] >= int(D["min_tiles"])]
    contrasts = [(a, b) for a, b in D["descriptive_contrasts"]]
    missing = sorted({c for p in contrasts for c in p} - set(scores))
    if missing:
        sys.exit(f"cells without route scores: {missing}")
    print(f"contrasts: {len(contrasts)} (all declared)   routes: {len(routes_all)} "
          f"primary / {len(routes_sens)} sensitivity A   near-tie delta {D['near_tie_delta']}")

    # pre-registered CIs, joined on for context
    pre_ci = {}
    p = OUT_DIR / "descriptive_contrast_paired_bootstrap.csv"
    if p.is_file():
        for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
            pre_ci[(r["cell_a"], r["cell_b"])] = r

    rows = []
    for a, b in contrasts:
        rec = {"cell_a": a, "cell_b": b, "status": STATUS}
        for tag, routes, delta in (("primary", routes_all, 0.0),
                                   ("sensitivity_A_route_size", routes_sens, 0.0),
                                   ("sensitivity_B_near_tie", routes_all, D["near_tie_delta"])):
            d = np.array([scores[a][r] - scores[b][r] for r in routes], dtype=np.float64)
            w = PB.wilcoxon_signed_rank(d, near_tie_delta=delta)
            pre = "" if tag == "primary" else tag.split("_")[1] + "_"
            if tag == "primary":
                st = PB.sign_test(d, near_tie_delta=0.0)
                eff = PB.bootstrap_median_diff(d, B=D["B"], seed=D["seed"], ci=D["ci"])
                rec.update({"n_routes": len(routes),
                            "p_uncorrected": w["p_value"],
                            "effective_n": w["effective_n"],
                            "n_exact_ties": w["n_exact_ties"],
                            "rank_biserial": w["rank_biserial"],
                            "sign_wins": st["n_positive"], "sign_n": st["effective_n"],
                            "sign_p": st["p_value"],
                            "median_route_diff": eff["median_difference"],
                            "median_ci_low": eff["ci_low"], "median_ci_high": eff["ci_high"]})
            else:
                rec[f"{pre}p_uncorrected"] = w["p_value"]
                rec[f"{pre}effective_n"] = w["effective_n"]
        c = pre_ci.get((a, b), {})
        rec["prereg_pooled_diff"] = c.get("difference", "")
        rec["prereg_ci_low"] = c.get("ci_low", "")
        rec["prereg_ci_high"] = c.get("ci_high", "")
        rec["prereg_ci_excludes_zero"] = c.get("excludes_zero", "")
        rows.append(rec)

    # ONE family of all 21 -- the only structure not chosen with the results in view
    ps = [r["p_uncorrected"] for r in rows]
    h = PB.holm_bonferroni(ps, alpha=D["alpha"])
    for r, adj, rej, thr in zip(rows, h["p_adjusted"], h["reject"], h["step_threshold"]):
        r["holm_p_adjusted"], r["holm_reject"], r["holm_threshold"] = adj, rej, thr
    for tag in ("A", "B"):
        key = f"{'route' if tag == 'A' else 'near'}_p_uncorrected"
        if key in rows[0]:
            hh = PB.holm_bonferroni([r[key] for r in rows], alpha=D["alpha"])
            for r, adj, rej in zip(rows, hh["p_adjusted"], hh["reject"]):
                r[f"sens{tag}_holm_p"], r[f"sens{tag}_holm_reject"] = adj, rej

    for r in rows:
        flips = [t for t in ("A", "B")
                 if f"sens{t}_holm_reject" in r
                 and bool(r[f"sens{t}_holm_reject"]) != bool(r["holm_reject"])]
        r["fragile"] = bool(flips)
        r["fragile_reason"] = "|".join(f"sensitivity_{t}" for t in flips)

    out = C.assert_writes_are_local(OUT_DIR / "posthoc_exploratory_contrasts.csv")
    keys = sorted({k for r in rows for k in r})
    keys = ["cell_a", "cell_b", "status"] + [k for k in keys if k not in
                                             ("cell_a", "cell_b", "status")]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} rows)")

    n_rej = sum(1 for r in rows if r["holm_reject"])
    n_ci = sum(1 for r in rows if str(r["prereg_ci_excludes_zero"]) == "True")
    prov = C.assert_writes_are_local(OUT_DIR / "posthoc_provenance.json")
    prov.write_text(json.dumps({
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": STATUS,
        "authorised_by": "author instruction 2026-09-08, after results were seen",
        "pre_registered": ["the 21 contrasts (D1, locked 2026-08-25)", "per-route metric (D3)",
                           "route and tie rules (D2)", "seed and bootstrap (declaration s4)"],
        "post_hoc": ["the decision to compute significance tests on these contrasts at all"],
        "family_structure": "ONE family of all 21 declared contrasts; Holm alpha 0.05, sharpest "
                            "threshold 0.05/21 = 0.00238. Chosen because any narrower grouping "
                            "would itself be a post-hoc, data-informed selection.",
        "declaration_version": str(dec.get("declaration_version")),
        "declaration_status": dec.get("status"),
        "n_contrasts": len(rows), "n_holm_reject": n_rej,
        "n_prereg_ci_excludes_zero": n_ci,
        "confirmatory_layer_untouched": True,
        "confirmatory_sha256_before": before,
    }, indent=2), encoding="utf-8")
    print(f"wrote {prov}")

    print(f"\n{'contrast':<46}{'p_unc':>9}{'holm':>9}{'rej':>5}{'wins':>7}"
          f"{'rb':>7}{'preCI excl 0':>14}")
    for r in sorted(rows, key=lambda x: x["p_uncorrected"]):
        print(f"{s(r['cell_a']) + ' - ' + s(r['cell_b']):<46}"
              f"{r['p_uncorrected']:>9.5f}{r['holm_p_adjusted']:>9.4f}"
              f"{('YES' if r['holm_reject'] else '-'):>5}"
              f"{str(r['sign_wins']) + '/' + str(r['sign_n']):>7}"
              f"{r['rank_biserial']:>+7.2f}"
              f"{('yes' if str(r['prereg_ci_excludes_zero']) == 'True' else '-'):>14}")

    after = hashlib.sha256(CONFIRMATORY.read_bytes()).hexdigest()
    assert before == after, "CONFIRMATORY LAYER CHANGED -- this must never happen"
    print(f"\nconfirmatory layer untouched: wilcoxon_by_family.csv sha256 {after[:16]} (unchanged)")
    print(f"{n_rej} of {len(rows)} survive Holm across all 21; "
          f"{n_ci} have a pre-registered CI excluding zero")
    print("\nREAD AS EXPLORATORY. These p-values are not pre-registered and support no claim.")


if __name__ == "__main__":
    main()
