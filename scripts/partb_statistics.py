#!/usr/bin/env python
"""
Part B -- the route-level inferential layer (Great Plan 3.1 section 5.2, work order 2026-08-24 task 3).

BUILT AND SELF-TESTED ONLY. This module refuses to compute a single p-value on real scores until
the author's locked `2026-08-25_pre_declarations.md` is on disk AND `--go` is passed. That refusal
is the point: pre-registration protects declarations-before-RESULTS, and descriptive route medians
have already been seen (E6 landed 23/8), so the declaration must be dated and locked before any
test statistic exists. `--selftest` runs on synthetic data with known answers and touches nothing.

WHAT IT COMPUTES, per declared pair, over per-route paired scores
  - Wilcoxon signed-rank, EXACT by sign-flip enumeration where N permits, with the exact-tie count
    and the effective N printed beside every p-value. This is not decoration: seven of sixteen
    routes carry three classes or fewer and 83-31 is single-class, so effective N is expected to
    land near 9-12 rather than 16, and at effective N = 9 the smallest attainable two-sided p is
    0.0039 -- which clears Holm's first threshold with almost nothing to spare, and at 8 cannot
    clear it at all. Non-significance under Holm at small effective N is a declared, expected,
    reportable outcome.
  - The exact two-sided sign test, as the declared robustness companion beside every Wilcoxon.
  - Holm-Bonferroni within each declared family, and nothing across families.
  - For each multi-pair family, a Friedman omnibus with mean ranks and Kendall's W.
  - Effect sizes: the median paired route difference with a bootstrap CI (primary), and the
    rank-biserial correlation (secondary). A p-value without a magnitude is what the literature
    review says the field does badly.
  - Whole-route block-bootstrap CIs for the pooled Macro-IoU of declared cells. The resampling unit
    is the ROUTE, and the statistic is recomputed from the resampled routes' confusion matrices
    through the project's own scoring function -- not from a mean of route scores, which would be a
    different and non-comparable quantity.
  - McNemar on pooled pixels as CONTEXT ONLY, framed as near-automatic by construction. It needs
    per-pixel agreement, which confusion matrices cannot supply, so it streams prediction pairs and
    is off unless asked for.

WHAT IT DELIBERATELY DOES NOT DO
  - It never chooses a family, a route rule, a metric or a bootstrap B. Every one of those comes
    from the declaration file. Where this docstring and the declaration disagree, THE DECLARATION
    WINS, by the work order's own wording.
  - It never tests an arm cell. `descriptive_cells` in the declaration are reported with CIs only
    and are asserted to appear in no family.
  - No sklearn: its LDA and lbfgs solvers hard-abort this environment (0xC0000409). numpy only,
    with scipy used solely for cross-checks inside the self-test.

    python partb_statistics.py --selftest
    python partb_statistics.py --emit-template
    python partb_statistics.py --declaration ../2026-08-25_pre_declarations.md --dry-run
    python partb_statistics.py --declaration ../2026-08-25_pre_declarations.md --go
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402

sys.path.insert(0, r"c:\thesis\ML_sdfi_fastai2\src\ML_sdfi_fastai2")
import analyse.per_category_metrics as pcm  # noqa: E402

ROUTE_METRICS_CSV = C.TABLES / "route_cell_metrics.csv"
ROUTE_CMS_NPZ = C.TABLES / "route_cell_cms.npz"
OUT_DIR = C.TABLES / "part_b"
EXACT_MAX_N = 22                 # 2**22 sign patterns is the practical ceiling for enumeration
TOL = 1e-12
CLASS_ORDER = ["unknown", "asfalt", "fliser", "grus", "ubefestet", "green_roof",
               "drivhus", "betonflade", "brosten", "unknown2", "solceller"]

TEMPLATE = """```yaml
# Machine-readable block for partb_statistics.py. This block is AUTHORITATIVE: where it and the
# runner's defaults disagree, this wins. Nothing here may change after the file is dated and locked.
declaration_id: 2026-08-25_pre_declarations
locked_utc: null                 # fill in when locking; the runner refuses to run while null

# D3 -- the per-route metric. Column name in route_cell_metrics.csv.
per_route_metric: macro_iou_present_classes

# D2 -- the route rule.
route_rule:
  primary: all_routes            # every route in the split
  sensitivity:
    name: drop_routes_under_100_tiles
    min_tiles: 100               # removes 83-34 (10 tiles) and 85-48 (150 tiles)
  report_effective_n: true       # exact-tie count and effective N beside every p-value
  tie_definition: exact_zero_difference

# D1 -- the Holm-Bonferroni families. Weighted cells only. Pairs are (a, b); the reported
# difference is a - b.
alpha: 0.05
families:
  - name: model_ranking_within_best_channel
    pairs: []                    # 6 pairs -- fill from the declaration
  - name: channel_effect_within_best_model
    pairs: []                    # 3 pairs, frozen channel configs only
  - name: weighting_within_best_model_and_channel
    pairs: []                    # 1 pair

# Omnibus layer (Demsar 2006): Friedman + mean ranks + Kendall's W per multi-pair family.
omnibus:
  enabled: true
  min_pairs_for_omnibus: 2

# Declared descriptive-only cells. Asserted to appear in NO family.
descriptive_cells:
  - convnext_upernet_rgb_ndsm
  - convnext_upernet_6ch_corrected
  - unet_resnet34_6ch_corrected
  - convnext_upernet_ortorgb
  - convnext_upernet_rgb_dsm_dtm_corrected

# Block bootstrap for pooled Macro-IoU. The unit is the whole route.
bootstrap:
  unit: route
  B: 10000
  seed: 20260825
  ci: 0.95
  cells: []                      # headline cells + the descriptive arms

# McNemar, context only. Off by default: it needs per-pixel agreement, not confusion matrices.
mcnemar:
  enabled: false
  pairs: []
```"""


# ==================================================================================================
# statistics -- pure numpy
# ==================================================================================================
def _avg_ranks(x):
    """Ranks of x, 1-based, ties averaged."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def wilcoxon_signed_rank(diffs, near_tie_delta=0.0):
    """Two-sided Wilcoxon signed-rank on paired differences.

    Exact zeros are DROPPED (the classical treatment) and counted; what remains is the effective N.
    The exact p is the sign-flip permutation p conditional on the observed |d| ranks, which is the
    correct exact reference when average ranks are used for ties in |d|.

    `near_tie_delta` implements declaration D2 Sensitivity B: |d| < delta is treated as a tie and
    dropped. It defaults to 0.0 so the PRIMARY analysis is the untouched standard test and delta is
    a labelled sensitivity rather than a tunable knob, exactly as D2 requires.
    """
    d = np.asarray(diffs, dtype=np.float64)
    n_total = len(d)
    ties = int((d == 0).sum())
    near = int(((np.abs(d) < near_tie_delta) & (d != 0)).sum()) if near_tie_delta > 0 else 0
    nz = d[np.abs(d) >= near_tie_delta] if near_tie_delta > 0 else d[d != 0]
    nz = nz[nz != 0]
    n = len(nz)
    out = {"n_pairs": n_total, "n_exact_ties": ties, "n_near_ties": near,
           "near_tie_delta": float(near_tie_delta), "effective_n": n,
           "w_plus": None, "w_minus": None, "statistic": None, "p_value": None,
           "p_method": None, "rank_biserial": None,
           "min_attainable_two_sided_p": None}
    if n == 0:
        out["p_method"] = "undefined (every pair is an exact tie)"
        return out

    ranks = _avg_ranks(np.abs(nz))
    w_plus = float(ranks[nz > 0].sum())
    w_minus = float(ranks[nz < 0].sum())
    total = w_plus + w_minus
    out["w_plus"], out["w_minus"] = w_plus, w_minus
    out["statistic"] = min(w_plus, w_minus)
    out["rank_biserial"] = (w_plus - w_minus) / total if total else 0.0
    # Smallest two-sided p this N can ever produce: every difference in the same direction.
    out["min_attainable_two_sided_p"] = min(1.0, 2.0 * 0.5 ** n)

    if n <= EXACT_MAX_N:
        signs = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.float64) * 2 - 1
        dist = signs @ ranks                                    # W+ - W- under every sign flip
        obs = w_plus - w_minus
        out["p_value"] = float((np.abs(dist) >= abs(obs) - 1e-12).mean())
        out["p_method"] = f"exact sign-flip enumeration over 2^{n} assignments"
    else:
        mu = n * (n + 1) / 4.0
        _u, cnt = np.unique(np.abs(nz), return_counts=True)
        tie_corr = float(((cnt ** 3 - cnt).sum())) / 48.0
        sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - tie_corr)
        z = (w_plus - mu) / sd if sd > 0 else 0.0
        out["p_value"] = float(math.erfc(abs(z) / math.sqrt(2)))
        out["p_method"] = f"normal approximation with tie correction (z = {z:.4f})"
    return out


def sign_test(diffs, near_tie_delta=0.0):
    """Exact two-sided sign test. Zeros dropped, as in the Wilcoxon, so the two share an N.

    D2 declares the same near-tie rule applies to this test identically, hence the parameter.
    """
    d = np.asarray(diffs, dtype=np.float64)
    nz = d[np.abs(d) >= near_tie_delta] if near_tie_delta > 0 else d[d != 0]
    nz = nz[nz != 0]
    n = len(nz)
    k = int((nz > 0).sum())
    if n == 0:
        return {"effective_n": 0, "n_positive": 0, "p_value": None,
                "p_method": "undefined (every pair is an exact tie)"}
    # two-sided exact binomial at p = 0.5: sum of all outcomes no more likely than the observed
    probs = np.array([math.comb(n, i) for i in range(n + 1)], dtype=np.float64) / (2.0 ** n)
    p = float(probs[probs <= probs[k] + 1e-15].sum())
    return {"effective_n": n, "n_positive": k, "p_value": min(1.0, p),
            "p_method": "exact binomial, two-sided"}


def holm_bonferroni(pvals, alpha=0.05):
    """Holm step-down. Returns adjusted p-values (monotone) and reject flags, in input order."""
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    adj = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    thresholds = np.empty(m, dtype=np.float64)
    for rank, idx in enumerate(order):
        thresholds[idx] = alpha / (m - rank)
    return {"p_adjusted": adj.tolist(), "reject": (adj <= alpha).tolist(),
            "step_threshold": thresholds.tolist(), "m": m, "alpha": alpha}


def friedman(block_matrix):
    """Friedman omnibus over blocks x treatments (routes x cells), plus mean ranks and Kendall's W.

    Demsar 2006's protocol for comparing classifiers over multiple data sets, with the data sets
    being routes here. Ties within a block get average ranks and the tie-corrected statistic.
    """
    x = np.asarray(block_matrix, dtype=np.float64)
    n, k = x.shape                                    # n blocks, k treatments
    if k < 2:
        raise ValueError("Friedman needs at least two treatments")
    ranks = np.vstack([_avg_ranks(row) for row in x])
    rbar = ranks.mean(axis=0)
    # tie correction (Conover): denominator uses the observed rank variance
    a = float((ranks ** 2).sum())
    b = n * k * (k + 1) ** 2 / 4.0
    num = (k - 1) * float(((ranks.sum(axis=0) - n * (k + 1) / 2.0) ** 2).sum())
    chi2 = num / (a - b) if (a - b) > 0 else 0.0
    df = k - 1
    return {"n_blocks": n, "k_treatments": k, "chi2": float(chi2), "df": df,
            "p_value": float(_chi2_sf(chi2, df)),
            "mean_ranks": rbar.tolist(),
            "kendalls_w": float(chi2 / (n * (k - 1))) if n and k > 1 else None}


def _chi2_sf(x, df):
    """Upper tail of a chi-square. Regularised upper incomplete gamma, series/CF, no scipy."""
    if x <= 0:
        return 1.0
    a, xx = df / 2.0, x / 2.0
    if xx < a + 1.0:                                   # series for P(a,x), then Q = 1 - P
        term, s, n = 1.0 / a, 1.0 / a, 0
        while n < 10000:
            n += 1
            term *= xx / (a + n)
            s += term
            if abs(term) < abs(s) * 1e-16:
                break
        return 1.0 - s * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
    tiny = 1e-300                                      # continued fraction for Q(a,x)
    b, c, d = xx + 1.0 - a, 1.0 / tiny, 1.0 / (xx + 1.0 - a)
    h, i = d, 0
    while i < 10000:
        i += 1
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return h * math.exp(-xx + a * math.log(xx) - math.lgamma(a))


def bootstrap_median_diff(diffs, B=10000, seed=0, ci=0.95):
    """Percentile bootstrap CI for the median paired route difference. Unit = the pair (route)."""
    d = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    meds = np.median(d[idx], axis=1)
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {"median_difference": float(np.median(d)),
            "ci_low": float(np.percentile(meds, lo)),
            "ci_high": float(np.percentile(meds, hi)),
            "B": B, "seed": seed, "ci": ci, "unit": "route"}


def bootstrap_pooled_macro_iou(route_cms, B=10000, seed=0, ci=0.95,
                               codes=CLASS_ORDER, ignore_index=0, report_only=(9,)):
    """Whole-route block bootstrap of the POOLED Macro-IoU.

    route_cms is (n_routes, n_class, n_class). Routes are resampled with replacement, their
    confusion matrices summed, and the statistic recomputed through the project's own scoring
    function -- so the bootstrap statistic is the same quantity as the headline, not an average of
    route-level scores.
    """
    cms = np.asarray(route_cms, dtype=np.int64)
    n = cms.shape[0]
    rng = np.random.default_rng(seed)
    point = pcm.metrics_from_confusion(cms.sum(axis=0), codes, ignore_index=ignore_index,
                                       report_only=report_only)["macro_iou"]
    vals = np.empty(B, dtype=np.float64)
    for b in range(B):
        pick = rng.integers(0, n, size=n)
        m = pcm.metrics_from_confusion(cms[pick].sum(axis=0), codes,
                                       ignore_index=ignore_index, report_only=report_only)
        vals[b] = np.nan if m["macro_iou"] is None else m["macro_iou"]
    good = vals[~np.isnan(vals)]
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {"macro_iou": float(point),
            "ci_low": float(np.percentile(good, lo)), "ci_high": float(np.percentile(good, hi)),
            "B": B, "n_resamples_valid": int(len(good)), "seed": seed, "ci": ci,
            "unit": "route (whole-route block bootstrap)"}


def bootstrap_paired_pooled_diff(cms_a, cms_b, B=10000, seed=0, ci=0.95,
                                 codes=CLASS_ORDER, ignore_index=0, report_only=(9,)):
    """Paired whole-route block bootstrap of the POOLED Macro-IoU DIFFERENCE (cell A minus cell B).

    Declaration section 4: "the same route resample is applied to both cells in each replicate".
    That pairing is what makes the interval an interval on the difference rather than on two
    independent quantities, and it matters because the two cells share the routes exactly.
    """
    a = np.asarray(cms_a, dtype=np.int64)
    b = np.asarray(cms_b, dtype=np.int64)
    assert a.shape == b.shape, f"route matrices differ in shape: {a.shape} vs {b.shape}"
    n = a.shape[0]
    rng = np.random.default_rng(seed)

    def macro(cm):
        return pcm.metrics_from_confusion(cm, codes, ignore_index=ignore_index,
                                          report_only=report_only)["macro_iou"]

    point_a, point_b = macro(a.sum(axis=0)), macro(b.sum(axis=0))
    vals = np.empty(B, dtype=np.float64)
    for i in range(B):
        pick = rng.integers(0, n, size=n)             # ONE resample, used for both cells
        ma, mb = macro(a[pick].sum(axis=0)), macro(b[pick].sum(axis=0))
        vals[i] = np.nan if (ma is None or mb is None) else (ma - mb)
    good = vals[~np.isnan(vals)]
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return {"macro_iou_a": float(point_a), "macro_iou_b": float(point_b),
            "difference": float(point_a - point_b),
            "ci_low": float(np.percentile(good, lo)), "ci_high": float(np.percentile(good, hi)),
            "excludes_zero": bool(np.percentile(good, lo) > 0 or np.percentile(good, hi) < 0),
            "B": B, "n_resamples_valid": int(len(good)), "seed": seed, "ci": ci,
            "unit": "route (paired: same resample applied to both cells)"}


def mcnemar(b, c):
    """McNemar on the two discordant counts. Exact binomial, plus the corrected chi-square.

    CONTEXT ONLY. On 1.2e10 pooled pixels any non-trivial discordance is significant by
    construction, so this number carries no weight in the claim structure.
    """
    b, c = int(b), int(c)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_exact": 1.0, "chi2": 0.0, "p_chi2": 1.0,
                "note": "no discordant pixels"}
    probs = np.array([math.comb(n, i) for i in range(n + 1)], dtype=np.float64) / (2.0 ** n) \
        if n <= 1000 else None
    if probs is not None:
        p_exact = float(probs[probs <= probs[b] + 1e-15].sum())
    else:
        z = (abs(b - c) - 1) / math.sqrt(n)
        p_exact = float(math.erfc(abs(z) / math.sqrt(2)))
    chi2 = (abs(b - c) - 1) ** 2 / n
    return {"b": b, "c": c, "p_exact": min(1.0, p_exact), "chi2": float(chi2),
            "p_chi2": float(_chi2_sf(chi2, 1)),
            "note": "context only -- near-automatic at pooled-pixel N"}


# ==================================================================================================
# declaration
# ==================================================================================================
def load_declaration(path):
    """Parse and validate the author's declaration block.

    The schema is the one in `2026-08-25_pre_declarations.md` section 6, which is AUTHORITATIVE:
    `status` gates execution (not a timestamp), `families` is a mapping of family name to a list of
    pairs, `per_route_metric` is a mapping, and the route rules are flat keys. An earlier draft of
    this runner assumed a different shape and would have refused the real file on lock day.
    """
    text = Path(path).read_text(encoding="utf-8")
    blocks = re.findall(r"```(?:yaml|yml|json)\s*\n(.*?)```", text, flags=re.S)
    if not blocks:
        sys.exit(f"{path}: no fenced yaml/json block found -- see --emit-template")
    body = blocks[0]
    try:
        import yaml
        dec = yaml.safe_load(body)
    except ImportError:
        try:
            dec = json.loads(body)
        except Exception:
            sys.exit("PyYAML is not installed and the block is not JSON -- "
                     "install pyyaml or write the block as fenced json")
    if not isinstance(dec, dict):
        sys.exit(f"{path}: the declaration block did not parse to a mapping")

    for key in ("status", "alpha", "per_route_metric", "families", "bootstrap"):
        if key not in dec:
            sys.exit(f"{path}: declaration is missing required key '{key}'")

    status = str(dec["status"]).strip().upper()
    if status != "LOCKED":
        sys.exit(f"{path}: status is '{dec['status']}', not LOCKED. The declaration must be locked "
                 f"before any test statistic is computed. Refusing to run.")

    fams = dec["families"]
    if not isinstance(fams, dict):
        sys.exit(f"{path}: 'families' must be a mapping of family name -> list of pairs")
    for name, pairs in fams.items():
        for p in pairs:
            if not (isinstance(p, (list, tuple)) and len(p) == 2):
                sys.exit(f"{path}: family {name} has a malformed pair {p!r}; expected [a, b]")

    desc = set(dec.get("descriptive_cells", []))
    in_family = {c for pairs in fams.values() for p in pairs for c in p}
    leaked = in_family & desc
    if leaked:
        sys.exit(f"{path}: descriptive-only cells appear inside a Holm family: {sorted(leaked)}. "
                 f"Declaration D1 puts the arms outside every family. Refusing to run.")

    for a, b in dec.get("descriptive_contrasts", []):
        if a not in desc:
            sys.exit(f"{path}: descriptive contrast [{a}, {b}] names {a}, which is not in "
                     f"descriptive_cells. Refusing to run.")
    return dec


def load_declaration_from(write_fn, doc, path):
    """Self-test helper: write a declaration dict to `path` and parse it back."""
    write_fn(doc)
    return load_declaration(path)


def normalised(dec):
    """Flatten the declaration into the handful of values the runner actually consumes."""
    prm = dec["per_route_metric"]
    boot = dec.get("bootstrap", {})
    return {
        "metric": prm["name"] if isinstance(prm, dict) else prm,
        "ignore_index": (prm.get("ignore_index", 0) if isinstance(prm, dict) else 0),
        "report_only": tuple(
            CLASS_ORDER.index(c) for c in (prm.get("report_only", ["unknown2"])
                                           if isinstance(prm, dict) else ["unknown2"])),
        "alpha": float(dec["alpha"]),
        "near_tie_delta": float(dec.get("near_tie_delta", 0.0) or 0.0),
        "min_tiles": dec.get("route_min_tiles_sensitivity"),
        "families": dec["families"],
        "descriptive_cells": list(dec.get("descriptive_cells", [])),
        "descriptive_contrasts": [tuple(x) for x in dec.get("descriptive_contrasts", [])],
        "friedman_families": list((dec.get("friedman") or {}).get("families", [])),
        "B": int(boot.get("B", 10000)),
        "seed": int(boot.get("seed", 0)),
        "ci": 0.95 if str(boot.get("ci", "percentile_95")).endswith("95") else float(boot["ci"]),
        "sign_near_tie": bool((dec.get("sign_test") or {}).get("near_tie_sensitivity", True)),
    }


def load_route_scores(metric_col):
    with open(ROUTE_METRICS_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if metric_col not in rows[0]:
        sys.exit(f"declared per_route_metric '{metric_col}' is not a column of "
                 f"{ROUTE_METRICS_CSV.name}; available: {sorted(rows[0])}")
    tiles, scores = {}, {}
    for r in rows:
        scores.setdefault(r["cell"], {})[r["route"]] = (
            float(r[metric_col]) if r[metric_col] != "" else np.nan)
        tiles[r["route"]] = int(r["tiles"])
    return scores, tiles


# ==================================================================================================
# self-test
# ==================================================================================================
def selftest():
    print("=" * 92)
    print("partb_statistics selftest -- synthetic data with known answers, NO real scores touched")
    print("=" * 92)
    ok = lambda label, detail="": print(f"  [OK] {label}" + (f"  --  {detail}" if detail else ""))

    # --- Wilcoxon: hand-checkable case, all differences positive ---
    d = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    w = wilcoxon_signed_rank(d)
    assert w["effective_n"] == 5 and w["n_exact_ties"] == 0
    assert w["w_plus"] == 15.0 and w["w_minus"] == 0.0
    assert abs(w["p_value"] - 2 / 32) < 1e-12, w["p_value"]          # 2 of 32 sign patterns
    assert abs(w["rank_biserial"] - 1.0) < 1e-12
    assert abs(w["min_attainable_two_sided_p"] - 2 / 32) < 1e-12
    ok("Wilcoxon exact, all-positive N=5", f"p = {w['p_value']:.6f} = 2/32, rank-biserial 1.0")

    # --- THE FORCED ALL-TIE ROUTE: zeros must drop, effective N must fall, and it must be visible
    d2 = np.array([0.1, 0.2, 0.0, 0.3, 0.0, 0.4, 0.5])
    w2 = wilcoxon_signed_rank(d2)
    assert w2["n_pairs"] == 7 and w2["n_exact_ties"] == 2 and w2["effective_n"] == 5
    assert abs(w2["p_value"] - w["p_value"]) < 1e-12, "ties must not change the surviving test"
    ok("forced all-tie routes dropped, effective N reported",
       f"7 pairs -> {w2['n_exact_ties']} ties -> effective N {w2['effective_n']}")

    everything_tied = wilcoxon_signed_rank(np.zeros(6))
    assert everything_tied["effective_n"] == 0 and everything_tied["p_value"] is None
    assert "undefined" in everything_tied["p_method"]
    ok("every pair tied -> undefined rather than a fabricated p")

    # --- the effective-N power arithmetic the declaration rests on ---
    for n, want in ((16, 2 / 2 ** 16), (12, 2 / 2 ** 12), (9, 2 / 2 ** 9), (8, 2 / 2 ** 8)):
        got = wilcoxon_signed_rank(np.arange(1, n + 1) / 10.0)["min_attainable_two_sided_p"]
        assert abs(got - want) < 1e-15, (n, got, want)
    p9 = wilcoxon_signed_rank(np.arange(1, 10) / 10.0)["min_attainable_two_sided_p"]
    p8 = wilcoxon_signed_rank(np.arange(1, 9) / 10.0)["min_attainable_two_sided_p"]
    assert p9 < 0.05 / 10 < p8, (p9, p8)          # clears Holm's first threshold at 9, not at 8
    ok("effective-N power arithmetic reproduces the declaration",
       f"min two-sided p: N=9 -> {p9:.4f} (clears 0.0050), N=8 -> {p8:.4f} (cannot)")

    # --- ties inside |d| use average ranks ---
    wt = wilcoxon_signed_rank(np.array([0.2, -0.2, 0.5]))
    assert abs(wt["w_plus"] - (1.5 + 3)) < 1e-12 and abs(wt["w_minus"] - 1.5) < 1e-12
    ok("average ranks on tied |d|", f"W+ {wt['w_plus']}, W- {wt['w_minus']}")

    # --- sign test ---
    s = sign_test(np.array([1, 1, 1, 1, 1.0]))
    assert abs(s["p_value"] - 2 / 32) < 1e-12
    s2 = sign_test(np.array([1, 1, 1, -1, 0, 0.0]))
    assert s2["effective_n"] == 4 and s2["n_positive"] == 3
    assert abs(s2["p_value"] - (2 * (4 + 1) / 16)) < 1e-12, s2["p_value"]
    ok("exact sign test", f"5 positives p = {s['p_value']:.6f}; 3/4 with 2 ties p = {s2['p_value']:.4f}")

    # --- Holm ---
    h = holm_bonferroni([0.001, 0.008, 0.039, 0.041, 0.042], alpha=0.05)
    assert h["p_adjusted"][0] == 0.005 and abs(h["p_adjusted"][1] - 0.032) < 1e-12
    assert h["reject"] == [True, True, False, False, False], h["reject"]
    hm = holm_bonferroni([0.04, 0.01], alpha=0.05)
    assert hm["p_adjusted"][0] >= hm["p_adjusted"][1], "adjusted p must be monotone in rank"
    ok("Holm-Bonferroni step-down", f"adjusted {[round(x, 4) for x in h['p_adjusted']]}")

    # --- Friedman + Kendall's W ---
    perfect = np.array([[1.0, 2, 3]] * 8)                # every block ranks identically
    f = friedman(perfect)
    assert abs(f["kendalls_w"] - 1.0) < 1e-9, f["kendalls_w"]
    assert f["mean_ranks"] == [1.0, 2.0, 3.0]
    ok("Friedman: perfect concordance -> Kendall's W = 1", f"chi2 {f['chi2']:.3f}, df {f['df']}")
    rng = np.random.default_rng(7)
    noise = rng.normal(size=(40, 4))
    fn = friedman(noise)
    assert fn["p_value"] > 0.05 and fn["kendalls_w"] < 0.2, (fn["p_value"], fn["kendalls_w"])
    ok("Friedman: pure noise -> not significant, W near 0",
       f"p {fn['p_value']:.3f}, W {fn['kendalls_w']:.3f}")

    # --- chi-square tail, against known values ---
    for x, df, want in ((3.841458820694124, 1, 0.05), (5.991464547107979, 2, 0.05),
                        (11.070497693516351, 5, 0.05), (0.0, 3, 1.0)):
        got = _chi2_sf(x, df)
        assert abs(got - want) < 1e-6, (x, df, got, want)
    ok("chi-square upper tail matches table values to 1e-6")

    # --- bootstrap of the median difference ---
    b = bootstrap_median_diff(np.array([0.01] * 20), B=500, seed=1)
    assert b["median_difference"] == 0.01 and b["ci_low"] == 0.01 and b["ci_high"] == 0.01
    b2 = bootstrap_median_diff(rng.normal(0.5, 0.1, 50), B=2000, seed=2)
    assert b2["ci_low"] < b2["median_difference"] < b2["ci_high"]
    ok("bootstrap CI brackets the point estimate", f"[{b2['ci_low']:.4f}, {b2['ci_high']:.4f}]")

    # --- whole-route block bootstrap on synthetic route confusion matrices ---
    n_routes = 16
    cms = np.zeros((n_routes, 11, 11), dtype=np.int64)
    for r in range(n_routes):
        cms[r, 1, 1] = 800 + 10 * r          # asfalt correct
        cms[r, 1, 4] = 200                   # asfalt -> ubefestet
        cms[r, 4, 4] = 5000
        cms[r, 0, 1] = 77                    # ignored label pixels, must not enter anything
    bp = bootstrap_pooled_macro_iou(cms, B=200, seed=3)
    direct = pcm.metrics_from_confusion(cms.sum(axis=0), CLASS_ORDER,
                                        ignore_index=0, report_only=(9,))["macro_iou"]
    assert abs(bp["macro_iou"] - direct) < 1e-12
    assert bp["ci_low"] <= bp["macro_iou"] <= bp["ci_high"]
    assert bp["unit"].startswith("route")
    ok("whole-route block bootstrap recomputes the POOLED statistic",
       f"macro-IoU {bp['macro_iou']:.4f}, CI [{bp['ci_low']:.4f}, {bp['ci_high']:.4f}]")

    identical = np.repeat(cms[:1], n_routes, axis=0)
    bi = bootstrap_pooled_macro_iou(identical, B=100, seed=4)
    assert abs(bi["ci_high"] - bi["ci_low"]) < 1e-12, "identical routes must give a zero-width CI"
    ok("identical routes -> zero-width CI (the bootstrap is resampling routes, not pixels)")

    # --- McNemar ---
    m = mcnemar(10, 0)
    assert abs(m["p_exact"] - 2 / 1024) < 1e-12, m["p_exact"]
    assert mcnemar(0, 0)["p_exact"] == 1.0
    big = mcnemar(5_000_000, 4_900_000)
    assert big["p_chi2"] < 1e-9, big
    ok("McNemar exact and chi-square", f"10/0 -> p {m['p_exact']:.6f}; "
       f"5.0M/4.9M -> p {big['p_chi2']:.2e} (near-automatic, as declared)")

    # --- near-tie sensitivity (declaration D2, Sensitivity B) ---
    d_near = np.array([0.0004, -0.0002, 0.05, 0.07, 0.09, 0.11, 0.13])
    w_prim = wilcoxon_signed_rank(d_near, near_tie_delta=0.0)
    w_sens = wilcoxon_signed_rank(d_near, near_tie_delta=0.001)
    assert w_prim["effective_n"] == 7 and w_prim["n_near_ties"] == 0
    assert w_sens["effective_n"] == 5 and w_sens["n_near_ties"] == 2
    # The sensitivity COSTS power, and that is the declared point of reporting n' beside every p.
    # Here the floor on attainable p rises from 2/2^7 to 2/2^5, and the realised p rises with it:
    # primary 4/128 = 0.03125 (the one discordant sign carries rank 1, so only 4 sign patterns are
    # at least as extreme) against sensitivity 2/32 = 0.0625 (a clean sweep of just 5 pairs).
    # Dropping two near-ties removes the discordance but costs more than it buys.
    assert (w_sens["min_attainable_two_sided_p"] > w_prim["min_attainable_two_sided_p"]), \
        "dropping near-ties must raise the floor on attainable p"
    assert abs(w_prim["p_value"] - 4 / 128) < TOL, w_prim["p_value"]
    assert abs(w_sens["p_value"] - 2 / 32) < TOL, w_sens["p_value"]
    s_sens = sign_test(d_near, near_tie_delta=0.001)
    assert s_sens["effective_n"] == 5 and s_sens["n_positive"] == 5
    ok("near-tie sensitivity drops |d| < delta, reports the count, and COSTS power",
       f"n' 7 -> 5; p {w_prim['p_value']:.5f} -> {w_sens['p_value']:.5f}; "
       f"floor {w_prim['min_attainable_two_sided_p']:.5f} -> "
       f"{w_sens['min_attainable_two_sided_p']:.5f}")
    assert wilcoxon_signed_rank(d_near)["effective_n"] == 7, \
        "the PRIMARY test must be untouched by the sensitivity default"
    ok("primary analysis unaffected by the near-tie default (delta is a labelled sensitivity)")

    # --- rank-biserial: the declaration's formula and the implemented one must agree ---
    rb_impl = w_prim["rank_biserial"]
    nprime = w_prim["effective_n"]
    rb_decl = (w_prim["w_plus"] - w_prim["w_minus"]) / (nprime * (nprime + 1) / 2)
    assert abs(rb_impl - rb_decl) < TOL, (rb_impl, rb_decl)
    ok("rank-biserial matches the declaration's (W+ - W-)/(n'(n'+1)/2) form",
       f"{rb_impl:.6f}")

    # --- paired pooled-difference bootstrap: same resample applied to both cells ---
    cms_a = cms.copy()
    cms_b = cms.copy()
    cms_b[:, 1, 1] -= 200                       # cell B is uniformly worse on asfalt
    cms_b[:, 1, 4] += 200
    pd_ = bootstrap_paired_pooled_diff(cms_a, cms_b, B=300, seed=5)
    assert pd_["difference"] > 0 and pd_["ci_low"] > 0 and pd_["excludes_zero"]
    assert "same resample" in pd_["unit"]
    same = bootstrap_paired_pooled_diff(cms_a, cms_a.copy(), B=100, seed=6)
    assert same["difference"] == 0.0 and same["ci_low"] == 0.0 and same["ci_high"] == 0.0, same
    ok("paired pooled-difference bootstrap", "identical cells -> exactly zero-width CI at 0; "
       f"uniformly-worse cell -> {pd_['difference']:+.4f}, CI excludes 0")

    # --- declaration guards, against the AUTHOR'S ACTUAL SCHEMA ---
    import tempfile
    base = {
        "declaration_version": "2026-08-25", "status": "DRAFT", "alpha": 0.05,
        "near_tie_delta": 0.001, "route_min_tiles_sensitivity": 100,
        "per_route_metric": {"name": "macro_iou_present_classes",
                             "presence_rule": "support_pixels_gt_0",
                             "ignore_index": 0, "report_only": ["unknown2"]},
        "families": {"F1": [["convnext_upernet_rgb", "swin_upernet_rgb"]]},
        "descriptive_cells": ["convnext_upernet_rgb_ndsm"],
        "descriptive_contrasts": [["convnext_upernet_rgb_ndsm", "convnext_upernet_rgb"]],
        "friedman": {"families": ["F1"]},
        "bootstrap": {"B": 10, "seed": 1, "unit": "route", "ci": "percentile_95"},
        "sign_test": {"near_tie_sensitivity": True},
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "dec.md"

        def write(doc):
            p.write_text("```yaml\n" + json.dumps(doc) + "\n```", encoding="utf-8")

        write(base)
        try:
            load_declaration(p)
        except SystemExit as e:
            assert "not LOCKED" in str(e), str(e)
            ok("status: DRAFT refused")
        else:
            sys.exit("FAIL: a DRAFT declaration was accepted")

        locked = dict(base, status="LOCKED")
        dec = load_declaration_from(write, locked, p)
        D = normalised(dec)
        assert D["metric"] == "macro_iou_present_classes"
        assert D["near_tie_delta"] == 0.001 and D["min_tiles"] == 100
        assert D["report_only"] == (9,), D["report_only"]
        assert D["ci"] == 0.95 and D["B"] == 10
        assert list(D["families"]) == ["F1"] and D["friedman_families"] == ["F1"]
        ok("LOCKED declaration in the author's schema parses",
           "families-as-mapping, per_route_metric mapping, percentile_95, flat route rules")

        bad = dict(locked, families={"F1": [["convnext_upernet_rgb", "convnext_upernet_rgb_ndsm"]]})
        try:
            load_declaration_from(write, bad, p)
        except SystemExit as e:
            assert "outside every family" in str(e), str(e)
            ok("descriptive arm cell inside a Holm family refused")
        else:
            sys.exit("FAIL: an arm cell was allowed into a family")

        bad2 = dict(locked, descriptive_contrasts=[["not_declared_cell", "convnext_upernet_rgb"]])
        try:
            load_declaration_from(write, bad2, p)
        except SystemExit as e:
            assert "not in descriptive_cells" in str(e), str(e)
            ok("descriptive contrast naming an undeclared cell refused")
        else:
            sys.exit("FAIL: an undeclared contrast cell was accepted")

    # --- cross-check against scipy where it exists, as an independent witness ---
    try:
        from scipy import stats as sps
        x = np.array([0.12, -0.03, 0.44, 0.09, -0.21, 0.33, 0.05, 0.18, -0.02, 0.27, 0.31, 0.08])
        mine = wilcoxon_signed_rank(x)
        theirs = sps.wilcoxon(x, alternative="two-sided", method="exact")
        assert abs(mine["p_value"] - float(theirs.pvalue)) < 1e-9, (mine["p_value"], theirs.pvalue)
        assert abs(mine["statistic"] - float(theirs.statistic)) < 1e-9
        fr = sps.friedmanchisquare(*noise.T)
        assert abs(friedman(noise)["p_value"] - float(fr.pvalue)) < 1e-6
        assert abs(friedman(noise)["chi2"] - float(fr.statistic)) < 1e-6
        ok("independent cross-check vs scipy", "wilcoxon exact p and friedman chi2/p agree")
    except ImportError:
        print("  [skip] scipy not available for the cross-check")

    print("\nSELFTEST PASSED -- machinery verified, no real scores were read")


# ==================================================================================================
def validate_schema(path):
    """Parse a DRAFT declaration and report exactly what the runner would consume. Computes nothing.

    Deliberately does NOT require LOCKED: this is the pre-lock check. Everything else that
    `load_declaration` asserts is applied, so a structural problem surfaces while it is still free
    to fix.
    """
    text = Path(path).read_text(encoding="utf-8")
    blocks = re.findall(r"```(?:yaml|yml|json)\s*\n(.*?)```", text, flags=re.S)
    if not blocks:
        sys.exit(f"{path}: FAIL -- no fenced yaml/json block found")
    import yaml
    try:
        dec = yaml.safe_load(blocks[0])
    except Exception as exc:                                  # noqa: BLE001
        sys.exit(f"{path}: FAIL -- the block is not valid YAML: {type(exc).__name__}: {exc}")
    if len(blocks) > 1:
        print(f"  NOTE: {len(blocks)} fenced blocks found; the runner reads the FIRST one only.")

    problems, notes = [], []
    for key in ("status", "alpha", "per_route_metric", "families", "bootstrap"):
        if key not in dec:
            problems.append(f"missing required key '{key}'")
    if problems:
        for p_ in problems:
            print(f"  FAIL  {p_}")
        sys.exit(f"{path}: schema invalid")

    print(f"declaration    : {dec.get('declaration_version')}   status: {dec.get('status')}")
    if str(dec["status"]).strip().upper() != "LOCKED":
        notes.append(f"status is '{dec['status']}' -- the runner will refuse until it reads LOCKED")

    if not isinstance(dec["families"], dict):
        sys.exit("  FAIL  'families' must be a mapping of family name -> list of pairs")

    D = normalised(dec)
    scores, tiles = load_route_scores(D["metric"])
    scored = set(scores)
    routes_sens = [r for r in sorted(tiles)
                   if D["min_tiles"] is None or tiles[r] >= int(D["min_tiles"])]

    print(f"per-route metric: {D['metric']}   (column present in {ROUTE_METRICS_CSV.name}: yes)")
    print(f"alpha {D['alpha']}   near-tie delta {D['near_tie_delta']}   "
          f"route min tiles {D['min_tiles']}")
    print(f"routes primary {len(tiles)}, sensitivity A {len(routes_sens)} "
          f"(drops {[r for r in sorted(tiles) if r not in routes_sens]})")
    print(f"bootstrap: B={D['B']} seed={D['seed']} ci={D['ci']}")

    n_pairs = 0
    print("\nfamilies:")
    for name, pairs in D["families"].items():
        n_pairs += len(pairs)
        thr = D["alpha"] / len(pairs) if pairs else float("nan")
        print(f"  {name:<40} {len(pairs)} pairs   Holm sharpest threshold {thr:.5f}")
        for p_ in pairs:
            if len(p_) != 2:
                problems.append(f"{name}: malformed pair {p_!r}")
                continue
            for c in p_:
                if c not in scored:
                    problems.append(f"{name}: FAMILY cell '{c}' has no route scores yet "
                                    f"(fatal at run time)")
    print(f"  total {n_pairs} tests across {len(D['families'])} families")

    desc = set(D["descriptive_cells"])
    in_fam = {c for pairs in D["families"].values() for p_ in pairs for c in p_}
    if in_fam & desc:
        problems.append(f"descriptive cells inside a family: {sorted(in_fam & desc)}")
    not_scored = sorted(desc - scored)
    print(f"\ndescriptive cells: {len(desc)} declared, {len(desc & scored)} scored, "
          f"{len(not_scored)} not yet scored (will be skipped with notice)")
    for c in not_scored:
        print(f"    pending: {c}")
    for a, b in D["descriptive_contrasts"]:
        if a not in desc:
            problems.append(f"contrast [{a}, {b}]: '{a}' is not in descriptive_cells")
    runnable = [(a, b) for a, b in D["descriptive_contrasts"] if a in scored and b in scored]
    print(f"descriptive contrasts: {len(D['descriptive_contrasts'])} declared, "
          f"{len(runnable)} runnable now")

    fri = [f for f in D["friedman_families"] if f not in D["families"]]
    if fri:
        problems.append(f"friedman names families that do not exist: {fri}")

    print()
    if problems:
        for p_ in problems:
            print(f"  PROBLEM  {p_}")
    for n_ in notes:
        print(f"  NOTE     {n_}")
    if not problems:
        print("SCHEMA OK -- the runner can consume this block as written.")
    else:
        print(f"{len(problems)} problem(s) -- fix BEFORE locking; afterwards it needs an amendment.")
    return 0 if not problems else sys.exit(1)


def run(dec, dry_run):
    D = normalised(dec)
    scores, tiles = load_route_scores(D["metric"])
    routes_all = sorted(tiles)
    routes_sens = [r for r in routes_all
                   if D["min_tiles"] is None or tiles[r] >= int(D["min_tiles"])]
    scored = set(scores)

    print(f"per-route metric : {D['metric']}")
    print(f"routes, primary  : {len(routes_all)}  {routes_all}")
    if D["min_tiles"]:
        dropped = [r for r in routes_all if r not in routes_sens]
        print(f"routes, sens. A  : {len(routes_sens)}  (drops {dropped} at < {D['min_tiles']} tiles)")
    print(f"near-tie delta   : {D['near_tie_delta']}  (sensitivity B)")
    print(f"cells with route scores : {len(scored)}")

    plan = [(name, tuple(p)) for name, pairs in D["families"].items() for p in pairs]
    print(f"\ndeclared families: {list(D['families'])}")
    print(f"declared pairs   : {len(plan)}")

    # A family cell that is not scored is FATAL: dropping a pair would silently change the family
    # size and therefore every Holm threshold in it. A descriptive cell that is not scored is not
    # fatal -- it is a cell still in the GPU queue -- so it is skipped with a notice.
    fam_missing = sorted({c for _n, pr in plan for c in pr} - scored)
    if fam_missing:
        sys.exit(f"declared FAMILY cells absent from {ROUTE_METRICS_CSV.name}: {fam_missing}\n"
                 f"Holm thresholds depend on the family size, so a partial family is not a valid "
                 f"run. Refusing.")

    desc_missing = sorted(set(D["descriptive_cells"]) - scored)
    if desc_missing:
        print(f"\n  NOTICE: {len(desc_missing)} declared descriptive cell(s) not yet scored -- "
              f"skipped, not fatal:")
        for c in desc_missing:
            print(f"    - {c}")
        print("  Re-run after they land; the family results above are unaffected by their absence.")
    contrasts = [(a, b) for a, b in D["descriptive_contrasts"]
                 if a in scored and b in scored]
    skipped_contrasts = [(a, b) for a, b in D["descriptive_contrasts"]
                         if a not in scored or b not in scored]
    print(f"descriptive contrasts: {len(contrasts)} runnable, {len(skipped_contrasts)} skipped "
          f"(cell not yet scored)")
    print(f"bootstrap        : unit=route B={D['B']} seed={D['seed']} ci={D['ci']}")

    if dry_run:
        print("\nDRY RUN -- declaration parsed and validated, nothing computed. Pass --go to run.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results, friedman_out = [], {}

    for fam_name, pairs in D["families"].items():
        rows, pvals = [], []
        for a, b in pairs:
            variants = [("primary", routes_all, 0.0),
                        ("sensitivity_A_route_size", routes_sens, 0.0),
                        ("sensitivity_B_near_tie", routes_all, D["near_tie_delta"])]
            for tag, routes, delta in variants:
                d = np.array([scores[a][r] - scores[b][r] for r in routes], dtype=np.float64)
                w = wilcoxon_signed_rank(d, near_tie_delta=delta)
                s = sign_test(d, near_tie_delta=delta if D["sign_near_tie"] else 0.0)
                eff = bootstrap_median_diff(d, B=D["B"], seed=D["seed"], ci=D["ci"])
                rows.append({"family": fam_name, "cell_a": a, "cell_b": b, "route_set": tag,
                             "n_routes": len(routes),
                             **{f"wilcoxon_{k}": v for k, v in w.items()},
                             "sign_p_value": s["p_value"], "sign_n_positive": s["n_positive"],
                             "sign_effective_n": s["effective_n"],
                             **{f"effect_{k}": v for k, v in eff.items()}})
                if tag == "primary":
                    pvals.append(w["p_value"])

        # D2 says a pair is fragile when primary and either sensitivity disagree in significance
        # AFTER HOLM. So each route set gets its own Holm correction within the family, and the
        # comparison is decision-vs-decision. Comparing a raw sensitivity p against the primary's
        # Holm decision would mix corrected and uncorrected quantities and misreport fragility in
        # both directions.
        for tag in ("primary", "sensitivity_A_route_size", "sensitivity_B_near_tie"):
            subset = [r for r in rows if r["route_set"] == tag]
            ps = [r["wilcoxon_p_value"] for r in subset]
            if ps and all(p is not None for p in ps):
                h = holm_bonferroni(ps, alpha=D["alpha"])
                for r, adj, rej, thr in zip(subset, h["p_adjusted"], h["reject"],
                                            h["step_threshold"]):
                    r["holm_p_adjusted"], r["holm_reject"], r["holm_threshold"] = adj, rej, thr

        prim = [r for r in rows if r["route_set"] == "primary"]
        for r in prim:
            decision = bool(r.get("holm_reject"))
            sibs = [x for x in rows if (x["cell_a"], x["cell_b"]) == (r["cell_a"], r["cell_b"])
                    and x["route_set"] != "primary"]
            flips = [x["route_set"] for x in sibs if bool(x.get("holm_reject")) != decision]
            # The sign test is declared uncorrected, so it is compared at alpha, as D2 states.
            sign_disagrees = ((r["sign_p_value"] is not None
                               and (r["sign_p_value"] <= D["alpha"]) != decision))
            r["fragile"] = bool(flips or sign_disagrees)
            r["fragile_reason"] = "|".join(
                flips + (["sign_test"] if sign_disagrees else [])) or ""
        results.extend(rows)

        cells = sorted({c for p in pairs for c in p})
        if fam_name in D["friedman_families"] and len(cells) >= 2:
            mat = np.array([[scores[c][r] for c in cells] for r in routes_all])
            fr = friedman(mat)
            fr.update({"family": fam_name, "cells": cells, "route_set": "primary",
                       "rejects_at_alpha": bool(fr["p_value"] <= D["alpha"]),
                       "narrative_downgraded_to_descriptive": bool(fr["p_value"] > D["alpha"])})
            friedman_out[fam_name] = fr
            (OUT_DIR / f"friedman_{fam_name}.json").write_text(json.dumps(fr, indent=2),
                                                               encoding="utf-8")
            print(f"\nFriedman [{fam_name}]  chi2 {fr['chi2']:.4f}  df {fr['df']}  "
                  f"p {fr['p_value']:.5f}  Kendall's W {fr['kendalls_w']:.4f}"
                  + ("" if fr["rejects_at_alpha"] else
                     "   -> FAILS TO REJECT: family narrative downgraded to descriptive (D1)"))
            for c, mr in zip(cells, fr["mean_ranks"]):
                print(f"    mean rank {mr:6.3f}  {c}")

    if results:
        out = C.assert_writes_are_local(OUT_DIR / "wilcoxon_by_family.csv")
        keys = sorted({k for r in results for k in r})
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
        print(f"\nwrote {out}  ({len(results)} rows)")

    # ---- bootstrap CIs: standalone per cell, and paired on the difference for the contrasts ----
    z = np.load(ROUTE_CMS_NPZ, allow_pickle=False)
    idx = {str(c): i for i, c in enumerate(z["cells"])}
    ci_rows = []
    family_cells = sorted({c for _n, pr in plan for c in pr})
    for cell in family_cells + [c for c in D["descriptive_cells"] if c in scored]:
        if cell not in idx:
            print(f"  [skip] no route matrices cached for {cell}")
            continue
        r = bootstrap_pooled_macro_iou(z["cms"][idx[cell]], B=D["B"], seed=D["seed"], ci=D["ci"],
                                       ignore_index=D["ignore_index"],
                                       report_only=D["report_only"])
        r["cell"] = cell
        r["role"] = "family" if cell in family_cells else "descriptive"
        ci_rows.append(r)
        print(f"  {cell:<44} macro-IoU {r['macro_iou']:.4f}  "
              f"CI [{r['ci_low']:.4f}, {r['ci_high']:.4f}]  ({r['role']})")
    if ci_rows:
        out = C.assert_writes_are_local(OUT_DIR / "pooled_macro_iou_route_bootstrap.csv")
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(ci_rows[0]))
            w.writeheader()
            w.writerows(ci_rows)
        print(f"wrote {out}")

    diff_rows = []
    for a, b in contrasts:
        if a not in idx or b not in idx:
            print(f"  [skip] no route matrices cached for the contrast {a} - {b}")
            continue
        r = bootstrap_paired_pooled_diff(z["cms"][idx[a]], z["cms"][idx[b]], B=D["B"],
                                         seed=D["seed"], ci=D["ci"],
                                         ignore_index=D["ignore_index"],
                                         report_only=D["report_only"])
        r.update({"cell_a": a, "cell_b": b, "test": "NONE -- descriptive, outside every family"})
        diff_rows.append(r)
        print(f"  {a:<44} - {b:<44} diff {r['difference']:+.4f}  "
              f"CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
    if diff_rows:
        out = C.assert_writes_are_local(OUT_DIR / "descriptive_contrast_paired_bootstrap.csv")
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(diff_rows[0]))
            w.writeheader()
            w.writerows(diff_rows)
        print(f"wrote {out}  ({len(diff_rows)} contrasts)")

    (OUT_DIR / "run_provenance.json").write_text(json.dumps({
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "declaration_version": dec.get("declaration_version"),
        "declaration_status": dec.get("status"),
        "per_route_metric": D["metric"], "alpha": D["alpha"],
        "near_tie_delta": D["near_tie_delta"],
        "routes_primary": routes_all, "routes_sensitivity_A": routes_sens,
        "families": {k: v for k, v in D["families"].items()},
        "descriptive_cells_scored": [c for c in D["descriptive_cells"] if c in scored],
        "descriptive_cells_skipped_not_yet_scored": desc_missing,
        "descriptive_contrasts_run": [list(c) for c in contrasts],
        "descriptive_contrasts_skipped": [list(c) for c in skipped_contrasts],
        "friedman": {k: {kk: v[kk] for kk in ("chi2", "df", "p_value", "kendalls_w",
                                              "rejects_at_alpha")}
                     for k, v in friedman_out.items()},
        "excluded_tests": dec.get("excluded_tests", []),
        "source": str(ROUTE_METRICS_CSV),
        # default=str: YAML turns an unquoted 2026-08-25 into a datetime.date, which json cannot
        # serialise. Coerced rather than quoted in the declaration, because the declaration is
        # locked and must not be edited to suit the runner.
    }, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'run_provenance.json'}")


def main():
    ap = argparse.ArgumentParser(description="Part B route-level statistics")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit-template", action="store_true")
    ap.add_argument("--declaration")
    ap.add_argument("--validate-schema", action="store_true",
                    help="parse a DRAFT declaration and report what the runner would consume, "
                         "without requiring LOCKED and without computing anything. Use this before "
                         "locking: after lock, a fix needs a dated amendment.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--go", action="store_true",
                    help="required to compute anything on real scores")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.emit_template:
        print(TEMPLATE)
        return
    if not args.declaration:
        sys.exit("no --declaration. This module will not compute a test statistic on real scores "
                 "without the author's locked pre-declaration file. Use --selftest or "
                 "--emit-template.")
    if args.validate_schema:
        return validate_schema(args.declaration)
    dec = load_declaration(args.declaration)
    if not (args.go or args.dry_run):
        sys.exit("declaration parsed, but --go was not passed. Refusing to compute p-values.")
    run(dec, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
