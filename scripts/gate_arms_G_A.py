#!/usr/bin/env python
"""
Gates G-C and the arm-G dataloader sanity check for the 2026-08-24 arms (work order 2026-08-24).

  G-C  split dry-run       -- the frozen partition still reads 6,439 / 6,437 / 6,438, is a clean
                              partition of the 19,314 active tiles, and no route straddles folds.
                              This re-asserts the LOCKED split; it never regenerates it.

  G-E  arm-G dataloader    -- arm G is the only cell in the project that reads its pixels from the
                              spring leaf-off orthophoto (`OrtoRGB`) instead of the
                              skraafoto-programme nadir product (`rgb`). Two things have to be true
                              before 28 GPU-hours are spent on it: every one of the 19,314 tiles
                              resolves in the OrtoRGB folder, and the bands arrive at the network in
                              a sane range under the ImageNet constants the arm deliberately reuses.
                              Both are measured here on the real code path, on CPU.

`load_all_datasources_for_image` resolves each band as `<item>.parent.parent / <datatype> / <name>`,
so the item list is built from `path_to_images` (splitted/rgb) while the pixels come from
`splitted/OrtoRGB`. That indirection is exactly why this gate exists: nothing else in the pipeline
would notice if the folder were incomplete.

CPU only. Read-only. No GPU job, no training.

    python gate_arms_G_A.py
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402

REPO = Path(r"c:\thesis\ML_sdfi_fastai2")
CONFIG_G = REPO / "configs" / "matrix_configs" / "train" / "convnext_upernet_ortorgb_fold0.ini"
CONFIG_A = REPO / "configs" / "matrix_configs" / "train" / "convnext_upernet_rgb_dsm_dtm_corrected_fold0.ini"
SPLIT_DIR = C.LOGS_AND_MODELS / "route_class_audit"

N_BATCHES = 3
failures = []


def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status:4}] {label}" + (f"  --  {detail}" if detail else ""))
    if not ok:
        failures.append(label)
    return ok


# ==================================================================================================
# G-C -- split dry-run
# ==================================================================================================
def gate_gc():
    print("\n=== G-C: frozen split dry-run (read-only; the split is LOCKED) ===")
    active = C.tile_names()
    check(f"all.txt active tiles = {C.EXPECTED_TILES}", len(active) == C.EXPECTED_TILES,
          f"got {len(active)}")

    folds = {}
    for f in range(C.NFOLDS):
        p = SPLIT_DIR / f"fold_{f}_valid.txt"
        folds[f] = C.read_tile_list(str(p))
        check(f"fold {f} held-out count = {C.EXPECTED_FOLD_COUNTS[f]}",
              len(folds[f]) == C.EXPECTED_FOLD_COUNTS[f], f"got {len(folds[f])}")

    sets = {f: set(v) for f, v in folds.items()}
    for f, v in folds.items():
        check(f"fold {f} has no duplicate tiles", len(v) == len(sets[f]))
    for a in range(C.NFOLDS):
        for b in range(a + 1, C.NFOLDS):
            inter = sets[a] & sets[b]
            check(f"folds {a}/{b} disjoint", not inter, f"{len(inter)} shared")

    union = set().union(*sets.values())
    check("folds partition all.txt exactly", union == set(active),
          f"union {len(union)} vs active {len(active)}")

    # Route leakage: every route must live in exactly one fold.
    route_folds = {}
    for f, names in folds.items():
        for n in names:
            route_folds.setdefault(C.parse_route(n), set()).add(f)
    straddling = {r: sorted(fs) for r, fs in route_folds.items() if len(fs) > 1}
    check(f"no route straddles folds ({len(route_folds)} routes)", not straddling, str(straddling))

    # And the same partition as fold_assignment.csv, which is what every scoring script reads.
    r2f = C.load_fold_assignment(str(SPLIT_DIR / "fold_assignment.csv"))
    mismatch = {r: (sorted(fs)[0], r2f.get(r)) for r, fs in route_folds.items()
                if r2f.get(r) != sorted(fs)[0]}
    check(f"fold_assignment.csv agrees with the fold_*_valid.txt lists ({len(r2f)} routes)",
          not mismatch, str(mismatch))
    print(f"         route -> fold: {dict(sorted((r, sorted(fs)[0]) for r, fs in route_folds.items()))}")


# ==================================================================================================
# G-E -- arm G dataloader sanity on the real code path
# ==================================================================================================
def gate_ge():
    print("\n=== G-E: arm G (`ortorgb`) dataloader sanity ===")
    import numpy as np
    import torch  # noqa: F401
    from fastai.torch_core import default_device

    sys.path.insert(0, str(REPO / "src" / "ML_sdfi_fastai2"))
    import utils.utils as sdfi_utils
    import sdfi_dataset

    # 1. Every tile must exist in the OrtoRGB folder. The loader stacks by filename and would
    #    otherwise die thousands of tiles into training.
    active = C.tile_names()
    orto_dir = C.SPLITTED_DIR / "OrtoRGB"
    present = set(os.listdir(orto_dir))
    missing = [n for n in active if n not in present]
    check(f"all {len(active)} active tiles resolve in splitted/OrtoRGB", not missing,
          f"{len(missing)} missing, first: {missing[:3]}")

    # 2. Pull real batches through the real dataloader, on CPU.
    default_device(False)
    prev = os.getcwd()
    os.chdir(REPO)                       # matrix configs carry paths relative to the repo root
    try:
        cfg = sdfi_utils.load_settings_from_config_file(str(CONFIG_G))
        n_in = sdfi_utils.n_in_from_settings(cfg)
        check("config reports n_in = 3", n_in == 3, f"got {n_in}")
        check("datatypes = ['OrtoRGB']", cfg["datatypes"] == ["OrtoRGB"], str(cfg["datatypes"]))
        dls = sdfi_dataset.get_dataset(cfg)
        # dls.valid is unaugmented (every transform is registered split_idx=0), so these statistics
        # describe the pixels themselves rather than the augmentation.
        batches = [dls.valid.one_batch() for _ in range(N_BATCHES)]
    finally:
        os.chdir(prev)

    xb = torch.cat([b[0] for b in batches], dim=0)
    yb = torch.cat([b[1] for b in batches], dim=0)
    check(f"batch shape is (N, 3, 1000, 1000)", tuple(xb.shape[1:]) == (3, 1000, 1000),
          str(tuple(xb.shape)))
    check("batch dtype is float32", xb.dtype == torch.float32, str(xb.dtype))
    check("labels are integer class codes", yb.dtype in (torch.int64, torch.uint8, torch.int32),
          str(yb.dtype))
    check("no NaN / inf in the batch", bool(torch.isfinite(xb).all()))

    means = np.asarray(cfg["means"])
    stds = np.asarray(cfg["stds"])
    print(f"\n  {len(xb)} tiles from splitted/OrtoRGB, post-normalisation "
          f"(ImageNet constants, as the frozen rgb cell):")
    print(f"    {'band':<12}{'min':>9}{'max':>9}{'mean':>9}{'std':>9}   {'implied raw/255 mean':>22}")
    band_rows = []
    for c in range(xb.shape[1]):
        v = xb[:, c]
        mn, mx, mu, sd = float(v.min()), float(v.max()), float(v.mean()), float(v.std())
        raw_mu = mu * stds[c] + means[c]
        band_rows.append({"band": f"OrtoRGB_{'RGB'[c]}", "min": mn, "max": mx,
                          "mean": mu, "std": sd, "implied_div255_mean": raw_mu})
        print(f"    {'OrtoRGB_' + 'RGB'[c]:<12}{mn:>9.4f}{mx:>9.4f}{mu:>9.4f}{sd:>9.4f}"
              f"{raw_mu:>23.4f}")

    # Sanity envelope. uint8 imagery divided by 255 then standardised by the ImageNet constants
    # lands, per band, inside roughly [-2.5, +3.0]; the effective std should be near 0.9, which is
    # the "healthy" figure the channel audit reports for the Orto bands under these constants.
    lo = min(r["min"] for r in band_rows)
    hi = max(r["max"] for r in band_rows)
    check("post-normalisation range within [-2.5, 3.0]", -2.5 <= lo and hi <= 3.0,
          f"[{lo:.3f}, {hi:.3f}]")
    for r in band_rows:
        check(f"{r['band']} effective std in [0.5, 1.5]", 0.5 <= r["std"] <= 1.5,
              f"{r['std']:.4f}")
        check(f"{r['band']} implied raw/255 mean in [0.1, 0.9]",
              0.1 <= r["implied_div255_mean"] <= 0.9, f"{r['implied_div255_mean']:.4f}")

    out = C.assert_writes_are_local(C.TABLES / "gate_arms_G_A_ortorgb_batch_stats.json")
    out.write_text(json.dumps({
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "config": str(CONFIG_G),
        "source_folder": str(orto_dir),
        "tiles_in_sample": int(len(xb)),
        "means_used": list(map(float, means)),
        "stds_used": list(map(float, stds)),
        "note": "ImageNet constants reused deliberately; arm G swaps the image source only",
        "bands": band_rows,
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


# ==================================================================================================
def main():
    print("Gates for arms G (ortorgb) and A (rgb_dsm_dtm_corrected) -- 2026-08-24 work order")
    for p in (CONFIG_G, CONFIG_A):
        if not p.is_file():
            sys.exit(f"missing config: {p}")
    gate_gc()
    gate_ge()
    print("\n" + ("ALL GATES PASS" if not failures else f"FAILURES: {failures}"))
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
