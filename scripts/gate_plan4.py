#!/usr/bin/env python
"""
Great Plan 4, Step 2 -- gates for the five final ConvNeXt arms. CPU only, read-only, no training.

  G-diff   every new config differs from its frozen template in exactly the declared keys, carries
           exactly the declared channel block, and its training and inference halves agree. A
           silent mismatch between the two would train on one stack and infer on another.
  G-C      the frozen split still reads 6,439 / 6,437 / 6,438, partitions all.txt, and no route
           straddles folds. Re-asserts the lock; never regenerates it. Reuses gate_arms_G_A.gate_gc.
  G-E      on the REAL code path (`sdfi_dataset.get_dataset`), for each cell's fold-0 config:
           every active tile resolves in every datatype folder it reads; three DISTINCT batches
           (iterated, not `one_batch()` -- on the unshuffled valid loader that returns the same
           batch every time) have the declared shape, are finite, and every band sits in a sane
           post-normalisation envelope. Imagery: effective std 0.5 to 1.5. nDSM: decoded back to
           metres it must lie in [0, 300], i.e. the floor-at-0 clamp is intact.
  G-width  ConvNeXt+UPerNet instantiates and forwards one 1 x n_in x 1000 x 1000 tensor at
           n_in = 3 and 4, through the same wrapper class the trainer uses.

    python gate_plan4.py            # all gates
    python gate_plan4.py --skip-width
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402
import gate_arms_G_A as GA  # noqa: E402  (G-C is reused verbatim)
from plan4_build_configs import CELLS, TRAIN, INFER, TRAIN_KEYS, INFER_KEYS, fmt_list  # noqa: E402

REPO = Path(r"C:\thesis\ML_sdfi_fastai2")
N_BATCHES = 3
failures: list[str] = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL':4}] {label}" + (f"  --  {detail}" if detail else ""))
    if not ok:
        failures.append(label)
    return ok


def kv(path: Path) -> dict:
    """`key = value` lines of an ini, comments ignored. Exact text values, no parsing."""
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.lstrip().startswith(("#", ";", "[")):
            continue
        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


# ==================================================================================================
def gate_diff():
    print("\n=== G-diff: configs against their frozen templates ===")
    for cell, (dts, chans, means, stds) in CELLS.items():
        want = {"datatypes": fmt_list(dts), "channels": fmt_list(chans),
                "means": fmt_list(means), "stds": fmt_list(stds)}
        for k in range(3):
            job = f"convnext_upernet_{cell}_fold{k}"
            t_new, t_tpl = kv(TRAIN / f"{job}.ini"), kv(TRAIN / f"convnext_upernet_rgb_fold{k}.ini")
            i_new = kv(INFER / f"infer_{job}.ini")
            i_tpl = kv(INFER / f"infer_convnext_upernet_ortorgb_fold{k}.ini")

            t_changed = {x for x in set(t_new) | set(t_tpl) if t_new.get(x) != t_tpl.get(x)}
            i_changed = {x for x in set(i_new) | set(i_tpl) if i_new.get(x) != i_tpl.get(x)}
            check(f"{job} train: only declared keys differ",
                  t_changed <= TRAIN_KEYS, f"undeclared: {sorted(t_changed - TRAIN_KEYS)}")
            check(f"{job} infer: only declared keys differ",
                  i_changed <= INFER_KEYS, f"undeclared: {sorted(i_changed - INFER_KEYS)}")
            check(f"{job} channel block = declaration",
                  all(t_new.get(x) == v for x, v in want.items()),
                  str({x: t_new.get(x) for x in want}))
            check(f"{job} train and infer channel blocks identical",
                  all(t_new.get(x) == i_new.get(x) for x in want))
            check(f"{job} job_name / model_to_load / output_folder point at its own job",
                  t_new.get("job_name") == f'"{job}"'
                  and i_new.get("model_to_load", "").endswith(f"/{job}/models/{job}.pth")
                  and i_new.get("output_folder", "").endswith(f"/{job}/models/example_dataset"))
            check(f"{job} trains and infers on fold {k}",
                  t_new.get("path_to_valid_txt", "").endswith(f"fold_{k}_valid.txt")
                  and i_new.get("path_to_all_benchmarkset_txt", "").endswith(f"fold_{k}_valid.txt"))


# ==================================================================================================
def gate_ge():
    print("\n=== G-E: real-path dataloader, every cell ===")
    import numpy as np
    import torch
    from fastai.torch_core import default_device

    sys.path.insert(0, str(REPO / "src" / "ML_sdfi_fastai2"))
    import utils.utils as sdfi_utils
    import sdfi_dataset

    active = C.tile_names()
    listing: dict[str, set] = {}
    report = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "cells": {}}
    default_device(False)

    for cell, (dts, chans, _m, _s) in CELLS.items():
        print(f"\n  -- {cell}")
        for d in dts:
            if d not in listing:
                listing[d] = set(os.listdir(C.SPLITTED_DIR / d))
            miss = [n for n in active if n not in listing[d]]
            check(f"{cell}: all {len(active)} tiles resolve in splitted/{d}", not miss,
                  f"{len(miss)} missing, first {miss[:2]}")

        cfg_path = TRAIN / f"convnext_upernet_{cell}_fold0.ini"
        prev = os.getcwd()
        os.chdir(REPO)                       # configs carry repo-relative paths
        try:
            cfg = sdfi_utils.load_settings_from_config_file(str(cfg_path))
            n_in = sdfi_utils.n_in_from_settings(cfg)
            dls = sdfi_dataset.get_dataset(cfg)
            it = iter(dls.valid)             # unaugmented, unshuffled: iterate for distinct tiles
            batches = [next(it) for _ in range(N_BATCHES)]
        finally:
            os.chdir(prev)

        want_n = sum(len(c) for c in chans)
        check(f"{cell}: n_in = {want_n}", n_in == want_n, f"got {n_in}")
        xb = torch.cat([b[0] for b in batches], 0)
        check(f"{cell}: batch shape (N, {want_n}, 1000, 1000)",
              tuple(xb.shape[1:]) == (want_n, 1000, 1000), str(tuple(xb.shape)))
        check(f"{cell}: float32 and finite", xb.dtype == torch.float32
              and bool(torch.isfinite(xb).all()))

        means, stds = np.asarray(cfg["means"]), np.asarray(cfg["stds"])
        names = [f"{d}_{j}" for d, cs in zip(dts, chans) for j in cs]
        rows = []
        for c in range(xb.shape[1]):
            v = xb[:, c]
            mn, mx, mu, sd = (float(v.min()), float(v.max()), float(v.mean()), float(v.std()))
            rows.append({"band": names[c], "min": mn, "max": mx, "mean": mu, "std": sd})
            if names[c].startswith("nDSM"):
                # decode: x_norm = (m/255 - mean)/std  ->  metres = (x_norm*std + mean)*255
                lo_m = (mn * stds[c] + means[c]) * C.INT_TO_FLOAT_DIV
                hi_m = (mx * stds[c] + means[c]) * C.INT_TO_FLOAT_DIV
                rows[-1].update({"decoded_min_m": lo_m, "decoded_max_m": hi_m})
                check(f"{cell}: {names[c]} decodes to metres in [0, 300]",
                      -0.01 <= lo_m and hi_m <= 300.0, f"[{lo_m:.3f}, {hi_m:.3f}] m")
            else:
                check(f"{cell}: {names[c]} effective std in [0.5, 1.5]", 0.5 <= sd <= 1.5,
                      f"{sd:.4f}")
                check(f"{cell}: {names[c]} range within [-2.5, 3.0]", -2.5 <= mn and mx <= 3.0,
                      f"[{mn:.3f}, {mx:.3f}]")
            print(f"       {names[c]:<12} min {mn:8.4f} max {mx:8.4f} mean {mu:8.4f} std {sd:7.4f}")
        report["cells"][cell] = {"config": str(cfg_path), "tiles_in_sample": int(len(xb)),
                                 "means": list(map(float, means)),
                                 "stds": list(map(float, stds)), "bands": rows}

    out = C.assert_writes_are_local(C.TABLES / "plan4" / "gate_batch_stats.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")


# ==================================================================================================
def gate_width():
    print("\n=== G-width: ConvNeXt+UPerNet forward at n_in 3 and 4, 1000 x 1000, CPU ===")
    for p in (str(REPO / "src" / "ML_sdfi_fastai2"), str(REPO / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    torch.set_grad_enabled(False)
    import train
    for n_in in (3, 4):
        try:
            m = train.ConvNeXtUPerNetWrapper("convnext_base", num_classes=11, n_in=n_in,
                                             pretrained=True).eval()
            y = m(torch.randn(1, n_in, 1000, 1000))
            check(f"n_in = {n_in} -> logits (1, 11, 1000, 1000)", tuple(y.shape) == (1, 11, 1000, 1000),
                  str(tuple(y.shape)))
        except Exception as e:   # noqa: BLE001 -- report every failure, do not stop at the first
            check(f"n_in = {n_in} forward", False, f"{type(e).__name__}: {e}")


def gate_production(cell):
    """The full-pool config must be the winner's fold-0 config with ONLY the monitor list and the
    job name changed, and the monitor list must be the 25 naive tiles, all of them in the pool."""
    from plan4_build_configs import PROD_KEYS, PROD_INFER_KEYS, NAIVE_VALID
    print(f"\n=== G-prod: full-pool production config for convnext_upernet_{cell} ===")
    job = f"convnext_upernet_{cell}_fullpool"
    new, tpl = kv(TRAIN / f"{job}.ini"), kv(TRAIN / f"convnext_upernet_{cell}_fold0.ini")
    changed = {x for x in set(new) | set(tpl) if new.get(x) != tpl.get(x)}
    check(f"{job} train: exactly {sorted(PROD_KEYS)} differ from fold 0", changed == PROD_KEYS,
          f"differ: {sorted(changed)}")
    check(f"{job} valid list is the naive monitor", new.get("path_to_valid_txt") == NAIVE_VALID)
    inew, itpl = kv(INFER / f"infer_{job}.ini"), kv(INFER / f"infer_convnext_upernet_{cell}_fold0.ini")
    ichanged = {x for x in set(inew) | set(itpl) if inew.get(x) != itpl.get(x)}
    check(f"infer_{job}: exactly {sorted(PROD_INFER_KEYS)} differ", ichanged == PROD_INFER_KEYS,
          f"differ: {sorted(ichanged)}")
    monitor = C.read_tile_list(str(C.NAIVE_VALID_TXT))
    pool = set(C.tile_names())
    check("monitor list holds 25 tiles, all in the pool",
          len(monitor) == 25 and set(monitor) <= pool, f"{len(monitor)} tiles")
    check("production trains on 19,289 tiles", len(pool - set(monitor)) == 19289,
          f"{len(pool - set(monitor))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-width", action="store_true")
    ap.add_argument("--production", metavar="CELL", help="gate only the full-pool config, e.g. rgb")
    args = ap.parse_args()
    if args.production:
        gate_production(args.production)
        print("\n" + ("ALL GATES PASS" if not failures else f"FAILURES ({len(failures)}): {failures}"))
        sys.exit(0 if not failures else 1)
    print("Great Plan 4 gates -- five ConvNeXt arms")
    gate_diff()
    GA.failures.clear()
    GA.gate_gc()
    failures.extend(f"G-C: {f}" for f in GA.failures)
    gate_ge()
    if not args.skip_width:
        gate_width()
    print("\n" + ("ALL GATES PASS" if not failures else f"FAILURES ({len(failures)}): {failures}"))
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
