#!/usr/bin/env python
"""
Great Plan 4, Step 1 -- build the configs for the five new ConvNeXt+UPerNet cells, and later the
full-pool production config.

WHAT IT DOES
    Every new config is a COPY of a frozen template in which only declared keys are replaced, line
    by line. Everything else -- comments, protocol, weights, paths, precision flags -- stays
    byte-identical to the template. That is the same discipline the `ortorgb` and `rgb_ndsm` arms
    followed (their configs differ from the frozen `convnext_upernet_rgb_fold<k>.ini` only in the
    channel block and `job_name`, verified by diff 2026-10-06).

    training templates   configs/matrix_configs/train/convnext_upernet_rgb_fold<k>.ini
    inference templates  configs/matrix_configs/infer/infer_convnext_upernet_ortorgb_fold<k>.ini

    train keys replaced  datatypes, channels, means, stds, job_name
    infer keys replaced  datatypes, channels, means, stds, model_to_load, output_folder, job_name

    --production CELL builds the full-pool production pair from that cell's fold-0 configs, replacing
    only path_to_valid_txt (the 25-tile naive valid.txt, a loss monitor only: SaveModelCallback runs
    with every_epoch=True and learn.save writes the last epoch) and job_name. The inference template
    is for KDS's recipe and is not run.

SAFETY
    Writes only into the two config folders, only NEW files. Refuses to overwrite anything.

    python plan4_build_configs.py               # dry run: print the plan, write nothing
    python plan4_build_configs.py --write       # the 15 + 15 CV configs
    python plan4_build_configs.py --production rgb --write
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(r"C:\thesis\ML_sdfi_fastai2")
TRAIN = REPO / "configs" / "matrix_configs" / "train"
INFER = REPO / "configs" / "matrix_configs" / "infer"

IN3_M = [0.485, 0.456, 0.406]
IN3_S = [0.229, 0.224, 0.225]
# copied verbatim from convnext_upernet_rgb_ndsm_fold0.ini (measured on the clamped nDSM rasters)
NDSM_M = 0.010076483036168348
NDSM_S = 0.022261953346304587

# The five cells, locked in 2026-10-06_plan4_declarations.md. Order = queue order.
CELLS = {
    "ortorgb_ndsm": (["OrtoRGB", "nDSM"], [[0, 1, 2], [0]], IN3_M + [NDSM_M], IN3_S + [NDSM_S]),
    "cir":          (["cir"],            [[0, 1, 2]],      IN3_M,            IN3_S),
    "ortocir":      (["OrtoCIR"],        [[0, 1, 2]],      IN3_M,            IN3_S),
    "ortocir_ndsm": (["OrtoCIR", "nDSM"], [[0, 1, 2], [0]], IN3_M + [NDSM_M], IN3_S + [NDSM_S]),
    "cir_ndsm":     (["cir", "nDSM"],    [[0, 1, 2], [0]], IN3_M + [NDSM_M], IN3_S + [NDSM_S]),
}

TRAIN_KEYS = {"datatypes", "channels", "means", "stds", "job_name"}
INFER_KEYS = {"datatypes", "channels", "means", "stds", "model_to_load", "output_folder", "job_name"}
PROD_KEYS = {"path_to_valid_txt", "job_name"}
NAIVE_VALID = "../multi_channel_dataset_creation/example_dataset/data/valid.txt"


def fmt_list(v):
    """Python-literal list exactly as the existing configs write it."""
    if isinstance(v, list) and v and isinstance(v[0], str):
        return "[" + ", ".join(f'"{x}"' for x in v) + "]"
    return repr(v).replace("'", '"')


def replace_keys(text: str, values: dict) -> tuple[str, set]:
    """Replace `key = ...` lines for the given keys. Returns new text and the keys actually hit."""
    out, hit = [], set()
    for line in text.splitlines(keepends=True):
        m = re.match(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(.*?)(\r?\n)?$", line)
        if m and m.group(2) in values and not line.lstrip().startswith("#"):
            key = m.group(2)
            if key in hit:
                raise RuntimeError(f"key {key} appears twice in the template")
            eol = m.group(5) or ""
            out.append(f"{m.group(1)}{key}{m.group(3)}{values[key]}{eol}")
            hit.add(key)
        else:
            out.append(line)
    return "".join(out), hit


def build(template: Path, dest: Path, values: dict, expected: set, plan: list):
    if not template.is_file():
        sys.exit(f"missing template {template}")
    if dest.exists():
        sys.exit(f"refusing to overwrite existing {dest}")
    with open(template, encoding="utf-8", newline="") as fh:   # newline="" keeps CRLF intact
        text = fh.read()
    new, hit = replace_keys(text, values)
    if hit != expected:
        sys.exit(f"{template.name}: replaced {sorted(hit)}, expected exactly {sorted(expected)}")
    plan.append((template, dest, new))


def cv_plan():
    plan = []
    for cell, (dts, chans, means, stds) in CELLS.items():
        for k in range(3):
            job = f"convnext_upernet_{cell}_fold{k}"
            block = {"datatypes": fmt_list(dts), "channels": fmt_list(chans),
                     "means": fmt_list(means), "stds": fmt_list(stds)}
            build(TRAIN / f"convnext_upernet_rgb_fold{k}.ini", TRAIN / f"{job}.ini",
                  {**block, "job_name": f'"{job}"'}, TRAIN_KEYS, plan)
            mdir = f"../logs_and_models/spatial_matrix/convnext_upernet/{job}"
            build(INFER / f"infer_convnext_upernet_ortorgb_fold{k}.ini", INFER / f"infer_{job}.ini",
                  {**block,
                   "model_to_load": f"{mdir}/models/{job}.pth",
                   "output_folder": f"{mdir}/models/example_dataset",
                   "job_name": f'"infer_{job}"'}, INFER_KEYS, plan)
    return plan


PROD_INFER_KEYS = {"model_to_load", "path_to_all_benchmarkset_txt", "output_folder", "job_name"}


def production_plan(cell):
    plan = []
    job = f"convnext_upernet_{cell}_fullpool"
    build(TRAIN / f"convnext_upernet_{cell}_fold0.ini", TRAIN / f"{job}.ini",
          {"path_to_valid_txt": NAIVE_VALID, "job_name": f'"{job}"'}, PROD_KEYS, plan)
    # Inference template for KDS's recipe. NOT run (Plan 4 Q8). The tile list and output folder are
    # deliberate placeholders: a run against them fails fast on the missing file, so this config
    # can never silently produce in-sample predictions over the training pool.
    build(INFER / f"infer_convnext_upernet_{cell}_fold0.ini", INFER / f"infer_{job}.ini",
          {"model_to_load": f"../logs_and_models/spatial_matrix/convnext_upernet/{job}/models/{job}.pth",
           "path_to_all_benchmarkset_txt": "REPLACE_WITH_YOUR_TILE_LIST.txt",
           "output_folder": "REPLACE_WITH_YOUR_OUTPUT_FOLDER",
           "job_name": f'"infer_{job}"'}, PROD_INFER_KEYS, plan)
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--production", default=None, help="cell suffix, e.g. rgb or rgb_ndsm")
    args = ap.parse_args()

    plan = production_plan(args.production) if args.production else cv_plan()
    for tpl, dest, _ in plan:
        print(f"{'WRITE' if args.write else 'would write'}  {dest.name:<52} from {tpl.name}")
    if not args.write:
        print(f"\ndry run: {len(plan)} configs, nothing written. Re-run with --write.")
        return
    for _, dest, text in plan:
        with open(dest, "x", encoding="utf-8", newline="") as fh:  # "x": fail if it exists
            fh.write(text)
    print(f"\nwrote {len(plan)} configs")


if __name__ == "__main__":
    main()
