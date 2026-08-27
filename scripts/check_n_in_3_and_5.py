#!/usr/bin/env python
"""
Gate G-A for the 2026-08-24 arms G and A: confirm ConvNeXt-base+UPerNet instantiates and forwards
one batch at n_in=3 (arm G, `ortorgb`) and n_in=5 (arm A, `rgb_dsm_dtm_corrected`), at the REAL tile
geometry of 1000x1000.

Width 5 has never been exercised by this project -- 3, 4, 6 and 10 have. The risk is the same one
`check_n_in_4.py` was written for: the UPerNet `_adapt_input_channels` stem path rebuilds for
n_in != 3 and would otherwise throw on the first forward pass part-way into a multi-day run. Width 3
is included as the positive control -- it is the untouched pretrained stem, so a failure there means
the gate itself is wrong rather than the arm.

Geometry matters: the data is 1000x1000 at 0.1 m GSD, one hectare per tile. `check_n_in_10.py` used
a synthetic 256x256 shape, which is where the incorrect "256x256 tiles" line in the 2026-07-28
handoff came from.

Only ConvNeXt is exercised, because both arms are ConvNeXt+UPerNet cells. CPU only. No training,
no GPU job.

    python check_n_in_3_and_5.py
"""
import pathlib
import sys

REPO_SRC = pathlib.Path(r"c:\thesis\ML_sdfi_fastai2\src")
for p in (str(REPO_SRC / "ML_sdfi_fastai2"), str(REPO_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
torch.set_grad_enabled(False)

import train  # noqa: E402  (the trainer module holds the wrapper classes)

N_CLASS, HW = 11, 1000
CASES = [
    (3, "arm G  ortorgb                (positive control: untouched 3-band stem)"),
    (5, "arm A  rgb_dsm_dtm_corrected  (NEW WIDTH -- never exercised before)"),
]

results = []
for n_in, label in CASES:
    x = torch.randn(1, n_in, HW, HW)
    try:
        model = train.ConvNeXtUPerNetWrapper("convnext_base", num_classes=N_CLASS, n_in=n_in,
                                             pretrained=True).eval()
        logits = model(x)
        ok = tuple(logits.shape) == (1, N_CLASS, HW, HW)
        results.append((n_in, label, "PASS" if ok else "BAD SHAPE", tuple(logits.shape)))
    except Exception as e:                                    # noqa: BLE001 - report, do not raise
        results.append((n_in, label, "FAIL", f"{type(e).__name__}: {e}"))

print(f"\n=== G-A (2026-08-24 arms): ConvNeXt-base+UPerNet forward at {HW}x{HW} ===")
all_ok = True
for n_in, label, status, detail in results:
    print(f"  [{status:9}] n_in={n_in}  1x{n_in}x{HW}x{HW} -> expect 1x{N_CLASS}x{HW}x{HW}")
    print(f"              {label}")
    print(f"              got {detail}")
    if status != "PASS":
        all_ok = False
print("\nALL PASS" if all_ok else "\nSOME CHECKS FAILED (see above)")
sys.exit(0 if all_ok else 1)
