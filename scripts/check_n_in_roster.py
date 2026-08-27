#!/usr/bin/env python
"""
Gate G-A for the Option 2 full-roster arms (Great Plan 3.1 s11.5, addendum task A2).

Confirms that resnet34+UNet, Swin+UPerNet and SegFormer-B1 each instantiate and forward one batch
at n_in = 3 (the `ortorgb` cells) and n_in = 5 (the `rgb_dsm_dtm_corrected` cells), at the REAL tile
geometry of 1000x1000.

WIDTH 5 IS THE KNOWN GAP. `check_n_in_4.py` covered all four models at width 4 and
`check_n_in_3_and_5.py` covered ConvNeXt at widths 3 and 5, so width 5 has only ever been forwarded
through the ConvNeXt wrapper. The risk is the same one those gates were written for: the stem /
patch-embed adapters rebuild for n_in != 3 and would otherwise throw on the first forward pass
part-way into a multi-day run. Width 3 is the positive control -- it is the untouched pretrained
stem, so a failure there means the gate is wrong rather than the arm.

**GPU DISCIPLINE.** The A100 is busy with the training queue. CUDA is hidden from torch before it is
imported and the gate asserts torch cannot see a device, so this cannot contend for the GPU even by
accident. CPU only, no training, no CUDA context.

    python check_n_in_roster.py
"""
import os
import pathlib
import sys

# Must precede the torch import: hiding the device afterwards does not work. Use "-1", not "": on
# Windows an empty environment variable is equivalent to an unset one, so "" leaves the GPU visible.
# The assertion below caught exactly that, which is why it is an assertion and not a comment.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

REPO_SRC = pathlib.Path(r"c:\thesis\ML_sdfi_fastai2\src")
for p in (str(REPO_SRC / "ML_sdfi_fastai2"), str(REPO_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
torch.set_grad_enabled(False)

if torch.cuda.is_available():
    sys.exit("REFUSING TO RUN: torch can still see a CUDA device. The training queue owns the GPU.")
print(f"torch {torch.__version__}  cuda visible: {torch.cuda.is_available()}  (CPU-only gate)")

import train  # noqa: E402

N_CLASS, HW = 11, 1000
WIDTHS = [(3, "ortorgb", "positive control: untouched 3-band stem"),
          (5, "rgb_dsm_dtm_corrected", "NEW WIDTH for this wrapper")]


def build_resnet(n_in):
    from torchvision.models import resnet34
    from fastai.vision.learner import create_unet_model
    return create_unet_model(resnet34, N_CLASS, (HW, HW), pretrained=True, n_in=n_in)


BUILDERS = [
    ("resnet34+UNet          (segformer_train.py)", build_resnet),
    ("Swin-base+UPerNet      (train.py)",
     lambda n_in: train.SwinUPerNetWrapper("openmmlab/upernet-swin-base", num_classes=N_CLASS,
                                           n_in=n_in, pretrained=True)),
    ("SegFormer-b1           (segformer_train.py)",
     lambda n_in: train.SegFormerWrapper("nvidia/segformer-b1-finetuned-ade-512-512",
                                         num_classes=N_CLASS, n_in=n_in, pretrained=True)),
]

results = []
for label, build in BUILDERS:
    for n_in, chan, why in WIDTHS:
        try:
            model = build(n_in).eval()
            logits = model(torch.randn(1, n_in, HW, HW))
            ok = tuple(logits.shape) == (1, N_CLASS, HW, HW)
            results.append((label, n_in, chan, why, "PASS" if ok else "BAD SHAPE",
                            tuple(logits.shape)))
            del model, logits
        except Exception as e:                                # noqa: BLE001 - report, do not raise
            results.append((label, n_in, chan, why, "FAIL", f"{type(e).__name__}: {e}"))

print(f"\n=== G-A (Option 2 roster): forward at {HW}x{HW} -> expect 1x{N_CLASS}x{HW}x{HW} ===")
all_ok = True
for label, n_in, chan, why, status, detail in results:
    print(f"  [{status:9}] {label}  n_in={n_in}  ({chan})")
    print(f"              {why}; got {detail}")
    if status != "PASS":
        all_ok = False
print("\nALL PASS" if all_ok else "\nSOME CHECKS FAILED (see above)")
sys.exit(0 if all_ok else 1)
