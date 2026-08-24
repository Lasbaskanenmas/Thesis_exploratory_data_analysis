#!/usr/bin/env python
"""
Gate G-A: confirm all four models instantiate and forward one batch at n_in=4, at the REAL tile
geometry of 1000x1000.

Width 4 has never been exercised by this project -- 3, 6 and 10 have. The risk is the patch-embed /
stem adapters in the three transformer + UPerNet wrappers, which rebuild for n_in != 3 and would
otherwise throw on the first forward pass part-way into a multi-day run.

Geometry matters here. `check_n_in_10.py` used a synthetic 256x256 shape, which is where the
incorrect "256x256 tiles" line in the 2026-07-28 handoff came from. The data is 1000x1000 at 0.1 m
GSD, one hectare per tile, and that is what this gate uses.

CPU only. No training, no GPU job.

    python check_n_in_4.py
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

N_IN, N_CLASS, HW = 4, 11, 1000
x = torch.randn(1, N_IN, HW, HW)
results = []


def run(name, build):
    try:
        model = build().eval()
        logits = model(x)
        ok = tuple(logits.shape) == (1, N_CLASS, HW, HW)
        results.append((name, "PASS" if ok else "BAD SHAPE", tuple(logits.shape)))
    except Exception as e:
        results.append((name, "FAIL", f"{type(e).__name__}: {e}"))


run("SegFormer-b1 (segformer-b1)",
    lambda: train.SegFormerWrapper("nvidia/segformer-b1-finetuned-ade-512-512",
                                   num_classes=N_CLASS, n_in=N_IN, pretrained=True))
run("ConvNeXt-base+UPerNet (convnext_base_upernet)",
    lambda: train.ConvNeXtUPerNetWrapper("convnext_base", num_classes=N_CLASS, n_in=N_IN,
                                         pretrained=True))
run("Swin-base+UPerNet (swin-base-upernet)",
    lambda: train.SwinUPerNetWrapper("openmmlab/upernet-swin-base", num_classes=N_CLASS, n_in=N_IN,
                                     pretrained=True))


def build_resnet():
    from torchvision.models import resnet34
    from fastai.vision.learner import create_unet_model
    return create_unet_model(resnet34, N_CLASS, (HW, HW), pretrained=True, n_in=N_IN)


run("resnet34+UNet (resnet34)", build_resnet)

print(f"\n=== G-A: n_in=4 single forward (1x{N_IN}x{HW}x{HW} -> expect 1x{N_CLASS}x{HW}x{HW}) ===")
all_ok = True
for name, status, detail in results:
    print(f"  [{status:9}] {name}\n              {detail}")
    if status != "PASS":
        all_ok = False
print("\nALL PASS" if all_ok else "\nSOME CHECKS FAILED (see above)")
sys.exit(0 if all_ok else 1)
