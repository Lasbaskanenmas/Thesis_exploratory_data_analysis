#!/usr/bin/env python
"""
E1 -- Class structure and separability: how the nine classes are distributed, how they co-occur,
how large their instances are, and how far a context-free per-pixel classifier can get.

WHY THIS EXISTS
    Imbalance is already quantified (class_pixel_audit), but imbalance alone does not say whether
    the task is hard. Two further properties decide that:
      - are the classes SEPARABLE in the input at all, and
      - is the difficulty about rarity, about instance size, or about spectral overlap?
    Finding F5b reports brosten near 0 IoU despite 120 million pixels of support, and hypothesises
    confusion with the visually similar paved classes. That hypothesis is testable BEFORE any model
    is involved, by asking whether brosten, fliser and asfalt are distinguishable from pixel values
    at all.

WHAT IT MEASURES
    1. Class co-occurrence within tiles, and the classes-per-tile distribution.
    2. Instance-size distribution per class, via connected components. This separates "rare because
       there are few instances" from "rare because the instances are small" -- different problems
       needing different interventions.
    3. Per-class spectral signatures across the 10 model channels.
    4. Pairwise Jeffries-Matusita separability between classes, from Gaussian fits.
    5. A LINEAR PROBE: multinomial logistic regression on raw pixel values, no spatial context.

    The probe is the point. It is a context-free floor, and the gap between it and the CNNs is the
    measured value of spatial context. Two design choices keep it honest:
      - it is FOLD-BLOCKED, trained on two folds and tested on the held-out third, rotating, so it
        obeys the same spatial protocol as the matrix. Training and testing on pixels from the same
        tiles would reproduce exactly the leak this thesis exists to criticise.
      - it is PRIOR-MATCHED. Pixels are sampled roughly evenly per class, so importance weights
        restore the real class frequencies during fitting, and the pooled confusion matrix is
        reweighted to true support before metrics. Without that its IoU would not be comparable to
        the models' numbers.

READ-ONLY over the source data. Writes only under exploratory_data_analysis/.

    python eda_separability.py [--blob-scan/--no-blob-scan] [--blob-sample N]
                               [--procs N] [--selftest]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Cap BLAS threading BEFORE numpy/sklearn load. On this Windows box the threaded lbfgs solver
# aborts the process (exit 0xC0000409) part-way through the probe fit; single-threaded BLAS is
# stable and costs nothing at 10 features.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402

SAMPLE_NPZ = C.CACHE / "class_pixel_sample.npz"


# ----------------------------------------------------------------------------------------------
# 2. Instance sizes
# ----------------------------------------------------------------------------------------------
def blob_worker(fname: str):
    import cv2
    try:
        lab = C.read_label(C.LABEL_DIR / fname)
        out = {}
        for cls in C.PREDICTED:
            m = (lab == cls).astype(np.uint8)
            if not m.any():
                continue
            n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            if n <= 1:
                continue
            out[cls] = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
        return ("OK", fname, out)
    except Exception as exc:                       # noqa: BLE001
        return ("FAIL", fname, f"{type(exc).__name__}: {exc}")


def choose_blob_tiles(df, sample_n: int, rng) -> list[str]:
    """Every tile holding a rare class, plus a random sample for the common ones.

    The inventory already records which tile contains which class, so the rare classes can be
    covered exhaustively for the price of scanning a few thousand tiles.
    """
    rare_mask = np.zeros(len(df), dtype=bool)
    for cls in C.RARE + [7]:                       # + betonflade
        rare_mask |= df[C.present_col(C.CODES[cls])].to_numpy() > 0
    rare_tiles = df.loc[rare_mask, "filename"].tolist()
    rest = df.loc[~rare_mask, "filename"].to_numpy()
    take = min(sample_n, len(rest))
    common = rng.choice(rest, take, replace=False).tolist() if take else []
    return sorted(set(rare_tiles) | set(common))


# ----------------------------------------------------------------------------------------------
# 4. Separability
# ----------------------------------------------------------------------------------------------
def jeffries_matusita(mu1, cov1, mu2, cov2) -> float:
    """JM distance from Gaussian fits. 0 = identical, 2 = perfectly separable."""
    cov = 0.5 * (cov1 + cov2)
    d = (mu1 - mu2).reshape(-1, 1)
    eps = 1e-8 * np.eye(len(mu1))
    try:
        term1 = float((0.125 * d.T @ np.linalg.solve(cov + eps, d)).item())
        s1 = np.linalg.slogdet(cov + eps)[1]
        s2 = np.linalg.slogdet(cov1 + eps)[1]
        s3 = np.linalg.slogdet(cov2 + eps)[1]
        bd = term1 + 0.5 * (s1 - 0.5 * (s2 + s3))
    except np.linalg.LinAlgError:
        return float("nan")
    bd = max(bd, 0.0)
    return float(2.0 * (1.0 - np.exp(-bd)))


def lda_fit_predict(Xtr, ytr, Xte, priors: dict):
    """Closed-form linear discriminant analysis, numpy only.

    Deliberately NOT sklearn. On this machine sklearn's LinearDiscriminantAnalysis.fit and the
    logistic lbfgs solver both ABORT THE PROCESS (Windows 0xC0000409) even on a trivial 50,000 x 10
    random array -- a broken LAPACK path in the environment, reproducible outside this project. A
    hard abort cannot be caught in Python, so the probe is implemented directly against the numpy
    routines that are demonstrably sound here (the same ones the Jeffries-Matusita block uses).

    Standard LDA with a pooled within-class covariance:
        delta_c(x) = x . S^-1 mu_c  -  0.5 mu_c . S^-1 mu_c  +  ln prior_c
    Class priors are supplied from the TRUE pixel frequencies, which is how the roughly balanced
    pixel sample is corrected back to the real problem.
    """
    classes = np.unique(ytr)
    d = Xtr.shape[1]
    means = np.stack([Xtr[ytr == c].mean(axis=0) for c in classes])

    # pooled within-class scatter
    S = np.zeros((d, d), dtype=np.float64)
    n = 0
    for i, c in enumerate(classes):
        Z = Xtr[ytr == c].astype(np.float64) - means[i]
        S += Z.T @ Z
        n += len(Z)
    S /= max(n - len(classes), 1)
    S += 1e-6 * np.eye(d)

    W = np.linalg.solve(S, means.T)                     # (d, n_classes)
    logpri = np.array([np.log(max(priors[int(c)], 1e-300)) for c in classes])
    scores = Xte.astype(np.float64) @ W - 0.5 * np.sum(means * W.T, axis=1) + logpri
    return classes[np.argmax(scores, axis=1)]


def load_sample():
    if not SAMPLE_NPZ.is_file():
        sys.exit(f"missing {SAMPLE_NPZ}\nrun eda_channel_stats.py first (module E2)")
    z = np.load(SAMPLE_NPZ, allow_pickle=False)
    bands = [str(b) for b in z["bands"]]
    X, y, fold = [], [], []
    for cls in C.PREDICTED:
        name = C.CODES[cls]
        if name not in z:
            continue
        X.append(z[name])
        y.append(np.full(len(z[name]), cls, dtype=np.int32))
        fold.append(z[f"fold_{name}"] if f"fold_{name}" in z
                    else np.full(len(z[name]), -1, dtype=np.int8))
    return np.concatenate(X), np.concatenate(y), np.concatenate(fold), bands


def main() -> None:
    ap = argparse.ArgumentParser(description="E1 -- class structure and separability")
    ap.add_argument("--blob-scan", action="store_true", default=True)
    ap.add_argument("--no-blob-scan", dest="blob_scan", action="store_false")
    ap.add_argument("--blob-sample", type=int, default=3000)
    ap.add_argument("--procs", type=int, default=min(16, os.cpu_count() or 8))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    C.ensure_out_dirs()
    df = C.load_tile_inventory()
    payload = {}

    # ------------------------------------------------------------------------------------------
    C.banner("1. CO-OCCURRENCE AND CLASSES PER TILE")
    pres = np.stack([df[C.present_col(C.CODES[c])].to_numpy() > 0 for c in C.PREDICTED], axis=1)
    names = [C.CODES[c] for c in C.PREDICTED]
    co = pres.astype(np.int64).T @ pres.astype(np.int64)

    print("tiles containing each pair of classes (diagonal = tiles containing the class)\n")
    print(f"{'':<12}" + "".join(f"{n[:7]:>9}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:<12}" + "".join(f"{co[i, j]:>9,}" for j in range(len(names))))

    print("\nconditional: given the row class is present, share of those tiles also holding col\n")
    print(f"{'':<12}" + "".join(f"{n[:7]:>9}" for n in names))
    for i, n in enumerate(names):
        row = "".join(f"{100 * co[i, j] / co[i, i]:>8.1f}%" if co[i, i] else f"{'-':>9}"
                      for j in range(len(names)))
        print(f"{n:<12}{row}")

    vc = df["n_predicted_classes_present"].value_counts().sort_index()
    print(f"\nclasses present per tile:")
    for k, v in vc.items():
        print(f"  {k} classes : {v:6,} tiles ({100 * v / len(df):5.2f}%)")
    print(f"\n{100 * vc.get(0, 0) / len(df) + 100 * vc.get(1, 0) / len(df):.1f}% of tiles hold at")
    print("most ONE predicted class. The task is mostly locally homogeneous, which is why a")
    print("tile-level oracle scores so well (module E5, bound B3) and why rare classes cannot be")
    print("learned from context within a tile -- they simply are not there to see.")

    payload["cooccurrence"] = {"classes": names, "matrix": co.tolist()}
    payload["classes_per_tile"] = {str(k): int(v) for k, v in vc.items()}

    cocsv = C.assert_writes_are_local(C.TABLES / "class_cooccurrence.csv")
    with open(cocsv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([""] + names)
        for i, n in enumerate(names):
            w.writerow([n] + co[i].tolist())
    print(f"\nwrote {cocsv}")

    # ------------------------------------------------------------------------------------------
    C.banner("2. INSTANCE SIZE  (connected components)")
    if not args.blob_scan:
        print("skipped (--no-blob-scan)")
    else:
        rng = np.random.default_rng(11)
        tiles = choose_blob_tiles(df, args.blob_sample, rng)
        print(f"scanning {len(tiles):,} tiles: every tile holding a rare class, plus a random")
        print(f"sample of {args.blob_sample:,} others. Rare classes are therefore covered in full.\n")
        blobs = {c: [] for c in C.PREDICTED}
        fails = []
        from multiprocessing import Pool
        with Pool(processes=args.procs) as pool:
            for i, res in enumerate(pool.imap_unordered(blob_worker, tiles, chunksize=8), 1):
                if res[0] != "OK":
                    fails.append(res[1])
                    continue
                for cls, areas in res[2].items():
                    blobs[cls].append(areas)
                if i % 1000 == 0:
                    print(f"  {i:,} / {len(tiles):,}")

        print(f"\n{'class':<12}{'instances':>12}{'median px':>11}{'p90':>10}{'max':>12}"
              f"{'median m2':>12}")
        rows = []
        for cls in C.PREDICTED:
            if not blobs[cls]:
                continue
            a = np.concatenate(blobs[cls])
            med = float(np.median(a))
            rec = {"class": C.CODES[cls], "instances": int(len(a)), "median_px": med,
                   "p90_px": float(np.percentile(a, 90)), "max_px": int(a.max()),
                   "median_m2": med * (C.GSD_M ** 2)}
            rows.append(rec)
            print(f"{C.CODES[cls]:<12}{len(a):>12,}{med:>11,.0f}"
                  f"{np.percentile(a, 90):>10,.0f}{a.max():>12,}{med * C.GSD_M ** 2:>12.2f}")
        print("\nAn instance here is a connected run of one class inside one tile, so objects")
        print("crossing a tile edge are counted once per tile. Read it as a size distribution,")
        print("not an object census.")
        payload["instance_sizes"] = rows
        bcsv = C.assert_writes_are_local(C.TABLES / "instance_sizes.csv")
        with open(bcsv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {bcsv}")

    # ------------------------------------------------------------------------------------------
    C.banner("3-5. SPECTRAL SIGNATURES, SEPARABILITY, AND THE LINEAR PROBE")
    X, y, fold, bands = load_sample()
    print(f"sampled pixels : {len(X):,} across {len(set(y.tolist()))} classes, "
          f"{len(bands)} channels")
    print(f"channels       : {bands}")
    have_folds = bool((fold >= 0).all())
    print(f"fold labels    : {'present' if have_folds else 'MISSING -- probe would be in-sample'}")

    present = [c for c in C.PREDICTED if (y == c).sum() >= 50]
    pnames = [C.CODES[c] for c in present]

    print(f"\nper-class mean value by channel")
    print(f"{'class':<12}" + "".join(f"{b[:9]:>10}" for b in bands))
    mus, covs = {}, {}
    for c in present:
        Xi = X[y == c]
        mus[c] = Xi.mean(0)
        covs[c] = np.cov(Xi, rowvar=False) + 1e-6 * np.eye(Xi.shape[1])
        print(f"{C.CODES[c]:<12}" + "".join(f"{v:>10.2f}" for v in mus[c]))

    print(f"\nJeffries-Matusita separability (0 identical, 2 perfectly separable)")
    print(f"{'':<12}" + "".join(f"{n[:9]:>10}" for n in pnames))
    jm = np.zeros((len(present), len(present)))
    for i, a in enumerate(present):
        for j, b in enumerate(present):
            jm[i, j] = 0.0 if i == j else jeffries_matusita(mus[a], covs[a], mus[b], covs[b])
        print(f"{C.CODES[a]:<12}" + "".join(f"{jm[i, j]:>10.3f}" for j in range(len(present))))

    pairs = [(pnames[i], pnames[j], jm[i, j])
             for i in range(len(present)) for j in range(i + 1, len(present))]
    pairs.sort(key=lambda t: t[2])
    print(f"\nleast separable pairs:")
    for a, b, v in pairs[:6]:
        print(f"  {a:<12} vs {b:<12} JM = {v:.3f}")
    payload["jm_matrix"] = {"classes": pnames, "matrix": jm.tolist()}

    jcsv = C.assert_writes_are_local(C.TABLES / "class_separability_jm.csv")
    with open(jcsv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([""] + pnames)
        for i, n in enumerate(pnames):
            w.writerow([n] + [f"{v:.6f}" for v in jm[i]])
    print(f"\nwrote {jcsv}")

    # ---- the probe ---------------------------------------------------------------------------
    C.banner("LINEAR PROBE  (per-pixel, no spatial context, fold-blocked, prior-matched)")
    audit = C.load_class_pixel_audit()
    true_support = {c: audit[C.CODES[c]]["pixel_count"] for c in present}
    tot_support = sum(true_support.values())
    sampled = {c: int((y == c).sum()) for c in present}
    # importance weight restores the real class prior from the roughly balanced sample
    w_of = {c: (true_support[c] / tot_support) / (sampled[c] / len(X)) for c in present}

    print("class priors restored to the true pixel frequencies:")
    for c in present:
        print(f"  {C.CODES[c]:<12} sampled {sampled[c]:>8,}  true share "
              f"{100 * true_support[c] / tot_support:>7.3f}%  (sampling bias {w_of[c]:>9.4f}x)")

    cm = np.zeros((C.NCLASS, C.NCLASS), dtype=np.float64)
    folds = sorted(set(fold.tolist())) if have_folds else [-1]
    if have_folds:
        for f in folds:
            te = fold == f
            tr = ~te
            if te.sum() == 0 or tr.sum() == 0:
                continue
            mu = X[tr].mean(axis=0)
            sd = X[tr].std(axis=0) + 1e-9
            Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
            pred = lda_fit_predict(Xtr, y[tr], Xte,
                                   {c: true_support[c] / tot_support for c in present})
            for a, b in zip(y[te], pred):
                cm[a, b] += 1.0
            print(f"  fold {f}: trained on {tr.sum():,} px, tested on {te.sum():,} px")
    else:
        print("  no fold labels -- skipping (re-run E2 to regenerate the sample with folds)")

    if cm.sum():
        # Rows are class-balanced by construction. Rescale each row to the class's true pixel
        # support so the matrix describes the real distribution and the IoU is comparable to the
        # models' pooled numbers.
        cm_w = np.zeros_like(cm)
        for c in present:
            r = cm[c].sum()
            if r > 0:
                cm_w[c] = cm[c] / r * true_support[c]
        m = C.metrics_from_confusion(np.rint(cm_w).astype(np.int64), C.CODES,
                                     ignore_index=C.IGNORE_INDEX,
                                     report_only=(C.UNKNOWN2_INDEX,), split_tag="linear_probe")
        print(f"\nprior-reweighted, spatially blocked linear probe:")
        print(f"  Macro-IoU {m['macro_iou']:.4f}   macro-F1 {m['macro_f1']:.4f}   "
              f"overall acc {m['overall_accuracy']:.4f}")
        print(f"\n{'class':<12}{'probe IoU':>11}{'probe recall':>14}")
        for c in present:
            pc = m["per_class"][C.CODES[c]]
            iou = "n/a" if pc["iou"] is None else f"{pc['iou']:.4f}"
            rec = "n/a" if pc["recall"] is None else f"{pc['recall']:.4f}"
            print(f"{C.CODES[c]:<12}{iou:>11}{rec:>14}")
        print("\nThis is what pixel VALUES alone support, with no spatial context whatsoever.")
        print("The distance from here to the CNNs is the measured contribution of context, and")
        print("any class the probe already fails on is failing for reasons no architecture change")
        print("will fix.")
        payload["linear_probe"] = {k: v for k, v in m.items()
                                   if k != "confusion_matrix_rows_label_cols_pred"}
        payload["linear_probe"]["balanced_confusion"] = cm.tolist()

    print(f"\nwrote {C.write_json(payload, C.TABLES / 'separability.json')}")


def selftest() -> None:
    C.banner("E1 selftest (synthetic, no files touched)")

    # Identical distributions -> JM 0; far-apart ones -> JM near 2.
    mu = np.zeros(4); cov = np.eye(4)
    assert abs(jeffries_matusita(mu, cov, mu, cov)) < 1e-9
    far = jeffries_matusita(mu, cov, mu + 50.0, cov)
    assert far > 1.99, far
    print(f"JM distance      : identical 0.000, far apart {far:.3f}   OK")

    mid = jeffries_matusita(mu, cov, mu + 1.5, cov)
    assert 0.1 < mid < 1.99, mid
    print(f"  partially overlapping pair gives {mid:.3f}, between the extremes   OK")

    # Importance weights must restore the true prior from a balanced sample.
    sampled = {1: 1000, 4: 1000}
    true_sup = {1: 100, 4: 900}
    tot = sum(true_sup.values()); n = sum(sampled.values())
    w = {c: (true_sup[c] / tot) / (sampled[c] / n) for c in sampled}
    eff = {c: w[c] * sampled[c] for c in sampled}
    assert abs(eff[4] / eff[1] - 9.0) < 1e-9, eff
    print("prior matching   : balanced 1:1 sample reweighted to the true 1:9   OK")

    # The hand-rolled LDA must recover two well-separated Gaussians almost perfectly, and must
    # shift its decision boundary when the priors change.
    rng = np.random.default_rng(5)
    Xa = rng.normal(loc=[0, 0], scale=0.6, size=(4000, 2))
    Xb = rng.normal(loc=[3, 3], scale=0.6, size=(4000, 2))
    Xtr = np.vstack([Xa, Xb]); ytr = np.array([1] * 4000 + [4] * 4000)
    pred = lda_fit_predict(Xtr, ytr, Xtr, {1: 0.5, 4: 0.5})
    acc = (pred == ytr).mean()
    assert acc > 0.99, acc
    print(f"numpy LDA        : separated Gaussians at accuracy {acc:.4f}   OK")

    # An overwhelming prior on one class must pull points that sit just inside the OTHER class's
    # territory back across the boundary. Picked at 1.9 rather than the 1.5 midpoint, where a
    # balanced fit ties and argmax decides arbitrarily.
    amb = np.full((200, 2), 1.9)
    bal = lda_fit_predict(Xtr, ytr, amb, {1: 0.5, 4: 0.5})
    skew = lda_fit_predict(Xtr, ytr, amb, {1: 0.999, 4: 0.001})
    assert (bal == 4).all(), f"balanced priors should call this class 4, got {bal[0]}"
    assert (skew == 1).all(), f"a dominant prior should claim it, got {skew[0]}"
    print("  priors shift the boundary as they must                              OK")

    # Row reweighting to true support must preserve the conditional confusion structure.
    cm = np.zeros((C.NCLASS, C.NCLASS))
    cm[1] = [0, 60, 0, 0, 40, 0, 0, 0, 0, 0, 0]
    out = cm[1] / cm[1].sum() * 100
    assert abs(out[1] - 60.0) < 1e-9 and abs(out[4] - 40.0) < 1e-9
    print("row reweighting  : 60/40 split preserved after rescaling to support   OK")

    print("\nSELFTEST PASSED")


if __name__ == "__main__":
    main()
