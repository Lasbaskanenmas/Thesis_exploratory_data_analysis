#!/usr/bin/env python
"""
Backup verification for the GitHub-backed repos, without git.

WHY THIS SHAPE. `git` is not installed on this machine -- the PATH carries
`C:\\Program Files\\Git\\cmd` but that directory does not exist -- and neither dulwich nor pygit2
is in the environment. So this reads `.git` directly:

  - `.git/index` (v2/v3) is parsed for the list of TRACKED paths and each entry's blob SHA-1.
  - each required file's current bytes are hashed as a git blob and compared to its index entry,
    which distinguishes "tracked and unmodified" from "tracked but changed since it was staged".
  - `refs/heads/<branch>` is compared with `refs/remotes/origin/<branch>` to establish whether the
    local branch has actually been pushed.

WHAT THIS CAN AND CANNOT PROVE, stated because it matters for a backup claim:
  it proves a path is tracked, that its on-disk content matches what is staged, and that the local
  branch tip equals the known remote tip. It does NOT walk the pushed commit's tree, so if work was
  committed and then the ref moved in a way this cannot see, that is outside what is checked. Any
  required file that is untracked, modified-since-staged, or sitting under an ignored path is
  listed explicitly rather than assumed safe.

Read-only. Writes one JSON report under exploratory_data_analysis/.

    python verify_git_backup.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402

REPOS = {
    "exploratory_data_analysis": Path(r"c:\thesis\exploratory_data_analysis"),
    "logs_and_models": Path(r"c:\thesis\logs_and_models"),
    "ML_sdfi_fastai2": Path(r"c:\thesis\ML_sdfi_fastai2"),
}

# What the author asked to be confirmed present in the backup.
REQUIRED = {
    "exploratory_data_analysis": [
        "results/tables/part_b/wilcoxon_by_family.csv",
        "results/tables/part_b/pooled_macro_iou_route_bootstrap.csv",
        "results/tables/part_b/descriptive_contrast_paired_bootstrap.csv",
        "results/tables/part_b/run_provenance.json",
        "results/tables/part_b/mcnemar_family_pairs.csv",
        "results/tables/part_b/mcnemar_provenance.json",
        "results/findings/2026-08-25_part_b_route_level_statistics.md",
        "results/findings/2026-08-24_arms_G_A_build_record.md",
        "results/findings/2026-08-24_E4_label_quality.md",
        "results/findings/2026-08-24_high_ndsm_tiles.md",
        "results/tables/boundary_iou.csv",
        "results/tables/git_backup_verification.json",
        "results/tables/metric_breadth_by_cell.csv",
        "results/tables/metric_breadth_per_class.csv",
        "results/tables/metric_breadth_binary_collapse.csv",
        "results/tables/label_quality.json",
        "results/tables/boundary_error_profile.csv",
        "results/tables/learning_curve/lc100_per_class.json",
        "results/tables/learning_curve/learning_curve_per_class.csv",
        "results/tables/cross_cell_summary.csv",
        "results/tables/route_cell_metrics.csv",
        "results/tables/route_cell_cms.npz",
        "2026-08-25_pre_declarations.md",
    ],
    # Every scored cell's pooled JSON, discovered rather than listed, so a cell that lands after
    # this file was last edited cannot quietly escape the backup check.
    "logs_and_models": None,
}

MATRIX_ROOT = Path(r"c:\thesis\logs_and_models\spatial_matrix")


def discovered_oof_requirements():
    """Relative paths of every pooled_oof_metrics.json currently on disk."""
    out = []
    for p in sorted(MATRIX_ROOT.glob("*/oof_*/pooled_oof_metrics.json")):
        if p.parents[1].name.startswith("_"):
            continue
        out.append(p.relative_to(MATRIX_ROOT.parent).as_posix())
    return out


def read_index(repo: Path):
    """-> {path: (sha1_hex, size)} for a v2/v3 git index."""
    p = repo / ".git" / "index"
    if not p.is_file():
        return None
    b = p.read_bytes()
    if b[:4] != b"DIRC":
        raise ValueError(f"{p}: not a git index")
    ver, n = struct.unpack(">II", b[4:12])
    if ver not in (2, 3, 4):
        raise ValueError(f"{p}: unsupported index version {ver}")
    if ver == 4:
        raise ValueError(f"{p}: index v4 uses prefix compression; not parsed here")
    out, off = {}, 12
    for _ in range(n):
        start = off
        size = struct.unpack(">I", b[off + 36:off + 40])[0]
        sha = b[off + 40:off + 60].hex()
        flags = struct.unpack(">H", b[off + 60:off + 62])[0]
        off += 62
        namelen = flags & 0xFFF
        if flags & 0x4000:                      # extended flag -> 2 more bytes
            off += 2
        if namelen < 0xFFF:
            name = b[off:off + namelen].decode("utf-8", "replace")
            off += namelen
        else:
            end = b.index(b"\x00", off)
            name = b[off:end].decode("utf-8", "replace")
            off = end
        off += 1                                 # at least one NUL
        off = start + ((off - start + 7) // 8) * 8   # 8-byte alignment
        out[name] = (sha, size)
    return out


def _blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def blob_sha1_candidates(path: Path):
    """Every blob SHA-1 this working-tree file could legitimately have in the index.

    On Windows with `core.autocrlf`, git normalises CRLF to LF when it writes the blob, so the
    committed object is NOT a hash of the bytes on disk. The first version of this module hashed the
    raw bytes only and therefore reported every CRLF text file as MODIFIED -- 28 frozen
    `pooled_oof_metrics.json` files that had not been touched since July, among others. Checking the
    LF-normalised form as well is what git itself does when deciding whether a file is clean.

    Binary files are unaffected: git does not normalise them, so the raw hash is the one that
    matches and the extra candidate simply never fires.
    """
    data = path.read_bytes()
    out = [_blob(data)]
    if b"\r\n" in data:
        out.append(_blob(data.replace(b"\r\n", b"\n")))
    return out


def refs(repo: Path):
    g = repo / ".git"
    head = (g / "HEAD").read_text().strip()
    branch = head[len("ref: refs/heads/"):] if head.startswith("ref: refs/heads/") else "(detached)"
    packed = {}
    pr = g / "packed-refs"
    if pr.is_file():
        for line in pr.read_text().splitlines():
            if line and not line.startswith(("#", "^")):
                sha, _, name = line.partition(" ")
                packed[name.strip()] = sha.strip()

    def resolve(name):
        f = g / name
        if f.is_file():
            return f.read_text().strip()
        return packed.get(name)

    local = resolve(f"refs/heads/{branch}") if branch != "(detached)" else head
    remote = resolve(f"refs/remotes/origin/{branch}")
    url = None
    cfg = g / "config"
    if cfg.is_file():
        lines = cfg.read_text(errors="replace").splitlines()
        for i, ln in enumerate(lines):
            if ln.strip() == '[remote "origin"]':
                for nxt in lines[i + 1:]:
                    if nxt.startswith("["):
                        break
                    if nxt.strip().startswith("url"):
                        url = nxt.split("=", 1)[1].strip()
                        break
    return {"branch": branch, "local": local, "remote_origin": remote, "origin_url": url,
            "in_sync": bool(local and remote and local == remote)}


def main():
    report = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "repos": {}}
    print("Backup verification -- reading .git directly (git is not installed on this machine)\n")

    for name, repo in REPOS.items():
        print("=" * 92)
        print(f"{name}   {repo}")
        print("=" * 92)
        r = {"path": str(repo)}
        try:
            idx = read_index(repo)
        except Exception as exc:                        # noqa: BLE001
            print(f"  INDEX UNREADABLE: {type(exc).__name__}: {exc}")
            r["index_error"] = f"{type(exc).__name__}: {exc}"
            report["repos"][name] = r
            continue
        rf = refs(repo)
        r["refs"] = rf
        r["tracked_files"] = len(idx)
        print(f"  branch          : {rf['branch']}")
        print(f"  origin url      : {rf['origin_url']}")
        print(f"  local  tip      : {rf['local']}")
        print(f"  origin/<branch> : {rf['remote_origin']}")
        print(f"  pushed          : {'YES - local tip == origin tip' if rf['in_sync'] else 'NO / UNKNOWN - local tip differs from the last known origin tip'}")
        print(f"  tracked files   : {len(idx):,}")

        req = REQUIRED.get(name)
        if req is None:
            req = discovered_oof_requirements() if name == "logs_and_models" else []
        if req:
            print(f"\n  required files ({len(req)}):")
        rows = []
        for rel in req:
            disk = repo / rel
            exists = disk.is_file()
            entry = idx.get(rel.replace("\\", "/"))
            state, detail = "MISSING", ""
            if not exists and entry is None:
                state, detail = "ABSENT", "not on disk and not tracked"
            elif not exists:
                state, detail = "TRACKED-BUT-GONE", "in the index, not on disk"
            elif entry is None:
                state, detail = "UNTRACKED", "on disk but NOT in the backup"
            else:
                same = entry[0] in blob_sha1_candidates(disk)
                state = "OK" if same else "MODIFIED"
                detail = "staged content matches disk" if same else \
                         "on disk differs from what is staged/committed -- commit it"
            rows.append({"path": rel, "state": state, "detail": detail,
                         "on_disk": exists, "tracked": entry is not None})
            flag = "  " if state == "OK" else ">>"
            print(f"  {flag} [{state:<17}] {rel}")
            if detail and state != "OK":
                print(f"       {detail}")
        r["required"] = rows
        report["repos"][name] = r

        bad = [x for x in rows if x["state"] != "OK"]
        print(f"\n  {len(rows) - len(bad)} / {len(rows)} required files verified in the backup")
        if bad:
            print(f"  {len(bad)} NOT safely backed up -- listed above")
        print()

    out = C.assert_writes_are_local(C.TABLES / "git_backup_verification.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")

    allbad = [(n, x) for n, r in report["repos"].items() for x in r.get("required", [])
              if x["state"] != "OK"]
    print("\n" + ("ALL REQUIRED FILES VERIFIED" if not allbad else
                  f"{len(allbad)} REQUIRED FILE(S) OUTSIDE THE TRACKED SET -- see above"))


if __name__ == "__main__":
    main()
