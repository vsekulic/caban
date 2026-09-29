"""Carry max_proj re-check results (scripts/verify_yra_replays.py) from one drive's YrA recompute sidecars
to the other's, so MINISCOPE and MINIRAZER agree (2026-09-29).

    python scripts/copy_recheck_sidecars.py <source root> <destination SSTCa2 root> <backup dir>

<source root> mirrors <mouse>/<day>/<session>/Miniscope/<output>/YrA_recompute.json (a copy of the
checked drive's sidecars). A destination sidecar is replaced only when the source one equals it except
for the fields the re-check adds (report.max_proj_check, max_proj_checked_utc), carries a passing
check, and the destination has none yet; the destination's version is copied to <backup dir> first.
Sidecars already carrying a check at the destination are left alone. md5-verified after writing.
"""

import copy
import glob
import hashlib
import json
import os
import shutil
import sys

CHECK_FIELDS_TOP = ("max_proj_checked_utc",)


def md5(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def without_check(record: dict) -> dict:
    stripped = copy.deepcopy(record)
    for key in CHECK_FIELDS_TOP:
        stripped.pop(key, None)
    stripped["report"].pop("max_proj_check", None)
    return stripped


def main(src_root: str, dst_root: str, backup_dir: str) -> None:
    updated = skipped = 0
    for src in sorted(glob.glob(os.path.join(src_root, "*/*/*/Miniscope/*/YrA_recompute.json"))):
        rel = os.path.relpath(src, src_root)
        dst = os.path.join(dst_root, rel)
        new = json.load(open(src))
        if "max_proj_check" not in new["report"]:
            continue
        old = json.load(open(dst))
        if "max_proj_check" in old["report"]:
            skipped += 1
            continue
        if without_check(new) != old:
            raise ValueError("{}: differs from the destination beyond the re-check fields".format(rel))
        if not new["report"]["max_proj_check"]["passed"]:
            raise ValueError("{}: its check did not pass".format(rel))
        os.makedirs(os.path.join(backup_dir, os.path.dirname(rel)), exist_ok=True)
        shutil.copy2(dst, os.path.join(backup_dir, rel))
        shutil.copyfile(src, dst)
        if md5(dst) != md5(src):
            raise IOError("{}: copy does not verify".format(rel))
        updated += 1
    print("{} sidecars updated, {} already had a check; previous versions in {}".format(
        updated, skipped, backup_dir))


if __name__ == "__main__":
    main(*sys.argv[1:4])
