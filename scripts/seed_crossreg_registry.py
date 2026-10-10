"""Seed caban/crossreg_groupings.yaml with the seven published cross-registration groupings, and check it.

    python scripts/seed_crossreg_registry.py [--data-root /Volumes/MINISCOPE/SSTCa2] [--cache-root <npy_files>]

Membership comes from each mapping file's own ``session`` columns -- the ground truth of what was
registered -- not from output-folder names (plans/crossreg_batch_runner_plan.md §3). Each member is
matched to its session folder and to the output folder whose cells the mapping refers to: the
``minian_crossreg*`` folder tagged with the grouping when there is one, else the session's
``minian``. Checks, all hard failures:

- exactly one mapping file per (mouse, grouping), stray copies listed and checked identical;
- every mapping session resolves to exactly one session folder of that mouse;
- every unit id in a mapping column is a unit of the member's output folder;
- the analysis cache's copy of each mapping (FUTROLA, ``*/crossreg/<mouse>_crossreg_<N>_mappings_df``)
  equals the file on disk.
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd
import xarray as xr
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caban import utilities as ut

REGISTRY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "caban",
                        "crossreg_groupings.yaml")
PUBLISHED = {
    1: ("conditioning_day", "LT1, LT2 and TFC_cond of the conditioning day (Fig. 2)"),
    2: ("cond_to_testB", "TFC_cond -> Test_B"),
    3: ("testB_day", "LT1 and TFC_test_B of the Test_B day"),
    4: ("cond_testB_testB1wk", "TFC_cond -> Test_B -> Test_B_1wk"),
    5: ("testB1wk_day", "LT1 and TFC_test_B_1wk of the Test_B_1wk day"),
    6: ("cond_all_tests", "TFC_cond + all four tests"),
    7: ("cond_testA_testA1wk", "TFC_cond -> Test_A -> Test_A_1wk"),
}
MOUSE_RE = re.compile(r"G(0[5-9]|1\d|2[01])-")


def mapping_files(mouse_dir, n):
    names = ["mappings_crossreg_{}.csv".format(n), "mappings_crossreg{}.csv".format(n)]
    found = []
    for name in names:
        found += glob.glob(os.path.join(mouse_dir, name)) + glob.glob(os.path.join(mouse_dir, "*", name))
    return sorted(set(found))


def canonical(files, mouse_dir, n):
    """The published mapping file: in the mouse folder for 2-7, in the conditioning-day folder for 1;
    anything else (old_mappings/, '- Copy') is a stray copy."""
    if n == 1:
        keep = [f for f in files if os.path.dirname(f) != mouse_dir and "old_mappings" not in f]
    else:
        keep = [f for f in files if os.path.dirname(f) == mouse_dir]
    if len(keep) != 1:
        raise ValueError("{} grouping {}: cannot pick one mapping file from {}".format(mouse_dir, n, files))
    return keep[0]


def read_mapping(path):
    return pd.read_csv(path, header=[0, 1], index_col=0)


def same_values(a, b):
    """Value equality with NaN == NaN, blind to dtype: a 2023 cache holds object columns where
    today's pandas reads StringDtype, which DataFrame.equals counts as a difference."""
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return False
    for col in a.columns:
        x, y = pd.to_numeric(a[col], errors="coerce"), pd.to_numeric(b[col], errors="coerce")
        if not ((x == y) | (x.isna() & y.isna())).all():
            return False
        sx, sy = a[col][x.isna()].astype(object), b[col][y.isna()].astype(object)
        if not all((p == q) or (pd.isna(p) and pd.isna(q)) for p, q in zip(sx, sy)):
            return False
    return True


def units(output_dir):
    return set(xr.open_zarr(os.path.join(output_dir, "A.zarr"), consolidated=False)["unit_id"].values.astype(int))


def member_output(session_miniscope, n):
    tagged = [o for o in os.listdir(session_miniscope)
              if o.startswith("minian_crossreg") and str(n) in re.findall(r"crossreg(\d+)", o)
              and os.path.isdir(os.path.join(session_miniscope, o))]
    if len(tagged) > 1:
        raise ValueError("{}: several output folders tagged crossreg{}: {}".format(session_miniscope, n, tagged))
    if tagged:
        return tagged[0]
    if os.path.isdir(os.path.join(session_miniscope, "minian")):
        return "minian"
    raise FileNotFoundError("{}: no output folder for crossreg{}".format(session_miniscope, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/Volumes/MINISCOPE/SSTCa2")
    ap.add_argument("--cache-root", default=ut.NPY_SAVE_PATH)
    args = ap.parse_args()
    root = args.data_root
    mice = sorted(m for m in os.listdir(root) if MOUSE_RE.match(m))
    groupings, notes, problems = {}, [], []
    for n, (name, description) in PUBLISHED.items():
        members = {}
        for m in mice:
            mdir = os.path.join(root, m)
            files = mapping_files(mdir, n)
            if not files:
                continue
            path = canonical(files, mdir, n)
            df = read_mapping(path)
            for other in files:
                if other != path:
                    same = same_values(read_mapping(other), df)
                    notes.append("{} g{}: stray copy {} ({})".format(m[:3], n, os.path.relpath(other, root),
                                                                    "identical" if same else "DIFFERENT"))
            sessions = []
            for col in [c for c in df.columns if c[0] == "session"]:
                t = col[1]
                hits = [h for h in glob.glob(os.path.join(mdir, "*", t + "-*", "Miniscope"))]
                if len(hits) > 1:
                    # The mapping names a session by its start time only, and two sessions of this mouse
                    # started at the same second on different days: the one tagged for this grouping.
                    tagged = [h for h in hits if any(str(n) in re.findall(r"crossreg(\d+)", o)
                                                     for o in os.listdir(h) if o.startswith("minian_crossreg"))]
                    notes.append("{} g{}: session time {} is shared by {} sessions; resolved by the crossreg{} tag "
                                 "to {}".format(m[:3], n, t, len(hits), n, [os.path.relpath(h, root) for h in tagged]))
                    hits = tagged
                if len(hits) != 1:
                    problems.append("{} g{}: mapping session {} resolves to {} session folders".format(
                        m[:3], n, t, len(hits)))
                    continue
                out = member_output(hits[0], n)
                ids = set(df[col].dropna().astype(int))
                member = {"session": os.path.relpath(os.path.dirname(hits[0]), root), "output": out}
                if ids - units(os.path.join(hits[0], out)):
                    # The mapping refers to another run of this session: find the output that holds
                    # every one of its unit ids (an earlier run kept in prev/ or as -ORIG).
                    others = [os.path.relpath(os.path.dirname(a), hits[0])
                              for a in glob.glob(os.path.join(hits[0], "**", "A.zarr"), recursive=True)]
                    holds = [o for o in others if o != out and not (ids - units(os.path.join(hits[0], o)))]
                    if len(holds) != 1:
                        problems.append("{} g{} {}: {} mapping ids not in {}, and {} other outputs hold them all: {}".format(
                            m[:3], n, t, len(ids - units(os.path.join(hits[0], out))), out, len(holds), holds))
                    else:
                        member["output"] = holds[0]
                        member["stale"] = "refers to an earlier run ({}), not the current {}".format(holds[0], out)
                        notes.append("{} g{} {}: STALE -- mapping matches {}, not the current {}".format(
                            m[:3], n, t, holds[0], out))
                elif out == "minian":
                    notes.append("{} g{}: {} has no crossreg-tagged folder; its {} carries all {} mapping ids".format(
                        m[:3], n, t, out, len(ids)))
                sessions.append(member)
            # The loader builds every CrossRegMapping with savepath = <cache>/TFC_cond, so that copy is
            # the one analyses read; copies under other folders are orphans of an older code version.
            disk = pd.read_csv(path)
            cache_name = "{}_crossreg_{}_mappings_df_DataFrame.pkl".format(m[:3], n)
            cache = glob.glob(os.path.join(args.cache_root, "TFC_cond", "crossreg", cache_name))
            for c in cache:
                if not same_values(pd.read_pickle(c), disk):
                    problems.append("{} g{}: the loader's cached mapping {} differs from {}".format(m[:3], n, c, path))
            for c in glob.glob(os.path.join(args.cache_root, "*", "crossreg", cache_name)):
                if c not in cache:
                    notes.append("{} g{}: orphan cache copy {} ({}; not read by the loader)".format(
                        m[:3], n, os.path.relpath(c, args.cache_root),
                        "same as the file" if same_values(pd.read_pickle(c), disk) else "STALE"))
            members[m[:3]] = {"mapping": os.path.relpath(path, root), "sessions": sessions,
                              "cache_checked": len(cache)}
        groupings[n] = {"name": name, "description": description, "status": "published", "mice": members}
        print("grouping {} ({}): {} mice".format(n, name, len(members)))
    doc = {"note": ("Cross-registration groupings: the master list (plans/crossreg_batch_runner_plan.md §3). "
                    "Paths are relative to the data root (MINISCOPE/SSTCa2). 1-7 seeded from their mapping files "
                    "by scripts/seed_crossreg_registry.py; do not edit them by hand."),
           "groupings": groupings}
    with open(REGISTRY, "w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, width=120, allow_unicode=True)
    print("wrote", REGISTRY)
    for line in notes:
        print("  note:", line)
    if problems:
        for line in problems:
            print("  PROBLEM:", line)
        raise SystemExit("{} unresolved problems; the registry was written with them unflagged -- fix first".format(
            len(problems)))


if __name__ == "__main__":
    main()
