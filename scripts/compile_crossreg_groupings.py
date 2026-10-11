"""Compile caban/crossreg_groupings_spec.yaml into caban/crossreg_groupings.json, and check it.

    python scripts/compile_crossreg_groupings.py [--data-root /Volumes/MINISCOPE/SSTCa2] [--cache-root <npy_files>]

The spec is the short, human-readable definition; the JSON lists every session of every grouping per
mouse and is what code reads (plans/crossreg_batch_runner_plan.md §3).

Published groupings 1-7: membership from each mapping file's own ``session`` columns -- the ground
truth of what was registered -- each member matched to the output folder whose cells the mapping refers
to. Checked (hard failures): one published mapping file per (mouse, grouping), stray copies compared;
each mapping session resolves to one session folder; every mapped unit id is a unit of its member's
output; the loader's cached copy (``<cache>/TFC_cond/crossreg/``) equals the file by value.

Groupings 8+: from the spec's rules (conditioning-day roles, track days, LT/HC sessions) applied to the
session folders and session types (``session_queue.SESSION_LABELS`` for unlabelled folders).

Every G05-G21 session with video must be in a grouping or matched by the spec's ``not_used``; the
compiler fails otherwise.
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd
import xarray as xr
import fnmatch
import json

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caban import session_queue as sq
from caban import utilities as ut

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(HERE, "caban", "crossreg_groupings_spec.yaml")
REGISTRY = os.path.join(HERE, "caban", "crossreg_groupings.json")
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


def published_members(n, root, mice, cache_root, notes, problems):
    """Members of published grouping ``n``, per mouse, from the mapping files (checked)."""
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
        cache = glob.glob(os.path.join(cache_root, "TFC_cond", "crossreg", cache_name))
        for c in cache:
            if not same_values(pd.read_pickle(c), disk):
                problems.append("{} g{}: the loader's cached mapping {} differs from {}".format(m[:3], n, c, path))
        for c in glob.glob(os.path.join(cache_root, "*", "crossreg", cache_name)):
            if c not in cache:
                notes.append("{} g{}: orphan cache copy {} ({}; not read by the loader)".format(
                    m[:3], n, os.path.relpath(c, cache_root),
                    "same as the file" if same_values(pd.read_pickle(c), disk) else "STALE"))
        members[m[:3]] = {"mapping": os.path.relpath(path, root), "sessions": sessions,
                          "cache_checked": len(cache)}
    return members


# --- Rule-based groupings (8+) --------------------------------------------------------------------------
TRACK_DAY = re.compile(r"-track_day(\d)(-tests)?$")
TEST_DAY = re.compile(r"-TFC_test[_-](A|B)(_1wk)?$")


def output_of(miniscope):
    """The output folder a grouping uses for a session: its crossreg-tagged folder for production
    sessions, else ``minian``, else ``pending`` (not run yet)."""
    outs = [o for o in os.listdir(miniscope)
            if o.startswith("minian") and not o.endswith("-ORIG") and os.path.isdir(os.path.join(miniscope, o))
            and os.path.exists(os.path.join(miniscope, o, "A.zarr"))]
    tagged = [o for o in outs if o.startswith("minian_crossreg")]
    if len(tagged) > 1:
        raise ValueError("{}: several crossreg-tagged outputs {}".format(miniscope, tagged))
    if tagged:
        return tagged[0]
    return "minian" if "minian" in outs else "pending"


def all_sessions(root, mice):
    rows = []
    for m in mice:
        for day in sorted(os.listdir(os.path.join(root, m))):
            if not re.match(r"\d{4}_\d{2}_\d{2}", day):
                continue
            for sess in sorted(os.listdir(os.path.join(root, m, day))):
                mini = os.path.join(root, m, day, sess, "Miniscope")
                if not (os.path.isdir(mini) and glob.glob(os.path.join(mini, "*.avi"))):
                    continue
                label = sq.session_label(mini)
                kind = label if label is not None else (sess.split("-", 1)[1] if "-" in sess else "")
                rows.append({"mouse": m[:3], "day": day, "session": sess, "time": sess[:8], "kind": kind,
                             "rel": os.path.join(m, day, sess), "short": "{}/{}/{}".format(m[:3], day, sess),
                             "miniscope": mini})
    return rows


def rule_members(gid, g, spec, sessions):
    """{mouse: [session rows]} for one rule-based grouping."""
    roles = spec["roles"]
    out = {}
    for r in sessions:
        day, kind, mouse = r["day"], r["kind"], r["mouse"]
        take = False
        if "roles" in g:
            if day.endswith("-TFC_cond"):
                row = dict(zip(roles, spec["conditioning_day"][mouse]))
                want = roles if g["roles"] == "all" else g["roles"]
                take = any(row[k] == r["time"] for k in want)
        elif gid == 11:
            if TRACK_DAY.search(day):
                # Track days 4-5 exist for G08 and G09 only: extra sessions, so not here (grouping 15/16 only).
                take = kind == "LT1" and TRACK_DAY.search(day).group(1) in "123"
            elif day.endswith("-TFC_cond"):
                row = dict(zip(roles, spec["conditioning_day"][mouse]))
                take = r["time"] in (row["LT_pre"], row["LT_post"])
            else:
                take = TEST_DAY.search(day) is not None and kind in ("LT1", "LT2")
        elif 12 <= gid <= 16:
            mt = TRACK_DAY.search(day)
            take = mt is not None and int(mt.group(1)) == gid - 11 and kind in ("HC1", "LT1", "HC2")
        elif gid == 17:
            if TRACK_DAY.search(day):
                take = TRACK_DAY.search(day).group(1) in "123" and kind in ("HC1", "HC2")
            elif day.endswith("-TFC_cond"):
                row = dict(zip(roles, spec["conditioning_day"][mouse]))
                take = r["time"] in (row["HC_start"], row["HC_preinjection"], row["HC_preTFC"], row["HC_postTFC"])
            else:
                take = TEST_DAY.search(day) is not None and re.fullmatch(r"HC\d", kind) is not None
        elif "sessions" in g and isinstance(g["sessions"], list):
            take = mouse in g["mice"] and "{}/{}".format(day, r["session"]) in g["sessions"]
        if take:
            out.setdefault(mouse, []).append(r)
    if g.get("require_all"):
        need = len(g["roles"])
        out = {m: rs for m, rs in out.items() if len(rs) == need}
    return out


def check_roles(spec, sessions, problems):
    """Every time in the role table names exactly one session of that mouse's conditioning day."""
    for mouse, times in spec["conditioning_day"].items():
        if len(times) != len(spec["roles"]):
            problems.append("{}: role row has {} entries, not {}".format(mouse, len(times), len(spec["roles"])))
        day_rows = [r for r in sessions if r["mouse"] == mouse and r["day"].endswith("-TFC_cond")]
        for role, t in zip(spec["roles"], times):
            if t == "-":
                continue
            if sum(1 for r in day_rows if r["time"] == t) != 1:
                problems.append("{} {}: {} matches no single conditioning-day session".format(mouse, role, t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/Volumes/MINISCOPE/SSTCa2")
    ap.add_argument("--cache-root", default=ut.NPY_SAVE_PATH)
    args = ap.parse_args()
    root = args.data_root
    with open(SPEC) as fh:
        spec = yaml.safe_load(fh)
    mice = sorted(m for m in os.listdir(root) if MOUSE_RE.match(m))
    sessions = all_sessions(root, mice)
    notes, problems, used = [], [], set()
    check_roles(spec, sessions, problems)
    groupings = {}
    for gid, g in spec["groupings"].items():
        entry = {"name": g["name"], "about": g["about"]}
        if g.get("published"):
            members = published_members(gid, root, mice, args.cache_root, notes, problems)
            for mm in members.values():
                used.update(x["session"] for x in mm["sessions"])
            entry.update(status="published", mice=members)
        else:
            members = rule_members(gid, g, spec, sessions)
            entry.update(status="to compute", mice={
                m: {"sessions": [{"session": r["rel"], "output": output_of(r["miniscope"])} for r in rs]}
                for m, rs in sorted(members.items())})
            used.update(r["rel"] for rs in members.values() for r in rs)
        groupings[str(gid)] = entry
        n_sess = sum(len(v["sessions"]) for v in entry["mice"].values())
        print("grouping {:>3} {:<22} {:>2} mice, {:>3} sessions".format(gid, g["name"], len(entry["mice"]), n_sess))
    # Every session is in a grouping or in not_used.
    unused_rules = spec.get("not_used", [])
    unaccounted, not_used = [], []
    for r in sessions:
        if r["rel"] in used:
            continue
        why = [u["why"] for u in unused_rules if fnmatch.fnmatch(r["short"], u["session"])]
        if why:
            not_used.append({"session": r["rel"], "why": why[0]})
        else:
            unaccounted.append(r["short"])
    for u in unused_rules:
        if not any(fnmatch.fnmatch(r["short"], u["session"]) for r in sessions):
            problems.append("not_used entry {} matches no session".format(u["session"]))
    if unaccounted:
        problems.append("{} sessions in no grouping and not in not_used: {}".format(len(unaccounted), unaccounted))
    doc = {"note": ("Generated by scripts/compile_crossreg_groupings.py from caban/crossreg_groupings_spec.yaml -- "
                    "do not edit. Session paths are relative to the data root (MINISCOPE/SSTCa2); 'output' is the "
                    "output folder inside the session's Miniscope/ ('pending': not run yet)."),
           "groupings": groupings, "not_used": not_used}
    with open(REGISTRY, "w") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
    print("wrote {} ({} sessions accounted for: {} in groupings, {} not used)".format(
        REGISTRY, len(sessions), len(used), len(not_used)))
    for line in notes:
        print("  note:", line)
    if problems:
        for line in problems:
            print("  PROBLEM:", line)
        raise SystemExit("{} unresolved problems -- fix the spec or the data first".format(len(problems)))


if __name__ == "__main__":
    main()
