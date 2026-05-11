"""
SSTCa2_CFC.py — Contextual Fear Conditioning analysis.
Protocol: CFC (ctx A) -> +48hr Test A (ctx A) -> +24hr Test B (ctx B) -> +1wk Test A -> +1wk Test B
Loads freeze score data from CSV files in a structured directory.
Three analysis windows for test sessions: full 300s, first 180s, last 120s.
"""

import csv
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from itertools import combinations

# ═════════════════════════════════════════════
# CONFIGURATION — edit these parameters
# ═════════════════════════════════════════════

# Directory under freeze_data/ containing the CSV subdirectories
csv_dir = '2026-03-19_FGET'

# File inside csv_dir that lists mouse group assignments.
# Format: one line per mouse — "MouseName X" where X is E, I, or C.
# Blank lines are ignored.
groups_file = 'groups.txt'

saveit = True
for_paper = False

if for_paper:
    my_linewidth = 0.2
    my_markersize = 1
    font_size = 5
    my_sz = 12
else:
    my_linewidth = 1
    my_markersize = 6
    font_size = 18
    my_sz = 42

# Colors
my_r = np.array([1.0, 0.0, 0.0])
my_b = np.array([0.0, 0.0, 1.0])
my_k = np.array([0.0, 0.0, 0.0])
my_h = np.array([0.5, 0.5, 0.5])
my_light_grey = np.array([0.8, 0.8, 0.8])

GROUP_COLORS = {'E': my_r, 'I': my_b, 'C': my_k}
GROUP_LABELS = {'E': 'SST Exc', 'I': 'SST Inh', 'C': 'SST Ctl'}
GROUP_ORDER = ['E', 'I', 'C']

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = font_size

file_path = r"C:\Users\vlads\Dropbox\1-McHugh postdoc\3-PAPER\paper_plots\fig1\plots\CFC"

# ═════════════════════════════════════════════
# CFC CONSTANTS
# ═════════════════════════════════════════════
BIN_SIZE = 20           # seconds per bin
CFC_DURATION = 360      # seconds
TEST_DURATION = 300     # seconds
PRE_SHOCK_DURATION = 180  # seconds — baseline before first shock

CFC_BINS = CFC_DURATION // BIN_SIZE     # 18
TEST_BINS = TEST_DURATION // BIN_SIZE   # 15
PRE_SHOCK_BINS = PRE_SHOCK_DURATION // BIN_SIZE  # 9
POST_SHOCK_BINS = TEST_BINS - PRE_SHOCK_BINS     # 6

onset_cfc = np.arange(0, CFC_DURATION, BIN_SIZE)    # 18 bins
onset_test = np.arange(0, TEST_DURATION, BIN_SIZE)   # 15 bins

# Shock parameters: 3 shocks at 180, 240, 300 s (2 s each)
shock_onsets = np.array([180, 240, 300])
shock_offsets = shock_onsets + 2

# Directory experiment name → internal session key
EXP_TO_SESSION = {
    'CFC-A':     'CFC',
    'TestA':     'TestA_48hr',
    'TestB':     'TestB_48hr',
    'TestA_1wk': 'TestA_1wk',
    'TestB_1wk': 'TestB_1wk',
    'TestA-1wk': 'TestA_1wk',
    'TestB-1wk': 'TestB_1wk',
}

SESSION_LABELS = {
    'CFC':        'CFC Conditioning',
    'TestA_48hr': 'Test A (48 hr)',
    'TestB_48hr': 'Test B (48 hr)',
    'TestA_1wk':  'Test A (1 wk)',
    'TestB_1wk':  'Test B (1 wk)',
}

# Test-session analysis windows: (slice, label, file_prefix)
FIRST60_BINS = 60 // BIN_SIZE  # 3
LAST60_BINS = 60 // BIN_SIZE   # 3

WINDOWS = [
    (slice(None),                        'Full (300 s)', 'full'),
    (slice(0, FIRST60_BINS),             'First 60 s',   'first60'),
    (slice(0, PRE_SHOCK_BINS),           'First 180 s',  'first180'),
    (slice(PRE_SHOCK_BINS, TEST_BINS),   'Last 120 s',   'last120'),
    (slice(TEST_BINS - LAST60_BINS, TEST_BINS), 'Last 60 s', 'last60'),
]

# ═════════════════════════════════════════════
# STATISTICAL HELPERS
# ═════════════════════════════════════════════

def holm_bonferroni(pvals):
    """Holm–Bonferroni step-down correction on a list of p-values."""
    n = len(pvals)
    if n == 0:
        return []
    pvals = np.asarray(pvals, dtype=float) 
    order = np.argsort(pvals)
    adjusted = np.zeros(n)
    for rank, idx in enumerate(order):
        adjusted[idx] = min(pvals[idx] * (n - rank), 1.0)
    # enforce monotonicity in sorted order
    for rank in range(1, n):
        idx = order[rank]
        prev_idx = order[rank - 1]
        adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])
    return adjusted.tolist()


def do_stats_cfc(data_by_group, offset=0):
    """
    One-way ANOVA + Holm post-hoc for groups present in data_by_group.
    data_by_group: dict {group_label: 1-d array of per-animal values}
    Returns (group_pairs, adj_pvals, anova_F, anova_p)
    group_pairs are bar-position pairs (1-indexed + offset).
    """
    labels = [g for g in GROUP_ORDER if g in data_by_group and len(data_by_group[g]) > 0]
    if len(labels) < 2:
        return [], [], np.nan, np.nan

    arrays = [data_by_group[g] for g in labels]

    if len(labels) >= 3:
        F, p_omni = sp_stats.f_oneway(*arrays)
    else:
        F, p_omni = np.nan, np.nan

    raw_pvals = []
    pair_indices = []
    for i, j in combinations(range(len(labels)), 2):
        _, p = sp_stats.ttest_ind(arrays[i], arrays[j])
        raw_pvals.append(p)
        # bar positions: group index + 1 + offset
        pair_indices.append([i + 1 + offset, j + 1 + offset])

    adj_pvals = holm_bonferroni(raw_pvals)
    return pair_indices, adj_pvals, F, p_omni


def _ttest_paired_or_ind(a, b):
    """Paired t-test if same length, otherwise unpaired."""
    if len(a) == len(b):
        _, p = sp_stats.ttest_rel(a, b)
    else:
        _, p = sp_stats.ttest_ind(a, b)
    return p


# ═════════════════════════════════════════════
# PLOTTING HELPERS
# ═════════════════════════════════════════════

def _setup_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out')
    for spine in ax.spines.values():
        spine.set_linewidth(my_linewidth)


def sigstar_text(ax, groups, pvals, y_start=None, sep=2.5):
    """Draw significance brackets + stars above bars."""
    if y_start is None:
        y_start = ax.get_ylim()[1] * 0.85
    y = y_start
    for (g1, g2), p in zip(groups, pvals):
        if p < 0.001:
            txt = '***'
        elif p < 0.01:
            txt = '**'
        elif p < 0.05:
            txt = '*'
        else:
            txt = 'n.s.'
        ax.plot([g1, g1, g2, g2], [y, y + sep * 0.3, y + sep * 0.3, y],
                'k-', linewidth=0.8)
        ax.text((g1 + g2) / 2, y + sep * 0.35, txt,
                ha='center', va='bottom', fontsize=8)
        y += sep


def _save(fig, name):
    if saveit:
        os.makedirs(file_path, exist_ok=True)
        fig.savefig(os.path.join(file_path, f'{name}.svg'), format='svg')
        fig.savefig(os.path.join(file_path, f'{name}.png'), format='png', dpi=300)
        print(f"Saved {name}")


def _mean_sem(arr):
    """Mean and SEM across rows (axis=0). arr shape (N, bins)."""
    m = np.mean(arr, axis=0)
    s = np.std(arr, axis=0, ddof=0) / np.sqrt(arr.shape[0])
    return m, s


# ═════════════════════════════════════════════
# CSV LOADING
# ═════════════════════════════════════════════

def load_groups_file(base_dir):
    """Load groups.txt from base_dir. Returns dict {mouse_name: group_letter}."""
    gpath = os.path.join(base_dir, groups_file)
    if not os.path.isfile(gpath):
        print(f"ERROR: groups file not found: {gpath}")
        raise SystemExit(1)
    groups_map = {}
    with open(gpath, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Expected format: "MouseName X" where X is last token
            tokens = line.rsplit(None, 1)
            if len(tokens) != 2:
                print(f"  WARNING: cannot parse groups.txt line: '{line}' — skipping")
                continue
            name, grp = tokens[0].strip(), tokens[1].strip().upper()
            if grp not in GROUP_COLORS:
                print(f"  WARNING: unknown group '{grp}' for mouse '{name}' — skipping")
                continue
            groups_map[name] = grp
    return groups_map


def parse_freeze_csv(csv_path):
    """Parse a freeze_Index.csv file. Returns list of (mouse_name, np.array)."""
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)  # skip "% freeze" line

        header = next(reader)  # Onset row
        # Count consecutive numeric bin columns starting at index 1
        n_bins = 0
        for cell in header[1:]:
            try:
                float(cell.strip())
                n_bins += 1
            except (ValueError, IndexError):
                break

        next(reader)  # skip Duration row

        mice = []
        for row in reader:
            if not row or not row[0].strip():
                continue
            name = row[0].strip()
            scores = []
            for cell in row[1:1 + n_bins]:
                try:
                    scores.append(float(cell.strip()))
                except (ValueError, IndexError):
                    scores.append(0.0)
            mice.append((name, np.array(scores)))
    return mice


def load_from_csv_dir():
    """Scan csv_dir subdirectories, parse CSVs, assign groups from groups.txt.
    Returns (sessions, groups_map, mouse_cohort, mouse_sessions)."""
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'freeze_data', csv_dir)
    if not os.path.isdir(base):
        print(f"ERROR: directory not found: {base}")
        raise SystemExit(1)

    # Load group assignments
    groups_map = load_groups_file(base)

    # Track which cohort each mouse belongs to, and which sessions it appears in
    mouse_cohort = {}    # mouse_name -> cohort (e.g. 'FGET1')
    mouse_sessions = {}  # mouse_name -> set of session_keys

    # Collect raw data: {session_key: {group: [(name, array), ...]}}
    raw = {sk: {g: [] for g in GROUP_ORDER} for sk in SESSION_LABELS}

    for subdir in sorted(os.listdir(base)):
        subpath = os.path.join(base, subdir)
        if not os.path.isdir(subpath):
            continue

        # Parse directory name: "DATEPART FGETn EXP"
        parts = subdir.split()
        if len(parts) < 3:
            print(f"  Skipping unrecognised dir: {subdir}")
            continue
        cohort = parts[1]                    # e.g. "FGET1"
        exp = ' '.join(parts[2:])            # e.g. "CFC-A" or "TestA_1wk"

        session_key = EXP_TO_SESSION.get(exp)
        if session_key is None:
            print(f"  Skipping unknown experiment '{exp}' in: {subdir}")
            continue

        csv_path = os.path.join(subpath, 'freeze_Index.csv')
        if not os.path.isfile(csv_path):
            print(f"  WARNING: no freeze_Index.csv in {subdir}")
            continue

        mice = parse_freeze_csv(csv_path)
        for name, scores in mice:
            grp = groups_map.get(name)
            if grp is None:
                print(f"  WARNING: mouse '{name}' not in {groups_file} — skipping")
                continue
            raw[session_key][grp].append((name, scores))
            # Track cohort and sessions
            if name not in mouse_cohort:
                mouse_cohort[name] = cohort
            mouse_sessions.setdefault(name, set()).add(session_key)

        print(f"  Loaded {subdir:45s} → {SESSION_LABELS[session_key]} "
              f"({cohort}, {len(mice)} mice)")

    # Stack arrays, drop empty sessions
    sessions = {}
    for sk, by_group in raw.items():
        stacked = {}
        for g, items in by_group.items():
            if items:
                stacked[g] = np.vstack([s for _, s in items])
        if stacked:
            sessions[sk] = stacked

    return sessions, groups_map, mouse_cohort, mouse_sessions


# ═════════════════════════════════════════════
# LOAD DATA
# ═════════════════════════════════════════════

# Tee stdout to output.txt in the plots directory
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()

os.makedirs(file_path, exist_ok=True)
_log_file = open(os.path.join(file_path, 'output.txt'), 'w', encoding='utf-8')
sys.stdout = _Tee(sys.__stdout__, _log_file)

print(f"\nLoading CSV data from freeze_data/{csv_dir}/ ...")
sessions, groups_map, mouse_cohort, mouse_sessions = load_from_csv_dir()

# ─── Diagnostic: per-mouse group assignment ───
print(f"\n─── Group Assignments (from {groups_file}) ───")
for name in sorted(groups_map.keys()):
    grp = groups_map[name]
    print(f"  {name:25s} → {GROUP_LABELS[grp]} ({grp})")

# ─── Diagnostic: group rosters ───
print("\n─── Group Rosters ───")
for g in GROUP_ORDER:
    members = sorted(n for n, gr in groups_map.items() if gr == g)
    print(f"  {GROUP_LABELS[g]} ({g}): n={len(members)}")
    for m in members:
        coh = mouse_cohort.get(m, '?')
        sess = mouse_sessions.get(m, set())
        sess_str = ', '.join(sorted(sess)) if sess else 'NONE'
        print(f"    {m:25s}  [{coh}]  sessions: {sess_str}")

# ─── Diagnostic: per-cohort roster ───
all_session_keys = set()
for sk in SESSION_LABELS:
    if sk in sessions:
        all_session_keys.add(sk)

print("\n─── Cohort Rosters ───")
cohorts = sorted(set(mouse_cohort.values()))
for coh in cohorts:
    members = sorted(n for n, c in mouse_cohort.items() if c == coh)
    in_all = [m for m in members
              if mouse_sessions.get(m, set()) >= all_session_keys]
    print(f"  {coh}: {len(members)} mice total, "
          f"{len(in_all)} in ALL loaded sessions ({len(all_session_keys)} sessions)")
    for m in members:
        grp = groups_map.get(m, '?')
        sess = mouse_sessions.get(m, set())
        missing = all_session_keys - sess
        flag = '' if not missing else f'  *** MISSING: {", ".join(sorted(missing))}'
        print(f"    {m:25s}  group={grp}  sessions={len(sess)}/{len(all_session_keys)}{flag}")

# ─── Session summary ───
print("\n─── Loaded Sessions ───")
for sk in SESSION_LABELS:
    if sk in sessions:
        counts = {g: d.shape[0] for g, d in sessions[sk].items()}
        print(f"  {SESSION_LABELS[sk]:25s} — " +
              ", ".join(f"{GROUP_LABELS.get(g,g)} n={n}" for g, n in counts.items()))
    else:
        print(f"  {SESSION_LABELS[sk]:25s} — no data")

if not sessions:
    print("\nNo data loaded. Check directory structure and re-run.")
    raise SystemExit(0)


# ═════════════════════════════════════════════
# METRIC EXTRACTION HELPERS
# ═════════════════════════════════════════════

def session_mean(session_key, bin_slice=None):
    """Per-animal mean freeze score. Returns {group: 1-d array} or None."""
    data = sessions.get(session_key)
    if data is None:
        return None
    result = {}
    for g, arr in data.items():
        if bin_slice is not None:
            result[g] = np.mean(arr[:, bin_slice], axis=1)
        else:
            result[g] = np.mean(arr, axis=1)
    return result if result else None


def discrimination_index(data_A, data_B):
    """DI = (A - B) / (A + B) per animal; returns {group: array} or None."""
    if data_A is None or data_B is None:
        return None
    result = {}
    for g in GROUP_ORDER:
        a, b = data_A.get(g), data_B.get(g)
        if a is None or b is None:
            continue
        n = min(len(a), len(b))
        denom = a[:n] + b[:n]
        result[g] = np.where(denom != 0, (a[:n] - b[:n]) / denom, 0.0)
    return result if result else None


# ═════════════════════════════════════════════
# GENERIC PLOTTING FUNCTIONS
# ═════════════════════════════════════════════

def plot_bar_by_group(data_by_group, title_str, file_str, ylabel='Freezing (%)',
                       ylim_top=None, show_points=True):
    """Bar chart with one bar per group (E, I, C)."""
    if data_by_group is None:
        return
    labels = [g for g in GROUP_ORDER if g in data_by_group]
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(2, 2) if for_paper else (4, 4))
    positions = list(range(1, len(labels) + 1))
    means, sems = [], []
    for pos, g in zip(positions, labels):
        vals = data_by_group[g]
        m = np.mean(vals)
        se = np.std(vals, ddof=0) / np.sqrt(len(vals))
        means.append(m); sems.append(se)
        ax.bar(pos, m, width=0.8, color=GROUP_COLORS[g])
        ax.errorbar(pos, m, yerr=se, fmt='none', ecolor=GROUP_COLORS[g],
                     capsize=0, linewidth=my_linewidth)
        if show_points:
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals,
                       s=my_sz, color=my_h, edgecolors='none', zorder=3, alpha=0.7)

    pairs, pvals, *_ = do_stats_cfc(data_by_group)
    if pairs:
        y0 = max(m + s for m, s in zip(means, sems)) + 3
        sigstar_text(ax, pairs, pvals, y_start=y0)

    ax.set_xticks(positions)
    ax.set_xticklabels([GROUP_LABELS[g] for g in labels], rotation=-30, ha='left')
    ax.set_ylabel(ylabel)
    ax.set_title(title_str)
    if ylim_top is not None:
        ax.set_ylim([0, ylim_top])
    _setup_axes(ax)
    fig.tight_layout()
    _save(fig, file_str)


def plot_context_disc(data_A, data_B, title_str, file_str):
    """Side-by-side Ctx A / Ctx B bars for each group, with within-group t-test."""
    if data_A is None or data_B is None:
        return
    labels = [g for g in GROUP_ORDER if g in data_A and g in data_B]
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(2, 2) if for_paper else (4, 4))
    pos = 1
    for g in labels:
        vA, vB = data_A[g], data_B[g]
        mA, seA = np.mean(vA), np.std(vA, ddof=0) / np.sqrt(len(vA))
        mB, seB = np.mean(vB), np.std(vB, ddof=0) / np.sqrt(len(vB))
        ax.bar(pos,     mA, width=0.8, color=GROUP_COLORS[g])
        ax.errorbar(pos, mA, yerr=seA, fmt='none', ecolor=GROUP_COLORS[g],
                     capsize=0, linewidth=my_linewidth)
        ax.bar(pos + 1, mB, width=0.8, color=GROUP_COLORS[g], alpha=0.45)
        ax.errorbar(pos + 1, mB, yerr=seB, fmt='none', ecolor=GROUP_COLORS[g],
                     capsize=0, linewidth=my_linewidth)
        p = _ttest_paired_or_ind(vA, vB)
        sigstar_text(ax, [[pos, pos + 1]], [p],
                     y_start=max(mA + seA, mB + seB) + 2, sep=2.5)
        pos += 3

    tp, tl = [], []
    p2 = 1
    for g in labels:
        tp.extend([p2, p2 + 1]); tl.extend(['Ctx A', 'Ctx B']); p2 += 3
    ax.set_xticks(tp)
    ax.set_xticklabels(tl, rotation=-30, ha='left')
    ax.set_ylabel('Freezing (%)')
    ax.set_title(title_str)
    _setup_axes(ax)
    fig.tight_layout()
    _save(fig, file_str)


def plot_paired_comparison(data_48hr, data_1wk, title_str, file_str,
                            ylabel='Freezing (%)', ylim_top=None):
    """Grouped bars: 48hr vs 1wk for each group, with paired t-test."""
    if data_48hr is None or data_1wk is None:
        return
    labels = [g for g in GROUP_ORDER if g in data_48hr and g in data_1wk]
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(2, 2) if for_paper else (4, 4))
    pos = 1
    all_pairs, all_pvals = [], []
    for g in labels:
        v48, v1w = data_48hr[g], data_1wk[g]
        m48, se48 = np.mean(v48), np.std(v48, ddof=0) / np.sqrt(len(v48))
        m1w, se1w = np.mean(v1w), np.std(v1w, ddof=0) / np.sqrt(len(v1w))
        ax.bar(pos,     m48, width=0.8, color=GROUP_COLORS[g])
        ax.errorbar(pos, m48, yerr=se48, fmt='none', ecolor=GROUP_COLORS[g],
                     capsize=0, linewidth=my_linewidth)
        ax.bar(pos + 1, m1w, width=0.8, color=GROUP_COLORS[g], alpha=0.5)
        ax.errorbar(pos + 1, m1w, yerr=se1w, fmt='none', ecolor=GROUP_COLORS[g],
                     capsize=0, linewidth=my_linewidth)
        all_pairs.append([pos, pos + 1])
        all_pvals.append(_ttest_paired_or_ind(v48, v1w))
        pos += 3

    adj = holm_bonferroni(all_pvals)
    sigstar_text(ax, all_pairs, adj, y_start=ax.get_ylim()[1] * 0.85)

    tp, tl = [], []
    p2 = 1
    for g in labels:
        tp.extend([p2, p2 + 1]); tl.extend(['48hr', '1wk']); p2 += 3
    ax.set_xticks(tp)
    ax.set_xticklabels(tl, rotation=-30, ha='left')
    ax.set_ylabel(ylabel)
    ax.set_title(title_str)
    if ylim_top is not None:
        ax.set_ylim([0, ylim_top])
    _setup_axes(ax)
    fig.tight_layout()
    _save(fig, file_str)


def plot_two_windows(data_early, data_late, title_str, file_str,
                     label_early='Early', label_late='Late',
                     ylabel='Freezing (%)', ylim_top=None):
    """Grouped bars: early vs late window for each group, with paired t-test."""
    if data_early is None or data_late is None:
        return
    labels = [g for g in GROUP_ORDER if g in data_early and g in data_late]
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(2, 2) if for_paper else (4, 4))
    pos = 1
    all_pairs, all_pvals = [], []
    for g in labels:
        ve, vl = data_early[g], data_late[g]
        me, see = np.mean(ve), np.std(ve, ddof=0) / np.sqrt(len(ve))
        ml, sel = np.mean(vl), np.std(vl, ddof=0) / np.sqrt(len(vl))
        ax.bar(pos,     me, width=0.8, color=GROUP_COLORS[g])
        ax.errorbar(pos, me, yerr=see, fmt='none', ecolor=GROUP_COLORS[g],
                     capsize=0, linewidth=my_linewidth)
        ax.bar(pos + 1, ml, width=0.8, color=GROUP_COLORS[g], alpha=0.5)
        ax.errorbar(pos + 1, ml, yerr=sel, fmt='none', ecolor=GROUP_COLORS[g],
                     capsize=0, linewidth=my_linewidth)
        all_pairs.append([pos, pos + 1])
        all_pvals.append(_ttest_paired_or_ind(ve, vl))
        pos += 3

    adj = holm_bonferroni(all_pvals)
    sigstar_text(ax, all_pairs, adj, y_start=ax.get_ylim()[1] * 0.85)

    tp, tl = [], []
    p2 = 1
    for g in labels:
        tp.extend([p2, p2 + 1]); tl.extend([label_early, label_late]); p2 += 3
    ax.set_xticks(tp)
    ax.set_xticklabels(tl, rotation=-30, ha='left')
    ax.set_ylabel(ylabel)
    ax.set_title(title_str)
    if ylim_top is not None:
        ax.set_ylim([0, ylim_top])
    _setup_axes(ax)
    fig.tight_layout()
    _save(fig, file_str)


def _print_summary(label, d):
    """Print mean ± SEM for each group."""
    if d is None:
        return
    parts = []
    for g in GROUP_ORDER:
        v = d.get(g)
        if v is not None:
            parts.append(f"{GROUP_LABELS[g]}: {np.mean(v):.1f} ± "
                         f"{np.std(v, ddof=0) / np.sqrt(len(v)):.1f}")
    if parts:
        print(f"  {label:30s}  {', '.join(parts)}")


# ═══════════════════════════════════════════════════
# PLOT 1 — CFC conditioning time series (always)
# ═══════════════════════════════════════════════════

if 'CFC' in sessions:
    fig1, ax1 = plt.subplots(
        figsize=(1.25, 0.75) if for_paper else (6, 4))

    for s_on, s_off in zip(shock_onsets, shock_offsets):
        ax1.axvspan(s_on, s_off, color=my_r, alpha=0.4, linewidth=0)

    for g in GROUP_ORDER:
        arr = sessions['CFC'].get(g)
        if arr is None:
            continue
        m, se = _mean_sem(arr)
        ax1.errorbar(onset_cfc, m, yerr=se, fmt='o-',
                      color=GROUP_COLORS[g], markerfacecolor=GROUP_COLORS[g],
                      linewidth=my_linewidth, markersize=my_markersize,
                      capsize=0, label=GROUP_LABELS[g])

    ticks = np.arange(60, CFC_DURATION, 60)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([str(int(t // 60)) for t in ticks])
    ax1.set_xlim([0, onset_cfc.max()])
    ax1.set_ylim([0, 85])
    if not for_paper:
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('Freezing (%)')
        ax1.set_title('CFC Conditioning')
    _setup_axes(ax1)
    fig1.tight_layout()
    _save(fig1, '01_CFC_conditioning')


# ═══════════════════════════════════════════════════
# PLOT 2 — CFC pre-shock baseline (always first 180s)
# ═══════════════════════════════════════════════════

cfc_baseline = session_mean('CFC', slice(0, PRE_SHOCK_BINS))
plot_bar_by_group(cfc_baseline, 'CFC Baseline (pre-shock)', '02_CFC_baseline')


# ═══════════════════════════════════════════════════
# PLOT — Test A / Test B time series (binned freezing)
# ═══════════════════════════════════════════════════

for sk, title, fname in [('TestA_48hr', 'Test A — 48 hr (Context A)', '03_TestA_48hr_timeseries'),
                          ('TestB_48hr', 'Test B — 48 hr (Context B)', '04_TestB_48hr_timeseries'),
                          ('TestA_1wk',  'Test A — 1 wk (Context A)',  '05_TestA_1wk_timeseries'),
                          ('TestB_1wk',  'Test B — 1 wk (Context B)',  '06_TestB_1wk_timeseries')]:
    if sk not in sessions:
        continue
    fig_ts, ax_ts = plt.subplots(
        figsize=(1.25, 0.75) if for_paper else (6, 4))
    for g in GROUP_ORDER:
        arr = sessions[sk].get(g)
        if arr is None:
            continue
        m, se = _mean_sem(arr)
        ax_ts.errorbar(onset_test, m, yerr=se, fmt='o-',
                        color=GROUP_COLORS[g], markerfacecolor=GROUP_COLORS[g],
                        linewidth=my_linewidth, markersize=my_markersize,
                        capsize=0, label=GROUP_LABELS[g])
    ticks = np.arange(60, TEST_DURATION, 60)
    ax_ts.set_xticks(ticks)
    ax_ts.set_xticklabels([str(int(t // 60)) for t in ticks])
    ax_ts.set_xlim([0, onset_test.max()])
    ax_ts.set_ylim([0, 100])
    if not for_paper:
        ax_ts.set_xlabel('Time (min)')
        ax_ts.set_ylabel('Freezing (%)')
        ax_ts.set_title(title)
    _setup_axes(ax_ts)
    fig_ts.tight_layout()
    _save(fig_ts, fname)


# ═══════════════════════════════════════════════════
# TEST SESSION PLOTS — three analysis windows
# ═══════════════════════════════════════════════════

has_1wk = ('TestA_1wk' in sessions or 'TestB_1wk' in sessions)

for win_slice, win_label, win_prefix in WINDOWS:
    print(f"\n─── Analysis window: {win_label} ───")

    tA48 = session_mean('TestA_48hr', win_slice)
    tB48 = session_mean('TestB_48hr', win_slice)
    tA1w = session_mean('TestA_1wk',  win_slice)
    tB1w = session_mean('TestB_1wk',  win_slice)
    di48 = discrimination_index(tA48, tB48)
    di1w = discrimination_index(tA1w, tB1w)

    pfx = win_prefix

    # Test A 48hr
    plot_bar_by_group(tA48, f'Test A — 48 hr  [{win_label}]',
                      f'{pfx}_03_TestA_48hr')
    # Test B 48hr
    plot_bar_by_group(tB48, f'Test B — 48 hr  [{win_label}]',
                      f'{pfx}_04_TestB_48hr')
    # Context discrimination 48hr
    plot_context_disc(tA48, tB48,
                      f'Context Disc. 48 hr  [{win_label}]',
                      f'{pfx}_05_context_disc_48hr')
    # DI 48hr
    plot_bar_by_group(di48, f'DI — 48 hr  [{win_label}]',
                      f'{pfx}_06_DI_48hr',
                      ylabel='DI (A−B)/(A+B)', ylim_top=1.0)

    if has_1wk:
        plot_bar_by_group(tA1w, f'Test A — 1 wk  [{win_label}]',
                          f'{pfx}_07_TestA_1wk')
        plot_bar_by_group(tB1w, f'Test B — 1 wk  [{win_label}]',
                          f'{pfx}_08_TestB_1wk')
        plot_context_disc(tA1w, tB1w,
                          f'Context Disc. 1 wk  [{win_label}]',
                          f'{pfx}_09_context_disc_1wk')
        plot_bar_by_group(di1w, f'DI — 1 wk  [{win_label}]',
                          f'{pfx}_10_DI_1wk',
                          ylabel='DI (A−B)/(A+B)', ylim_top=1.0)
        plot_paired_comparison(tA48, tA1w,
                               f'Test A 48hr vs 1wk  [{win_label}]',
                               f'{pfx}_11_TestA_48hr_vs_1wk')
        plot_paired_comparison(tB48, tB1w,
                               f'Test B 48hr vs 1wk  [{win_label}]',
                               f'{pfx}_12_TestB_48hr_vs_1wk')

    # Summary for this window
    print(f"  Summary [{win_label}]:")
    _print_summary(f'Test A 48hr ({win_label})', tA48)
    _print_summary(f'Test B 48hr ({win_label})', tB48)
    _print_summary(f'DI 48hr ({win_label})', di48)
    if has_1wk:
        _print_summary(f'Test A 1wk ({win_label})', tA1w)
        _print_summary(f'Test B 1wk ({win_label})', tB1w)
        _print_summary(f'DI 1wk ({win_label})', di1w)


# ═══════════════════════════════════════════════════
# EARLY vs LATE COMPARISONS
# ═══════════════════════════════════════════════════

print("\n─── Early vs Late Comparisons ───")

# Precompute the windows needed
first60  = {sk: session_mean(sk, slice(0, FIRST60_BINS)) for sk in SESSION_LABELS}
last60   = {sk: session_mean(sk, slice(TEST_BINS - LAST60_BINS, TEST_BINS)) for sk in SESSION_LABELS}
first180 = {sk: session_mean(sk, slice(0, PRE_SHOCK_BINS)) for sk in SESSION_LABELS}
last120  = {sk: session_mean(sk, slice(PRE_SHOCK_BINS, TEST_BINS)) for sk in SESSION_LABELS}

for test_sk, test_label, tp in [('TestA_48hr', 'Test A 48hr', '48hr'),
                                 ('TestB_48hr', 'Test B 48hr', '48hr'),
                                 ('TestA_1wk',  'Test A 1wk',  '1wk'),
                                 ('TestB_1wk',  'Test B 1wk',  '1wk')]:
    if test_sk not in sessions:
        continue

    # First 60 vs Last 60
    plot_two_windows(first60[test_sk], last60[test_sk],
                     f'{test_label} — First 60s vs Last 60s',
                     f'evl_{test_sk}_first60_vs_last60',
                     label_early='First 60', label_late='Last 60')
    _print_summary(f'{test_label} First 60s', first60[test_sk])
    _print_summary(f'{test_label} Last 60s',  last60[test_sk])

    # First 180 vs Last 120
    plot_two_windows(first180[test_sk], last120[test_sk],
                     f'{test_label} — First 180s vs Last 120s',
                     f'evl_{test_sk}_first180_vs_last120',
                     label_early='First 180', label_late='Last 120')
    _print_summary(f'{test_label} First 180s', first180[test_sk])
    _print_summary(f'{test_label} Last 120s',  last120[test_sk])


# ═══════════════════════════════════════════════════
# OVERALL SUMMARY
# ═══════════════════════════════════════════════════

print("\n─── CFC Baseline ───")
_print_summary('CFC pre-shock (0-180s)', cfc_baseline)

print("\nDone.")

# Close log file and restore stdout
sys.stdout = sys.__stdout__
_log_file.close()
print(f"Log saved to {os.path.join(file_path, 'output.txt')}")

plt.show()
pass
