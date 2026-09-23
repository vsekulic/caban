#!/bin/bash
# Consolidate both Miniscope backup drives onto one APFS volume.
#
# THE ORIGINALS ARE TO BE WIPED AFTERWARDS, so this copies EVERYTHING and
# excludes nothing but `.DS_Store` (Finder window state, created by browsing the
# drives on this Mac -- not original data). Anything else that exists on a source
# drive lands on the destination, including the Windows recycle bins, which hold
# 36 genuinely deleted files totalling 451 MB -- among them a 370 MB .avi.
#
# Restartable: re-run and rsync resumes. Nothing is deleted from the sources.
#
#   *** DO NOT WIPE EITHER SOURCE DRIVE UNTIL `verify` PASSES CLEAN. ***
#   Until then the source drives are the only backup that exists.
#
# Measured 2026-09-22 across both drives:
#   data/                 3606.4 GB   828,261 files   (3262 GB of it .avi)
#   $RECYCLE.BIN (1a)        0.451 GB     163 files   (36 real deleted files)
#   $RECYCLE.BIN (1b)       ~0     GB      22 files
#   System Volume Information 0.028 GB       8 files   (Windows volume metadata)
#
# See plans/local_minian_pipeline_plan.md §5 for the layout rationale.

set -u

# ---- set this to the new volume, then run ----------------------------------
DEST=${DEST:-/Volumes/MINISCOPE}
# ----------------------------------------------------------------------------

SRC_A=/Volumes/1a-MINISCOPE-BAK
SRC_B=/Volumes/1b-MINISCOPE-BAK
LOG="$DEST/consolidate.log"
NEED_GB=3700   # refuse to start without room for the whole thing plus headroom

# macOS ships openrsync, not GNU rsync: no --info=progress2, no --no-perms/--no-owner/
# --no-group. `-rlt` already declines to copy perms/owner/group, so those were redundant.
# Per-file output is suppressed deliberately -- 828k lines would bury the per-mouse
# --stats blocks that make this log readable.
R="rsync -rlt --partial --stats --exclude=.DS_Store"

say() { echo -e "\n========== $* ==========" | tee -a "$LOG"; date | tee -a "$LOG"; }
die() { echo "ERROR: $*" >&2; exit 1; }

[ -d "$DEST" ]  || die "destination not mounted: $DEST   (run as: DEST=/Volumes/<name> $0)"
[ -d "$SRC_A" ] || die "source not mounted: $SRC_A"
[ -d "$SRC_B" ] || die "source not mounted: $SRC_B"
touch "$DEST/.write_test" 2>/dev/null || die "$DEST is not writable"
rm -f "$DEST/.write_test"

FREE_GB=$(df -g "$DEST" | awk 'NR==2 {print $4}')
[ "$FREE_GB" -ge "$NEED_GB" ] || die "only ${FREE_GB} GB free on $DEST, need >= ${NEED_GB} GB"

# `data/` is the one path present on both drives; the mouse directories beneath
# it are disjoint (1a: G01-G11, 1b: G12-G23) except `baseplating`, which exists
# on both with different contents and is therefore kept apart rather than merged.
copy_data() {
  local src="$1" tag="$2"
  local base="$src/data/vsekulic/OF_test"
  # The server-era `data/vsekulic/OF_test` prefix is dropped here: "OF test" means open
  # field test, which this project has never been. Everything lands under SSTCa2/ and is
  # sorted into FRAM/ or _misc/ afterwards -- instant on the same APFS volume.
  local out="$DEST/SSTCa2"
  mkdir -p "$out"
  for m in "$base"/*/ ; do
    local name; name=$(basename "$m")
    if [ "$name" = "baseplating" ]; then
      $R "$m" "$out/baseplating-from-$tag/" 2>&1 | tee -a "$LOG"
    else
      $R "$m" "$out/$name/" 2>&1 | tee -a "$LOG"
    fi
  done
}

# Everything at a drive's root that is not `data/`: the recycle bin, the Windows
# volume metadata, stray dotfiles. Namespaced per drive because both drives carry
# the same names.
copy_extras() {
  local src="$1" tag="$2"
  local out="$DEST/_drive_roots/$tag"
  mkdir -p "$out"
  $R --exclude='/data' "$src/" "$out/" 2>&1 | tee -a "$LOG"
}

WHAT=${1:-all}
DO_DATA=0; DO_EXTRAS=0; DO_VERIFY=0
case "$WHAT" in
  data)   DO_DATA=1 ;;
  extras) DO_EXTRAS=1 ;;
  verify) DO_VERIFY=1 ;;
  all)    DO_DATA=1; DO_EXTRAS=1; DO_VERIFY=1 ;;
  *)      die "usage: DEST=/Volumes/<name> $0 [data|extras|verify|all]" ;;
esac

if [ "$DO_DATA" = 1 ]; then
  say "STAGE 1: all of data/ from both drives -- 3606 GB, 828k files, expect 6-12 h"
  copy_data "$SRC_A" 1a
  copy_data "$SRC_B" 1b
  say "STAGE 1 done"
fi

if [ "$DO_EXTRAS" = 1 ]; then
  say "STAGE 2: drive roots -- recycle bins (451 MB of real deleted files), volume metadata"
  copy_extras "$SRC_A" 1a
  copy_extras "$SRC_B" 1b
  say "STAGE 2 done"
fi

if [ "$DO_VERIFY" = 1 ]; then
  say "STAGE 3: verify by checksum -- reads every byte on both sides, expect many hours"
  for pair in "$SRC_A/data/vsekulic/OF_test:1a" "$SRC_B/data/vsekulic/OF_test:1b"; do
    src="${pair%%:*}"; tag="${pair##*:}"
    for m in "$src"/*/ ; do
      name=$(basename "$m")
      [ "$name" = "baseplating" ] && name="baseplating-from-$tag"
      $R -n -c "$m" "$DEST/SSTCa2/$name/" 2>&1 | tee -a "$LOG"
    done
  done
  for pair in "$SRC_A:1a" "$SRC_B:1b"; do
    src="${pair%%:*}"; tag="${pair##*:}"
    $R -n -c --exclude='/data' "$src/" "$DEST/_drive_roots/$tag/" 2>&1 | tee -a "$LOG"
  done
  say "STAGE 3 done"
  echo "Any file PATH listed above (as opposed to the summary stats) differs and must" | tee -a "$LOG"
  echo "be re-copied. Only wipe a source drive once this stage lists no files." | tee -a "$LOG"
fi
