#!/usr/bin/env bash
# The Minian batch on the Razer, whole experiment days first (VS, 2026-09-28): every session of a day --
# HC, LT, CNO and the TFC/test session -- in this order of day groups, each group in mouse order:
#   TFC_cond, TFC_test_B, TFC_test_B_1wk, TFC_test_A, TFC_test_A_1wk, track days 1, 2, 3, then the rest
#   (track_day0/4/5, monitor1/2, the TFC_test_B_1wk-redux test day).
# One stream, 6 dask workers, staged through the copier (plans/session_staging_copier_plan.md); a failed
# session is recorded and the queue moves on. Sessions already done are skipped, so the script can be
# restarted at any point. Stop switch: touch ~/minian_stop -- the running session finishes and is copied
# back, no further session or group starts; delete the file before the next start.
# Patterns are regexes searched in <mouse>/<day>/<session> (run_minian_batch.py --labels): the trailing
# "/" ends the day name, so -TFC_test_B/ matches neither TFC_test_B_1wk nor TFC_test_B_1wk-redux.
# Run from the repo root in the caban env, inside screen and ~/bin/logrun.

STOP_FILE="$HOME/minian_stop"
DAY_GROUPS=(
  "TFC_cond days|-TFC_cond/"
  "TFC_test_B days|-TFC_test_B/"
  "TFC_test_B_1wk days|-TFC_test_B_1wk/"
  "TFC_test_A days|-TFC_test_A/ -TFC_test-A/"
  "TFC_test_A_1wk days|-TFC_test_A_1wk/"
  "track day 1|-track_day1/ -track_day1-tests/"
  "track day 2|-track_day2/"
  "track day 3|-track_day3/"
  "everything else|"
)

if [ -e "$STOP_FILE" ]; then
  echo "$STOP_FILE exists; delete it to start the batch"
  exit 1
fi
for group in "${DAY_GROUPS[@]}"; do
  name=${group%%|*}
  patterns=${group#*|}
  if [ -e "$STOP_FILE" ]; then
    echo "=== $STOP_FILE exists: stopping before the group '$name' ==="
    exit 0
  fi
  echo "=== batch: $name, $(date '+%F %T') ==="
  if [ -n "$patterns" ]; then
    # shellcheck disable=SC2086  # the patterns are meant to split into separate arguments
    python -u scripts/run_minian_batch.py --labels $patterns --n-workers 6
  else
    python -u scripts/run_minian_batch.py --n-workers 6
  fi
  echo "=== batch: $name ended with exit $?, $(date '+%F %T') ==="
done
