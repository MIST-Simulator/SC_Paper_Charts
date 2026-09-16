#!/usr/bin/env bash
# Run every experiment, then regenerate every figure.
#
#   bash run_all.sh                 # T1..T4 sequentially, full T3 sweep
#   bash run_all.sh --parallel      # all four task groups at once
#   bash run_all.sh --fast-eval     # reduced T3 search space (~2-4 h)
#   bash run_all.sh --plot-only     # skip the sweeps, plot from results/
#
# The task groups are mutually independent, so --parallel is safe; it needs
# roughly 4x the RAM of a sequential run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PARALLEL=0
PLOT_ONLY=0
T3_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --parallel)  PARALLEL=1 ;;
    --plot-only) PLOT_ONLY=1 ;;
    --fast-eval) T3_ARGS+=(--fast-eval) ;;
    -h|--help)   sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

LOG_DIR="$REPO_ROOT/results/logs"
mkdir -p "$LOG_DIR"

run_task() {
  local task="$1"; shift
  echo "==> [$task] starting experiment"
  python "run_${task}.py" "$@" 2>&1 | tee "$LOG_DIR/${task}.log"
  echo "==> [$task] experiment done"
}

if [[ "$PLOT_ONLY" -eq 0 ]]; then
  if [[ "$PARALLEL" -eq 1 ]]; then
    echo "==> Running T1-T4 in parallel (logs in $LOG_DIR)"
    run_task T1 & p1=$!
    run_task T2 & p2=$!
    run_task T3 "${T3_ARGS[@]+"${T3_ARGS[@]}"}" & p3=$!
    run_task T4 & p4=$!
    # Wait on each PID individually so one failure still surfaces its task.
    failed=0
    for pid_name in "T1:$p1" "T2:$p2" "T3:$p3" "T4:$p4"; do
      if ! wait "${pid_name#*:}"; then
        echo "error: ${pid_name%%:*} failed (see $LOG_DIR/${pid_name%%:*}.log)" >&2
        failed=1
      fi
    done
    [[ "$failed" -eq 0 ]] || exit 1
  else
    run_task T1
    run_task T2
    run_task T3 "${T3_ARGS[@]+"${T3_ARGS[@]}"}"
    run_task T4
  fi
else
  echo "==> --plot-only: skipping experiments"
fi

echo
echo "==> Regenerating figures"
for task in T1 T2 T3 T4; do
  python "plot_${task}.py"
done

echo
echo "==> All figures written to $REPO_ROOT/figures:"
ls -1 "$REPO_ROOT/figures"
