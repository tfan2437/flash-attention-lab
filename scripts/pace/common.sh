# Sourced by the PACE scripts after env.sh. start_run <name> [args...] creates
# runs/<YYYYMMDD-HHMMSS>-<sha7>[-dirty]-<name>/ with the command line and environment, tees the
# rest of the script's output into stdout.log, and writes summary.md when the script exits.

start_run() {
  local name="$1"
  shift
  mkdir -p runs bench/results profiling
  GIT_SHA="$(git rev-parse --short=7 HEAD 2>/dev/null || echo nogit)"
  if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    GIT_SHA="${GIT_SHA}-dirty"
  fi
  RUN_DIR="runs/$(date +%Y%m%d-%H%M%S)-${GIT_SHA}-${name}"
  mkdir -p "$RUN_DIR"
  echo "$name $*" >"$RUN_DIR/cmd.txt"
  python -m bench.env >"$RUN_DIR/env.json" 2>"$RUN_DIR/env.err" || true
  exec > >(tee -a "$RUN_DIR/stdout.log") 2>&1
  trap 'finish_run $?' EXIT
  echo "run dir: $RUN_DIR"
}

finish_run() {
  local status="$1"
  sleep 1 # let tee flush before the log is summarized
  {
    echo "# $(basename "$RUN_DIR")"
    echo
    echo "- command: $(cat "$RUN_DIR/cmd.txt")"
    echo "- exit status: $status"
    echo "- host: $(hostname), slurm job: ${SLURM_JOB_ID:-none}"
    echo
    echo '```'
    tail -n 60 "$RUN_DIR/stdout.log"
    echo '```'
  } >"$RUN_DIR/summary.md"
}
