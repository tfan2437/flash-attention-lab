# Sourced by the PACE scripts after env.sh.
#
# run_logged <name> <command...> creates runs/<YYYYMMDD-HHMMSS>-<sha7>[-dirty]-<name>/ with the
# command line and environment, runs the command with its output teed into stdout.log, writes
# summary.md, and returns the command's exit status. The pipeline is waited on in the foreground,
# so no output is lost when srun tears the job step down.

# Uncommitted changes to tracked files other than outputs (results and profiler digests), the
# same rule as bench/env.py: a run that regenerates a digest does not make later runs dirty.
code_dirty() {
  [ -n "$(git status --porcelain --untracked-files=no -- . ':!bench/results' ':!profiling' \
    2>/dev/null)" ]
}

start_run_dir() {
  local name="$1"
  shift
  mkdir -p runs bench/results profiling
  GIT_SHA="$(git rev-parse --short=7 HEAD 2>/dev/null || echo nogit)"
  if code_dirty; then
    GIT_SHA="${GIT_SHA}-dirty"
  fi
  RUN_DIR="runs/$(date +%Y%m%d-%H%M%S)-${GIT_SHA}-${name}"
  mkdir -p "$RUN_DIR"
  echo "$name $*" >"$RUN_DIR/cmd.txt"
  python -m bench.env >"$RUN_DIR/env.json" 2>"$RUN_DIR/env.err" || true
  export GIT_SHA RUN_DIR
}

write_summary() {
  local status="$1"
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

run_logged() {
  local name="$1"
  shift
  start_run_dir "$name" "$@"
  echo "run dir: $RUN_DIR" | tee "$RUN_DIR/stdout.log"
  set +e
  "$@" 2>&1 | tee -a "$RUN_DIR/stdout.log"
  local status="${PIPESTATUS[0]}"
  set -e
  write_summary "$status"
  return "$status"
}
