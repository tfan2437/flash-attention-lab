# Run logs

PACE scripts write one directory per invocation here (ignored by git):
`<timestamp>-<sha>-<name>/` with `cmd.txt`, `env.json`, `stdout.log`, and `summary.md`.
Outputs worth keeping are copied by the scripts to `bench/results/` and `profiling/`.
