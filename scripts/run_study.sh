#!/usr/bin/env bash
# Single unattended entry point for the whole executable study.
#
#   ./scripts/run_study.sh --download --acknowledge-fire-terms-unresolved
#
# Stage sequencing, resume behaviour, provenance hashes, and the run report all
# live in `warpaudit run-study`; this wrapper only chooses the interpreter,
# verifies the evaluation environment, and keeps a durable transcript so an
# overnight run is readable after the terminal is gone.
#
# To survive a disconnect, launch it detached:
#   nohup ./scripts/run_study.sh --download --acknowledge-fire-terms-unresolved \
#     > /dev/null 2>&1 &

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

config="configs/pilot.yaml"
run_verification=1
run_setup=0
forwarded=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      config="$2"
      shift 2
      ;;
    --no-verify)
      run_verification=0
      shift
      ;;
    --setup)
      run_setup=1
      shift
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: ./scripts/run_study.sh [--setup] [--no-verify] [run-study options]

  --setup       create .venv and bootstrap the isolated matcher environments
                first, so a fresh clone needs only this one command
  --no-verify   skip lint/tests/pip-check before the run (not recommended on a
                machine that has not run them yet)

Common run-study options:
  --download                            retrieve the checksum-pinned archives
  --acknowledge-fire-terms-unresolved   required for any local FIRE extraction
  --signed-off-by NAME --review-note T  pass the G1 gate and run the confirmatory
                                        evaluation; without both, the run stops
                                        at the M2 feasibility screen
  --acknowledge-infeasible              freeze a direction the screen blocks
  --development-only                    omit the confirmatory sweeps
  --dry-run                             print the resolved plan and exit
  --start-at STAGE / --only STAGE ...   resume or narrow the run
  --keep-going                          continue past a failed stage

Full option list: python -m warpaudit run-study --help
USAGE
      exit 0
      ;;
    *)
      forwarded+=("$1")
      shift
      ;;
  esac
done

if (( run_setup )) && [[ ! -x .venv/bin/python ]]; then
  echo "== creating the evaluation environment =="
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -c requirements-eval.lock -e '.[dev]'
  echo
fi

if [[ -x .venv/bin/python ]]; then
  eval_python=.venv/bin/python
else
  cat >&2 <<'MISSING'
No evaluation environment at .venv. Create it before starting a long run:

  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -c requirements-eval.lock -e '.[dev]'
MISSING
  exit 2
fi

if (( run_setup )); then
  # Idempotent: existing checkouts, weights, and environments are verified
  # rather than rebuilt, and the probes are the gate on whether this machine
  # reproduces the pinned matchers at all.
  echo "== bootstrapping the isolated matcher environments =="
  ./scripts/setup_matcher_envs.sh
  echo
fi

if [[ ! -d .pipeline-envs ]]; then
  cat >&2 <<'MISSING'
No isolated matcher environments at .pipeline-envs. The registration stages
cannot run without them. Bootstrap them on this machine first:

  ./scripts/setup_matcher_envs.sh
MISSING
  exit 2
fi

log_dir="reports/study_runs"
mkdir -p "$log_dir"
transcript="${log_dir}/run-$(date -u +%Y%m%dT%H%M%SZ).log"
ln -sfn "$(basename "$transcript")" "${log_dir}/latest.log"

echo "Project:    $project_root"
echo "Config:     $config"
echo "Transcript: $transcript"
echo

# The transcript is the record an operator reads in the morning; everything the
# run prints, including stage failures, has to reach it.
exec > >(tee -a "$transcript") 2>&1

if (( run_verification )); then
  echo "== verifying the evaluation environment =="
  ./scripts/verify_setup.sh
  echo
fi

echo "== running the study =="
set +e
"$eval_python" -m warpaudit run-study --config "$config" \
  ${forwarded[@]+"${forwarded[@]}"}
status=$?
set -e

echo
if (( status == 0 )); then
  echo "Study run completed. Summary: reports/STUDY_RUN.md"
else
  echo "Study run finished with exit status ${status}. Summary: reports/STUDY_RUN.md" >&2
  echo "Stage logs are under ${log_dir}/; rerun with --start-at STAGE to resume." >&2
fi
exit "$status"
