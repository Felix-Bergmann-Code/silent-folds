#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$workspace_dir"

if [[ -x .venv/bin/python ]]; then
  eval_python=.venv/bin/python
  lint_command=(.venv/bin/ruff)
else
  eval_python=python
  lint_command=(ruff)
fi

"$eval_python" -m warpaudit validate-config --config configs/pilot.yaml
"${lint_command[@]}" check warpaudit tests
"$eval_python" -m pytest -q
"$eval_python" -m pip check
