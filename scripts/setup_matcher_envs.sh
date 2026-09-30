#!/usr/bin/env bash
set -euo pipefail

# Reproducible environment bootstrap. If any pin changes, mark the corresponding
# pipeline [VERIFY] again until both probes pass on the intended hardware.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${1:-${project_root}/.venv/bin/python}"
env_root="${project_root}/.pipeline-envs"
vendor_root="${project_root}/vendor"
checkpoint_root="${project_root}/checkpoints"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python interpreter is not executable: ${python_bin}" >&2
  exit 2
fi

clone_exact() {
  local url="$1"
  local destination="$2"
  local commit="$3"
  if [[ ! -d "${destination}/.git" ]]; then
    git clone "${url}" "${destination}"
    git -C "${destination}" checkout --detach "${commit}"
  fi
  local actual
  actual="$(git -C "${destination}" rev-parse HEAD)"
  if [[ "${actual}" != "${commit}" ]]; then
    echo "Refusing existing ${destination}: expected ${commit}, found ${actual}" >&2
    exit 3
  fi
  if ! git -C "${destination}" diff --quiet; then
    echo "Refusing modified upstream checkout: ${destination}" >&2
    exit 3
  fi
}

download_exact() {
  local url="$1"
  local destination="$2"
  local expected="$3"
  mkdir -p "$(dirname "${destination}")"
  if [[ ! -f "${destination}" ]]; then
    local temporary="${destination}.partial"
    curl --fail --location --output "${temporary}" "${url}"
    mv "${temporary}" "${destination}"
  fi
  local actual
  actual="$(shasum -a 256 "${destination}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "Checksum mismatch for ${destination}: expected ${expected}, found ${actual}" >&2
    exit 4
  fi
}

mkdir -p "${env_root}" "${vendor_root}" "${checkpoint_root}"

clone_exact \
  https://github.com/verlab/accelerated_features.git \
  "${vendor_root}/accelerated_features" \
  e92685f57f8318b18725c5c8c0bd28c7fe188d9a
clone_exact \
  https://github.com/cvg/LightGlue.git \
  "${vendor_root}/LightGlue" \
  eb42fee2d71449efb0aa5c10549752b5d75384d8

download_exact \
  https://raw.githubusercontent.com/verlab/accelerated_features/e92685f57f8318b18725c5c8c0bd28c7fe188d9a/weights/xfeat.pt \
  "${checkpoint_root}/xfeat/xfeat.pt" \
  0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b
download_exact \
  https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth \
  "${checkpoint_root}/lightglue/hub/checkpoints/superpoint_v1.pth" \
  52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e
download_exact \
  https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth \
  "${checkpoint_root}/lightglue/hub/checkpoints/superpoint_lightglue_v0-1_arxiv.pth" \
  6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a

if [[ ! -x "${env_root}/xfeat/bin/python" ]]; then
  "${python_bin}" -m venv "${env_root}/xfeat"
fi
"${env_root}/xfeat/bin/python" -m pip install --upgrade pip
"${env_root}/xfeat/bin/python" -m pip install \
  --requirement "${project_root}/requirements-xfeat.lock"
"${env_root}/xfeat/bin/python" -m pip freeze > "${env_root}/xfeat.freeze.txt"

if [[ ! -x "${env_root}/sp_lightglue/bin/python" ]]; then
  "${python_bin}" -m venv "${env_root}/sp_lightglue"
fi
"${env_root}/sp_lightglue/bin/python" -m pip install --upgrade pip
"${env_root}/sp_lightglue/bin/python" -m pip install \
  --requirement "${project_root}/requirements-sp-lightglue.lock"
"${env_root}/sp_lightglue/bin/python" -m pip install \
  --no-deps --editable "${vendor_root}/LightGlue"
"${env_root}/sp_lightglue/bin/python" -m pip freeze > "${env_root}/sp_lightglue.freeze.txt"

"${project_root}/.venv/bin/python" -m warpaudit probe-adapter \
  --config "${project_root}/configs/pilot.yaml" --pipeline xfeat_h \
  --output "${project_root}/reports/xfeat_adapter_probe.json"
"${project_root}/.venv/bin/python" -m warpaudit probe-adapter \
  --config "${project_root}/configs/pilot.yaml" --pipeline sp_lg_h \
  --output "${project_root}/reports/sp_lg_adapter_probe.json"

echo "Matcher probes passed. Resolved package sets are under ${env_root}."
