#!/usr/bin/env bash
# Build an fplt-mc wheel into dist/ (repo root or mc/dist → copied to ../dist).
# Used by `make mc-ci` and GitHub Actions. Prefer GHA for release wheels.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MC="${ROOT}/mc"
DIST="${ROOT}/dist"
mkdir -p "${DIST}"

export CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)}"

cd "${ROOT}"
# Prefer explicit python (manylinux /opt/python), then uv, then python3.
PY="${FPLT_MC_PYTHON:-}"
if [[ -z "${PY}" ]] && command -v uv >/dev/null 2>&1 && [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
elif [[ -z "${PY}" ]] && command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
elif [[ -z "${PY}" ]] && command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
fi
if [[ -z "${PY}" ]]; then
  echo "build_wheel: no Python found" >&2
  exit 1
fi
"${PY}" -m pip install -q build scikit-build-core nanobind 2>/dev/null || true
"${PY}" -m build --wheel --outdir "${DIST}" "${MC}"

plat="$(uname -s)-$(uname -m)"
echo "Built wheel(s) for ${plat}:"
ls -la "${DIST}"/*.whl

# Linux: auditwheel repair into manylinux tag when auditwheel is available.
if [[ "$(uname -s)" == "Linux" ]] && command -v auditwheel >/dev/null 2>&1; then
  for whl in "${DIST}"/fplt_mc-*.whl; do
    if [[ "${whl}" == *manylinux* ]]; then
      continue
    fi
    auditwheel repair "${whl}" -w "${DIST}" || echo "auditwheel repair skipped for ${whl}"
  done
fi

# macOS: delocate when available.
if [[ "$(uname -s)" == "Darwin" ]] && command -v delocate-wheel >/dev/null 2>&1; then
  export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"
  for whl in "${DIST}"/fplt_mc-*.whl; do
    delocate-wheel -w "${DIST}" -v "${whl}" || echo "delocate skipped for ${whl}"
  done
fi
