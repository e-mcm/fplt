#!/usr/bin/env bash
# Fail if mc/** changed vs base but mc/pyproject.toml version did not.
# Usage: version_guard.sh [base_ref]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE="${1:-origin/main}"

cd "${ROOT}"
if ! git rev-parse --verify "${BASE}" >/dev/null 2>&1; then
  echo "version_guard: base ${BASE} missing; skip"
  exit 0
fi

changed="$(git diff --name-only "${BASE}"...HEAD -- mc/ || true)"
if [[ -z "${changed}" ]]; then
  echo "version_guard: no mc/ changes"
  exit 0
fi

if ! echo "${changed}" | grep -q '^mc/pyproject.toml$'; then
  # mc/ changed but pyproject.toml not in the diff at all
  if git diff --name-only "${BASE}"...HEAD -- mc/pyproject.toml | grep -q .; then
    :
  else
    echo "version_guard: mc/** changed but mc/pyproject.toml was not modified."
    echo "Bump project.version in mc/pyproject.toml (wheels are immutable)."
    exit 1
  fi
fi

# Ensure the version string itself changed.
old="$(git show "${BASE}:mc/pyproject.toml" 2>/dev/null | grep -E '^version = ' | head -1 || true)"
new="$(grep -E '^version = ' mc/pyproject.toml | head -1 || true)"
if [[ -n "${old}" && "${old}" == "${new}" ]]; then
  echo "version_guard: mc/** changed but version is still ${new}"
  echo "Bump project.version in mc/pyproject.toml."
  exit 1
fi

echo "version_guard: ok (${old:-<none>} -> ${new})"
