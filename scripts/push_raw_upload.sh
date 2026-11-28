#!/usr/bin/env bash
# Upload local raw captures to an rclone destination; set uploaded on success.
#
# Used by .github/actions/push-raw (#340). Sets ``uploaded=true`` on
# ``$GITHUB_OUTPUT`` only after ``rclone copy`` (and ``rclone check --one-way``)
# exit 0. Empty source directories count as success (nothing to transfer).
#
# Completeness (#353): upload everything except ``_source.json`` first, then
# ``_source.json`` as a completion marker (like ``_SUCCESS`` for runs). Defaults
# ``--transfers 32 --checkers 32`` (overridable via ``RCLONE_TRANSFERS`` /
# ``RCLONE_CHECKERS``) so many-small-object captures (vaastav) finish in time.
#
# Usage: push_raw_upload.sh <src_dir> <rclone_dest>
#   e.g. push_raw_upload.sh "$RUNNER_TEMP/raw" "fplt:fplt-data/raw"
set -euo pipefail

src="${1:?src directory required}"
dest="${2:?rclone destination required}"
transfers="${RCLONE_TRANSFERS:-32}"
checkers="${RCLONE_CHECKERS:-32}"

write_uploaded() {
  local value="$1"
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "uploaded=${value}" >>"$GITHUB_OUTPUT"
  fi
}

fail_upload() {
  local msg="$1"
  local rc="$2"
  echo "::error::${msg}"
  write_uploaded false
  exit "${rc}"
}

if [ ! -d "$src" ] || [ -z "$(find "$src" -type f | head -n 1)" ]; then
  echo "nothing new to upload"
  write_uploaded true
  exit 0
fi

# Defence in depth (#339): never upload '.'-prefixed path components
# (e.g. .vaastav_cache) into write-once raw/.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! python3 "${script_dir}/reject_raw_dot_paths.py" "$src"; then
  fail_upload "raw source contains path components starting with '.'" 1
fi

rclone_common=(
  --immutable
  --fast-list
  --stats-one-line
  -v
  --transfers "${transfers}"
  --checkers "${checkers}"
)

# Pass 1: payload without the completion marker (#353).
set +e
rclone copy "$src" "$dest" "${rclone_common[@]}" --exclude '**/_source.json'
copy_rc=$?
set -e

if [ "$copy_rc" -ne 0 ]; then
  fail_upload "rclone copy (payload) failed (exit ${copy_rc})" "$copy_rc"
fi

# Pass 2: ``_source.json`` last so a timed-out upload cannot look complete.
set +e
rclone copy "$src" "$dest" "${rclone_common[@]}" --include '**/_source.json'
marker_rc=$?
set -e

if [ "$marker_rc" -ne 0 ]; then
  fail_upload "rclone copy (_source.json) failed (exit ${marker_rc})" "$marker_rc"
fi

# Prefer a one-way check over parsing retry ERROR lines from the copy log (#340).
set +e
rclone check --one-way "$src" "$dest" --fast-list
check_rc=$?
set -e

if [ "$check_rc" -ne 0 ]; then
  fail_upload "rclone check --one-way failed (exit ${check_rc})" "$check_rc"
fi

write_uploaded true
