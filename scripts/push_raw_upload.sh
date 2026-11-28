#!/usr/bin/env bash
# Upload local raw captures to an rclone destination; set uploaded on success.
#
# Used by .github/actions/push-raw (#340). Sets ``uploaded=true`` on
# ``$GITHUB_OUTPUT`` only after ``rclone copy`` (and ``rclone check --one-way``)
# exit 0. Empty source directories count as success (nothing to transfer).
#
# Usage: push_raw_upload.sh <src_dir> <rclone_dest>
#   e.g. push_raw_upload.sh "$RUNNER_TEMP/raw" "fplt:fplt-data/raw"
set -euo pipefail

src="${1:?src directory required}"
dest="${2:?rclone destination required}"

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

set +e
rclone copy "$src" "$dest" --immutable --fast-list --stats-one-line -v
copy_rc=$?
set -e

if [ "$copy_rc" -ne 0 ]; then
  fail_upload "rclone copy failed (exit ${copy_rc})" "$copy_rc"
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
