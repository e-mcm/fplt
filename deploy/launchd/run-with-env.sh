#!/usr/bin/env bash
# Load secrets for launchd agents, then exec the real command.
#
# Default file: ~/.config/fplt/env (KEY=value lines, same names as .env.example).
# Override with FPLT_ENV_FILE. Missing file is fine — live squad / alerts stay no-op.
set -euo pipefail

env_file="${FPLT_ENV_FILE:-${HOME}/.config/fplt/env}"
if [[ -f "${env_file}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a
fi

exec "$@"
