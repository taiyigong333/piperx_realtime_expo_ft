#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$PROJECT_ROOT"
role="${1:-core}"
ssh_command="$(git config --get core.sshCommand || true)"
if [[ -n "$ssh_command" && -z "${GIT_SSH_COMMAND:-}" ]]; then export GIT_SSH_COMMAND="$ssh_command"; fi
git submodule update --init third_party/collector third_party/converter third_party/expo third_party/openpi
git submodule update --init --recursive third_party/collector
python3 scripts/bootstrap.py
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
case "$role" in
  core) "$UV" sync --python 3.11 --group dev ;;
  server) "$UV" sync --project server --python 3.11 ;;
  client) "$UV" sync --project client --python 3.10 ;;
  *) echo '用法：bash scripts/setup.sh core|server|client' >&2; exit 2 ;;
esac
