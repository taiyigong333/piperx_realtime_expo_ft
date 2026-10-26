#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$PROJECT_ROOT"
role="${1:-core}"
ssh_command="$(git config --get core.sshCommand || true)"
if [[ -n "$ssh_command" && -z "${GIT_SSH_COMMAND:-}" ]]; then export GIT_SSH_COMMAND="$ssh_command"; fi
git_options=()
# 某些现场网络无法访问GitHub HTTPS；只对本次命令切换SSH，不改远程URL或全局Git配置。
if [[ "${PIPERX_GITHUB_SSH:-0}" == 1 ]]; then
  git_options=(-c 'url.git@github.com:.insteadOf=https://github.com/')
fi
# 固定提交不需要完整历史，避免部署端下载上游历史中的大文件。
git "${git_options[@]}" submodule update --init --depth 1 third_party/collector third_party/converter third_party/expo third_party/openpi
git "${git_options[@]}" submodule update --init --recursive --depth 1 third_party/collector
python3 scripts/bootstrap.py
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
case "$role" in
  core) "$UV" sync --locked --python 3.11 --group dev ;;
  server) "$UV" sync --locked --project server --python 3.11 ;;
  client) "$UV" sync --locked --project client --python 3.10 ;;
  *) echo '用法：bash scripts/setup.sh core|server|client' >&2; exit 2 ;;
esac
