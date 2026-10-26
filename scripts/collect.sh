#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$PROJECT_ROOT"
config="${1:-configs/collection/rotating_square.yaml}"
config="$(realpath "$config")"
shift "$(( $# > 0 ? 1 : 0 ))"
# 复用原采集命令和数据规范；预检失败时不进入真实采集。
"$UV" run --project client --locked piper-x-collect preflight --config "$config"
"$UV" run --project client --locked piper-x-collect teleop-session --config "$config" "$@"
