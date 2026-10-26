#!/usr/bin/env bash
# 统一查找 uv，兼容远程非交互 SSH 未加载 ~/.local/bin 的情况。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "$UV" && -x "$HOME/.local/bin/uv" ]]; then UV="$HOME/.local/bin/uv"; fi
if [[ -z "$UV" ]]; then echo '请先安装 uv（https://docs.astral.sh/uv/）' >&2; exit 1; fi
