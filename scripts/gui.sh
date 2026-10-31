#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$PROJECT_ROOT"
exec "$UV" run --project client --locked python -m piperx_expo.client.gui "$@"
