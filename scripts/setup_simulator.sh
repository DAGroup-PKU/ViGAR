#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
exec python3 "$ROOT/vigar/source/recipes/simulation/robotwin/common/setup.py" "$@"
