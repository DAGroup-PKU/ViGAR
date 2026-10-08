#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
TARGET="${1:-$ROOT/.venv-model}"
TARGET="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$TARGET")"
command -v uv >/dev/null
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "The model environment requires Linux x86_64 and an NVIDIA GPU." >&2
  exit 1
fi
python3 - <<'CHECK'
import ctypes
libc = ctypes.CDLL(None)
libc.gnu_get_libc_version.restype = ctypes.c_char_p
version = tuple(map(int, libc.gnu_get_libc_version().decode().split('.')))
if version < (2, 34):
    raise SystemExit('glibc >= 2.34 is required; use Ubuntu 22.04 or newer.')
CHECK
export UV_PROJECT_ENVIRONMENT="$TARGET"
uv sync --project "$ROOT/vigar/source" --python 3.13 --extra gpu
"$TARGET/bin/python" "$ROOT/scripts/check_environment.py" --component models
