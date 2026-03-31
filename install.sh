#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$BUNDLE_ROOT/.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "python3 is required but was not found on PATH." >&2
  exit 1
fi

if [ ! -f "$BUNDLE_ROOT/pyproject.toml" ]; then
  echo "Expected pyproject.toml next to install.sh in $BUNDLE_ROOT." >&2
  exit 1
fi

if [ ! -d "$BUNDLE_ROOT/src/smallctl" ]; then
  echo "Expected source tree at $BUNDLE_ROOT/src/smallctl." >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip wheel setuptools
"$VENV_DIR/bin/python" -m pip install -e "$BUNDLE_ROOT"

cat <<EOF
smallctl is installed in: $VENV_DIR

Activate it with:
  source "$VENV_DIR/bin/activate"

Then run:
  smallctl --help

Copy "$BUNDLE_ROOT/.env.example" to "$BUNDLE_ROOT/.env" if you want local defaults.
EOF
