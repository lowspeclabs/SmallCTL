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

# Ensure optional but recommended system packages are available.
# These are used by apt/deb822 validation guards.
ensure_python_package() {
  local pkg="$1"
  local import_name="${2:-$pkg}"
  if ! "$VENV_DIR/bin/python" -c "import $import_name" 2>/dev/null; then
    echo "Package '$pkg' (import: $import_name) not found. Attempting install..."
    if "$VENV_DIR/bin/python" -m pip install "$pkg" 2>/dev/null; then
      echo "  -> installed $pkg via pip"
    else
      echo "  -> pip install failed for $pkg. You may need to install it via your system package manager:"
      echo "       apt-get install python3-${pkg#python-}   # Debian/Ubuntu"
      echo "       dnf install python3-$pkg                 # Fedora/RHEL"
    fi
  fi
}

ensure_python_package "python-debian" "debian.deb822"
# python-apt is typically a system package; warn if absent.
if ! "$VENV_DIR/bin/python" -c "import apt_pkg" 2>/dev/null; then
  echo "Note: 'python-apt' is not available. Install it via: apt-get install python3-apt"
fi

cat <<EOF
smallctl is installed in: $VENV_DIR

Activate it with:
  source "$VENV_DIR/bin/activate"

Then run:
  smallctl --help

Copy "$BUNDLE_ROOT/.env.example" to "$BUNDLE_ROOT/.env" if you want local defaults.
EOF
