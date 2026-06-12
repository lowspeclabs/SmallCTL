#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$BUNDLE_ROOT/.venv}"
SMALLCTL_DEV_MODE="${SMALLCTL_DEV_MODE:-}"

# ------------------------------------------------------------------
# CLI introspection flags  (Issue 6)
# ------------------------------------------------------------------
show_help() {
  cat <<EOF
Usage: install.sh [OPTION]

Install smallctl and its dependencies into an isolated virtual environment.

Options:
  -h, --help        Show this help message and exit
  -V, --version     Print the package version and exit
  --check           Check prerequisites without installing anything
  --no-env-setup    Skip the interactive .env setup wizard
  --env-only        Run the .env setup wizard and exit (skips install)
EOF
}

show_version() {
  "$PYTHON_BIN" -c "
import tomllib
with open('$BUNDLE_ROOT/pyproject.toml', 'rb') as f:
    print(tomllib.load(f)['project']['version'])
"
}

check_prereqs() {
  local ok=0

  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "FAIL: $PYTHON_BIN not found on PATH" >&2
    ok=1
  else
    local pyver
    pyver=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if "$PYTHON_BIN" -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)"; then
      echo "PASS: Python $pyver (>= 3.10)"
    else
      echo "FAIL: Python $pyver (< 3.10 required)" >&2
      ok=1
    fi
  fi

  if [ -f "$BUNDLE_ROOT/pyproject.toml" ]; then
    echo "PASS: pyproject.toml found"
  else
    echo "FAIL: pyproject.toml not found in $BUNDLE_ROOT" >&2
    ok=1
  fi

  if [ -d "$BUNDLE_ROOT/src/smallctl" ]; then
    echo "PASS: source tree found at src/smallctl"
  else
    echo "FAIL: src/smallctl not found in $BUNDLE_ROOT" >&2
    ok=1
  fi

  if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    echo "PASS: pip available"
  else
    echo "FAIL: pip not available for $PYTHON_BIN" >&2
    ok=1
  fi

  exit "$ok"
}

NO_ENV_SETUP=0
ENV_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --help|-h) show_help; exit 0 ;;
    --version|-V) show_version; exit 0 ;;
    --check) check_prereqs; exit 0 ;;
    --no-env-setup) NO_ENV_SETUP=1 ;;
    --env-only) ENV_ONLY=1 ;;
  esac
done

# --env-only skips the install and jumps straight to env setup
if [ "$ENV_ONLY" -eq 1 ]; then
  _setup_dotenv
  exit 0
fi

# ------------------------------------------------------------------
# Issue 1: Python presence and minimum version guard
# ------------------------------------------------------------------
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: $PYTHON_BIN is required but was not found on PATH." >&2
  exit 1
fi

PYVER=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if ! "$PYTHON_BIN" -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)"; then
  echo "Error: Python >= 3.10 required (found $PYVER)." >&2
  exit 1
fi
echo "Python $PYVER (>= 3.10 OK)"

# ------------------------------------------------------------------
# Source tree integrity (carried over from original)
# ------------------------------------------------------------------
if [ ! -f "$BUNDLE_ROOT/pyproject.toml" ]; then
  echo "Error: Expected pyproject.toml next to install.sh in $BUNDLE_ROOT." >&2
  exit 1
fi

if [ ! -d "$BUNDLE_ROOT/src/smallctl" ]; then
  echo "Error: Expected source tree at $BUNDLE_ROOT/src/smallctl." >&2
  exit 1
fi

# ------------------------------------------------------------------
# Issue 2: Virtual environment with stale-Python detection
# ------------------------------------------------------------------
if [ -d "$VENV_DIR" ]; then
  VENV_PYTHON="$VENV_DIR/bin/python"
  if [ -f "$VENV_PYTHON" ]; then
    VENV_VER=$("$VENV_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [ "$VENV_VER" != "$PYVER" ]; then
      echo "Python version mismatch: venv uses $VENV_VER but $PYTHON_BIN is $PYVER. Recreating venv..."
      rm -rf "$VENV_DIR"
    fi
  else
    echo "Incomplete venv detected. Recreating..."
    rm -rf "$VENV_DIR"
  fi
fi

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  echo "Created virtual environment at $VENV_DIR"
fi

# ------------------------------------------------------------------
# Upgrade core packaging tools
# ------------------------------------------------------------------
"$VENV_DIR/bin/python" -m pip install --upgrade pip wheel setuptools

# ------------------------------------------------------------------
# Issue 4: Install smallctl (non-editable for users, editable for devs)
# ------------------------------------------------------------------
if [ -n "$SMALLCTL_DEV_MODE" ]; then
  "$VENV_DIR/bin/python" -m pip install -e "$BUNDLE_ROOT"
  echo "smallctl installed in editable mode (SMALLCTL_DEV_MODE)."
else
  "$VENV_DIR/bin/python" -m pip install "$BUNDLE_ROOT"
  echo "smallctl installed."
fi

# ------------------------------------------------------------------
# Issue 3: Resolve python-debian (required for apt/deb822 validation)
# ------------------------------------------------------------------
_resolve_python_debian() {
  if "$VENV_DIR/bin/python" -c "import debian.deb822" 2>/dev/null; then
    return 0
  fi

  echo "Installing python-debian (required for apt sources validation)..."

  # Try pip first — works on most platforms with build dependencies
  if "$VENV_DIR/bin/python" -m pip install python-debian 2>/dev/null; then
    echo "  -> installed via pip"
    return 0
  fi

  # Debian/Ubuntu — use the pre-compiled system package
  if command -v apt-get >/dev/null 2>&1; then
    if [ "$(id -u)" -eq 0 ]; then
      apt-get install -y python3-debian && echo "  -> installed via apt" && return 0
    elif command -v sudo >/dev/null 2>&1; then
      sudo apt-get install -y python3-debian && echo "  -> installed via apt" && return 0
    fi
  fi

  # Fedora/RHEL
  if command -v dnf >/dev/null 2>&1; then
    if [ "$(id -u)" -eq 0 ]; then
      dnf install -y python3-debian && echo "  -> installed via dnf" && return 0
    elif command -v sudo >/dev/null 2>&1; then
      sudo dnf install -y python3-debian && echo "  -> installed via dnf" && return 0
    fi
  fi

  cat >&2 <<MSG
Error: Could not install python-debian. Please install it manually:
  Debian/Ubuntu:  sudo apt-get install python3-debian
  Fedora/RHEL:    sudo dnf install python3-debian
  pip (requires build tools: gcc, python3-dev, libbz2-dev):
                  pip install python-debian
MSG
  return 1
}

_resolve_python_debian

# ------------------------------------------------------------------
# .env setup wizard (runs if .env is missing and not suppressed)
# ------------------------------------------------------------------
_setup_dotenv() {
  local env_file="$BUNDLE_ROOT/.env"
  local example_file="$BUNDLE_ROOT/.env.example"

  if [ -f "$env_file" ]; then
    echo ".env already exists at $env_file"
    return 0
  fi

  if [ ! -f "$example_file" ]; then
    echo "No .env.example found — skipping env setup."
    return 0
  fi

  echo ""
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║            smallctl — First-Time Setup                  ║"
  echo "╚══════════════════════════════════════════════════════════╝"
  echo ""
  echo "No .env file found. Let's configure a few essentials."
  echo "Press Enter to accept the default shown in [brackets]."
  echo ""

  local default_endpoint
  local default_model
  local default_profile
  local default_context
  local default_reasoning
  default_endpoint=$(grep '^SMALLCTL_ENDPOINT=' "$example_file" | cut -d= -f2-)
  default_model=$(grep '^SMALLCTL_MODEL=' "$example_file" | cut -d= -f2-)
  default_profile=$(grep '^SMALLCTL_PROVIDER_PROFILE=' "$example_file" | cut -d= -f2-)
  default_context=$(grep '^SMALLCTL_CONTEXT_LIMIT=' "$example_file" | cut -d= -f2-)
  default_reasoning=$(grep '^SMALLCTL_REASONING_MODE=' "$example_file" | cut -d= -f2-)

  echo "── API Connection ────────────────────────────────────────"
  read -r -p "LLM API endpoint [$default_endpoint]: " input_endpoint
  local endpoint="${input_endpoint:-$default_endpoint}"

  read -r -p "Model name [$default_model]: " input_model
  local model="${input_model:-$default_model}"

  read -r -s -p "API key (leave blank if not needed): " input_key
  echo ""
  local api_key="${input_key:-}"

  echo ""
  echo "── Provider ──────────────────────────────────────────────"
  echo "Supported: generic, openai, ollama, vllm, lmstudio, openrouter"
  read -r -p "Provider profile [$default_profile]: " input_profile
  local profile="${input_profile:-$default_profile}"

  echo ""
  echo "── Performance ──────────────────────────────────────────"
  read -r -p "Context limit (tokens) [$default_context]: " input_context
  local context="${input_context:-$default_context}"

  read -r -p "Reasoning mode (auto/off/on) [$default_reasoning]: " input_reasoning
  local reasoning="${input_reasoning:-$default_reasoning}"

  cat > "$env_file" <<ENV
# smallctl configuration — generated by install.sh
SMALLCTL_ENDPOINT=$endpoint
SMALLCTL_MODEL=$model
SMALLCTL_PROVIDER_PROFILE=$profile
SMALLCTL_CONTEXT_LIMIT=$context
SMALLCTL_REASONING_MODE=$reasoning
ENV

  if [ -n "$api_key" ]; then
    echo "SMALLCTL_API_KEY=$api_key" >> "$env_file"
  fi

  echo ""
  echo "Created $env_file"
  echo "Edit it anytime to adjust settings."
}

if [ "$NO_ENV_SETUP" -eq 0 ] && [ "${SKIP_ENV_SETUP:-0}" != "1" ]; then
  _setup_dotenv
fi

# ------------------------------------------------------------------
# Completion message
# ------------------------------------------------------------------
cat <<EOF

smallctl is installed in: $VENV_DIR

Activate it with:
  source "$VENV_DIR/bin/activate"

Then run:
  smallctl --help
EOF

if [ -f "$BUNDLE_ROOT/.env.example" ]; then
  cat <<EOF

Copy "$BUNDLE_ROOT/.env.example" to "$BUNDLE_ROOT/.env" to configure local defaults.
EOF
fi