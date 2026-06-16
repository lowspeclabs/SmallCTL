#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$SCRIPT_DIR"

REQUIRED_PYTHON_MAJOR="${REQUIRED_PYTHON_MAJOR:-3}"
REQUIRED_PYTHON_MINOR="${REQUIRED_PYTHON_MINOR:-13}"
DEFAULT_PYTHON_BIN="python${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
VENV_DIR="${VENV_DIR:-$BUNDLE_ROOT/.venv}"
SMALLCTL_DEV_MODE="${SMALLCTL_DEV_MODE:-}"

show_help() {
  cat <<EOF_HELP
Usage: install.sh [OPTION]

Install smallctl and its dependencies into an isolated virtual environment.

Options:
  -h, --help        Show this help message and exit
  -V, --version     Print the package version and exit
  --check           Check prerequisites without installing anything
  --no-env-setup    Skip the interactive .env setup wizard
  --env-only        Run the .env setup wizard and exit (skips install)

Environment:
  PYTHON_BIN         Python executable to use (default: $DEFAULT_PYTHON_BIN)
  VENV_DIR           Virtual environment path (default: .venv)
  SMALLCTL_DEV_MODE  Install editable when non-empty
EOF_HELP
}

show_prereq_hint() {
  cat >&2 <<EOF_HINT

Install the startup prerequisites first.

Debian/Ubuntu:
  sudo apt-get update
  sudo apt-get install -y python${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR} python${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}-venv python${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}-dev python3-pip sshpass

Fedora/RHEL:
  sudo dnf install -y python${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR} python${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}-devel python3-pip sshpass

macOS (with Homebrew):
  brew install python sshpass

If Python ${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR} is installed somewhere else:
  PYTHON_BIN=/path/to/python${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR} ./install.sh
EOF_HINT
}

show_version() {
  "$PYTHON_BIN" -c "
import tomllib
with open('$BUNDLE_ROOT/pyproject.toml', 'rb') as f:
    print(tomllib.load(f)['project']['version'])
"
}

python_version() {
  "$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
}

python_is_supported() {
  "$PYTHON_BIN" -c "import sys; exit(0 if sys.version_info >= (${REQUIRED_PYTHON_MAJOR}, ${REQUIRED_PYTHON_MINOR}) else 1)"
}

check_python_command() {
  if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    return 0
  fi

  echo "FAIL: $PYTHON_BIN not found on PATH" >&2
  show_prereq_hint
  return 1
}

check_python_version() {
  local pyver
  pyver="$(python_version)"

  if python_is_supported; then
    echo "PASS: Python $pyver (>= ${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR})"
    return 0
  fi

  echo "FAIL: Python $pyver (< ${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR} required)" >&2
  show_prereq_hint
  return 1
}

check_python_venv() {
  if "$PYTHON_BIN" -c "import venv" >/dev/null 2>&1; then
    echo "PASS: venv module available"
    return 0
  fi

  echo "FAIL: venv module not available for $PYTHON_BIN" >&2
  show_prereq_hint
  return 1
}

check_python_pip_bootstrap() {
  if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    echo "PASS: pip available for $PYTHON_BIN"
    return 0
  fi

  if "$PYTHON_BIN" -c "import ensurepip" >/dev/null 2>&1; then
    echo "PASS: pip can be bootstrapped with ensurepip"
    return 0
  fi

  echo "FAIL: pip is not available and ensurepip cannot bootstrap it for $PYTHON_BIN" >&2
  show_prereq_hint
  return 1
}

check_source_tree() {
  local ok=0

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

  return "$ok"
}

check_sshpass() {
  if command -v sshpass >/dev/null 2>&1; then
    echo "PASS: sshpass available"
    return 0
  fi

  echo "WARN: sshpass not found. Password-based SSH tools will fail until it is installed." >&2
  return 1
}

install_sshpass() {
  if command -v sshpass >/dev/null 2>&1; then
    return 0
  fi

  echo "sshpass is required for password-based SSH. Attempting to install..."

  if command -v apt-get >/dev/null 2>&1; then
    if [ "$(id -u)" -eq 0 ]; then
      apt-get update && apt-get install -y sshpass && echo "  -> installed sshpass via apt" && return 0
    elif command -v sudo >/dev/null 2>&1; then
      sudo apt-get update && sudo apt-get install -y sshpass && echo "  -> installed sshpass via apt" && return 0
    fi
  fi

  if command -v dnf >/dev/null 2>&1; then
    if [ "$(id -u)" -eq 0 ]; then
      dnf install -y sshpass && echo "  -> installed sshpass via dnf" && return 0
    elif command -v sudo >/dev/null 2>&1; then
      sudo dnf install -y sshpass && echo "  -> installed sshpass via dnf" && return 0
    fi
  fi

  if command -v brew >/dev/null 2>&1; then
    brew install hudochenkov/sshpass/sshpass && echo "  -> installed sshpass via brew" && return 0
  fi

  echo "Error: Could not install sshpass automatically. Please install it manually:" >&2
  echo "  Debian/Ubuntu:  sudo apt-get install sshpass" >&2
  echo "  Fedora/RHEL:    sudo dnf install sshpass" >&2
  echo "  macOS:          brew install hudochenkov/sshpass/sshpass" >&2
  return 1
}

check_prereqs() {
  local ok=0

  check_python_command || ok=1
  if [ "$ok" -eq 0 ]; then
    check_python_version || ok=1
    check_python_venv || ok=1
    check_python_pip_bootstrap || ok=1
  fi
  check_source_tree || ok=1
  check_sshpass || ok=1

  exit "$ok"
}

setup_dotenv() {
  local env_file="$BUNDLE_ROOT/.env"
  local example_file="$BUNDLE_ROOT/.env.example"

  if [ -f "$env_file" ]; then
    echo ".env already exists at $env_file"
    return 0
  fi

  if [ ! -f "$example_file" ]; then
    echo "No .env.example found; skipping env setup."
    return 0
  fi

  echo ""
  echo "smallctl - First-Time Setup"
  echo ""
  echo "No .env file found. Configure a few essentials."
  echo "Press Enter to accept the default shown in brackets."
  echo ""

  local default_endpoint
  local default_model
  local default_profile
  local default_context
  local default_reasoning
  default_endpoint="$(grep '^SMALLCTL_ENDPOINT=' "$example_file" | cut -d= -f2-)"
  default_model="$(grep '^SMALLCTL_MODEL=' "$example_file" | cut -d= -f2-)"
  default_profile="$(grep '^SMALLCTL_PROVIDER_PROFILE=' "$example_file" | cut -d= -f2-)"
  default_context="$(grep '^SMALLCTL_CONTEXT_LIMIT=' "$example_file" | cut -d= -f2-)"
  default_reasoning="$(grep '^SMALLCTL_REASONING_MODE=' "$example_file" | cut -d= -f2-)"

  echo "API Connection"
  read -r -p "LLM API endpoint [$default_endpoint]: " input_endpoint
  local endpoint="${input_endpoint:-$default_endpoint}"

  read -r -p "Model name [$default_model]: " input_model
  local model="${input_model:-$default_model}"

  read -r -s -p "API key (leave blank if not needed): " input_key
  echo ""
  local api_key="${input_key:-}"

  echo ""
  echo "Provider"
  echo "Supported: generic, openai, ollama, vllm, lmstudio, openrouter"
  read -r -p "Provider profile [$default_profile]: " input_profile
  local profile="${input_profile:-$default_profile}"

  echo ""
  echo "Performance"
  read -r -p "Context limit (tokens) [$default_context]: " input_context
  local context="${input_context:-$default_context}"

  read -r -p "Reasoning mode (auto/off/on) [$default_reasoning]: " input_reasoning
  local reasoning="${input_reasoning:-$default_reasoning}"

  cat > "$env_file" <<ENV_FILE
# smallctl configuration - generated by install.sh
SMALLCTL_ENDPOINT=$endpoint
SMALLCTL_MODEL=$model
SMALLCTL_PROVIDER_PROFILE=$profile
SMALLCTL_CONTEXT_LIMIT=$context
SMALLCTL_REASONING_MODE=$reasoning
ENV_FILE

  if [ -n "$api_key" ]; then
    echo "SMALLCTL_API_KEY=$api_key" >> "$env_file"
  fi

  echo ""
  echo "Created $env_file"
  echo "Edit it anytime to adjust settings."
}

require_prereqs() {
  check_python_command
  check_python_version
  check_python_venv
  check_source_tree
}

ensure_host_deps() {
  install_sshpass
}

ensure_venv() {
  if [ -d "$VENV_DIR" ]; then
    local venv_python="$VENV_DIR/bin/python"
    if [ -f "$venv_python" ]; then
      local pyver
      local venv_ver
      pyver="$(python_version)"
      venv_ver="$("$venv_python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
      if [ "$venv_ver" != "$pyver" ]; then
        echo "Python version mismatch: venv uses $venv_ver but $PYTHON_BIN is $pyver. Recreating venv..."
        rm -rf "$VENV_DIR"
      fi
    else
      echo "Incomplete venv detected. Recreating..."
      rm -rf "$VENV_DIR"
    fi
  fi

  if [ ! -d "$VENV_DIR" ]; then
    if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
      echo "Error: failed to create virtual environment with $PYTHON_BIN." >&2
      show_prereq_hint
      exit 1
    fi
    echo "Created virtual environment at $VENV_DIR"
  fi
}

ensure_venv_pip() {
  if "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
    return 0
  fi

  echo "pip is missing inside $VENV_DIR. Trying ensurepip..."
  "$VENV_DIR/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true

  if "$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1; then
    return 0
  fi

  echo "Error: pip is required inside the virtual environment but is not available." >&2
  show_prereq_hint
  exit 1
}

install_smallctl() {
  "$VENV_DIR/bin/python" -m pip install --upgrade pip wheel setuptools

  if [ -n "$SMALLCTL_DEV_MODE" ]; then
    "$VENV_DIR/bin/python" -m pip install -e "$BUNDLE_ROOT"
    echo "smallctl installed in editable mode (SMALLCTL_DEV_MODE)."
  else
    "$VENV_DIR/bin/python" -m pip install "$BUNDLE_ROOT"
    echo "smallctl installed."
  fi
}

resolve_python_debian() {
  if "$VENV_DIR/bin/python" -c "import debian.deb822" 2>/dev/null; then
    return 0
  fi

  echo "Installing python-debian (required for apt sources validation)..."

  if "$VENV_DIR/bin/python" -m pip install python-debian 2>/dev/null; then
    echo "  -> installed via pip"
    return 0
  fi

  if command -v apt-get >/dev/null 2>&1; then
    if [ "$(id -u)" -eq 0 ]; then
      apt-get install -y python3-debian && echo "  -> installed via apt" && return 0
    elif command -v sudo >/dev/null 2>&1; then
      sudo apt-get install -y python3-debian && echo "  -> installed via apt" && return 0
    fi
  fi

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
  pip:            "$VENV_DIR/bin/python" -m pip install python-debian
MSG
  return 1
}

NO_ENV_SETUP=0
ENV_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --help|-h) show_help; exit 0 ;;
    --version|-V) show_version; exit 0 ;;
    --check) check_prereqs ;;
    --no-env-setup) NO_ENV_SETUP=1 ;;
    --env-only) ENV_ONLY=1 ;;
    *)
      echo "Unknown option: $arg" >&2
      show_help >&2
      exit 2
      ;;
  esac
done

if [ "$ENV_ONLY" -eq 1 ]; then
  setup_dotenv
  exit 0
fi

require_prereqs
ensure_host_deps
ensure_venv
ensure_venv_pip
install_smallctl
resolve_python_debian

if [ "$NO_ENV_SETUP" -eq 0 ] && [ "${SKIP_ENV_SETUP:-0}" != "1" ]; then
  setup_dotenv
fi

cat <<EOF_DONE

smallctl is installed in: $VENV_DIR

Activate it with:
  source "$VENV_DIR/bin/activate"

Then run:
  smallctl --help
EOF_DONE
