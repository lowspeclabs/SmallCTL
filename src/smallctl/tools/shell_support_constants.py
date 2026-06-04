from __future__ import annotations

import re
from pathlib import Path

_ARGPARSE_REQUIRED_ARGS_PATTERN = re.compile(
    r"(?:error:\s*)?the following arguments are required:\s*(.+)",
    re.IGNORECASE,
)
_YES_PIPE_PATTERN = re.compile(
    r"(?:^|[;&(]\s*)yes(?:\s+[^|;&]+)?\s*\|\s*(?P<target>[^;&|]+)",
    re.IGNORECASE | re.DOTALL,
)
_SINGLE_ANSWER_PIPE_PATTERN = re.compile(
    r"(?:^|[;&(]\s*)echo\s+(?P<answer>(?:-[A-Za-z]+\s+)?(?:['\"]?\s*[YyNn](?:[Ee][Ss]|[Oo])?\s*['\"]?))\s*\|\s*(?P<target>[^;&|]+)",
    re.IGNORECASE | re.DOTALL,
)
_INVALID_INPUT_MARKERS = (
    "invalid input, please try again",
    "answer not recognized",
)
_REMOTE_INSTALLER_PREFLIGHT_KEY = "_remote_installer_preflight"
_DEB822_FIELDS = ("Types:", "URIs:", "Suites:", "Components:")
_SHELL_CONTROL_TOKENS = {"&&", ";", "||", "|"}
_DISPOSABLE_PATH_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".cache",
}
_DISPOSABLE_PATH_SUFFIXES = {".pyc", ".pyo", ".tmp"}
_SOURCE_OR_TEST_SUFFIXES = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".go",
    ".rs",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".rb",
    ".php",
    ".cs",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".md",
    ".sh",
    ".sql",
}
_SOURCE_OR_TEST_DIR_NAMES = {
    "src",
    "lib",
    "app",
    "tests",
    "test",
    "scripts",
    "components",
    "pages",
    "packages",
}
_DETACHED_COMMAND_MARKERS = (
    "nohup ",
    "setsid ",
    "disown",
    "daemonize ",
    "systemd-run ",
    "tmux ",
    "screen ",
)
_FOLLOW_FLAGS = {"-f", "--follow", "--tail=follow"}
_INSPECTION_FLAGS = {"-h", "--help", "-v", "--version", "version"}
_SERVICE_MANAGER_COMMANDS = {"systemctl", "service", "supervisorctl", "rc-service", "launchctl"}
_PACKAGE_RUNNERS = {"npm", "pnpm", "yarn", "bun"}
_FOREGROUND_SUBCOMMANDS = {"dev", "develop", "serve", "server", "start", "watch"}
_FOREGROUND_BINARIES = {
    "air",
    "caddy",
    "flask",
    "gunicorn",
    "http-server",
    "nodemon",
    "rails",
    "redis-server",
    "uvicorn",
    "vite",
    "webpack-dev-server",
}
