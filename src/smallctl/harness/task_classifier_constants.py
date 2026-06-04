from __future__ import annotations

import re

IP_ADDRESS_PATTERN = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")

EXECUTION_ACTION_MARKERS = (
    "run",
    "exec",
    "shell",
    "terminal",
    "ping",
    "curl",
    "wget",
    "git",
)

OPERATIONAL_ACTION_VERBS = (
    "install",
    "setup",
    "set up",
    "spin up",
    "configure",
    "deploy",
    "provision",
    "restart",
    "start",
    "stop",
    "enable",
    "disable",
    "uninstall",
    "remove",
    "upgrade",
    "update",
)

WRITE_ACTION_MARKERS = (
    "patch",
    "edit",
    "modify",
    "fix",
    "update",
    "implement",
    "write",
    "create",
    "refactor",
)

WRITE_FILE_CREATION_MARKERS = (
    "build",
    "create",
    "make",
    "write",
    "generate",
    "implement",
    "save",
    "produce",
)

READONLY_SUGGESTION_MARKERS = (
    "list improvement",
    "list improvements",
    "improvements you would make",
    "improvements would you make",
    "what improvements",
    "recommend improvements",
    "suggest improvements",
    "suggest changes",
    "would change",
    "would improve",
)

CODE_TARGET_RE = re.compile(
    r"\b(?:file|script|code|source|module|repo|repository)\b|(?:^|[\s`'\"(])[\./\\A-Za-z0-9_-]+\.(?:py|sh|bash|ps1|js|ts|tsx|jsx|md|toml|yaml|yml|json)\b",
    re.IGNORECASE,
)

AUTHORING_TARGET_MARKERS = (
    "report",
    "summary",
    "summaries",
    "findings",
    "note",
    "notes",
    "writeup",
    "write-up",
    "readme",
    "documentation",
    "markdown",
    "text file",
    "doc",
    "docs",
)

AUTHORING_TARGET_RE = re.compile(
    r"(?:^|[\s`'\"(])[\./\\A-Za-z0-9_-]+\.(?:md|txt|text|rst)\b",
    re.IGNORECASE,
)

OPERATIONAL_ACTION_TARGETS = (
    "remote",
    "host",
    "server",
    "vm",
    "machine",
    "instance",
    "service",
    "daemon",
    "package",
    "packages",
    "apt",
    "apt-get",
    "yum",
    "dnf",
    "apk",
    "pacman",
    "brew",
    "ssh",
    "systemd",
    "caddy",
    "nginx",
    "apache",
    "postgres",
    "postgresql",
    "mysql",
    "mariadb",
    "redis",
    "docker",
    "kubernetes",
)

PLAN_ONLY_PHRASES = (
    "make a plan",
    "make a short plan",
    "create a plan",
    "create a short plan",
    "create a brief plan",
    "plan this",
    "plan this out",
    "make a plan first",
    "plan out",
    "before doing anything, create a short plan",
    "before doing anything, create a plan",
    "before doing anything, plan",
)

ANALYSIS_MARKERS = (
    "explain",
    "analyze",
    "analyse",
    "review",
    "reason about",
    "understand",
    "why",
    "what caused",
)

DEBUG_MARKERS = (
    "debug",
    "inspect",
    "investigate",
    "look at",
    "check",
    "trace",
    "failure",
    "failed",
    "error",
    "exception",
    "traceback",
    "stack trace",
    "log",
    "logs",
)

REMOTE_HINTS = (
    "remote",
    "ssh",
    "server",
    "host",
    "vm",
    "instance",
)

SSH_AUTH_MARKERS = (
    "pubkey auth",
    "public key auth",
    "public-key auth",
    "pubkey authentication",
    "public key authentication",
    "ssh key",
    "ssh keys",
    "authorized_keys",
    "authorized keys",
)

LOCAL_SHELL_OVERRIDE_RE = re.compile(
    r"\b(?:use\s+)?shell_exec\b.*\bno\s+(?:ssh|ssh_exec)\b"
    r"|"
    r"\bno\s+(?:ssh|ssh_exec)\b.*\b(?:use\s+)?shell_exec\b"
    r"|"
    r"\buse\s+shell\s+exec\b.*\bno\s+ssh\b"
    r"|"
    r"\buse\s+local\s+shell_exec\b"
    r"|"
    r"\bshell_exec\b.*\blocal\b"
    r"|"
    r"\blocal\b.*\bshell_exec\b",
    re.IGNORECASE,
)

READONLY_FILE_TARGETS = (
    "file",
    "files",
    "folder",
    "directory",
    "repo",
    "repository",
    "code",
    "source",
    "src",
    "log",
    "logs",
)

CAPABILITY_QUERY_STRONG_MARKERS = (
    "what tools",
    "available tools",
    "tool access",
    "what do you have access to",
    "which tools",
    "what capabilities",
    "what mode are you in",
    "which mode are you in",
    "what harness capabilities",
    "can you inspect the environment",
)

CAPABILITY_QUERY_CONTEXTUAL_MARKERS = (
    "what can you do",
)

CAPABILITY_QUERY_TARGETS = (
    "tool",
    "tools",
    "capability",
    "capabilities",
    "access",
    "available",
    "enabled",
    "mode",
    "environment",
    "harness",
)

CAPABILITY_QUERY_NEGATIVE_PHRASES = (
    "what tools would you use",
    "how do your tools work",
    "tooling seems broken",
    "tooling failure",
    "tool failure",
)

WEB_LOOKUP_MARKERS = (
    "latest",
    "current",
    "recent",
    "today",
    "web",
    "website",
    "internet",
    "online",
    "search the web",
    "search web",
    "web search",
    "docs",
    "documentation",
    "pricing",
    "release",
    "releases",
    "announcement",
    "news",
)

TOOL_PLAN_EVIDENCE_MARKERS = (
    "read through",
    "look through",
    "find where",
    "find out where",
    "summarize current implementation",
    "summarize the implementation",
    "identify files",
    "where does",
    "where is",
    "trace",
    "investigate",
    "multi-file",
    "codebase",
    "repo",
    "repository",
    "web research",
)
