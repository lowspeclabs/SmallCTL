from __future__ import annotations

import re

# Patterns that indicate the command is probing for a binary's presence.
_BINARY_PROBE_RE = re.compile(
    r"^(?:"
    r"which\s+\S+"
    r"|whereis\s+\S+"
    r"|\S+\s+(?:--version|version|info|status|--help)(?:\s|\||\>|\&|$)"
    r"|type\s+\S+"
    r"|command\s+-v\s+\S+"
    r"|dpkg\s+-l\s+\S+"
    r"|apt\s+(?:list|show|search)\s+\S+"
    r"|rpm\s+-q\s+\S+"
    r"|apk\s+info\s+\S+"
    r"|pgrep\b"
    r"|pidof\b"
    r")"
    r"(?:.*)?$",
    re.IGNORECASE,
)
_REMOVAL_ABSENCE_PROBE_RE = re.compile(
    r"\b(?:grep|egrep|fgrep|find|ls|systemctl|pgrep|ps)\b",
    re.IGNORECASE,
)
_REMOVAL_ABSENCE_PIPE_RE = re.compile(
    r"\|\s*(?:grep|egrep|fgrep)\b",
    re.IGNORECASE,
)
_LS_NO_SUCH_FILE_RE = re.compile(
    r"\bls:\s+cannot\s+access\b.*\b(?:no such file|not found)\b",
    re.IGNORECASE | re.DOTALL,
)
_FOG_RESOURCE_RE = re.compile(
    r"\bfog(?:project|server|\.service|[_-]?(?:nfs|scheduler|multicast|snapin|replicator|image|web|php|worker))?\b",
    re.IGNORECASE,
)

# Keywords that indicate a removal/uninstall task.
_REMOVAL_TASK_KEYWORDS = frozenset([
    "uninstall", "remove", "delete", "purge", "rm ", "rm -f",
    "stop and remove", "stop all", "clean up", "clean-up",
    "get rid of", "wipe", "tear down", "teardown", "disable",
])

# Strings in stderr/stdout that confirm exit-127 means "not found".
_NOT_FOUND_MARKERS = ("command not found", "not found", "no such file", "permission denied", "could not be found")
_SSH_AUTH_RECOVERY_KEY = "_ssh_auth_recovery_state"
_REMOTE_MUTATING_COMMAND_RE = re.compile(
    r"\bsed\s+-i\b"
    r"|"
    r"\bperl\s+-p(?:i|[^A-Za-z0-9_]*-i)\b"
    r"|"
    r"\bpython3?\s+-c\b.*\bopen\s*\([^)]*['\"]w"
    r"|"
    r"(?:^|\s)(?:\d?>|\d?>>|>>|>)\s*\S+"
    r"|"
    r"\btee(?:\s+-a)?\s+/\S+"
    r"|"
    r"\bcat\s*>\s*/\S+"
    r"|"
    r"\b(?:rm|truncate|install\s+-m)\b"
    r"|"
    r"\b(?:mv|cp)\b.+\s+/(?:etc|var|usr|opt|srv|root)/\S+",
    re.IGNORECASE | re.DOTALL,
)
_REMOTE_READBACK_COMMANDS = {"cat", "head", "tail"}
_REMOTE_FILE_PRESENCE_PROBE_RE = re.compile(
    r"(?:^|[;&|]\s*)test\s+-s\s+/\S+"
    r"|"
    r"(?:^|[;&|]\s*)sha256sum\s+/\S+"
    r"|"
    r"(?:^|[;&|]\s*)stat\s+(?:-[^\s]+\s+)*?/\S+",
    re.IGNORECASE,
)
_NGINX_VERIFIER_COMMAND_RE = re.compile(r"\bnginx\s+-t\b", re.IGNORECASE)
_NGINX_VERIFIER_FAILURE_RE = re.compile(
    r"nginx:\s*configuration\s*file\b.*\btest\s*failed\b"
    r"|"
    r"\[\s*emerg\s*\].*?\bin\s+/etc/nginx/"
    r"|"
    r"unexpected\s+end\s+of\s*file,\s*expecting\s+[\"']?[;}]",
    re.IGNORECASE | re.DOTALL,
)
_CURL_VERIFIER_FAILURE_RE = re.compile(
    r"\bcurl:\s*\(\d+\)"
    r"|"
    r"\bfailed to connect\b"
    r"|"
    r"\bconnection refused\b",
    re.IGNORECASE | re.DOTALL,
)
_ZERO_TESTS_RAN_RE = re.compile(
    r"\bNO TESTS RAN\b"
    r"|"
    r"\bRan\s+0\s+tests?\b",
    re.IGNORECASE,
)
_TEST_FAILURE_OUTPUT_RE = re.compile(
    r"\bFAILED\s*\((?:failures|errors|failed|passed|skipped|xfailed|xpassed)[^)]*\)"
    r"|"
    r"\b(?:FAIL|ERROR):\s+\S+"
    r"|"
    r"\b=+\s*(?:FAILURES|ERRORS|short test summary info)\s*=+"
    r"|"
    r"\b\d+\s+failed\b",
    re.IGNORECASE,
)
_TEST_FAILURE_SUMMARY_RE = re.compile(
    r"\bFAILED\s*\([^)]*\)"
    r"|"
    r"\b\d+\s+failed\b",
    re.IGNORECASE,
)
_TEST_FAILURE_COUNT_RE = re.compile(
    r"FAILED\s*\((?:[^)]*?failures\s*=\s*(\d+)[^)]*?)\)"
    r"|"
    r"(\d+)\s+failed",
    re.IGNORECASE,
)
_LONG_RUNNING_REMOTE_INSTALLER_COMMAND_RE = re.compile(
    r"\binstallfog\.sh\b"
    r"|"
    r"\b(?:apt-get|apt|dnf|yum|zypper|pacman)\b.*\b(?:install|upgrade|dist-upgrade|full-upgrade)\b"
    r"|"
    r"\b(?:docker\s+compose|docker-compose)\s+up\b"
    r"|"
    r"\b(?:make|ninja)\s+(?:install|build)\b",
    re.IGNORECASE | re.DOTALL,
)
_LONG_RUNNING_REMOTE_INSTALLER_OUTPUT_RE = re.compile(
    r"\binstallation started\b"
    r"|"
    r"\btesting internet connection\b"
    r"|"
    r"\binstalling\b.+\bas needed\b"
    r"|"
    r"\bbuilding dependency tree\b"
    r"|"
    r"\bsetting up\b\s+\S+"
    r"|"
    r"\bpulling\b.+\bimage\b",
    re.IGNORECASE | re.DOTALL,
)
_REMOTE_APPLICATION_BLOCKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "fogproject_account_exists",
        re.compile(
            r'The account\s+"fogproject"\s+already exists.*?'
            r"(?:Please remove the account\s+\"fogproject\".*?|"
            r"username=<usernameForSystem>\s+\./installfog\.sh\s+-y)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "account_exists",
        re.compile(
            r'The account\s+"[^"]+"\s+already exists.*?'
            r"(?:Please remove the account|set a new service username|userdel\s+\S+)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "command_not_found",
        re.compile(r"\b(?:bash|sh):\s+line\s+\d+:\s+\S+:\s+command not found\b", re.IGNORECASE),
    ),
    (
        "file_exists",
        re.compile(r"\bfailed to create symbolic link\b.*?\bFile exists\b", re.IGNORECASE | re.DOTALL),
    ),
    (
        "permission_denied",
        re.compile(r"\bpermission denied\b", re.IGNORECASE),
    ),
)
_INTERACTIVE_PROMPT_RE = re.compile(
    r"\b(?:Choice:\s*\[\d+\]|Are you sure you wish to continue|Should .*?\?\s*\([yYnN]/|"
    r"Sorry,\s+answer not recognized|Hit \[?Enter\]?)\b",
    re.IGNORECASE | re.DOTALL,
)
_RAW_SSH_COMMAND_RE = re.compile(r"^\s*(?:ssh\b|scp\b|sftp\b|sshpass\b)", re.IGNORECASE)
