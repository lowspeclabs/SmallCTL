from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from ..models.conversation import ConversationMessage

_IP_ADDRESS_PATTERN = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_EXECUTION_ACTION_MARKERS = (
    "run",
    "exec",
    "shell",
    "terminal",
    "ping",
    "curl",
    "wget",
    "git",
)
_OPERATIONAL_ACTION_VERBS = (
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
_WRITE_ACTION_MARKERS = (
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
_WRITE_FILE_CREATION_MARKERS = (
    "build",
    "create",
    "make",
    "write",
    "generate",
    "implement",
    "save",
    "produce",
)
_CODE_TARGET_RE = re.compile(
    r"\b(?:file|script|code|source|module|repo|repository)\b|(?:^|[\s`'\"(])[\./\\A-Za-z0-9_-]+\.(?:py|sh|bash|ps1|js|ts|tsx|jsx|md|toml|yaml|yml|json)\b",
    re.IGNORECASE,
)
_AUTHORING_TARGET_MARKERS = (
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
_AUTHORING_TARGET_RE = re.compile(
    r"(?:^|[\s`'\"(])[\./\\A-Za-z0-9_-]+\.(?:md|txt|text|rst)\b",
    re.IGNORECASE,
)
_OPERATIONAL_ACTION_TARGETS = (
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
_PLAN_ONLY_PHRASES = (
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
_ANALYSIS_MARKERS = (
    "explain",
    "analyze",
    "analyse",
    "review",
    "reason about",
    "understand",
    "why",
    "what caused",
)
_DEBUG_MARKERS = (
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
_REMOTE_HINTS = (
    "remote",
    "ssh",
    "server",
    "host",
    "vm",
    "instance",
)
_SSH_AUTH_MARKERS = (
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
_READONLY_FILE_TARGETS = (
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
_CAPABILITY_QUERY_STRONG_MARKERS = (
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
_CAPABILITY_QUERY_CONTEXTUAL_MARKERS = (
    "what can you do",
)
_CAPABILITY_QUERY_TARGETS = (
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
_CAPABILITY_QUERY_NEGATIVE_PHRASES = (
    "what tools would you use",
    "how do your tools work",
    "tooling seems broken",
    "tooling failure",
    "tool failure",
)
_WEB_LOOKUP_MARKERS = (
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


@dataclass(frozen=True)
class RuntimeIntent:
    label: str
    task_mode: str


@dataclass(frozen=True)
class RuntimePolicy:
    route_mode: str | None
    chat_requires_tools: bool


def is_smalltalk(task: str) -> bool:
    text = task.strip().lower()
    smalltalk = {
        "hi",
        "hello",
        "hey",
        "yo",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
        "how are you",
        "what's up",
        "whats up",
    }
    return text in smalltalk


def looks_like_plan_only_request(task: str) -> bool:
    text = task.strip().lower()
    if not text:
        return False
    return any(phrase in text for phrase in _PLAN_ONLY_PHRASES)


def has_remote_execution_target(task: str) -> bool:
    text = task.strip().lower()
    if not text:
        return False
    if _IP_ADDRESS_PATTERN.search(text):
        return True
    if "@" in text and any(marker in text for marker in ("ssh", "scp", "sftp")):
        return True
    return any(marker in text for marker in _REMOTE_HINTS)


def looks_like_debug_inspection_request(task: str) -> bool:
    text = task.strip().lower()
    if not text:
        return False
    if "tell me what failed" in text or "what failed" in text:
        return True
    has_debug_marker = any(marker in text for marker in _DEBUG_MARKERS)
    has_read_signal = any(
        marker in text
        for marker in (
            "inspect",
            "read",
            "show",
            "tell me",
            "summarize",
            "check",
            "look at",
        )
    )
    return has_debug_marker and (has_read_signal or needs_loop_for_content_lookup(task))


def looks_like_analysis_request(task: str) -> bool:
    text = task.strip().lower()
    if not text:
        return False
    if any(marker in text for marker in _ANALYSIS_MARKERS):
        return True
    if looks_like_readonly_chat_request(task) and not looks_like_debug_inspection_request(task):
        return True
    if needs_loop_for_content_lookup(task) and not looks_like_debug_inspection_request(task):
        return any(target in text for target in _READONLY_FILE_TARGETS)
    return False


def classify_task_mode(task: str) -> str:
    text = task.strip()
    if not text:
        return "chat"
    lowered = text.lower()
    if is_smalltalk(text):
        return "chat"
    if looks_like_plan_only_request(text):
        return "plan_only"
    if (
        looks_like_write_patch_request(text)
        or looks_like_write_file_request(text)
        or looks_like_author_write_request(text)
    ):
        return "local_execute"
    if has_remote_execution_target(text) and (
        looks_like_action_request(text)
        or looks_like_shell_request(text)
        or looks_like_debug_inspection_request(text)
        or needs_contextual_loop_escalation([], text)
    ):
        return "remote_execute"
    if looks_like_debug_inspection_request(text):
        return "debug_inspect"
    if looks_like_action_request(text) or looks_like_shell_request(text):
        return "local_execute"
    if looks_like_analysis_request(text):
        return "analysis"
    if "error" in lowered or "failed" in lowered or "failure" in lowered:
        return "analysis"
    return "chat"


def needs_loop_for_content_lookup(task: str) -> bool:
    text = task.strip().lower()
    if not text:
        return False

    file_markers = (
        "file",
        "log",
        "logs",
        ".log",
        ".jsonl",
        ".txt",
        ".md",
        ".py",
        "/",
        "\\",
        "code",
        "source",
        "src",
    )
    content_queries = (
        "what is",
        "what's",
        "show",
        "read",
        "tell me",
        "summarize",
        "contents",
        "content",
        "line ",
        "lines ",
        "bug",
        "error",
        "inconsistent",
        "inconsistency",
        "issue",
        "debug",
        "still",
    )
    asks_for_specific_line = bool(re.search(r"\bline(?:s)?\s+\d+\b", text))
    asks_for_range = bool(re.search(r"\b\d+\s*-\s*\d+\b", text))
    asks_for_log_or_file_content = any(marker in text for marker in file_markers) and any(
        query in text for query in content_queries
    )
    asks_for_command_execution = (
        bool(re.search(r"\b(run|execute|exec)\b", text))
        and bool(
            re.search(
                r"\b(ls|dir|pwd|cd|cat|type|findstr|grep|git\s+status|get-childitem|pytest|python|powershell|pwsh|cmd)\b",
                text,
            )
        )
    )
    asks_for_directory_contents = (
        any(
            phrase in text
            for phrase in (
                "what files",
                "which files",
                "list files",
                "show files",
                "what folders",
                "which folders",
                "list folders",
                "show folders",
                "list directory",
                "show directory",
                "directory contents",
                "folder contents",
                "current directory",
                "this directory",
                "current folder",
                "this folder",
                "what is in this directory",
                "what is in the current directory",
                "what is in this folder",
                "what is in the current folder",
            )
        )
        or bool(
            re.search(
                r"\b(list|show|what|which)\b.*\b(files?|folders?|directories?|contents?)\b",
                text,
            )
        )
    )
    asks_where_specific_line_is = asks_for_specific_line and any(
        phrase in text for phrase in ("what is", "what's", "show", "read")
    )
    return (
        asks_for_specific_line
        or asks_for_range
        or asks_for_log_or_file_content
        or asks_where_specific_line_is
        or asks_for_directory_contents
        or asks_for_command_execution
    )


def looks_like_execution_followup(text: str) -> bool:
    followup_phrases = (
        "use the command",
        "use that command",
        "run it",
        "run that",
        "execute it",
        "execute that",
        "try again",
        "use the shell command",
        "run the shell command",
        "execute the shell command",
    )
    return any(phrase in text for phrase in followup_phrases)


def looks_like_action_request(task: str) -> bool:
    text = task.strip().lower()
    if _looks_like_ssh_auth_request(text):
        return True
    if any(marker in text for marker in _EXECUTION_ACTION_MARKERS):
        return True
    has_operational_verb = any(verb in text for verb in _OPERATIONAL_ACTION_VERBS)
    has_operational_target = any(target in text for target in _OPERATIONAL_ACTION_TARGETS)
    if has_operational_verb and (has_operational_target or bool(_IP_ADDRESS_PATTERN.search(text))):
        return True
    return False


def looks_like_write_patch_request(task: str) -> bool:
    text = str(task or "").strip().lower()
    if not text:
        return False
    if re.search(r"\b(?:new|fresh)\s+(?:script|file|module|code)\b", text) and looks_like_write_file_request(text):
        return False
    has_write_action = any(marker in text for marker in _WRITE_ACTION_MARKERS)
    return bool(has_write_action and _CODE_TARGET_RE.search(text))


def looks_like_write_file_request(task: str) -> bool:
    text = str(task or "").strip().lower()
    if not text:
        return False
    has_creation_marker = any(marker in text for marker in _WRITE_FILE_CREATION_MARKERS)
    has_code_target = "script" in text or bool(_CODE_TARGET_RE.search(text))
    return bool(has_creation_marker and has_code_target)


def looks_like_author_write_request(task: str) -> bool:
    text = str(task or "").strip().lower()
    if not text:
        return False
    has_creation_marker = any(marker in text for marker in _WRITE_FILE_CREATION_MARKERS)
    has_authoring_target = any(marker in text for marker in _AUTHORING_TARGET_MARKERS)
    has_authoring_path = bool(_AUTHORING_TARGET_RE.search(text))
    return bool(
        has_creation_marker
        and (has_authoring_target or has_authoring_path)
        and not looks_like_write_patch_request(text)
        and not looks_like_write_file_request(text)
    )


def needs_memory_persistence(task: str) -> bool:
    text = task.strip().lower()
    if not text:
        return False
    memory_markers = (
        "save this in memory",
        "save memory",
        "remember this",
        "store this in memory",
        "store this",
        "note this",
        "pin this",
        "persist this",
        "keep this in memory",
        "write this down",
    )
    return any(marker in text for marker in memory_markers)


def looks_like_shell_request(task: str) -> bool:
    text = task.strip().lower()
    if not text:
        return False
    if looks_like_write_patch_request(text):
        return False
    if _looks_like_ssh_auth_request(text):
        return True
    shell_markers = (
        "bash",
        "shell",
        "terminal",
        "command",
        "command line",
        "run ",
        "execute",
        "exec",
        "install",
        "setup",
        "set up",
        "configure",
        "deploy",
        "provision",
        "restart",
        "stop",
        "enable",
        "disable",
        "upgrade",
        "scan",
        "nmap",
        "ssh",
        "scp",
        "sftp",
        "ping",
        "curl",
        "wget",
        "traceroute",
        "tracepath",
        "netstat",
        "route",
        "ip route",
        "ip addr",
        "tcpdump",
        "netcat",
        "nc ",
        "dig",
        "nslookup",
        "whoami",
        "ps ",
        "top ",
        "lsof",
        "df ",
        "du ",
    )
    if any(marker in text for marker in shell_markers):
        return True
    if any(verb in text for verb in _OPERATIONAL_ACTION_VERBS) and (
        any(target in text for target in _OPERATIONAL_ACTION_TARGETS)
        or bool(_IP_ADDRESS_PATTERN.search(text))
    ):
        return True
    return bool(
        re.search(
            r"\b(run|execute|exec|launch|invoke|start)\b.*\b(command|shell|terminal|script|scan|nmap|port|ports|ssh|ping|curl|wget)\b",
            text,
        )
    )


def looks_like_readonly_chat_request(task: str) -> bool:
    text = task.strip().lower()
    if not text:
        return False
    if looks_like_execution_followup(text):
        return False
    readonly_markers = (
        "what",
        "which",
        "show",
        "read",
        "find",
        "search",
        "grep",
        "list",
        "current",
        "status",
        "where",
        "how many",
        "inspect",
        "check",
        "look at",
        "can you see",
        "tell me",
        "summarize",
    )
    readonly_targets = (
        "file",
        "files",
        "folder",
        "directory",
        "repo",
        "repository",
        "cwd",
        "working directory",
        "log",
        "logs",
        "artifact",
        "artifacts",
        "process",
        "cpu",
        "ram",
        "memory",
        "host",
        "system",
        "status",
        "code",
        "source",
        "src",
        "web",
        "website",
        "internet",
        "online",
        "docs",
        "documentation",
        "pricing",
        "release",
        "releases",
        "announcement",
        "news",
    )
    has_readonly_marker = any(marker in text for marker in readonly_markers)
    has_target = any(target in text for target in readonly_targets)
    return has_readonly_marker and has_target


def classify_runtime_intent(
    task: str,
    *,
    recent_messages: Iterable[ConversationMessage],
) -> RuntimeIntent:
    text = str(task or "").strip()
    if not text:
        return RuntimeIntent(label="chat_only", task_mode="chat")

    task_mode = classify_task_mode(text)
    if is_smalltalk(text):
        return RuntimeIntent(label="smalltalk", task_mode="chat")
    if looks_like_capability_query(text, recent_messages=recent_messages):
        return RuntimeIntent(label="capability_query", task_mode=task_mode)
    if needs_memory_persistence(text):
        return RuntimeIntent(label="memory_persistence", task_mode=task_mode)
    if needs_contextual_loop_escalation(recent_messages, text):
        return RuntimeIntent(label="contextual_execute", task_mode=task_mode)
    if looks_like_author_write_request(text):
        return RuntimeIntent(label="author_write", task_mode=task_mode)
    if (
        looks_like_write_patch_request(text)
        or looks_like_write_file_request(text)
        or task_mode in {"local_execute", "remote_execute"}
        or looks_like_action_request(text)
        or looks_like_shell_request(text)
    ):
        return RuntimeIntent(label="execute", task_mode=task_mode)
    if needs_loop_for_content_lookup(text):
        return RuntimeIntent(label="content_lookup", task_mode=task_mode)
    if looks_like_readonly_chat_request(text):
        return RuntimeIntent(label="readonly_lookup", task_mode=task_mode)
    return RuntimeIntent(label="chat_only", task_mode=task_mode)


def runtime_policy_for_intent(intent: RuntimeIntent) -> RuntimePolicy:
    if intent.label == "smalltalk":
        return RuntimePolicy(route_mode="chat", chat_requires_tools=False)
    if intent.label in {
        "capability_query",
        "author_write",
        "memory_persistence",
        "contextual_execute",
        "execute",
        "content_lookup",
    }:
        return RuntimePolicy(route_mode="loop", chat_requires_tools=True)
    if intent.label == "readonly_lookup":
        return RuntimePolicy(route_mode=None, chat_requires_tools=True)
    return RuntimePolicy(route_mode=None, chat_requires_tools=False)


def looks_like_capability_query(
    task: str,
    *,
    recent_messages: Iterable[ConversationMessage],
) -> bool:
    text = str(task or "").strip().lower()
    if not text:
        return False
    if any(phrase in text for phrase in _CAPABILITY_QUERY_NEGATIVE_PHRASES):
        return False
    if any(marker in text for marker in _CAPABILITY_QUERY_STRONG_MARKERS):
        return True
    if any(marker in text for marker in _CAPABILITY_QUERY_CONTEXTUAL_MARKERS) and any(
        scope in text
        for scope in (
            "harness",
            "environment",
            "tool",
            "tools",
            "capability",
            "capabilities",
            "access",
            "available",
            "enabled",
            "mode",
            "right now",
            "on this turn",
        )
    ):
        return True

    has_target = any(target in text for target in _CAPABILITY_QUERY_TARGETS)
    if not has_target:
        return False

    query_markers = (
        "what",
        "which",
        "can you",
        "do you",
        "are",
        "is",
        "enabled",
        "available",
        "access",
        "current",
        "right now",
    )
    if any(marker in text for marker in query_markers) and any(
        scope in text
        for scope in (
            "harness",
            "environment",
            "right now",
            "available now",
            "available on this turn",
            "access to",
            "enabled right now",
            "mode",
        )
    ):
        return True

    if _recent_capability_context(recent_messages) and any(
        marker in text for marker in ("what about", "which", "what", "are", "is", "enabled", "available")
    ):
        return True
    return False


def _recent_capability_context(messages: Iterable[ConversationMessage]) -> bool:
    recent_contents = [
        str(message.content or "").strip().lower()
        for message in reversed(list(messages))
        if str(getattr(message, "content", "") or "").strip()
    ][:4]
    if not recent_contents:
        return False
    return any(
        any(token in content for token in ("tools", "capabilities", "access", "mode", "harness", "environment"))
        for content in recent_contents
    )


def recent_assistant_proposed_command(messages: Iterable[ConversationMessage]) -> bool:
    recent_assistants = [
        message.content or ""
        for message in reversed(list(messages))
        if message.role == "assistant" and (message.content or "").strip()
    ][:2]
    if not recent_assistants:
        return False
    command_pattern = re.compile(
        r"```(?:bash|sh|shell|zsh|pwsh|powershell)?\s*\n.+?```",
        re.IGNORECASE | re.DOTALL,
    )
    shell_tokens = re.compile(
        r"\b(top|ps|ls|pwd|cd|cat|grep|find|git|pytest|python|bash|sh|systemctl|journalctl)\b",
        re.IGNORECASE,
    )
    for content in recent_assistants:
        if command_pattern.search(content):
            return True
        if shell_tokens.search(content):
            return True
    return False


def recent_assistant_referenced_tool_name(
    messages: Iterable[ConversationMessage],
    tool_name: str,
) -> bool:
    target = str(tool_name or "").strip().lower()
    if not target:
        return False
    for message in reversed(list(messages)):
        if message.role != "assistant" or not message.content:
            continue
        if target in message.content.lower():
            return True
    return False


def needs_contextual_loop_escalation(
    messages: Iterable[ConversationMessage],
    task: str,
) -> bool:
    text = task.strip().lower()
    if not text:
        return False
    if not looks_like_execution_followup(text):
        return False
    if recent_assistant_proposed_command(messages):
        return True
    return recent_assistant_referenced_tool_name(messages, "shell_exec")


def looks_like_complex_task(task: str) -> bool:
    """Return True when a task implies multiple steps, cross-file work, or
    mixed local/remote operations that benefit from structured planning."""
    text = str(task or "").strip().lower()
    if not text:
        return False

    # Multi-step sequencing language
    sequence_markers = (
        "step 1", "step 2", "step 3", "phase 1", "phase 2",
        "first ", "second ", "third ", "then ", "next ", "after that", "finally",
        "and then", "followed by", "subsequently", "before ", "after ",
    )
    if sum(1 for m in sequence_markers if m in text) >= 2:
        return True

    # Large-scale restructuring
    restructuring_markers = (
        "refactor", "redesign", "restructure", "reorganize",
        "project-wide", "across the codebase", "throughout the project",
        "migrate", "deprecate", "modernize",
    )
    if any(m in text for m in restructuring_markers):
        return True

    # Multiple distinct operational verbs
    operational_verbs = (
        "install", "setup", "set up", "configure", "deploy",
        "debug", "investigate", "find", "trace",
        "fix", "patch", "repair", "resolve",
        "implement", "write", "create", "build",
        "test", "verify", "validate", "check",
    )
    verb_hits = sum(1 for v in operational_verbs if v in text)
    if verb_hits >= 3:
        return True

    # Both remote and local targets
    has_remote = any(m in text for m in _REMOTE_HINTS) or bool(_IP_ADDRESS_PATTERN.search(text))
    has_local = any(m in text for m in ("file", "files", "code", "script", "patch", "edit", "module", "repo"))
    if has_remote and has_local:
        return True

    # Multiple file targets
    file_extensions = re.findall(r"[\./\\A-Za-z0-9_-]+\.(?:py|sh|bash|ps1|js|ts|tsx|jsx|md|toml|yaml|yml|json|go|rs|java|kt|cpp|c|h|rb|php)", text)
    if len(set(file_extensions)) >= 3:
        return True

    # Debug-investigate-fix-verify chain
    if any(m in text for m in ("debug", "investigate", "find", "trace", "diagnose")) and any(
        m in text for m in ("fix", "patch", "repair", "resolve", "solve")
    ) and any(m in text for m in ("test", "verify", "validate", "check")):
        return True

    return False


def _looks_like_ssh_auth_request(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    if not any(marker in normalized for marker in _SSH_AUTH_MARKERS):
        return False
    return any(
        token in normalized
        for token in (
            "auth",
            "authentication",
            "key",
            "keys",
            "ssh",
            "pubkey",
            "public key",
        )
    )
