from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Iterable

from ..interrupt_replies import is_interrupt_response
from ..models.conversation import ConversationMessage
from .task_classifier_constants import (
    AUTHORING_TARGET_MARKERS,
    AUTHORING_TARGET_RE,
    CAPABILITY_QUERY_CONTEXTUAL_MARKERS,
    CAPABILITY_QUERY_NEGATIVE_PHRASES,
    CAPABILITY_QUERY_STRONG_MARKERS,
    CAPABILITY_QUERY_TARGETS,
    CODE_TARGET_RE,
    EXECUTION_ACTION_MARKERS,
    IP_ADDRESS_PATTERN,
    LOCAL_SHELL_OVERRIDE_RE,
    OPERATIONAL_ACTION_TARGETS,
    OPERATIONAL_ACTION_VERBS,
    READONLY_SUGGESTION_MARKERS,
    SSH_AUTH_MARKERS,
    TOOL_PLAN_EVIDENCE_MARKERS,
    WEB_LOOKUP_MARKERS,
    WRITE_ACTION_MARKERS,
    WRITE_FILE_CREATION_MARKERS,
)
from .task_classifier_content_lookup import needs_loop_for_content_lookup
from .task_classifier_support import (
    has_remote_execution_target,
    is_smalltalk,
    looks_like_analysis_request,
    looks_like_debug_inspection_request,
    looks_like_execution_followup,
    looks_like_plan_only_request,
    looks_like_readonly_chat_request,
    task_is_local_coding_target,
    task_is_local_ssh_file_target,
    task_is_local_system_target,
    task_uses_local_tool_for_remote_api,
)


@dataclass(frozen=True, slots=True)
class TaskClassificationRule:
    """Declarative rule for classify_task_mode precedence table."""

    name: str
    check: Callable[[str], bool]
    mode: str

@dataclass(frozen=True)
class RuntimeIntent:
    label: str
    task_mode: str


@dataclass(frozen=True)
class RuntimePolicy:
    route_mode: str | None
    chat_requires_tools: bool



_TASK_CLASSIFICATION_RULES: list[TaskClassificationRule] = [
    TaskClassificationRule(
        "local_shell_override", lambda t: bool(LOCAL_SHELL_OVERRIDE_RE.search(t)), "local_execute"
    ),
    TaskClassificationRule(
        "local_coding_target", task_is_local_coding_target, "local_execute"
    ),
    TaskClassificationRule(
        "local_ssh_file_target", task_is_local_ssh_file_target, "local_execute"
    ),
    TaskClassificationRule(
        "local_tool_remote_api",
        task_uses_local_tool_for_remote_api,
        "local_execute",
    ),
    TaskClassificationRule(
        "hybrid_execute",
        lambda t: has_remote_execution_target(t)
        and any(marker in t.lower() for marker in ("local ", "locally", "./", "../")),
        "hybrid_execute",
    ),
    TaskClassificationRule(
        "remote_coding_target",
        lambda t: has_remote_execution_target(t) and has_code_target(t),
        "remote_execute",
    ),
    TaskClassificationRule(
        "local_system_target", task_is_local_system_target, "local_execute"
    ),
    TaskClassificationRule(
        "smalltalk", is_smalltalk, "chat"
    ),
    TaskClassificationRule(
        "plan_only", looks_like_plan_only_request, "plan_only"
    ),
    TaskClassificationRule(
        "remote_execute",
        lambda t: has_remote_execution_target(t)
        and (
            looks_like_action_request(t)
            or looks_like_shell_request(t)
            or looks_like_debug_inspection_request(t)
            or needs_contextual_loop_escalation([], t)
        ),
        "remote_execute",
    ),
    TaskClassificationRule(
        "write_patch",
        lambda t: looks_like_write_patch_request(t)
        or looks_like_write_file_request(t)
        or looks_like_author_write_request(t)
        or looks_like_implementation_followup(t),
        "local_execute",
    ),
    TaskClassificationRule(
        "debug_inspect", looks_like_debug_inspection_request, "debug_inspect"
    ),
    TaskClassificationRule(
        "action_or_shell",
        lambda t: looks_like_action_request(t) or looks_like_shell_request(t),
        "local_execute",
    ),
    TaskClassificationRule(
        "analysis", looks_like_analysis_request, "analysis"
    ),
]


def has_code_target(task: str) -> bool:
    text = str(task or "").strip()
    if not text:
        return False
    return bool(CODE_TARGET_RE.search(text.lower()))


def classify_task_mode(task: str) -> str:
    text = task.strip()
    if not text:
        return "chat"
    for rule in _TASK_CLASSIFICATION_RULES:
        if rule.check(text):
            return rule.mode
    lowered = text.lower()
    if "error" in lowered or "failed" in lowered or "failure" in lowered:
        return "analysis"
    return "chat"


def looks_like_action_request(task: str) -> bool:
    text = task.strip().lower()
    if _looks_like_ssh_auth_request(text):
        return True
    if any(marker in text for marker in EXECUTION_ACTION_MARKERS):
        return True
    has_operational_verb = any(verb in text for verb in OPERATIONAL_ACTION_VERBS)
    has_operational_target = any(target in text for target in OPERATIONAL_ACTION_TARGETS)
    if has_operational_verb and (has_operational_target or bool(IP_ADDRESS_PATTERN.search(text))):
        return True
    return False


def looks_like_write_patch_request(task: str) -> bool:
    text = str(task or "").strip().lower()
    if not text:
        return False
    if looks_like_numbered_implementation_followup(text):
        return True
    if re.search(r"\b(?:new|fresh)\s+(?:script|file|module|code)\b", text) and looks_like_write_file_request(text):
        return False
    has_write_action = any(marker in text for marker in WRITE_ACTION_MARKERS)
    return bool(has_write_action and CODE_TARGET_RE.search(text))


def looks_like_numbered_implementation_followup(task: str) -> bool:
    text = str(task or "").strip().lower()
    if not text:
        return False
    action = r"(?:apply|do|fix|fixes|implement|patch|proceed\s+with|start\s+(?:with|on)|use|choose|pick)"
    subject = r"(?:fix|fixes|proposal|proposals|option|options|improvement|improvements|change|changes|item|items|step|steps|issue|issues)"
    if re.search(rf"\b{action}\b(?:\s+the)?\s+{subject}\s*#?\d+\b", text):
        return True
    if re.search(rf"\b{action}\b\s+#\d+\b", text):
        return True
    if re.search(rf"\b{subject}\s*#?\d+\b", text) and re.search(r"\b(?:apply|implement|patch|fix|fixes|ux|cli|code|script)\b", text):
        return True
    # Typo-tolerant fallback: a short action word followed by a hash/number
    # can still be a numbered follow-up even if the user abbreviates or misspells
    # the subject (e.g. "apply and implement fi #3").
    loose_action = r"(?:apply|do|fix|fixes|implement|patch|use|make|add)"
    if re.search(rf"\b{loose_action}\b(?:\s+\S+){{0,5}}\s+#\d+\b", text):
        return True
    return False


def looks_like_implementation_followup(task: str) -> bool:
    """Detect direct commands to apply/make unnumbered improvements/fixes.

    Matches phrases like "apply the bug and robustness fixes" or
    "make the UX improvements", but avoids analysis requests such as
    "list improvements you would make".
    """
    text = str(task or "").strip().lower()
    if not text:
        return False
    if looks_like_numbered_implementation_followup(text):
        return False
    # Reject pure analysis/question phrasing before checking action verbs.
    if re.search(r"\b(list|describe|explain|what|which|how|should|would)\s+(?:the\s+)?(?:fix|fixes|improvement|improvements|change|changes)\b", text):
        return False
    action = r"(?:apply|do|fix|implement|patch|proceed\s+with|start\s+(?:with|on)|use|make|add)"
    subject = r"(?:fix|fixes|proposal|proposals|improvement|improvements|change|changes)"
    # Require action followed by subject within a short window, optionally
    # separated by modifiers such as "the", "bug and robustness", "UX", etc.
    if re.search(rf"\b{action}\b(?:\s+\S+){{0,8}}\s+{subject}\b", text):
        return True
    return False


def looks_like_write_file_request(task: str) -> bool:
    text = str(task or "").strip().lower()
    if not text:
        return False
    if any(marker in text for marker in READONLY_SUGGESTION_MARKERS):
        return False
    has_creation_marker = any(marker in text for marker in WRITE_FILE_CREATION_MARKERS)
    has_code_target = "script" in text or bool(CODE_TARGET_RE.search(text))
    return bool(has_creation_marker and has_code_target)


def looks_like_author_write_request(task: str) -> bool:
    text = str(task or "").strip().lower()
    if not text:
        return False
    has_creation_marker = any(marker in text for marker in WRITE_FILE_CREATION_MARKERS)
    has_authoring_target = any(marker in text for marker in AUTHORING_TARGET_MARKERS)
    has_authoring_path = bool(AUTHORING_TARGET_RE.search(text))
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
    if any(verb in text for verb in OPERATIONAL_ACTION_VERBS) and (
        any(target in text for target in OPERATIONAL_ACTION_TARGETS)
        or bool(IP_ADDRESS_PATTERN.search(text))
    ):
        return True
    return bool(
        re.search(
            r"\b(run|execute|exec|launch|invoke|start)\b.*\b(command|shell|terminal|script|scan|nmap|port|ports|ssh|ping|curl|wget)\b",
            text,
        )
    )


def looks_like_tool_plan_candidate(task: str) -> bool:
    text = str(task or "").strip().lower()
    if not text:
        return False
    if looks_like_plan_only_request(text):
        return False
    if looks_like_shell_request(text) or looks_like_action_request(text):
        return False
    if any(marker in text for marker in TOOL_PLAN_EVIDENCE_MARKERS):
        return True
    if (
        looks_like_write_patch_request(text)
        or looks_like_write_file_request(text)
        or looks_like_author_write_request(text)
    ):
        return False
    if any(marker in text for marker in WEB_LOOKUP_MARKERS) and looks_like_readonly_chat_request(text):
        return True
    mode = classify_task_mode(text)
    return mode in {"analysis", "debug_inspect"} and (
        needs_loop_for_content_lookup(text) or looks_like_readonly_chat_request(text)
    )


def classify_runtime_intent(
    task: str,
    *,
    recent_messages: Iterable[ConversationMessage],
    pending_interrupt: dict | None = None,
) -> RuntimeIntent:
    text = str(task or "").strip()
    if not text:
        return RuntimeIntent(label="chat_only", task_mode="chat")

    if is_interrupt_response(pending_interrupt, text):
        return RuntimeIntent(label="interrupt_continuation", task_mode="loop")

    task_mode = classify_task_mode(text)
    if is_smalltalk(text):
        return RuntimeIntent(label="smalltalk", task_mode="chat")
    if looks_like_capability_query(text, recent_messages=recent_messages):
        return RuntimeIntent(label="capability_query", task_mode=task_mode)
    if needs_memory_persistence(text):
        return RuntimeIntent(label="memory_persistence", task_mode=task_mode)
    if needs_contextual_loop_escalation(recent_messages, text):
        return RuntimeIntent(label="contextual_execute", task_mode=task_mode)
    if looks_like_author_write_request(text) or looks_like_implementation_followup(text):
        return RuntimeIntent(label="author_write", task_mode=task_mode)
    if looks_like_readonly_chat_request(text):
        return RuntimeIntent(label="readonly_lookup", task_mode=task_mode)
    if (
        looks_like_write_patch_request(text)
        or looks_like_write_file_request(text)
        or task_mode in {"local_execute", "remote_execute"}
        or looks_like_implementation_followup(text)
        or looks_like_action_request(text)
        or looks_like_shell_request(text)
    ):
        return RuntimeIntent(label="execute", task_mode=task_mode)
    if needs_loop_for_content_lookup(text):
        return RuntimeIntent(label="content_lookup", task_mode=task_mode)
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
        "interrupt_continuation",
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
    if any(phrase in text for phrase in CAPABILITY_QUERY_NEGATIVE_PHRASES):
        return False
    if any(marker in text for marker in CAPABILITY_QUERY_STRONG_MARKERS):
        return True
    if any(marker in text for marker in CAPABILITY_QUERY_CONTEXTUAL_MARKERS) and any(
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

    has_target = any(target in text for target in CAPABILITY_QUERY_TARGETS)
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


_LOG_LEVEL_LABELS_RE = re.compile(
    r"\b(?:info|warn|warning|error|debug|trace|fatal)\b",
    re.IGNORECASE,
)
_LOG_LEVEL_SPEC_RE = re.compile(
    r"levels?\s*[:=]\s*[^\n]*" + _LOG_LEVEL_LABELS_RE.pattern,
    re.IGNORECASE,
)


def _text_without_log_level_specs(text: str) -> str:
    """Remove lines/segments that enumerate log levels.

    Requirement lists often contain words like DEBUG as a severity label, not
    as an operational verb. Stripping those segments prevents false-positive
    complexity classification (e.g. an HTML dashboard spec listing log
    levels INFO/WARN/ERROR/DEBUG).
    """
    return _LOG_LEVEL_SPEC_RE.sub("", text)


def looks_like_complex_task(task: str) -> bool:
    """Return True when a task implies multiple steps, cross-file work, or
    mixed local/remote operations that benefit from structured planning."""
    text = str(task or "").strip().lower()
    if not text:
        return False
    if "do not connect" in text and "remote" in text and ("./" in text or "local" in text):
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
    verb_search_text = _text_without_log_level_specs(text)
    verb_hits = sum(1 for v in operational_verbs if v in verb_search_text)
    if verb_hits >= 3:
        return True

    # Both remote and local targets
    has_remote = has_remote_execution_target(text)
    has_local = any(m in text for m in ("file", "files", "code", "script", "patch", "edit", "module", "repo"))
    if has_remote and has_local:
        return True

    # Multiple file targets. Match extension-only tokens to avoid alternation
    # ordering bugs such as treating `.conf` as `.c` because `c` matches first.
    ext_tokens = re.findall(r"(?<![A-Za-z0-9_])[\./\\A-Za-z0-9_-]+\.(\w+)", text)
    code_extensions = {
        "py", "sh", "bash", "ps1", "js", "ts", "tsx", "jsx", "md", "toml",
        "yaml", "yml", "json", "go", "rs", "java", "kt", "cpp", "c", "html",
        "htm", "h", "rb", "php",
    }
    distinct_exts = {ext for ext in ext_tokens if ext in code_extensions}
    if len(distinct_exts) >= 3:
        return True

    # Debug-investigate-fix-verify chain
    chain_search_text = _text_without_log_level_specs(text)
    if any(m in chain_search_text for m in ("debug", "investigate", "find", "trace", "diagnose")) and any(
        m in chain_search_text for m in ("fix", "patch", "repair", "resolve", "solve")
    ) and any(m in chain_search_text for m in ("test", "verify", "validate", "check")):
        return True

    return False


def _looks_like_ssh_auth_request(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    if not any(marker in normalized for marker in SSH_AUTH_MARKERS):
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
