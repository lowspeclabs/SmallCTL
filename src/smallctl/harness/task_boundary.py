from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..context.policy import estimate_text_tokens
from ..remote_scope import handoff_supports_remote_continuation, task_matches_remote_continuation
from ..models.conversation import ConversationMessage
from ..state import (
    EpisodicSummary,
    PromptBudgetSnapshot,
    RunBrief,
    WorkingMemory,
    clip_text_value,
    json_safe_value,
)
from ..task_targets import extract_task_target_paths
from ..normalization import dedupe_keep_tail
from ..state_memory import trim_recent_messages
from .followup_signals import (
    assistant_message_proposes_concrete_implementation,
    is_affirmative_followup,
    recent_assistant_requested_action_confirmation,
)
from .task_classifier import (
    classify_task_mode,
    looks_like_write_file_request,
    looks_like_write_patch_request,
)
from .task_intent import derive_task_contract, next_action_for_task

_SYSTEM_FOLLOW_UP_SPLIT_RE = re.compile(r"\nFollow-up:\s*", re.IGNORECASE)
_INLINE_CONTINUE_TASK_PREFIX_RE = re.compile(
    r"^\s*Continue current task:\s*(?P<body>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_INLINE_USER_WRAP_MARKER_RE = re.compile(
    r"\.\s*User\s+(?P<kind>follow-up|correction):\s*",
    re.IGNORECASE,
)
_FOLLOWUP_FILLERS = {"please", "pls", "now", "again", "just", "then", "more", "further"}
_NUMBERED_OPTION_RE = re.compile(r"^\s*(\d+)[.)]\s+(.+?)\s*$")
_INLINE_NUMBERED_OPTION_RE = re.compile(r"(?:^|\s)(\d+)[.)]\s+(.+?)(?=(?:\s+\d+[.)]\s+)|$)")
_ORDINAL_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
_OPTION_ACTION_WORDS = re.compile(
    r"\b(stream|streaming|md5|hash|patch|edit|modify|fix|update|implement|add|replace|refactor|test|skip|handle|read|write|calculate)\b",
    re.IGNORECASE,
)
_TARGET_NEGATION_RE = re.compile(r"\b(?:instead\s+of|rather\s+than|without|avoid|do\s+not|don't|dont)\b", re.IGNORECASE)
_TARGET_REPLACEMENT_RE = re.compile(
    r"\b(?:in|as|using|with)\s+(?:rust|go|typescript|javascript|python|bash|shell)\b",
    re.IGNORECASE,
)
_TARGET_LANGUAGE_RE = re.compile(r"\b(?:rust|go|typescript|javascript|python|bash|shell)\b", re.IGNORECASE)
_ORDINAL_FOLLOWUP_RE = re.compile(
    r"\b(?:start\s+(?:with|by|on)|do|use|choose|pick|implement|patch|apply)\s+"
    r"(?:option\s+|proposal\s+|#)?(\d+)\b",
    re.IGNORECASE,
)
_ORDINAL_PREFIX_RE = re.compile(
    r"^\s*(?:start\s+(?:with|by|on)|do|use|choose|pick|implement|patch|apply)\s+"
    r"(?:option\s+|proposal\s+|#)?\d+[.)]?\s*[,;:]?\s*",
    re.IGNORECASE,
)
_SEQUENTIAL_REMOTE_FOLLOWUP_RE = re.compile(
    r"\b(?:now|next|then|proceed|continue|move)\s+(?:to|on|with|do|edit|modify|fix|update|implement|patch|write)\b|"
    r"\b(?:do|edit|modify|fix|update|implement|patch|write)\s+(?:next|now|then)\b",
    re.IGNORECASE,
)
_GENERIC_EDIT_LEAD_RE = re.compile(
    r"^\s*(?:patch|edit|modify|fix|update|implement|apply)\b[^,.;:]*[,;:]\s*",
    re.IGNORECASE,
)
_GENERIC_TARGET_RE = re.compile(r"\b(?:script|file|module|code|python\s+file)\b", re.IGNORECASE)
_FOLLOWUP_ACTION_RE = re.compile(
    r"\b(?:add|apply|change|choose|decide|decision|edit|fix|implement|make|modify|patch|replace|resolve|update|write)\b",
    re.IGNORECASE,
)
_CONTEXTUAL_REFERENCE_RE = re.compile(
    r"\b(?:"
    r"this|that|it|same\s+(?:file|script|module|code)|"
    r"the\s+(?:file|script|module|code|change|fix|patch)|"
    r"loop(?:ing)?|stuck|repetitive|repeat(?:ing|ed)?|"
    r"you(?:'ve| have)\s+read\s+(?:the\s+)?(?:file\s+)?enough|"
    r"read\s+(?:the\s+)?(?:file\s+)?enough"
    r")\b",
    re.IGNORECASE,
)
_QUALITY_FOLLOWUP_RE = re.compile(
    r"\b(?:"
    r"still|inconsistent|inconsistency|wrong|off|broken|"
    r"not\s+(?:fixed|right|consistent|working)|"
    r"does(?:n't| not)\s+(?:look|match|work)|"
    r"mismatch(?:ed)?|regress(?:ed|ion)?"
    r")\b",
    re.IGNORECASE,
)
_QUALITY_TARGET_RE = re.compile(
    r"\b(?:css|code|file|files|layout|module|page|pages|script|site|style|styles|theme|theming|ui)\b",
    re.IGNORECASE,
)
_GUARD_RECOVERY_NUDGE_RE = re.compile(
    r"\b(?:loop(?:ing)?|stuck|repetitive|repeat(?:ing|ed)?|decide|decision|choose|resolve)\b",
    re.IGNORECASE,
)
_GUARD_FAILURE_RE = re.compile(
    r"\b(?:guard\s+tripped|loop\s+detected|repeated\s+tool|stagnation|stuck\s+in\s+(?:a\s+)?loop|max_consecutive_errors)\b",
    re.IGNORECASE,
)
_CORRECTIVE_TOOL_NAMES = (
    "file_patch",
    "ast_patch",
    "file_write",
    "file_append",
    "shell_exec",
    "ssh_exec",
    "task_complete",
)
_CORRECTIVE_RESTEER_RE = re.compile(
    r"\b(?:use|try|call|prefer|switch\s+to|move\s+to)\s+`?"
    r"(?:file_patch|ast_patch|file_write|file_append|shell_exec|ssh_exec|task_complete)"
    r"`?\b(?:\s+(?:instead|now|next))?"
    r"|"
    r"\b(?:not|don't|dont)\s+`?"
    r"(?:file_patch|ast_patch|file_write|file_append|shell_exec|ssh_exec|task_complete)"
    r"`?\b.*\b(?:use|try|call|prefer|switch\s+to|move\s+to)\s+`?"
    r"(?:file_patch|ast_patch|file_write|file_append|shell_exec|ssh_exec|task_complete)"
    r"`?\b",
    re.IGNORECASE,
)
_TASK_BOUNDARY_GUARD_SCRATCHPAD_KEYS = (
    "_tool_attempt_history",
    "_repeat_guard_one_shot_fingerprints",
    "_artifact_read_recovery_nudged",
    "_artifact_read_recovery_query",
    "_artifact_read_synthesis_nudged",
    "_artifact_summary_exit_nudged",
    "_artifact_evidence_unavailable_nudged",
    "_file_read_recovery_nudged",
    "_plan_artifact_read_suppressed",
    "_chunk_write_loop_guard",
    "_chunk_write_loop_guard_config",
    "_chunk_write_loop_guard_read_scheduled",
)
_ACTION_CONFIRMATION_PROMPTS = (
    "would you like me to",
    "do you want me to",
    "should i",
    "shall i",
    "want me to",
    "ready for me to",
)
_AFFIRMATIVE_REMOTE_CONTINUATION_TEXT = "proceed with the approved remote execution steps now"
_IPV4_HOST_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_USER_AT_HOST_RE = re.compile(
    r"\b(?P<user>[A-Za-z0-9._-]+)@(?P<host>[A-Za-z0-9.-]+|\d{1,3}(?:\.\d{1,3}){3})\b",
    re.IGNORECASE,
)
_REMOTE_OPERATIONAL_VERBS = (
    "boot",
    "bring up",
    "create",
    "install",
    "configure",
    "deploy",
    "provision",
    "restart",
    "start",
    "stop",
    "enable",
    "disable",
    "launch",
    "pull",
    "push",
    "run",
    "spin",
    "spin up",
    "switch",
    "switch to",
    "try",
    "upgrade",
    "update",
    "use",
    "setup",
    "set up",
)
_REMOTE_OPERATIONAL_TARGETS = (
    "app",
    "application",
    "container",
    "docker",
    "compose",
    "deployment",
    "image",
    "service",
    "stack",
    "daemon",
    "systemd",
    "startup",
    "server",
    "host",
    "remote",
    "package",
    "packages",
    "apt",
    "apt-get",
    "yum",
    "dnf",
    "apk",
    "pacman",
    "brew",
    "nginx",
    "apache",
    "postgres",
    "postgresql",
    "mysql",
    "mariadb",
    "redis",
    "tracker",
    "vikunja",
)
_REMOTE_DEPLOYMENT_CONTEXT_TARGETS = (
    "container",
    "docker",
    "compose",
    "image",
    "service",
    "deployment",
    "app",
    "application",
    "package",
    "stack",
    "tracker",
)
_REMOTE_CLARIFICATION_PHRASES = (
    "does not have to be",
    "doesn't have to be",
    "does not need to be",
    "doesn't need to be",
    "do not have to be",
    "dont have to be",
    "not have to be",
    "not need to be",
    "will do",
    "any app",
    "any application",
    "any image",
    "any container",
    "any service",
    "exactly",
    "exact name",
    "exact image",
    "called exactly",
)
_REMOTE_LIVE_CORRECTION_PHRASES = (
    "actually use ssh",
    "check again",
    "check it again",
    "do it live",
    "redo the remote action",
    "do not rely on past records",
    "don't rely on past records",
    "dont rely on past records",
    "do not rely on prior records",
    "don't rely on prior records",
    "re-run on the host",
    "rerun on the host",
    "run it live",
    "redo it live",
    "fresh ssh",
    "fresh run",
    "verify again",
)
_REMOTE_LIVE_CORRECTION_HINTS = (
    "actually",
    "again",
    "fresh",
    "live",
    "redo",
    "rerun",
    "re-run",
    "retry",
    "retest",
    "verify",
)
_REMOTE_DIAGNOSTIC_TARGETS = (
    "404",
    "500",
    "502",
    "503",
    "apache",
    "config",
    "configuration",
    "document root",
    "docroot",
    "error",
    "htaccess",
    "live",
    "nginx",
    "not live",
    "page",
    "pages",
    "rewrite",
    "route",
    "routing",
    "serve",
    "serving",
    "server block",
    "site",
    "site structure",
    "vhost",
)
_REMOTE_DIAGNOSTIC_HINTS = (
    "404 error",
    "500 error",
    "502 error",
    "503 error",
    "been update",
    "been updated",
    "does have",
    "error",
    "installed",
    "is live",
    "missing",
    "not live",
    "not serving",
    "serving",
    "updated",
)
_REMOTE_DIAGNOSTIC_QUESTION_RE = re.compile(
    r"^(?:has|have|is|are|does|do|did|why|what|which|where|when|can|could|would)\b",
    re.IGNORECASE,
)
_REMOTE_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![\w/])/(?:(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+(?:\.[A-Za-z0-9._-]+)?)"
)
_REMOTE_RESIDUE_MARKERS = (
    "htmleof",
    "wrote `htmleof`",
    "wc -l <",
    "written $(",
    "here-doc",
    "heredoc",
    "<< '",
    "<< \"",
)
_REMOTE_CORRECTIVE_CLEANUP_PHRASES = (
    "please fix",
    "fix this",
    "remove this",
    "clean this up",
    "clean it up",
    "trim this",
    "remove this from the bottom of the page",
    "remove this from the end of the page",
    "very bottom of the page",
    "very end of the page",
    "bottom of the page",
    "end of the page",
    "stuck to the very bottom",
    "stuck to the bottom",
    "trailing text",
    "trailing shell echo",
)
_SEMANTIC_RECENT_TAIL_TOKEN_CAP = 320


def _normalize_task_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _collapse_task_chain(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parts = [part.strip() for part in _SYSTEM_FOLLOW_UP_SPLIT_RE.split(text) if part.strip()]
    candidate = parts[-1] if parts else text
    inline = _canonicalize_inline_task_wrapper(candidate)
    return inline if inline else candidate


def _base_task_from_task_chain(value: Any) -> str:
    text = _collapse_task_chain(value)
    parsed = _parse_inline_task_wrapper(text)
    if parsed is None:
        return text
    base = str(parsed.get("base") or "").strip()
    return base or text


def _is_remote_followup_wrapper(value: Any) -> bool:
    return _normalize_task_text(_collapse_task_chain(value)).startswith("continue remote task over ssh")


def _canonicalize_inline_task_wrapper(value: Any) -> str:
    parsed = _parse_inline_task_wrapper(value)
    if parsed is None:
        return ""
    base = str(parsed.get("base") or "").strip()
    latest_suffix_text = str(parsed.get("latest_suffix_text") or "").strip()
    latest_suffix_kind = str(parsed.get("latest_suffix_kind") or "").strip().lower()
    if not base:
        return str(value or "").strip()
    if not latest_suffix_text:
        return f"Continue current task: {base}"

    label = "User correction" if latest_suffix_kind == "correction" else "User follow-up"
    return f"Continue current task: {base}. {label}: {latest_suffix_text}"


def _parse_inline_task_wrapper(value: Any) -> dict[str, str] | None:
    text = str(value or "").strip()
    if not text:
        return None

    current = text
    saw_wrapper = False
    latest_suffix_kind = ""
    latest_suffix_text = ""
    while True:
        match = _INLINE_CONTINUE_TASK_PREFIX_RE.match(current)
        if match is None:
            break
        saw_wrapper = True
        body = str(match.group("body") or "").strip()
        if not body:
            break
        suffix_markers = list(_INLINE_USER_WRAP_MARKER_RE.finditer(body))
        if suffix_markers:
            suffix = suffix_markers[-1]
            suffix_text = body[suffix.end() :].strip()
            suffix_kind = str(suffix.group("kind") or "").strip().lower()
            if suffix_text:
                latest_suffix_text = suffix_text
                latest_suffix_kind = suffix_kind
            current = body[: suffix.start()].strip().rstrip(".")
            if not current:
                break
            continue
        current = body
        break

    if not saw_wrapper:
        return None

    base = current.strip()
    return {
        "base": base or text,
        "latest_suffix_kind": latest_suffix_kind,
        "latest_suffix_text": latest_suffix_text,
    }


def _clean_option_title(value: str) -> str:
    title = str(value or "").strip()
    if not title:
        return ""
    bold = re.match(r"^\*\*(.+?)\*\*(?:\s*[-:]\s*(.*))?$", title)
    if bold:
        head = str(bold.group(1) or "").strip()
        tail = str(bold.group(2) or "").strip()
        return f"{head} - {tail}" if tail else head
    return title


def _extract_action_options_from_text(text: str, inherited_paths: list[str]) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    def _append_option(index: int, raw_title: str) -> None:
        title = _clean_option_title(raw_title)
        if not title or not _OPTION_ACTION_WORDS.search(title):
            return
        key = (index, title.lower())
        if key in seen:
            return
        seen.add(key)
        paths = extract_task_target_paths(title) or inherited_paths
        options.append(
            {
                "index": index,
                "title": title,
                "target_paths": list(paths),
            }
        )

    for line in str(text or "").splitlines():
        match = _NUMBERED_OPTION_RE.match(line)
        if not match:
            continue
        _append_option(int(match.group(1)), str(match.group(2) or ""))
    if options:
        return options
    flattened = re.sub(r"\s+", " ", str(text or "").strip())
    for match in _INLINE_NUMBERED_OPTION_RE.finditer(flattened):
        try:
            index = int(match.group(1))
        except (TypeError, ValueError):
            continue
        _append_option(index, str(match.group(2) or ""))
    return options


def _merge_action_options(
    existing: list[dict[str, Any]],
    extracted: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for option in [*existing, *extracted]:
        if not isinstance(option, dict):
            continue
        title = str(option.get("title") or "").strip()
        try:
            index = int(option.get("index") or 0)
        except (TypeError, ValueError):
            index = 0
        if not title or index <= 0:
            continue
        key = (index, title.lower())
        if key in seen:
            continue
        seen.add(key)
        target_paths = option.get("target_paths")
        if isinstance(target_paths, list):
            cleaned_paths = [str(path).strip() for path in target_paths if str(path).strip()]
        else:
            cleaned_paths = []
        merged.append({"index": index, "title": title, "target_paths": cleaned_paths})
    return merged


def _blocks_inherited_target(suffix: str, inherited_paths: list[str]) -> bool:
    text = str(suffix or "").strip().lower()
    if not text or not inherited_paths:
        return False
    path_names = {str(path).strip().lower() for path in inherited_paths if str(path).strip()}
    path_basenames = {Path(path).name.lower() for path in path_names if path}
    mentions_inherited_path = any(path in text for path in path_names | path_basenames)
    mentions_generic_code_target = bool(re.search(r"\b(?:python\s+file|script|file|module|code)\b", text))
    if _TARGET_NEGATION_RE.search(text) and (mentions_inherited_path or mentions_generic_code_target):
        return True
    if _TARGET_NEGATION_RE.search(text) and (_TARGET_REPLACEMENT_RE.search(text) or _TARGET_LANGUAGE_RE.search(text)):
        return True
    return False


def _normalize_remote_host(value: Any) -> str:
    return str(value or "").strip().lower()


def _coerce_remote_target(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    host = _normalize_remote_host(value.get("host"))
    if not host:
        return None
    user = str(value.get("user") or "").strip()
    return {"host": host, "user": user}


def _merge_remote_targets(targets: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for target in targets:
        normalized = _coerce_remote_target(target)
        if normalized is None:
            continue
        key = (normalized["host"], normalized["user"].lower())
        if key in seen:
            continue
        seen.add(key)
        merged.append(normalized)
    return merged


def _format_remote_target(target: dict[str, Any]) -> str:
    host = _normalize_remote_host(target.get("host"))
    user = str(target.get("user") or "").strip()
    if not host:
        return ""
    return f"{user}@{host}" if user else host


class TaskBoundaryService:
    def __init__(self, harness: Any):
        self.harness = harness

    def _consume_session_restored_flag(self) -> bool:
        scratchpad = getattr(getattr(self.harness, "state", None), "scratchpad", None)
        if not isinstance(scratchpad, dict):
            return False
        return bool(scratchpad.pop("_session_restored", False))

    def _active_task_scope_payload(self) -> dict[str, Any] | None:
        payload = getattr(self.harness, "_active_task_scope", None)
        if isinstance(payload, dict) and payload:
            return dict(payload)
        state = getattr(self.harness, "state", None)
        scratchpad = getattr(state, "scratchpad", None)
        if not isinstance(scratchpad, dict):
            return None
        stored = scratchpad.get("_active_task_scope")
        if not isinstance(stored, dict) or not stored:
            return None
        restored = dict(stored)
        self.harness._active_task_scope = restored
        sequence = stored.get("sequence")
        try:
            restored_sequence = int(sequence)
        except (TypeError, ValueError):
            restored_sequence = 0
        current_sequence = int(getattr(self.harness, "_task_sequence", 0) or 0)
        if restored_sequence > current_sequence:
            self.harness._task_sequence = restored_sequence
        return restored

    def _clip_task_summary_text(self, value: Any, *, limit: int = 240) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        clipped, truncated = clip_text_value(text, limit=limit)
        return f"{clipped} [truncated]" if truncated else clipped

    def _extract_task_terminal_message(self, result: dict[str, Any] | None) -> str:
        if not isinstance(result, dict) or not result:
            return ""
        message = result.get("message")
        if isinstance(message, dict):
            candidate = (
                message.get("message")
                or message.get("question")
                or message.get("status")
            )
            if candidate:
                return self._clip_task_summary_text(candidate)
        if isinstance(message, str) and message.strip():
            return self._clip_task_summary_text(message)
        reason = str(result.get("reason") or "").strip()
        if reason:
            return self._clip_task_summary_text(reason)
        error = result.get("error")
        if isinstance(error, dict):
            candidate = error.get("message")
            if candidate:
                return self._clip_task_summary_text(candidate)
        return ""

    def _task_duration_seconds(self, started_at: str, finished_at: str) -> float:
        try:
            started = datetime.fromisoformat(str(started_at))
            finished = datetime.fromisoformat(str(finished_at))
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, round((finished - started).total_seconds(), 3))

    def _write_task_summary(self, payload: dict[str, Any]) -> str:
        summary_path_text = str(payload.get("summary_path") or "").strip()
        if not summary_path_text:
            return ""
        path = Path(summary_path_text)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(json_safe_value(payload), indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
            return str(path)
        except Exception:
            logger = getattr(self.harness, "log", logging.getLogger("smallctl.harness"))
            if logger is not None:
                logger.exception("failed to write task summary")
            return ""

    def _append_task_episodic_summary(self, payload: dict[str, Any]) -> None:
        task_id = str(payload.get("task_id") or "").strip()
        if not task_id:
            return
        existing_ids = {
            str(getattr(summary, "summary_id", "") or "").strip()
            for summary in getattr(self.harness.state, "episodic_summaries", []) or []
        }
        summary_id = f"{task_id}-summary"
        if summary_id in existing_ids:
            return

        task_text = self._clip_task_summary_text(
            payload.get("effective_task") or payload.get("raw_task"),
            limit=180,
        )
        message = self._clip_task_summary_text(payload.get("message"), limit=180)
        reason = self._clip_task_summary_text(payload.get("reason"), limit=140)
        status = str(payload.get("status") or "").strip()
        notes = [f"Task {task_id} {status}: {task_text}".strip()]
        if message:
            notes.append(message)
        elif reason:
            notes.append(f"Reason: {reason}")

        artifacts = [str(item).strip() for item in (payload.get("artifact_ids") or []) if str(item).strip()]
        if not artifacts:
            count = int(payload.get("artifact_count") or 0)
            if count:
                artifacts = list((getattr(self.harness.state, "artifacts", {}) or {}).keys())[-min(count, 5):]

        self.harness.state.episodic_summaries.append(
            EpisodicSummary(
                summary_id=summary_id,
                created_at=str(payload.get("finished_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")),
                decisions=[f"status={status}"] if status else [],
                files_touched=[
                    str(path).strip()
                    for path in (payload.get("target_paths") or [])
                    if str(path).strip()
                ],
                failed_approaches=[reason] if status in {"aborted", "failed"} and reason else [],
                remaining_plan=[str(payload.get("replacement_task") or "").strip()]
                if payload.get("replacement_task")
                else [],
                artifact_ids=artifacts,
                notes=notes,
                full_summary_artifact_id=None,
            )
        )
        self.harness.state.episodic_summaries = self.harness.state.episodic_summaries[-12:]

    @staticmethod
    def _normalize_target_path(value: Any) -> str:
        text = str(value or "").strip().strip("`")
        if not text:
            return ""
        text = text.replace("\\", "/")
        while text.startswith("./"):
            text = text[2:]
        return text.rstrip("/").lower()

    def _target_paths_overlap(self, left: list[str], right: list[str]) -> bool:
        left_norm = {self._normalize_target_path(path) for path in left if self._normalize_target_path(path)}
        right_norm = {self._normalize_target_path(path) for path in right if self._normalize_target_path(path)}
        if not left_norm or not right_norm:
            return False
        if left_norm & right_norm:
            return True
        left_names = {Path(path).name.lower() for path in left_norm if path}
        right_names = {Path(path).name.lower() for path in right_norm if path}
        return bool(left_names & right_names)

    def _known_target_paths(self) -> list[str]:
        paths: list[str] = []
        handoff = self.last_task_handoff()
        for source in (
            handoff.get("target_paths"),
            handoff.get("remote_target_paths"),
            self.harness.state.scratchpad.get("_task_target_paths"),
            getattr(self.harness, "_active_task_scope", {}).get("target_paths")
            if isinstance(getattr(self.harness, "_active_task_scope", None), dict)
            else [],
        ):
            if not isinstance(source, list):
                continue
            paths.extend(str(path).strip() for path in source if str(path).strip())
        return dedupe_keep_tail(paths, limit=12)

    def _extract_remote_absolute_paths(self, *texts: Any) -> list[str]:
        collected: list[str] = []
        seen: set[str] = set()
        for text_value in texts:
            text = str(text_value or "")
            if not text:
                continue
            for match in _REMOTE_ABSOLUTE_PATH_RE.finditer(text):
                normalized = self._normalize_target_path(match.group(0))
                if not normalized or not normalized.startswith("/") or normalized in seen:
                    continue
                seen.add(normalized)
                collected.append(normalized)
        return collected

    def _recent_remote_target_paths(self, *, handoff: dict[str, Any] | None = None) -> list[str]:
        candidates: list[str] = []
        payload = handoff if isinstance(handoff, dict) else self.last_task_handoff()
        stored_paths = payload.get("remote_target_paths") if isinstance(payload, dict) else None
        if isinstance(stored_paths, list):
            candidates.extend(str(path).strip() for path in stored_paths if str(path).strip())

        for record in reversed(self._tool_execution_record_items()):
            if str(record.get("tool_name") or "").strip() != "ssh_exec":
                continue
            result = record.get("result")
            if not isinstance(result, dict):
                continue
            metadata = result.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            if not (
                bool(result.get("success"))
                or bool(metadata.get("ssh_transport_succeeded"))
                or str(metadata.get("failure_kind") or "").strip() == "remote_command"
            ):
                continue
            args = record.get("args")
            if isinstance(args, dict):
                candidates.extend(self._extract_remote_absolute_paths(args.get("command")))
            candidates.extend(self._extract_remote_absolute_paths(metadata.get("command")))

        return dedupe_keep_tail(candidates, limit=12)

    def _session_ssh_targets(self) -> list[dict[str, str]]:
        scratchpad = getattr(self.harness.state, "scratchpad", {})
        targets = scratchpad.get("_session_ssh_targets")
        if not isinstance(targets, dict):
            return []
        collected: list[dict[str, str]] = []
        for key, value in targets.items():
            if not isinstance(value, dict):
                continue
            host = _normalize_remote_host(value.get("host") or key)
            if not host:
                continue
            collected.append({"host": host, "user": str(value.get("user") or "").strip()})
        return _merge_remote_targets(collected)

    def _confirmed_session_ssh_targets(self) -> list[dict[str, str]]:
        scratchpad = getattr(self.harness.state, "scratchpad", {})
        targets = scratchpad.get("_session_ssh_targets")
        if not isinstance(targets, dict):
            return []
        collected: list[dict[str, str]] = []
        for key, value in targets.items():
            if not isinstance(value, dict) or not bool(value.get("confirmed")):
                continue
            host = _normalize_remote_host(value.get("host") or key)
            if not host:
                continue
            collected.append({"host": host, "user": str(value.get("user") or "").strip()})
        return _merge_remote_targets(collected)

    def _remote_targets_from_texts(self, *texts: Any) -> list[dict[str, str]]:
        collected: list[dict[str, str]] = []
        for text_value in texts:
            text = str(text_value or "").strip()
            if not text:
                continue
            seen_hosts: set[str] = set()
            for match in _USER_AT_HOST_RE.finditer(text):
                host = _normalize_remote_host(match.group("host"))
                if not host:
                    continue
                seen_hosts.add(host)
                collected.append({"host": host, "user": str(match.group("user") or "").strip()})
            for match in _IPV4_HOST_RE.finditer(text):
                host = _normalize_remote_host(match.group(0))
                if not host or host in seen_hosts:
                    continue
                seen_hosts.add(host)
                collected.append({"host": host, "user": ""})
        return _merge_remote_targets(collected)

    @staticmethod
    def _remote_target_matches_known_target(
        candidate: dict[str, Any],
        known_targets: list[dict[str, str]],
    ) -> bool:
        candidate_host = _normalize_remote_host(candidate.get("host"))
        candidate_user = str(candidate.get("user") or "").strip().lower()
        if not candidate_host:
            return False
        for known in known_targets:
            known_host = _normalize_remote_host(known.get("host"))
            if known_host != candidate_host:
                continue
            known_user = str(known.get("user") or "").strip().lower()
            if candidate_user and known_user and candidate_user != known_user:
                continue
            return True
        return False

    def _handoff_remote_targets(self, handoff: dict[str, Any]) -> list[dict[str, str]]:
        stored_targets = handoff.get("ssh_targets")
        if isinstance(stored_targets, list):
            coerced = _merge_remote_targets(stored_targets)
            if coerced:
                return coerced

        session_by_host = {
            target["host"]: target
            for target in self._session_ssh_targets()
            if target.get("host")
        }
        inferred = self._remote_targets_from_texts(
            handoff.get("effective_task"),
            handoff.get("current_goal"),
            handoff.get("raw_task"),
        )
        if inferred:
            enriched: list[dict[str, str]] = []
            for target in inferred:
                session_target = session_by_host.get(target["host"])
                if session_target is not None:
                    enriched.append(session_target)
                else:
                    enriched.append(target)
            return _merge_remote_targets(enriched)

        if str(handoff.get("task_mode") or "").strip() == "remote_execute":
            session_targets = self._session_ssh_targets()
            if len(session_targets) == 1:
                return session_targets
        return []

    def _handoff_mentions_remote_context(self, handoff: dict[str, Any]) -> bool:
        texts = (
            handoff.get("effective_task"),
            handoff.get("current_goal"),
            handoff.get("raw_task"),
        )
        for text_value in texts:
            text = str(text_value or "").strip().lower()
            if not text:
                continue
            if "ssh" in text or "remote host" in text or "remote" in text:
                return True
            if _USER_AT_HOST_RE.search(text) or _IPV4_HOST_RE.search(text):
                return True
        return False

    def _handoff_has_remote_context(self, handoff: dict[str, Any]) -> bool:
        task_mode = str(handoff.get("task_mode") or "").strip()
        if task_mode == "remote_execute":
            return True
        ssh_target = handoff.get("ssh_target")
        if isinstance(ssh_target, dict) and str(ssh_target.get("host") or "").strip():
            return True
        ssh_targets = handoff.get("ssh_targets")
        if isinstance(ssh_targets, list) and any(
            isinstance(target, dict) and str(target.get("host") or "").strip()
            for target in ssh_targets
        ):
            return True
        remote_target_paths = handoff.get("remote_target_paths")
        if isinstance(remote_target_paths, list) and any(str(path).strip() for path in remote_target_paths):
            return True
        next_required_tool = handoff.get("next_required_tool")
        if isinstance(next_required_tool, dict) and str(next_required_tool.get("tool_name") or "").strip() == "ssh_exec":
            return True
        if task_mode == "debug_inspect" and (
            self._handoff_mentions_remote_context(handoff) or bool(self._handoff_remote_targets(handoff))
        ):
            return True

        inferred_mode = classify_task_mode(
            str(handoff.get("effective_task") or handoff.get("current_goal") or handoff.get("raw_task") or "")
        )
        if inferred_mode == "remote_execute":
            return True
        return inferred_mode == "debug_inspect" and (
            self._handoff_mentions_remote_context(handoff) or bool(self._handoff_remote_targets(handoff))
        )

    def _handoff_mentions_remote_deployment_context(self, handoff: dict[str, Any]) -> bool:
        texts = (
            handoff.get("effective_task"),
            handoff.get("current_goal"),
            handoff.get("raw_task"),
        )
        for text_value in texts:
            text = str(text_value or "").strip().lower()
            if not text:
                continue
            if any(target in text for target in _REMOTE_DEPLOYMENT_CONTEXT_TARGETS):
                return True
        return False

    def _looks_like_remote_operational_followup(self, task: str) -> bool:
        text = str(task or "").strip().lower()
        if not text:
            return False
        if looks_like_write_patch_request(text) or looks_like_write_file_request(text):
            return False
        if extract_task_target_paths(text) or self._extract_remote_absolute_paths(text):
            return False
        has_remote_verb = any(verb in text for verb in _REMOTE_OPERATIONAL_VERBS)
        has_remote_target = any(target in text for target in _REMOTE_OPERATIONAL_TARGETS)
        return has_remote_verb and has_remote_target

    def _looks_like_remote_clarification_followup(
        self,
        task: str,
        handoff: dict[str, Any],
    ) -> bool:
        text = str(task or "").strip().lower()
        if not text:
            return False
        if looks_like_write_patch_request(text) or looks_like_write_file_request(text):
            return False
        if extract_task_target_paths(text) or self._extract_remote_absolute_paths(text):
            return False
        if self._remote_targets_from_texts(text):
            return False
        if not self._handoff_has_remote_context(handoff):
            return False
        if not self._handoff_mentions_remote_deployment_context(handoff):
            return False
        has_clarification_phrase = any(phrase in text for phrase in _REMOTE_CLARIFICATION_PHRASES)
        has_deployment_target = any(target in text for target in _REMOTE_DEPLOYMENT_CONTEXT_TARGETS)
        return has_clarification_phrase and has_deployment_target

    def _looks_like_remote_live_correction_followup(
        self,
        task: str,
        handoff: dict[str, Any],
    ) -> bool:
        text = str(task or "").replace("\u2019", "'").strip().lower()
        if not text:
            return False
        if self._looks_like_remote_artifact_cleanup_followup(text, handoff):
            return True
        if looks_like_write_patch_request(text) or looks_like_write_file_request(text):
            return False
        if extract_task_target_paths(text) or self._extract_remote_absolute_paths(text):
            return False
        if not (self.has_task_local_context() or handoff):
            return False
        if not (
            self._handoff_has_remote_context(handoff)
            or self._handoff_mentions_remote_context(handoff)
            or bool(self._session_ssh_targets())
        ):
            return False
        if any(phrase in text for phrase in _REMOTE_LIVE_CORRECTION_PHRASES):
            return True
        has_live_correction_language = any(marker in text for marker in _REMOTE_LIVE_CORRECTION_HINTS)
        has_remote_anchor = any(token in text for token in ("ssh", "remote", "host", "server"))
        has_reliance_negation = any(
            phrase in text
            for phrase in (
                "don't rely",
                "do not rely",
                "dont rely",
                "do not trust",
                "don't trust",
            )
        )
        return has_live_correction_language and (has_remote_anchor or has_reliance_negation)

    def _looks_like_remote_diagnostic_followup(
        self,
        task: str,
        handoff: dict[str, Any],
    ) -> bool:
        text = str(task or "").replace("\u2019", "'").strip().lower()
        if not text:
            return False
        if looks_like_write_patch_request(text) or looks_like_write_file_request(text):
            return False
        explicit_paths = extract_task_target_paths(text)
        if explicit_paths:
            return False
        if not self._handoff_has_remote_context(handoff):
            return False
        if not (
            self._handoff_mentions_remote_deployment_context(handoff)
            or bool(self._recent_remote_target_paths(handoff=handoff))
            or bool(self._confirmed_session_ssh_targets())
        ):
            return False

        referenced_remote_paths = self._extract_remote_absolute_paths(text)
        if referenced_remote_paths:
            recent_remote_paths = self._recent_remote_target_paths(handoff=handoff)
            if recent_remote_paths and self._target_paths_overlap(referenced_remote_paths, recent_remote_paths):
                return True

        has_target = any(target in text for target in _REMOTE_DIAGNOSTIC_TARGETS)
        if not has_target:
            return False
        has_hint = any(hint in text for hint in _REMOTE_DIAGNOSTIC_HINTS)
        has_question = "?" in text or _REMOTE_DIAGNOSTIC_QUESTION_RE.search(text) is not None
        return has_hint or has_question

    def _looks_like_remote_artifact_cleanup_followup(
        self,
        task: str,
        handoff: dict[str, Any],
    ) -> bool:
        text = str(task or "").replace("\u2019", "'").strip()
        if not text:
            return False
        if not (
            self._handoff_has_remote_context(handoff)
            or self._handoff_mentions_remote_context(handoff)
            or bool(self._session_ssh_targets())
        ):
            return False

        lowered = text.lower()
        recent_remote_paths = self._recent_remote_target_paths(handoff=handoff)
        referenced_remote_paths = self._extract_remote_absolute_paths(text)
        if referenced_remote_paths:
            if not recent_remote_paths:
                return False
            if not self._target_paths_overlap(referenced_remote_paths, recent_remote_paths):
                return False

        has_cleanup_phrase = any(phrase in lowered for phrase in _REMOTE_CORRECTIVE_CLEANUP_PHRASES)
        has_cleanup_verb = bool(re.search(r"\b(?:fix|remove|clean(?:\s+up)?|trim)\b", lowered))
        has_location_hint = any(
            phrase in lowered
            for phrase in ("bottom of the page", "end of the page", "very bottom", "very end", "trailing")
        )
        has_residue_marker = any(marker in lowered for marker in _REMOTE_RESIDUE_MARKERS)
        return (has_cleanup_phrase or (has_cleanup_verb and (has_location_hint or has_residue_marker))) and (
            has_residue_marker or bool(referenced_remote_paths)
        )

    def _looks_like_remote_contextual_site_followup(
        self,
        task: str,
        handoff: dict[str, Any],
    ) -> bool:
        text = str(task or "").strip()
        if not text:
            return False
        if not self._handoff_has_remote_context(handoff):
            return False
        return task_matches_remote_continuation(self.harness.state, text)

    def _is_remote_correction_followup(self, task: str) -> bool:
        return self._looks_like_remote_live_correction_followup(task, self.last_task_handoff())

    def _remote_followup_resolution(self, task: str) -> dict[str, Any] | None:
        text = str(task or "").strip()
        if not text:
            return None

        handoff = self.last_task_handoff()
        if not handoff:
            return None

        is_operational_followup = self._looks_like_remote_operational_followup(text)
        is_clarification_followup = self._looks_like_remote_clarification_followup(text, handoff)
        is_live_correction_followup = self._looks_like_remote_live_correction_followup(text, handoff)
        is_diagnostic_followup = self._looks_like_remote_diagnostic_followup(text, handoff)
        is_artifact_cleanup_followup = self._looks_like_remote_artifact_cleanup_followup(text, handoff)
        is_contextual_site_followup = self._looks_like_remote_contextual_site_followup(text, handoff)
        if not (
            is_operational_followup
            or is_clarification_followup
            or is_live_correction_followup
            or is_diagnostic_followup
            or is_artifact_cleanup_followup
            or is_contextual_site_followup
        ):
            return None
        if not handoff_supports_remote_continuation(self.harness.state):
            return None

        mission_task = self._current_or_handoff_continuity_task()
        chosen_targets = self._handoff_remote_targets(handoff)
        session_targets = self._session_ssh_targets()
        explicit_targets = self._remote_targets_from_texts(text)
        if is_operational_followup and not explicit_targets and not self._confirmed_session_ssh_targets():
            return None
        if (
            (
                is_live_correction_followup
                or is_diagnostic_followup
                or is_artifact_cleanup_followup
                or is_contextual_site_followup
            )
            and not explicit_targets
            and not (chosen_targets or self._confirmed_session_ssh_targets())
        ):
            return None
        if explicit_targets:
            known_targets = _merge_remote_targets(chosen_targets + session_targets)
            if not known_targets:
                return None
            if not all(
                self._remote_target_matches_known_target(target, known_targets)
                for target in explicit_targets
            ):
                return None
        session_labels = [_format_remote_target(target) for target in session_targets]
        session_labels = [label for label in session_labels if label]

        if len(chosen_targets) == 1:
            target = chosen_targets[0]
            label = _format_remote_target(target)
            effective_task = f"Continue remote task over SSH on {label}. User follow-up: {text}"
            return {
                "effective_task": effective_task,
                "mission_task": mission_task,
                "target_status": "resolved",
                "host": target["host"],
                "user": target["user"],
                "active_sessions": session_labels,
            }

        if len(session_targets) == 1:
            target = session_targets[0]
            label = _format_remote_target(target)
            effective_task = f"Continue remote task over SSH on {label}. User follow-up: {text}"
            return {
                "effective_task": effective_task,
                "mission_task": mission_task,
                "target_status": "resolved",
                "host": target["host"],
                "user": target["user"],
                "active_sessions": session_labels,
            }

        if session_labels:
            effective_task = (
                "Continue remote task over SSH. "
                f"Active SSH sessions: {', '.join(session_labels[:3])}. "
                f"User follow-up: {text}. Resolve which host before executing remote commands."
            )
            return {
                "effective_task": effective_task,
                "mission_task": mission_task,
                "target_status": "ambiguous",
                "host": "",
                "user": "",
                "active_sessions": session_labels,
            }

        if (is_live_correction_followup or is_diagnostic_followup) and self._handoff_has_remote_context(handoff):
            effective_task = (
                "Continue remote task over SSH. "
                f"User follow-up: {text}. Resolve which host before executing remote commands."
            )
            return {
                "effective_task": effective_task,
                "mission_task": mission_task,
                "target_status": "ambiguous",
                "host": "",
                "user": "",
                "active_sessions": session_labels,
            }

        return None

    def _apply_remote_followup_metadata(self, raw_task: str, resolution: dict[str, Any]) -> None:
        self.harness.state.scratchpad["_resolved_remote_followup"] = {
            "raw_task": str(raw_task or "").strip(),
            "effective_task": str(resolution.get("effective_task") or "").strip(),
            "mission_task": _collapse_task_chain(resolution.get("mission_task") or ""),
            "target_status": str(resolution.get("target_status") or "").strip(),
            "host": _normalize_remote_host(resolution.get("host")),
            "user": str(resolution.get("user") or "").strip(),
            "active_sessions": list(resolution.get("active_sessions") or []),
        }

    def _affirmative_remote_execution_followup_resolution(self, task: str) -> dict[str, Any] | None:
        text = str(task or "").strip()
        if not text or not self._is_affirmative_followup(text):
            return None

        handoff = self.last_task_handoff()
        if not handoff or not self._handoff_has_remote_context(handoff):
            return None
        if not self._recent_assistant_requested_action_confirmation() and not self._can_assume_remote_affirmative_continuation():
            return None

        mission_task = self._current_or_handoff_continuity_task()
        chosen_targets = self._handoff_remote_targets(handoff)
        session_targets = self._session_ssh_targets()
        session_labels = [_format_remote_target(target) for target in session_targets]
        session_labels = [label for label in session_labels if label]

        followup_text = _AFFIRMATIVE_REMOTE_CONTINUATION_TEXT
        if len(chosen_targets) == 1:
            target = chosen_targets[0]
            label = _format_remote_target(target)
            return {
                "effective_task": f"Continue remote task over SSH on {label}. User follow-up: {followup_text}",
                "mission_task": mission_task,
                "target_status": "resolved",
                "host": target["host"],
                "user": target["user"],
                "active_sessions": session_labels,
            }

        if len(session_targets) == 1:
            target = session_targets[0]
            label = _format_remote_target(target)
            return {
                "effective_task": f"Continue remote task over SSH on {label}. User follow-up: {followup_text}",
                "mission_task": mission_task,
                "target_status": "resolved",
                "host": target["host"],
                "user": target["user"],
                "active_sessions": session_labels,
            }

        if session_labels:
            return {
                "effective_task": (
                    "Continue remote task over SSH. "
                    f"Active SSH sessions: {', '.join(session_labels[:3])}. "
                    f"User follow-up: {followup_text}. Resolve which host before executing remote commands."
                ),
                "mission_task": mission_task,
                "target_status": "ambiguous",
                "host": "",
                "user": "",
                "active_sessions": session_labels,
            }

        return {
            "effective_task": (
                "Continue remote task over SSH. "
                f"User follow-up: {followup_text}. Resolve which host before executing remote commands."
            ),
            "mission_task": mission_task,
            "target_status": "ambiguous",
            "host": "",
            "user": "",
            "active_sessions": session_labels,
        }

    def begin_task_scope(self, *, raw_task: str, effective_task: str) -> dict[str, Any]:
        self._consume_session_restored_flag()
        normalized_raw = str(raw_task or "").strip()
        normalized_effective = str(effective_task or normalized_raw).strip()
        current = self._active_task_scope_payload()
        if current:
            current_effective = str(
                current.get("effective_task") or current.get("raw_task") or ""
            ).strip()
            current_raw = str(current.get("raw_task") or "").strip()
            if (
                (current_effective and current_effective == normalized_effective)
                or (current_raw and current_raw == normalized_raw)
            ):
                return current
            self.finalize_task_scope(
                terminal_event="task_aborted",
                status="aborted",
                reason="replaced_by_new_task",
                replacement_task=normalized_effective,
            )

        prior_sequence = getattr(self.harness, "_task_sequence", 0)
        if not prior_sequence:
            state = getattr(self.harness, "state", None)
            scratchpad = getattr(state, "scratchpad", None)
            if isinstance(scratchpad, dict):
                prior_sequence = scratchpad.get("_task_sequence", 0)
        try:
            sequence = int(prior_sequence) + 1
        except (TypeError, ValueError):
            sequence = 1
        self.harness._task_sequence = sequence

        task_id = f"task-{sequence:04d}"
        summary_path = ""
        if getattr(self.harness, "run_logger", None) is not None:
            summary_path = str(
                (self.harness.run_logger.run_dir / "tasks" / task_id / "task_summary.json").resolve()
            )
        scope = {
            "task_id": task_id,
            "sequence": sequence,
            "raw_task": normalized_raw,
            "effective_task": normalized_effective,
            "target_paths": extract_task_target_paths(normalized_effective),
            "started_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "start_step_count": int(getattr(self.harness.state, "step_count", 0) or 0),
            "start_token_usage": int(getattr(self.harness.state, "token_usage", 0) or 0),
            "summary_path": summary_path,
        }
        self.harness._active_task_scope = dict(scope)
        self.harness.state.scratchpad["_task_sequence"] = sequence
        self.harness.state.scratchpad["_active_task_scope"] = json_safe_value(scope)
        self.harness.state.scratchpad["_active_task_id"] = task_id
        return dict(scope)

    def finalize_task_scope(
        self,
        *,
        terminal_event: str,
        status: str,
        reason: str = "",
        result: dict[str, Any] | None = None,
        replacement_task: str = "",
    ) -> dict[str, Any] | None:
        scope = self._active_task_scope_payload()
        if not scope:
            return None

        finished_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        result_status = str((result or {}).get("status") or "").strip().lower()
        summary_terminal_event = terminal_event or "task_finalize"
        summary_text = self._extract_task_terminal_message(result)
        if not summary_text and reason:
            summary_text = self._clip_task_summary_text(reason)

        start_step_count = int(scope.get("start_step_count") or 0)
        start_token_usage = int(scope.get("start_token_usage") or 0)
        current_step_count = int(getattr(self.harness.state, "step_count", 0) or 0)
        current_token_usage = int(getattr(self.harness.state, "token_usage", 0) or 0)

        payload = {
            "task_id": str(scope.get("task_id") or "").strip(),
            "sequence": int(scope.get("sequence") or 0),
            "raw_task": str(scope.get("raw_task") or "").strip(),
            "effective_task": str(scope.get("effective_task") or "").strip(),
            "terminal_event": summary_terminal_event,
            "status": str(status or result_status or "stopped").strip(),
            "result_status": result_status,
            "reason": self._clip_task_summary_text(reason),
            "message": summary_text,
            "started_at": str(scope.get("started_at") or "").strip(),
            "finished_at": finished_at,
            "duration_seconds": self._task_duration_seconds(
                str(scope.get("started_at") or "").strip(),
                finished_at,
            ),
            "step_count": max(0, current_step_count - start_step_count),
            "token_usage": max(0, current_token_usage - start_token_usage),
            "current_phase": str(getattr(self.harness.state, "current_phase", "") or "").strip(),
            "active_tool_profiles": list(getattr(self.harness.state, "active_tool_profiles", []) or []),
            "target_paths": list(scope.get("target_paths") or []),
            "artifact_count": len(getattr(self.harness.state, "artifacts", {}) or {}),
            "recent_error_count": len(getattr(self.harness.state, "recent_errors", []) or []),
            "last_recent_error": self._clip_task_summary_text(
                (getattr(self.harness.state, "recent_errors", []) or [""])[-1]
                if getattr(self.harness.state, "recent_errors", [])
                else "",
                limit=180,
            ),
            "summary_path": str(scope.get("summary_path") or "").strip(),
        }
        if replacement_task:
            payload["replacement_task"] = replacement_task
        error = (result or {}).get("error")
        if isinstance(error, dict):
            payload["error_type"] = str(error.get("type") or "").strip()

        summary_path = self._write_task_summary(payload)
        payload["summary_path"] = summary_path
        self._append_task_episodic_summary(payload)

        if terminal_event:
            self.harness._runlog(
                terminal_event,
                "task ended without normal completion",
                task_id=payload["task_id"],
                status=payload["status"],
                result_status=result_status,
                reason=payload["reason"],
                replacement_task=replacement_task,
                summary_path=summary_path,
                raw_task=payload["raw_task"],
                effective_task=payload["effective_task"],
            )

        self.harness._active_task_scope = None
        self.harness.state.scratchpad.pop("_active_task_scope", None)
        self.harness.state.scratchpad.pop("_active_task_id", None)
        return payload

    def reset_task_boundary_state(
        self,
        *,
        reason: str,
        new_task: str = "",
        previous_task: str = "",
        preserve_memory: bool = False,
        preserve_summaries: bool = False,
        preserve_recent_tail: bool = False,
        semantic_recent_tail: bool = False,
        preserve_guard_context: bool = False,
    ) -> None:
        preserved_previous_task = str(
            previous_task
            or self.harness.state.run_brief.original_task
            or self.harness.state.scratchpad.get("_last_task_text")
            or ""
        ).strip()
        preserved_scratchpad: dict[str, Any] = {}
        for key in (
            "_model_name",
            "_model_is_small",
            "_max_steps",
            "strategy",
            "_session_notepad",
            "_session_ssh_targets",
            "_last_task_text",
            "_last_task_handoff",
            "_resolved_remote_followup",
            "_task_boundary_previous_task",
            "_task_sequence",
            "_web_result_index",
            "_web_search_artifact_results",
            "_web_last_search_result_ids",
            "_web_last_search_fetch_ids",
            "_web_last_search_artifact_id",
            "_web_fetch_id_counter",
            "_web_budget",
        ):
            if key in self.harness.state.scratchpad:
                preserved_scratchpad[key] = self.harness.state.scratchpad[key]
        if preserve_guard_context:
            for key in _TASK_BOUNDARY_GUARD_SCRATCHPAD_KEYS:
                if key in self.harness.state.scratchpad:
                    preserved_scratchpad[key] = self.harness.state.scratchpad[key]
        if preserved_previous_task:
            preserved_scratchpad["_task_boundary_previous_task"] = preserved_previous_task

        background_processes = json_safe_value(self.harness.state.background_processes)
        preserved_artifacts = dict(self.harness.state.artifacts)
        if preserve_recent_tail and semantic_recent_tail:
            recent_tail = self._semantic_recent_tail_messages(token_cap=_SEMANTIC_RECENT_TAIL_TOKEN_CAP)
        else:
            recent_tail = trim_recent_messages(
                list(self.harness.state.recent_messages),
                limit=self.harness.state.recent_message_limit,
            )
        preserved_recent_errors = list(self.harness.state.recent_errors[-6:])
        preserved_summaries = list(self.harness.state.episodic_summaries)
        preserved_context_briefs = list(self.harness.state.context_briefs)
        preserved_tool_history = list(self.harness.state.tool_history)
        current_memory = self.harness.state.working_memory
        preserved_memory = WorkingMemory(
            current_goal=str(current_memory.current_goal or ""),
            plan=list(current_memory.plan),
            decisions=list(current_memory.decisions),
            open_questions=list(current_memory.open_questions),
            known_facts=list(current_memory.known_facts),
            known_fact_meta=list(current_memory.known_fact_meta),
            failures=list(current_memory.failures),
            failure_meta=list(current_memory.failure_meta),
            next_actions=list(current_memory.next_actions),
            next_action_meta=list(current_memory.next_action_meta),
        )

        self.harness.state.current_phase = self.harness._initial_phase
        self.harness.state.step_count = 0
        self.harness.state.inactive_steps = 0
        self.harness.state.latest_verdict = None

        self.harness.state.scratchpad = preserved_scratchpad
        self.harness.state.recent_messages = recent_tail if preserve_recent_tail else []
        self.harness.state.recent_errors = preserved_recent_errors if preserve_guard_context else []
        self.harness.state.run_brief = RunBrief()
        if preserve_memory and preserved_previous_task:
            self.harness.state.run_brief.original_task = preserved_previous_task
        self.harness.state.working_memory = preserved_memory if preserve_memory else WorkingMemory()
        self.harness.state.acceptance_ledger = {}
        self.harness.state.acceptance_waivers = []
        self.harness.state.acceptance_waived = False
        self.harness.state.last_verifier_verdict = None
        self.harness.state.last_failure_class = ""

        self.harness.state.files_changed_this_cycle = []
        self.harness.state.repair_cycle_id = ""
        self.harness.state.stagnation_counters = {}
        self.harness.state.scratchpad.pop("_confabulation_nudged", None)
        self.harness.state.draft_plan = None
        self.harness.state.active_plan = None
        self.harness.state.plan_resolved = False
        self.harness.state.plan_artifact_id = ""
        self.harness.state.planning_mode_enabled = self.harness._configured_planning_mode
        self.harness.state.planner_requested_output_path = ""
        self.harness.state.planner_requested_output_format = ""
        self.harness.state.planner_resume_target_mode = "loop"
        self.harness.state.planner_interrupt = None
        # Artifacts are durable session handles. Keep them across task switches so
        # follow-up and resteered tasks can still inspect outputs from earlier work.
        self.harness.state.artifacts = preserved_artifacts
        if self.harness.state.write_session and self.harness.state.write_session.status != "complete":
            self.harness._runlog(
                "write_session_abandoned",
                "incomplete write session abandoned on task switch",
                session_id=self.harness.state.write_session.write_session_id,
                stage_target=self.harness.state.write_session.write_target_path,
                status=self.harness.state.write_session.status,
            )
            from ..graph.tool_outcomes import _register_write_session_stage_artifact
            _register_write_session_stage_artifact(self.harness, self.harness.state.write_session)
            self.harness.state.write_session = None
        self.harness.state.episodic_summaries = preserved_summaries if preserve_summaries else []
        self.harness.state.context_briefs = preserved_context_briefs if preserve_summaries else []
        self.harness.state.prompt_budget = PromptBudgetSnapshot()
        self.harness.state.retrieval_cache = []
        self.harness.state.task_mode = ""
        self.harness.state.active_intent = ""
        self.harness.state.secondary_intents = []
        self.harness.state.intent_tags = []
        self.harness.state.retrieved_experience_ids = []
        self.harness.state.tool_execution_records = {}
        self.harness.state.pending_interrupt = None
        self.harness.state.tool_history = preserved_tool_history if preserve_guard_context else []
        self.harness.state.background_processes = background_processes if isinstance(background_processes, dict) else {}
        self.harness.state.warm_experiences = []
        self.harness.state.touch()
        self.harness._runlog(
            "task_boundary_reset",
            "reset task-local state for new task",
            reason=reason,
            previous_task=previous_task,
            new_task=new_task,
            preserve_memory=preserve_memory,
            preserve_summaries=preserve_summaries,
            preserve_recent_tail=preserve_recent_tail,
            semantic_recent_tail=semantic_recent_tail,
            preserve_guard_context=preserve_guard_context,
        )

    def maybe_reset_for_new_task(self, task: str, *, raw_task: str | None = None) -> None:
        previous_task = _collapse_task_chain(
            self.harness.state.run_brief.original_task
            or self.harness.state.scratchpad.get("_last_task_text")
            or ""
        )
        if not previous_task:
            return
        new_task = _collapse_task_chain(task)
        if not new_task or _normalize_task_text(new_task) == _normalize_task_text(previous_task):
            return
        has_task_local_context = self.has_task_local_context()
        if has_task_local_context:
            session_restored = self._consume_session_restored_flag()
            remote_correction = self._is_remote_correction_followup(raw_task or task)
            same_scope_followup = self._is_same_scope_transition(
                raw_task=raw_task or task,
                effective_task=new_task,
                previous_task=previous_task,
            ) or remote_correction
            preserve_recent_tail = same_scope_followup or session_restored or remote_correction
            if previous_task:
                self.store_task_handoff(raw_task=previous_task, effective_task=previous_task)
            reset_reason = "task_soft_switch" if same_scope_followup else "task_switch"
            if session_restored and not same_scope_followup:
                reset_reason = "task_resume_switch"
            self.finalize_task_scope(
                terminal_event="task_aborted",
                status="aborted",
                reason="replaced_by_new_task",
                replacement_task=new_task,
            )
            self.reset_task_boundary_state(
                reason=reset_reason,
                new_task=new_task,
                previous_task=previous_task,
                preserve_memory=same_scope_followup,
                preserve_summaries=same_scope_followup,
                preserve_recent_tail=preserve_recent_tail,
                semantic_recent_tail=same_scope_followup or remote_correction,
                preserve_guard_context=same_scope_followup,
            )

    def has_task_local_context(self) -> bool:
        return self.has_resettable_context() or self.has_durable_context()

    def has_resettable_context(self) -> bool:
        return bool(
            self.harness.state.recent_messages
            or self.harness.state.recent_errors
            or self.harness.state.run_brief.task_contract
            or self.harness.state.run_brief.current_phase_objective
            or self.harness.state.working_memory.current_goal
            or self.harness.state.working_memory.plan
            or self.harness.state.working_memory.open_questions
            or self.harness.state.working_memory.next_actions
            or self.harness.state.acceptance_ledger
            or self.harness.state.acceptance_waivers
            or self.harness.state.scratchpad.get("_task_complete")
            or self.harness.state.scratchpad.get("_task_failed")
            or self.harness.state.scratchpad.get("_tool_attempt_history")
        )

    def has_durable_context(self) -> bool:
        return bool(
            self.harness.state.artifacts
            or self.harness.state.episodic_summaries
            or self.harness.state.context_briefs
            or self.harness.state.working_memory.decisions
            or self.harness.state.working_memory.known_facts
            or self.harness.state.working_memory.failures
        )

    def last_task_handoff(self) -> dict[str, Any]:
        payload = self.harness.state.scratchpad.get("_last_task_handoff")
        if not isinstance(payload, dict):
            return {}
        return dict(payload)

    def maybe_nudge_internal_divergence(self) -> bool:
        """Detect internal task divergence and emit a recovery nudge."""
        if self.harness.state.scratchpad.get("_task_divergence_nudged"):
            return False
        handoff = self.last_task_handoff()
        task_mode = handoff.get("task_mode", "")
        # Only detect remote -> local divergence for now
        if task_mode != "remote_execute":
            return False
        history = self.harness.state.scratchpad.get("_tool_attempt_history", [])
        if not isinstance(history, list) or len(history) < 3:
            return False
        recent = history[-5:]
        local_only = 0
        for item in recent:
            tool_name = str(item.get("tool_name", ""))
            if tool_name in {"file_read", "dir_list", "find_files"}:
                local_only += 1
            elif tool_name.startswith("ssh_"):
                return False
        if local_only < 3:
            return False
        original_task = self._clip_task_summary_text(
            self.harness.state.run_brief.original_task or handoff.get("raw_task", "")
        )
        self.harness.state.append_message(
            ConversationMessage(
                role="system",
                content=(
                    f"TASK DIVERGENCE WARNING: Your original task is: {original_task}. "
                    "You appear to have switched to working on local files instead of the remote target. "
                    "Do NOT abandon the original task. Return to the remote work immediately."
                ),
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "task_divergence",
                },
            )
        )
        self.harness._runlog(
            "task_divergence_nudge",
            "nudged model back to original remote task after internal task switch detected",
            original_task=original_task,
        )
        self.harness.state.scratchpad["_task_divergence_nudged"] = True
        return True

    def _is_continue_like_followup(self, task: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", " ", str(task or "").strip().lower()).strip()
        if not normalized:
            return False
        tokens = [token for token in normalized.split() if token not in _FOLLOWUP_FILLERS]
        collapsed = " ".join(tokens)
        return collapsed in {
            "continue",
            "cntinue",
            "conitnue",
            "continune",
            "cotinue",
            "keep going",
            "resume",
            "proceed",
            "go on",
            "carry on",
        }

    def _is_affirmative_followup(self, task: str) -> bool:
        return is_affirmative_followup(task, fillers=_FOLLOWUP_FILLERS)

    def _recent_assistant_requested_action_confirmation(self) -> bool:
        return recent_assistant_requested_action_confirmation(
            list(self.harness.state.recent_messages),
            prompts=_ACTION_CONFIRMATION_PROMPTS,
        )

    def _can_assume_remote_affirmative_continuation(self) -> bool:
        handoff = self.last_task_handoff()
        if not handoff or not self._handoff_has_remote_context(handoff):
            return False
        if len(self._confirmed_session_ssh_targets()) != 1:
            return False
        for message in reversed(self.harness.state.recent_messages[-4:]):
            if getattr(message, "role", "") != "assistant":
                continue
            text = str(getattr(message, "content", "") or "").strip()
            if assistant_message_proposes_concrete_implementation(text):
                return True
        return False

    def _has_contextual_reference_to_current_task(self, task: str) -> bool:
        text = str(task or "").replace("\u2019", "'").strip()
        if not text:
            return False
        known_paths = self._known_target_paths()
        explicit_paths = extract_task_target_paths(text)
        if explicit_paths and known_paths and self._target_paths_overlap(explicit_paths, known_paths):
            return True
        recent_remote_paths = self._recent_remote_target_paths()
        explicit_remote_paths = self._extract_remote_absolute_paths(text)
        if (
            explicit_remote_paths
            and recent_remote_paths
            and self._target_paths_overlap(explicit_remote_paths, recent_remote_paths)
        ):
            return True
        if explicit_paths or explicit_remote_paths:
            return False
        if not (self.has_task_local_context() or self.last_task_handoff()):
            return False
        if _CONTEXTUAL_REFERENCE_RE.search(text) and (
            _FOLLOWUP_ACTION_RE.search(text) or _GENERIC_TARGET_RE.search(text)
        ):
            return True
        if known_paths and _GENERIC_TARGET_RE.search(text) and _FOLLOWUP_ACTION_RE.search(text):
            return True
        return False

    def _has_recent_guard_failure_context(self) -> bool:
        candidate_texts: list[str] = []
        recent_errors = getattr(self.harness.state, "recent_errors", None)
        if isinstance(recent_errors, list):
            candidate_texts.extend(str(error or "") for error in recent_errors[-6:])
        scratchpad = getattr(self.harness.state, "scratchpad", None)
        if isinstance(scratchpad, dict):
            for key in ("_task_failed_message", "_last_guard_error"):
                value = scratchpad.get(key)
                if value:
                    candidate_texts.append(str(value))
            for key in _TASK_BOUNDARY_GUARD_SCRATCHPAD_KEYS:
                if key in scratchpad:
                    candidate_texts.append(key)
        return any(_GUARD_FAILURE_RE.search(text) for text in candidate_texts if text)

    def _is_quality_followup(self, task: str) -> bool:
        text = str(task or "").replace("\u2019", "'").strip()
        if not text or not _QUALITY_FOLLOWUP_RE.search(text):
            return False
        if not (self.has_task_local_context() or self.last_task_handoff()):
            return False
        known_paths = self._known_target_paths()
        explicit_paths = extract_task_target_paths(text)
        explicit_remote_paths = self._extract_remote_absolute_paths(text)
        if explicit_paths and known_paths:
            return self._target_paths_overlap(explicit_paths, known_paths)
        if explicit_remote_paths and self._recent_remote_target_paths():
            return self._target_paths_overlap(explicit_remote_paths, self._recent_remote_target_paths())
        if explicit_paths or explicit_remote_paths:
            return False
        return bool(
            known_paths
            or _QUALITY_TARGET_RE.search(text)
            or _CONTEXTUAL_REFERENCE_RE.search(text)
        )

    def _is_guard_recovery_followup(self, task: str) -> bool:
        text = str(task or "").replace("\u2019", "'").strip()
        if not text or not self._has_recent_guard_failure_context():
            return False
        explicit_paths = extract_task_target_paths(text)
        known_paths = self._known_target_paths()
        if explicit_paths and known_paths and not self._target_paths_overlap(explicit_paths, known_paths):
            return False
        explicit_remote_paths = self._extract_remote_absolute_paths(text)
        recent_remote_paths = self._recent_remote_target_paths()
        if (
            explicit_remote_paths
            and recent_remote_paths
            and not self._target_paths_overlap(explicit_remote_paths, recent_remote_paths)
        ):
            return False
        return bool(_GUARD_RECOVERY_NUDGE_RE.search(text))

    def _is_corrective_resteer_followup(self, task: str) -> bool:
        text = str(task or "").replace("\u2019", "'").strip()
        if not text:
            return False
        if not (self.has_task_local_context() or self.last_task_handoff()):
            return False
        if _CORRECTIVE_RESTEER_RE.search(text):
            return True
        lowered = text.lower()
        if any(f"`{tool}` instead" in lowered or f"{tool} instead" in lowered for tool in _CORRECTIVE_TOOL_NAMES):
            return True
        return self._looks_like_remote_live_correction_followup(text, self.last_task_handoff())

    def _current_or_handoff_task(self) -> str:
        handoff = self.last_task_handoff()
        return _base_task_from_task_chain(
            self.harness.state.working_memory.current_goal
            or handoff.get("current_goal")
            or handoff.get("effective_task")
            or self.harness.state.run_brief.original_task
            or self.harness.state.scratchpad.get("_last_task_text")
            or ""
        )

    def _current_or_handoff_continuity_task(self) -> str:
        handoff = self.last_task_handoff()
        return _collapse_task_chain(
            self.harness.state.working_memory.current_goal
            or handoff.get("current_goal")
            or self.harness.state.run_brief.original_task
            or handoff.get("effective_task")
            or self.harness.state.scratchpad.get("_last_task_text")
            or ""
        )

    def _remote_followup_mission_task(self, raw_task: str) -> str:
        resolved_remote = self.harness.state.scratchpad.get("_resolved_remote_followup")
        if not isinstance(resolved_remote, dict):
            return ""
        if _normalize_task_text(resolved_remote.get("raw_task")) != _normalize_task_text(raw_task):
            return ""
        mission_task = _collapse_task_chain(resolved_remote.get("mission_task") or "")
        if mission_task:
            return mission_task
        return self._current_or_handoff_continuity_task() or _collapse_task_chain(
            self.harness.state.scratchpad.get("_task_boundary_previous_task") or ""
        )

    def _resolved_corrective_resteer_task(self, raw_task: str) -> str:
        candidate = self._current_or_handoff_task()
        if not candidate:
            return str(raw_task or "").strip()
        correction = str(raw_task or "").strip()
        return f"Continue current task: {candidate}. User correction: {correction}"

    def _is_same_scope_transition(
        self,
        *,
        raw_task: str,
        effective_task: str,
        previous_task: str,
    ) -> bool:
        if self._looks_like_remote_artifact_cleanup_followup(raw_task, self.last_task_handoff()):
            return True
        if self._is_contextual_followup(raw_task):
            return True
        resolved = self.harness.state.scratchpad.get("_resolved_followup")
        if isinstance(resolved, dict) and resolved.get("target_inheritance") == "inherited":
            return True
        known_paths = self._known_target_paths()
        candidate_paths = (
            extract_task_target_paths(effective_task)
            or extract_task_target_paths(raw_task)
            or self._extract_remote_absolute_paths(effective_task)
            or self._extract_remote_absolute_paths(raw_task)
        )
        if candidate_paths and self._target_paths_overlap(candidate_paths, known_paths):
            return True
        previous_paths = extract_task_target_paths(previous_task) or self._extract_remote_absolute_paths(previous_task)
        if candidate_paths and self._target_paths_overlap(candidate_paths, previous_paths):
            return True
        # Remote same-directory / same-host transitions for multi-file operational sequences
        if candidate_paths and self._confirmed_session_ssh_targets():
            for cand in candidate_paths:
                if not cand.startswith("/"):
                    continue
                cand_parent = str(Path(cand).parent).lower()
                for known in known_paths:
                    if str(Path(known).parent).lower() == cand_parent:
                        return True
                for prev in previous_paths:
                    if str(Path(prev).parent).lower() == cand_parent:
                        return True
        # Sequential language ("now do ...", "next edit ...") on a confirmed remote session
        if candidate_paths and self._is_sequential_remote_followup(raw_task, candidate_paths):
            return True
        return False

    def _is_sequential_remote_followup(self, task: str, candidate_paths: list[str]) -> bool:
        if not candidate_paths or not self._confirmed_session_ssh_targets():
            return False
        if not any(path.startswith("/") for path in candidate_paths):
            return False
        if not self.has_task_local_context():
            return False
        return bool(_SEQUENTIAL_REMOTE_FOLLOWUP_RE.search(task))

    def _is_contextual_followup(self, task: str) -> bool:
        if self._is_continue_like_followup(task):
            return True
        if self._ordinal_followup_index(task) is not None and self.last_task_handoff().get("action_options"):
            return True
        if self._remote_followup_resolution(task) is not None:
            return True
        if self._is_corrective_resteer_followup(task):
            return True
        if self._has_contextual_reference_to_current_task(task):
            return True
        if self._is_quality_followup(task):
            return True
        if self._is_guard_recovery_followup(task):
            return True
        if not self._is_affirmative_followup(task):
            return False
        if not (self.has_task_local_context() or self.last_task_handoff()):
            return False
        return self._recent_assistant_requested_action_confirmation()

    def _ordinal_followup_index(self, task: str) -> int | None:
        text = str(task or "").strip().lower()
        if not text:
            return None
        match = _ORDINAL_FOLLOWUP_RE.search(text)
        if match:
            try:
                return int(match.group(1))
            except (TypeError, ValueError):
                return None
        for word, index in _ORDINAL_WORDS.items():
            if re.search(rf"\b{word}\s+(?:one|option|proposal)?\b", text):
                return index
        return None

    def _selected_action_option(self, task: str, handoff: dict[str, Any]) -> dict[str, Any] | None:
        index = self._ordinal_followup_index(task)
        if index is None:
            return None
        options = handoff.get("action_options")
        if not isinstance(options, list):
            return None
        for option in options:
            if not isinstance(option, dict):
                continue
            try:
                option_index = int(option.get("index") or 0)
            except (TypeError, ValueError):
                option_index = 0
            if option_index == index:
                return dict(option)
        return None

    def _message_is_semantic_tail_candidate(self, message: Any) -> bool:
        role = str(getattr(message, "role", "") or "").strip().lower()
        if role not in {"user", "assistant"}:
            return False
        metadata = getattr(message, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
        if metadata.get("hidden_from_prompt") or metadata.get("is_recovery_nudge"):
            return False
        text = str(getattr(message, "content", "") or "").strip()
        if not text:
            return False
        return True

    def _semantic_recent_tail_messages(self, *, token_cap: int) -> list[Any]:
        candidates = [
            message
            for message in self.harness.state.recent_messages
            if self._message_is_semantic_tail_candidate(message)
        ]
        if not candidates:
            return []

        selected: list[Any] = []
        last_assistant_index = max(
            (index for index, message in enumerate(candidates) if str(getattr(message, "role", "")).lower() == "assistant"),
            default=-1,
        )
        if last_assistant_index >= 0:
            assistant_message = candidates[last_assistant_index]
            user_index = max(
                (
                    index
                    for index in range(last_assistant_index - 1, -1, -1)
                    if str(getattr(candidates[index], "role", "")).lower() == "user"
                ),
                default=-1,
            )
            if user_index >= 0:
                selected.append(candidates[user_index])
            selected.append(assistant_message)
        else:
            user_index = max(
                (index for index, message in enumerate(candidates) if str(getattr(message, "role", "")).lower() == "user"),
                default=-1,
            )
            if user_index >= 0:
                selected.append(candidates[user_index])

        if not selected:
            return []

        total_tokens = sum(
            estimate_text_tokens(str(getattr(message, "content", "") or ""))
            for message in selected
        )
        if total_tokens <= token_cap:
            return selected

        if len(selected) == 2:
            user_message, assistant_message = selected
            assistant_tokens = estimate_text_tokens(str(getattr(assistant_message, "content", "") or ""))
            if assistant_tokens <= token_cap:
                return [assistant_message]
            user_tokens = estimate_text_tokens(str(getattr(user_message, "content", "") or ""))
            if user_tokens <= token_cap:
                return [user_message]

        return [selected[-1]]

    def _strip_ordinal_prefix(self, task: str) -> str:
        text = str(task or "").strip()
        if not text:
            return ""
        text = _ORDINAL_PREFIX_RE.sub("", text, count=1)
        text = _GENERIC_EDIT_LEAD_RE.sub("", text, count=1)
        return text.strip()

    def _resolve_option_target_paths(
        self,
        raw_task: str,
        option: dict[str, Any],
        handoff: dict[str, Any],
    ) -> dict[str, Any]:
        suffix = self._strip_ordinal_prefix(raw_task)
        explicit_paths = extract_task_target_paths(suffix)
        inherited_paths = option.get("target_paths")
        if not isinstance(inherited_paths, list) or not inherited_paths:
            inherited_paths = handoff.get("target_paths")
        if not isinstance(inherited_paths, list):
            inherited_paths = []

        if explicit_paths:
            return {
                "target_paths": list(explicit_paths),
                "target_inheritance": "explicit_override",
                "blocked_target_paths": [],
                "suffix": suffix,
            }

        if _blocks_inherited_target(suffix, list(inherited_paths)):
            return {
                "target_paths": [],
                "target_inheritance": "blocked_by_user_constraint",
                "blocked_target_paths": list(inherited_paths),
                "suffix": suffix,
            }

        return {
            "target_paths": list(inherited_paths),
            "target_inheritance": "inherited",
            "blocked_target_paths": [],
            "suffix": suffix,
        }

    def _resolved_option_task(
        self,
        raw_task: str,
        option: dict[str, Any],
        handoff: dict[str, Any],
    ) -> str:
        title = str(option.get("title") or "").strip()
        target_info = self._resolve_option_target_paths(raw_task, option, handoff)
        path_text = ", ".join(
            str(path).strip() for path in target_info.get("target_paths", []) if str(path).strip()
        )
        suffix = str(target_info.get("suffix") or "").strip()
        pieces: list[str] = []
        if path_text:
            pieces.append(
                f"Patch {path_text} to implement proposal #{option.get('index')}: {title}."
            )
        else:
            pieces.append(f"Implement proposal #{option.get('index')}: {title}.")
        if target_info.get("target_inheritance") == "blocked_by_user_constraint":
            blocked = ", ".join(
                str(path).strip()
                for path in target_info.get("blocked_target_paths", [])
                if str(path).strip()
            )
            if suffix:
                pieces.append(f"User constraint: {suffix[0].upper() + suffix[1:]}.")
            if blocked:
                pieces.append(
                    f"Do not assume {blocked} is the edit target; identify the appropriate target or ask before editing."
                )
        elif suffix:
            pieces.append(suffix[0].upper() + suffix[1:])
        return " ".join(piece.strip() for piece in pieces if piece.strip()).strip()

    def _apply_resolved_followup_metadata(
        self,
        raw_task: str,
        option: dict[str, Any],
        target_info: dict[str, Any],
        effective_task: str,
    ) -> None:
        self.harness.state.scratchpad["_resolved_followup"] = {
            "raw_task": str(raw_task or "").strip(),
            "option_index": option.get("index"),
            "option_title": str(option.get("title") or "").strip(),
            "target_paths": list(target_info.get("target_paths") or []),
            "target_inheritance": str(target_info.get("target_inheritance") or "").strip(),
            "blocked_target_paths": list(target_info.get("blocked_target_paths") or []),
            "effective_task": str(effective_task or "").strip(),
        }

    def resolve_followup_task(self, task: str) -> str:
        raw_task = str(task or "").strip()
        if not raw_task:
            return raw_task
        self.harness.state.scratchpad.pop("_resolved_remote_followup", None)

        handoff = self.last_task_handoff()
        option = self._selected_action_option(raw_task, handoff)
        if option is not None:
            target_info = self._resolve_option_target_paths(raw_task, option, handoff)
            resolved = self._resolved_option_task(raw_task, option, handoff)
            self._apply_resolved_followup_metadata(raw_task, option, target_info, resolved)
            return resolved

        remote_resolution = self._remote_followup_resolution(raw_task)
        if remote_resolution is not None:
            self._apply_remote_followup_metadata(raw_task, remote_resolution)
            return str(remote_resolution.get("effective_task") or raw_task).strip()

        affirmative_remote_resolution = self._affirmative_remote_execution_followup_resolution(raw_task)
        if affirmative_remote_resolution is not None:
            self._apply_remote_followup_metadata(raw_task, affirmative_remote_resolution)
            return str(affirmative_remote_resolution.get("effective_task") or raw_task).strip()

        if self._is_corrective_resteer_followup(raw_task):
            resolved = self._resolved_corrective_resteer_task(raw_task)
            self.harness.state.scratchpad["_resolved_resteer"] = {
                "raw_task": raw_task,
                "effective_task": resolved,
                "kind": "corrective_tool_resteer",
            }
            return resolved

        if not self._is_contextual_followup(raw_task):
            return raw_task

        continuity_candidate = self._current_or_handoff_continuity_task()
        candidate = _base_task_from_task_chain(
            handoff.get("current_goal")
            or handoff.get("effective_task")
            or self.harness.state.run_brief.original_task
            or self.harness.state.scratchpad.get("_last_task_text")
            or ""
        )
        if self._is_continue_like_followup(raw_task):
            return continuity_candidate or candidate or raw_task
        if not candidate:
            return raw_task

        if not (self.has_task_local_context() or handoff):
            return raw_task

        if (
            self._has_contextual_reference_to_current_task(raw_task)
            or self._is_quality_followup(raw_task)
            or self._is_guard_recovery_followup(raw_task)
        ):
            return f"Continue current task: {candidate}. User follow-up: {raw_task}"

        return candidate

    def store_task_handoff(self, *, raw_task: str, effective_task: str) -> None:
        effective = _collapse_task_chain(effective_task)
        if not effective:
            return
        target_paths = extract_task_target_paths(effective)
        current_goal = _collapse_task_chain(self.harness.state.working_memory.current_goal or effective)
        task_mode = classify_task_mode(effective)
        active_scope = self._active_task_scope_payload()
        active_task_id = str(active_scope.get("task_id") or "").strip() if active_scope else ""
        ssh_targets = self._handoff_remote_targets(
            {
                "raw_task": str(raw_task or "").strip(),
                "effective_task": effective,
                "current_goal": current_goal,
                "task_mode": task_mode,
            }
        )
        remote_target_paths = self._recent_remote_target_paths()
        if task_mode == "remote_execute" or ssh_targets:
            remote_target_paths = dedupe_keep_tail(
                remote_target_paths
                + self._extract_remote_absolute_paths(effective, current_goal, raw_task),
                limit=12,
            )
        previous = self.last_task_handoff()
        previous_paths = previous.get("target_paths") if isinstance(previous.get("target_paths"), list) else []
        same_task = _normalize_task_text(previous.get("effective_task")) == _normalize_task_text(effective)
        same_target = bool(set(previous_paths) & set(target_paths))
        existing_options = previous.get("action_options") if (same_task or same_target) else []
        if not isinstance(existing_options, list):
            existing_options = []
        last_good_artifact_ids = self._continuation_artifact_ids(previous)
        recent_research_artifact_ids = self._continuation_research_artifact_ids(previous)
        next_required_tool = self._continuation_next_required_tool(previous)
        last_failed_tool = self._continuation_last_failed_tool(previous)
        ssh_target = self._continuation_primary_ssh_target(ssh_targets, previous)
        self.harness.state.scratchpad["_last_task_handoff"] = {
            "task_id": active_task_id or str(previous.get("task_id") or "").strip(),
            "raw_task": str(raw_task or "").strip(),
            "effective_task": effective,
            "current_goal": current_goal,
            "task_mode": task_mode,
            "active_tool_profiles": list(getattr(self.harness.state, "active_tool_profiles", []) or []),
            "ssh_target": ssh_target,
            "ssh_targets": list(ssh_targets),
            "target_paths": list(target_paths),
            "remote_target_paths": list(remote_target_paths),
            "action_options": list(existing_options),
            "last_good_artifact_ids": list(last_good_artifact_ids),
            "recent_research_artifact_ids": list(recent_research_artifact_ids),
            "next_required_tool": next_required_tool,
            "last_failed_tool": last_failed_tool,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def _tool_execution_record_items(self) -> list[dict[str, Any]]:
        records = getattr(self.harness.state, "tool_execution_records", None)
        if not isinstance(records, dict):
            return []
        items = [record for record in records.values() if isinstance(record, dict)]
        return sorted(
            items,
            key=lambda record: (
                int(record.get("step_count") or 0),
                str(record.get("operation_id") or ""),
            ),
        )

    def _continuation_artifact_ids(self, previous: dict[str, Any], *, limit: int = 4) -> list[str]:
        candidates: list[str] = []
        retrieval_cache = getattr(self.harness.state, "retrieval_cache", None)
        if isinstance(retrieval_cache, list):
            candidates.extend(str(item).strip() for item in retrieval_cache if str(item).strip())

        for record in reversed(self._tool_execution_record_items()):
            result = record.get("result")
            if not isinstance(result, dict) or not bool(result.get("success")):
                continue
            metadata = result.get("metadata")
            if not isinstance(metadata, dict):
                continue
            artifact_id = str(metadata.get("artifact_id") or "").strip()
            if artifact_id:
                candidates.append(artifact_id)

        previous_ids = previous.get("last_good_artifact_ids")
        if isinstance(previous_ids, list):
            candidates.extend(str(item).strip() for item in previous_ids if str(item).strip())
        return dedupe_keep_tail(candidates, limit=limit)

    def _continuation_research_artifact_ids(self, previous: dict[str, Any], *, limit: int = 2) -> list[str]:
        candidates: list[str] = []
        for record in reversed(self._tool_execution_record_items()):
            tool_name = str(record.get("tool_name") or "").strip()
            if tool_name not in {"web_search", "web_fetch"}:
                continue
            result = record.get("result")
            if not isinstance(result, dict) or not bool(result.get("success")):
                continue
            metadata = result.get("metadata")
            if not isinstance(metadata, dict):
                continue
            artifact_id = str(metadata.get("artifact_id") or "").strip()
            if artifact_id:
                candidates.append(artifact_id)

        previous_ids = previous.get("recent_research_artifact_ids")
        if isinstance(previous_ids, list):
            candidates.extend(str(item).strip() for item in previous_ids if str(item).strip())

        fallback_ids = previous.get("last_good_artifact_ids")
        if isinstance(fallback_ids, list):
            for artifact_id in fallback_ids:
                normalized_id = str(artifact_id or "").strip()
                artifact = self.harness.state.artifacts.get(normalized_id)
                if normalized_id and artifact is not None and str(getattr(artifact, "kind", "")).strip() in {
                    "web_search",
                    "web_fetch",
                }:
                    candidates.append(normalized_id)
        return dedupe_keep_tail(candidates, limit=limit)

    def _continuation_next_required_tool(self, previous: dict[str, Any]) -> dict[str, Any]:
        for record in reversed(self._tool_execution_record_items()):
            result = record.get("result")
            if not isinstance(result, dict):
                continue
            metadata = result.get("metadata")
            if not isinstance(metadata, dict):
                continue
            next_required_tool = metadata.get("next_required_tool")
            if isinstance(next_required_tool, dict) and str(next_required_tool.get("tool_name") or "").strip():
                normalized = json_safe_value(next_required_tool)
                return normalized if isinstance(normalized, dict) else {}
        previous_tool = previous.get("next_required_tool")
        if isinstance(previous_tool, dict):
            normalized = json_safe_value(previous_tool)
            return normalized if isinstance(normalized, dict) else {}
        return {}

    def _continuation_last_failed_tool(self, previous: dict[str, Any]) -> dict[str, Any]:
        for record in reversed(self._tool_execution_record_items()):
            result = record.get("result")
            if not isinstance(result, dict) or bool(result.get("success")):
                continue
            tool_name = str(record.get("tool_name") or "").strip()
            if not tool_name:
                continue
            return {
                "tool_name": tool_name,
                "error": clip_text_value(str(result.get("error") or "").strip(), limit=220)[0],
            }
        previous_failed = previous.get("last_failed_tool")
        if isinstance(previous_failed, dict) and str(previous_failed.get("tool_name") or "").strip():
            normalized = json_safe_value(previous_failed)
            return normalized if isinstance(normalized, dict) else {}
        return {}

    def _continuation_primary_ssh_target(
        self,
        ssh_targets: list[dict[str, str]],
        previous: dict[str, Any],
    ) -> dict[str, Any]:
        resolved_remote = self.harness.state.scratchpad.get("_resolved_remote_followup")
        if isinstance(resolved_remote, dict):
            host = str(resolved_remote.get("host") or "").strip().lower()
            user = str(resolved_remote.get("user") or "").strip()
            if host:
                payload: dict[str, Any] = {"host": host}
                if user:
                    payload["user"] = user
                return payload
        if len(ssh_targets) == 1:
            target = ssh_targets[0]
            host = str(target.get("host") or "").strip().lower()
            user = str(target.get("user") or "").strip()
            if host:
                payload = {"host": host}
                if user:
                    payload["user"] = user
                return payload
        previous_target = previous.get("ssh_target")
        if isinstance(previous_target, dict) and str(previous_target.get("host") or "").strip():
            normalized = json_safe_value(previous_target)
            return normalized if isinstance(normalized, dict) else {}
        return {}

    def refresh_task_handoff_action_options(self, assistant_text: str) -> None:
        handoff = self.last_task_handoff()
        if not handoff:
            return
        inherited_paths = handoff.get("target_paths")
        if not isinstance(inherited_paths, list):
            inherited_paths = []
        extracted_options = _extract_action_options_from_text(assistant_text, list(inherited_paths))
        if not extracted_options:
            return
        existing_options = handoff.get("action_options")
        if not isinstance(existing_options, list):
            existing_options = []
        handoff["action_options"] = _merge_action_options(existing_options, extracted_options)
        handoff["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.harness.state.scratchpad["_last_task_handoff"] = handoff

    def initialize_run_brief(self, task: str, *, raw_task: str | None = None) -> None:
        effective_task = _collapse_task_chain(task)
        source_task = str(raw_task or effective_task).strip()
        continue_like = self._is_contextual_followup(source_task)
        remote_mission_task = self._remote_followup_mission_task(source_task)
        previous_task = _collapse_task_chain(
            self.harness.state.scratchpad.pop("_task_boundary_previous_task", "") or ""
        )
        existing_task = _collapse_task_chain(self.harness.state.run_brief.original_task or "")
        canonical_task = remote_mission_task or effective_task or existing_task or previous_task

        self.harness.state.run_brief.original_task = canonical_task
        self.harness.state.run_brief.task_contract = derive_task_contract(canonical_task)
        task_mode_source = canonical_task
        if (
            isinstance(self.harness.state.scratchpad.get("_resolved_remote_followup"), dict)
            and classify_task_mode(canonical_task) != "remote_execute"
            and classify_task_mode(effective_task) == "remote_execute"
        ):
            task_mode_source = effective_task
        self.harness.state.task_mode = classify_task_mode(task_mode_source)

        existing_phase_objective = str(self.harness.state.run_brief.current_phase_objective or "").strip()
        if remote_mission_task and effective_task:
            self.harness.state.run_brief.current_phase_objective = (
                f"{self.harness.state.current_phase}: {effective_task}"
            )
        elif continue_like and existing_phase_objective:
            self.harness.state.run_brief.current_phase_objective = existing_phase_objective
        elif effective_task:
            self.harness.state.run_brief.current_phase_objective = (
                f"{self.harness.state.current_phase}: {effective_task}"
            )
        elif not existing_phase_objective:
            self.harness.state.run_brief.current_phase_objective = self.harness.state.current_phase

        existing_goal = _collapse_task_chain(self.harness.state.working_memory.current_goal or "")
        plan = self.harness.state.active_plan or self.harness.state.draft_plan
        plan_goal = _collapse_task_chain(getattr(plan, "goal", "") or "")
        if self._is_corrective_resteer_followup(source_task) and existing_goal:
            next_goal = existing_goal
        elif remote_mission_task:
            if plan_goal and _normalize_task_text(plan_goal) == _normalize_task_text(existing_goal):
                next_goal = plan_goal
            elif existing_goal and not _is_remote_followup_wrapper(existing_goal):
                next_goal = existing_goal
            else:
                next_goal = remote_mission_task
        elif continue_like and plan_goal and _normalize_task_text(plan_goal) == _normalize_task_text(existing_goal):
            next_goal = plan_goal
        else:
            next_goal = canonical_task or existing_goal
        self.harness.state.working_memory.current_goal = next_goal

        self.harness.state.scratchpad["_task_target_paths"] = extract_task_target_paths(effective_task)
        self.store_task_handoff(raw_task=source_task, effective_task=effective_task)
        if self._ordinal_followup_index(source_task) is not None:
            handoff = self.last_task_handoff()
            option = self._selected_action_option(source_task, handoff)
            if option is not None:
                target_info = self._resolve_option_target_paths(source_task, option, handoff)
                self._apply_resolved_followup_metadata(
                    source_task,
                    option,
                    target_info,
                    self._resolved_option_task(source_task, option, handoff),
                )
        if hasattr(self.harness.memory, "prime_write_policy"):
            self.harness.memory.prime_write_policy(effective_task)
        self.harness.state.working_memory.next_actions = dedupe_keep_tail(
            self.harness.state.working_memory.next_actions + [next_action_for_task(self.harness, effective_task)],
            limit=6,
        )

    def current_user_task(self) -> str:
        for message in reversed(self.harness.state.recent_messages):
            if message.role == "user" and message.content:
                content = str(message.content or "").strip()
                resolved = self.resolve_followup_task(content)
                return _collapse_task_chain(resolved or content)
        last_task = self.harness.state.scratchpad.get("_last_task_text")
        if isinstance(last_task, str) and last_task:
            return _collapse_task_chain(last_task)
        return _collapse_task_chain(self.harness.state.run_brief.original_task)
