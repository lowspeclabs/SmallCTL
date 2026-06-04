from __future__ import annotations

from difflib import get_close_matches
from pathlib import Path
from typing import Any

from .state import PendingToolCall
from .tool_call_parser import _extract_artifact_id_from_args

_WRITE_OUTPUT_KEYWORDS = (
    "save",
    "write",
    "store",
    "export",
    "persist",
    "record",
)


def _path_mentions_in_task_text(harness: Any, raw_path: str) -> bool:
    normalized_path = raw_path.strip().lower()
    path_name = Path(raw_path).name.lower()
    task_text = _merged_task_text(harness)
    if not task_text or not normalized_path:
        return False
    return normalized_path in task_text or (path_name and path_name in task_text)


def _nearby_missing_input_candidates(harness: Any, raw_path: str) -> list[str]:
    path = Path(raw_path)
    cwd = getattr(getattr(harness, "state", None), "cwd", None)
    base = Path(cwd) if isinstance(cwd, str) and cwd else Path.cwd()
    search_dir = path.parent if str(path.parent) not in {"", "."} else Path(".")
    if not search_dir.is_absolute():
        search_dir = base / search_dir
    try:
        names = [item.name for item in search_dir.iterdir() if item.is_file()]
    except OSError:
        return []
    matches = get_close_matches(path.name, names, n=3, cutoff=0.72)
    if not matches:
        return []
    prefix = "" if str(path.parent) in {"", "."} else f"{path.parent}/"
    return [f"{prefix}{name}" for name in matches]


def recent_assistant_texts(harness: Any, *, limit: int = 2) -> list[str]:
    texts: list[str] = []
    for message in reversed(getattr(harness.state, "recent_messages", [])):
        if getattr(message, "role", "") != "assistant":
            continue
        content = str(getattr(message, "content", "") or "").strip()
        if not content:
            continue
        texts.append(content)
        if len(texts) >= limit:
            break
    return texts


def looks_like_freeze_or_hang(harness: Any, assistant_text: str) -> bool:
    text = str(assistant_text or "").strip()
    if not text:
        return False
    recent = recent_assistant_texts(harness, limit=3)
    if not recent:
        return False
    historical = recent[1:] if recent and recent[0] == text else recent
    if text in historical:
        return True
    return False


def task_prefers_summary_synthesis(harness: Any) -> bool:
    merged = _merged_task_text(harness)
    if not merged:
        return False
    asks_for_summary = any(keyword in merged for keyword in ("table", "summary", "summarize", "report", "overview", "present"))
    asks_about_listing = any(
        keyword in merged
        for keyword in ("list", "listing", "files", "directories", "artifact", "results", "output", "current env", "cron", "job")
    )
    return asks_for_summary and asks_about_listing


def _merged_task_text(harness: Any) -> str:
    texts = [
        str(getattr(getattr(harness, "state", None), "run_brief", None).original_task or "")
        if getattr(getattr(harness, "state", None), "run_brief", None) is not None
        else "",
        str(getattr(getattr(harness, "state", None), "working_memory", None).current_goal or "")
        if getattr(getattr(harness, "state", None), "working_memory", None) is not None
        else "",
    ]
    current_user_task = getattr(harness, "_current_user_task", None)
    if callable(current_user_task):
        texts.append(str(current_user_task() or ""))
    return " ".join(text.strip().lower() for text in texts if text and text.strip())


def _pending_artifact_record(harness: Any, pending: PendingToolCall) -> Any | None:
    artifact_id = str(_extract_artifact_id_from_args(pending.args) or "").strip()
    if not artifact_id:
        return None
    artifacts = getattr(getattr(harness, "state", None), "artifacts", {})
    if not isinstance(artifacts, dict):
        return None
    return artifacts.get(artifact_id)


def artifact_prefers_summary_synthesis(harness: Any, pending: PendingToolCall) -> bool:
    if pending.tool_name not in {"artifact_read", "artifact_print"}:
        return False
    if task_prefers_summary_synthesis(harness):
        return True

    artifact = _pending_artifact_record(harness, pending)
    if artifact is None:
        return False

    artifact_kind = str(getattr(artifact, "tool_name", "") or getattr(artifact, "kind", "") or "").strip().lower()
    if artifact_kind not in {"web_search", "web_fetch"}:
        return False

    merged = _merged_task_text(harness)
    if not merged:
        return False

    asks_for_web_research_synthesis = any(
        keyword in merged
        for keyword in (
            "web search",
            "websearch",
            "research",
            "findings",
            "what it is",
            "what is ",
            "how it works",
            "how does it work",
            "explain",
            "detailed summary",
        )
    )
    return asks_for_web_research_synthesis


def _task_requests_written_output_path(harness: Any, raw_path: str) -> bool:
    if not raw_path:
        return False
    task_text = _merged_task_text(harness)
    if not task_text:
        return False
    if not _path_mentions_in_task_text(harness, raw_path):
        return False
    return any(keyword in task_text for keyword in _WRITE_OUTPUT_KEYWORDS)


def _has_prior_successful_evidence_for_output_write(harness: Any) -> bool:
    history = getattr(getattr(harness, "state", None), "tool_history", [])
    if not isinstance(history, list):
        return False
    for item in reversed(history):
        text = str(item or "")
        if "|success" not in text:
            continue
        tool_name = text.split("|", 1)[0]
        if tool_name not in {"file_read", "dir_list", "memory_update", "task_complete", "task_fail"}:
            return True
    return False
