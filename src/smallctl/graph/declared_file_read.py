from __future__ import annotations

import json
from typing import Any

from ..task_targets import primary_task_target_path
from .state import GraphRunState, PendingToolCall


_REASONING_FALLBACK_FLAG = "_assistant_text_from_reasoning_fallback"
_DECLARED_FILE_READ_MARKERS = (
    "call file_read",
    "use file_read",
    "invoke file_read",
    "run file_read",
    "file_read to read",
    "read the file",
    "read this file",
    "read that file",
    "read it first",
)
_DECLARED_ACTION_PREFIXES = (
    "let me",
    "i need to",
    "i'll",
    "i will",
    "i'm going to",
    "i am going to",
    "first i",
    "next i",
)


def _maybe_synthesize_declared_file_read(
    graph_state: GraphRunState,
    harness: Any,
    assistant_text: str,
) -> bool:
    if graph_state.pending_tool_calls:
        return False
    text = str(assistant_text or "").strip()
    if not text:
        return False

    low_text = text.lower()
    if "file_read" not in low_text and not any(marker in low_text for marker in _DECLARED_FILE_READ_MARKERS):
        return False
    if not any(prefix in low_text for prefix in _DECLARED_ACTION_PREFIXES) and "file_read" not in low_text:
        return False

    target_path = str(primary_task_target_path(harness) or "").strip()
    if not target_path:
        return False
    if target_path.lower() not in low_text and "file_read" not in low_text:
        return False

    args = {"path": target_path}
    graph_state.pending_tool_calls = [
        PendingToolCall(
            tool_name="file_read",
            args=args,
            raw_arguments=json.dumps(args, ensure_ascii=True, sort_keys=True),
            source="system",
            parser_metadata={"auto_repaired_from_text": "declared_file_read"},
        )
    ]
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", None)
    if isinstance(scratchpad, dict):
        scratchpad["_declared_file_read_synthesized"] = {
            "path": target_path,
            "assistant_text_preview": text[:240],
        }
        scratchpad.pop("_action_stalls", None)
        scratchpad.pop("_blank_message_nudges", None)
        scratchpad.pop("_small_model_continue_nudges", None)
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "declared_file_read_synthesized",
            "synthesized file_read from assistant text that declared a read action",
            path=target_path,
            assistant_text_preview=text[:240],
        )
    return True


def _consume_reasoning_fallback_flag(harness: Any) -> bool:
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return False
    return bool(scratchpad.pop(_REASONING_FALLBACK_FLAG, False))
