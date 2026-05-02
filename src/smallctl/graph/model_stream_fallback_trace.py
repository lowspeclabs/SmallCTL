from __future__ import annotations

import time
from typing import Any

from ..models.events import UIEvent, UIEventType
from ..state import clip_text_value
from .deps import GraphRuntimeDeps


def _record_text_write_fallback_state(
    harness: Any,
    *,
    status: str,
    reason: str,
    session: Any,
    current_section: str,
    remaining_sections: list[str],
    prompt: str,
    tool_names: list[str],
    assistant_text: str | None = None,
    extracted_code: str | None = None,
    next_section_name: str | None = None,
) -> None:
    harness.state.scratchpad["_last_text_write_fallback"] = {
        "status": status,
        "reason": reason,
        "write_session_id": str(getattr(session, "write_session_id", "") or "").strip(),
        "target_path": str(getattr(session, "write_target_path", "") or "").strip(),
        "current_section": current_section,
        "remaining_sections": list(remaining_sections),
        "tool_names": list(tool_names),
        "prompt_excerpt": clip_text_value(prompt, limit=400)[0],
        "assistant_excerpt": clip_text_value(assistant_text or "", limit=400)[0] if assistant_text else "",
        "extracted_code_excerpt": clip_text_value(extracted_code or "", limit=400)[0] if extracted_code else "",
        "next_section_name": next_section_name or "",
        "timestamp": time.time(),
    }


def _build_text_write_fallback_trace(
    *,
    session: Any,
    current_section: str,
    prompt: str,
    assistant_text: str,
    extracted_code: str,
    next_section_name: str,
    tool_names: list[str],
) -> str:
    prompt_excerpt, prompt_was_clipped = clip_text_value(prompt, limit=800)
    response_excerpt, response_was_clipped = clip_text_value(assistant_text, limit=800)
    code_excerpt, code_was_clipped = clip_text_value(extracted_code, limit=800)

    lines = [
        "Chat-mode fallback activated for a stalled write task.",
        f"Target path: `{getattr(session, 'write_target_path', '')}`.",
        f"Current section: `{current_section or 'imports'}`.",
        f"Next section: `{next_section_name or 'none'}`.",
        f"Observed tool calls: {', '.join(tool_names) if tool_names else 'none'}.",
        "",
        "Prompt sent to the model:",
        prompt_excerpt + (" [truncated]" if prompt_was_clipped else ""),
        "",
        "Assistant response:",
        response_excerpt + (" [truncated]" if response_was_clipped else ""),
        "",
        "Extraction method:",
        "first fenced code block, otherwise raw response",
        "",
        "Extracted script or key details:",
        code_excerpt if code_excerpt else "(no extractable script found)",
    ]
    if code_was_clipped and code_excerpt:
        lines[-1] += " [truncated]"
    return "\n".join(lines)


async def _emit_text_write_fallback_trace(
    harness: Any,
    deps: GraphRuntimeDeps,
    *,
    session: Any,
    current_section: str,
    prompt: str,
    assistant_text: str,
    extracted_code: str,
    next_section_name: str,
    tool_names: list[str],
) -> None:
    trace_text = _build_text_write_fallback_trace(
        session=session,
        current_section=current_section,
        prompt=prompt,
        assistant_text=assistant_text,
        extracted_code=extracted_code,
        next_section_name=next_section_name,
        tool_names=tool_names,
    )
    await harness._emit(
        deps.event_handler,
        UIEvent(
            event_type=UIEventType.ASSISTANT,
            content=trace_text,
            data={
                "kind": "print",
                "artifact_id": "chat-mode-fallback-trace",
                "speaker": "assistant",
            },
        ),
    )
