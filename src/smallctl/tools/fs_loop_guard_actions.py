from __future__ import annotations

from typing import Any


def next_required_read(path: str) -> dict[str, Any]:
    return {
        "tool_name": "file_read",
        "required_fields": ["path"],
        "required_arguments": {"path": path},
        "optional_fields": ["start_line", "end_line"],
        "notes": [
            "Read the current staged content before attempting another chunk write.",
            "Confirm the exact missing section or lines from the read result before writing again.",
        ],
    }


def forced_escape_action(path: str, session_id: str) -> dict[str, Any]:
    return {
        "strategy": "escape_write_dead_end",
        "allowed_operations": ["full_file_rewrite", "finalize_and_verify", "human_handoff"],
        "notes": [
            "Do not retry the same chunk write again.",
            "Either perform one complete file_write overwrite, finalize and verify the staged file, or ask_human for handoff.",
        ],
        "full_file_rewrite": {
            "tool_name": "file_write",
            "required_arguments": {
                "path": path,
                "write_session_id": session_id,
                "section_name": "full_file",
                "replace_strategy": "overwrite",
            },
        },
        "finalize_and_verify": {"tool_name": "finalize_write_session"},
        "human_handoff": {"tool_name": "ask_human"},
    }


def outline_handoff_question(path: str) -> str:
    return (
        f"LoopGuard outline required for `{path}`.\n"
        "- Bullet 1: next missing section\n"
        "- Bullet 2: section after that\n"
        "- Bullet 3: final section or verification step\n"
        "Reply `continue` to resume writing, or provide corrections first."
    )
