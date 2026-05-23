from __future__ import annotations

import re
import shlex
from typing import Any

from ..diagnostic_tasks import diagnostic_failure_completion_allowed
from ..state import LoopState, clip_text_value
from ..write_session_fsm import recent_write_session_events, record_write_session_event
from .common import fail, ok
from .fs_loop_guard import build_loop_guard_status


_WRITE_SESSION_SCHEMA_FAILURE_KEY = "_last_write_session_schema_failure"
_REMOTE_MUTATION_VERIFICATION_KEY = "_remote_mutation_requires_verification"
_MULTI_OBJECTIVE_LEDGER_KEY = "_multi_objective_ledger"
_ISSUE_LIST_CONTEXT_RE = re.compile(
    r"\b(?:fix|address|resolve|handle|correct).{0,80}\b(?:issues|bugs|findings|items|problems)\b",
    re.IGNORECASE | re.DOTALL,
)
_ISSUE_BULLET_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[.)]\s+)(?P<body>.+?)\s*$")
_SEVERITY_PREFIX_RE = re.compile(r"^(?:critical|high|medium|low|p[0-3])\s*:\s+", re.IGNORECASE)
_OBJECTIVE_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*(?:\(\))?")
_OBJECTIVE_STOPWORDS = {
    "about",
    "after",
    "again",
    "already",
    "also",
    "before",
    "both",
    "cannot",
    "complete",
    "could",
    "does",
    "done",
    "fail",
    "fails",
    "fix",
    "fixed",
    "following",
    "from",
    "handle",
    "high",
    "ignore",
    "ignores",
    "into",
    "issue",
    "issues",
    "large",
    "look",
    "medium",
    "method",
    "missing",
    "only",
    "patch",
    "patches",
    "probably",
    "prompt",
    "report",
    "safe",
    "should",
    "state",
    "still",
    "test",
    "that",
    "there",
    "this",
    "with",
    "works",
}
_REMOTE_BINARYISH_SUFFIXES = (
    ".asc",
    ".bin",
    ".crt",
    ".deb",
    ".der",
    ".gpg",
    ".gz",
    ".key",
    ".pem",
    ".pfx",
    ".tar",
    ".tgz",
    ".xz",
    ".zip",
)


def _normalized_verifier_verdict(state: LoopState) -> dict[str, Any] | None:
    verdict = state.current_verifier_verdict()
    if not isinstance(verdict, dict) or not verdict:
        return None
    stale = state.scratchpad.get("_last_verifier_stale_after_mutation")
    if isinstance(stale, dict) and stale:
        verdict = dict(verdict)
        raw_paths = stale.get("paths", [])
        if not isinstance(raw_paths, list):
            raw_paths = [raw_paths]
        paths = [str(path).strip() for path in raw_paths if str(path).strip()]
        verdict["stale"] = True
        verdict["stale_reason"] = str(stale.get("reason") or "file_changed_after_verifier")
        verdict["stale_after_tool"] = str(stale.get("tool_name") or "")
        if paths:
            verdict["stale_after_paths"] = paths
        command = str(verdict.get("command") or verdict.get("target") or "").strip()
        note = "Rerun the focused verifier after the latest file change."
        if command:
            note = f"Rerun the focused verifier after the latest file change: `{command}`."
        verdict["next_required_action"] = {
            "tool_name": "shell_exec",
            "notes": [note, "Do not poll `loop_status` waiting for a stale verifier verdict to change."],
        }
    return verdict


def _verifier_failure_summary(verifier_verdict: dict[str, Any] | None) -> str:
    if not isinstance(verifier_verdict, dict) or not verifier_verdict:
        return ""

    bits: list[str] = []
    target_text, clipped = clip_text_value(
        str(verifier_verdict.get("command") or verifier_verdict.get("target") or "").strip(),
        limit=180,
    )
    if target_text:
        suffix = " [truncated]" if clipped else ""
        bits.append(f"check={target_text}{suffix}")

    detail = ""
    acceptance_delta = verifier_verdict.get("acceptance_delta")
    if isinstance(acceptance_delta, dict):
        notes = acceptance_delta.get("notes")
        if isinstance(notes, list):
            detail = next((str(note).strip() for note in notes if str(note).strip()), "")
    if not detail:
        detail = str(
            verifier_verdict.get("key_stderr")
            or verifier_verdict.get("key_stdout")
            or ""
        ).strip()
    detail_text, clipped = clip_text_value(detail, limit=180)
    if detail_text:
        suffix = " [truncated]" if clipped else ""
        bits.append(f"details={detail_text}{suffix}")

    return " | ".join(bits)


def _verifier_requires_human_approval(verifier_verdict: dict[str, Any] | None) -> bool:
    if not isinstance(verifier_verdict, dict) or not verifier_verdict:
        return False
    if bool(verifier_verdict.get("approval_denied")):
        return True
    if str(verifier_verdict.get("verdict", "")).strip() == "needs_human":
        return True
    acceptance_delta = verifier_verdict.get("acceptance_delta")
    if isinstance(acceptance_delta, dict):
        status = str(acceptance_delta.get("status") or "").strip().lower()
        if status == "pending":
            notes = acceptance_delta.get("notes")
            if isinstance(notes, list):
                return any("denied by user" in str(note).strip().lower() for note in notes)
    return False


def _write_session_schema_failure(state: LoopState) -> dict[str, Any] | None:
    payload = state.scratchpad.get(_WRITE_SESSION_SCHEMA_FAILURE_KEY)
    if not isinstance(payload, dict) or not payload:
        return None
    return payload


def _remote_mutation_verification_requirement(state: LoopState) -> dict[str, Any] | None:
    payload = state.scratchpad.get(_REMOTE_MUTATION_VERIFICATION_KEY)
    if not isinstance(payload, dict) or not payload:
        return None
    if payload.get("failed_verification_attempts", 0) >= 3:
        return None
    if not _remote_mutation_has_pending_verifier(payload):
        return None
    return payload


def _remote_mutation_has_pending_verifier(requirement: dict[str, Any]) -> bool:
    guessed_paths = requirement.get("guessed_paths")
    if not isinstance(guessed_paths, list):
        guessed_paths = []
    verified_paths = {
        str(path).strip()
        for path in requirement.get("verified_paths", [])
        if str(path).strip()
    }
    if any(str(path).strip() and str(path).strip() not in verified_paths for path in guessed_paths):
        return True

    verified_directories = {
        str(path).strip().rstrip("/")
        for path in requirement.get("verified_directory_empty_checks", [])
        if str(path).strip()
    }
    return any(
        check["path"] not in verified_directories
        for check in _remote_mutation_directory_checks(requirement)
    )


def _remote_path_needs_presence_probe(path: str) -> bool:
    lowered = str(path or "").strip().lower()
    return bool(lowered) and lowered.endswith(_REMOTE_BINARYISH_SUFFIXES)


def _remote_presence_probe_command(path: str) -> str:
    quoted = shlex.quote(str(path or "").strip())
    return f"test -s {quoted} && sha256sum {quoted}"


def _remote_mutation_block_payload(requirement: dict[str, Any]) -> dict[str, Any]:
    guessed_paths = requirement.get("guessed_paths")
    if not isinstance(guessed_paths, list):
        guessed_paths = []
    verified_paths = {
        str(path).strip()
        for path in requirement.get("verified_paths", [])
        if str(path).strip()
    }
    pending_paths = [
        str(path).strip()
        for path in guessed_paths
        if str(path).strip() and str(path).strip() not in verified_paths
    ]
    directory_checks = _remote_mutation_directory_checks(requirement)
    verified_directories = {
        str(path).strip().rstrip("/")
        for path in requirement.get("verified_directory_empty_checks", [])
        if str(path).strip()
    }
    pending_directory_checks = [
        check for check in directory_checks if check["path"] not in verified_directories
    ]
    path_hint = ", ".join(str(path) for path in pending_paths[:3] if str(path).strip())
    first_path = next((str(path).strip() for path in pending_paths if str(path).strip()), "")
    first_directory_check = pending_directory_checks[0] if pending_directory_checks else {}
    host = str(requirement.get("host") or "").strip()
    user = str(requirement.get("user") or "").strip()
    mutation_type = str(requirement.get("mutation_type") or "").strip().lower()
    required_arguments: dict[str, Any] = {}
    first_path_needs_presence_probe = mutation_type != "deletion" and _remote_path_needs_presence_probe(first_path)
    if first_path:
        if first_path_needs_presence_probe:
            required_arguments["command"] = _remote_presence_probe_command(first_path)
        else:
            required_arguments["path"] = first_path
    elif first_directory_check:
        directory_path = str(first_directory_check.get("path") or "").strip()
        if directory_path:
            required_arguments["command"] = (
                f"find {directory_path} -mindepth 1 -maxdepth 1 -print -quit"
            )
    if host:
        required_arguments["host"] = host
    if user:
        required_arguments["user"] = user
    if host and "@" in host:
        required_arguments.pop("host", None)
        required_arguments["target"] = host

    if mutation_type == "deletion":
        if first_path:
            error = (
                "Cannot complete the task while a raw `ssh_exec` remote file deletion still needs meaningful verification. "
                "Verify the target is gone with `ssh_file_read`; a `not found` / `no such file` result counts as successful verification."
            )
            next_required_action = {
                "tool_names": ["ssh_file_read"],
                "required_fields": sorted(required_arguments),
                "required_arguments": required_arguments,
                "notes": [
                    "Read the deleted path directly.",
                    "A missing-file result is valid proof for deletion tasks and will clear the requirement.",
                ],
            }
        else:
            directory_path = str(first_directory_check.get("path") or "").strip()
            glob = str(first_directory_check.get("glob") or "").strip()
            error = (
                "Cannot complete the task while a raw `ssh_exec` remote glob deletion still needs meaningful verification. "
                "Verify the parent directory is empty with a read-only `ssh_exec` check."
            )
            next_required_action = {
                "tool_names": ["ssh_exec"],
                "required_fields": sorted(required_arguments),
                "required_arguments": required_arguments,
                "notes": [
                    f"Check that `{directory_path}` has no remaining entries from `{glob or directory_path + '/*'}`.",
                    "Empty stdout from the suggested find command is valid proof for glob deletion tasks.",
                ],
            }
    else:
        if first_path_needs_presence_probe:
            error = (
                "Cannot complete the task while a raw `ssh_exec` remote file mutation still needs meaningful verification. "
                "Verify the changed binary or key file exists and has content with a read-only `ssh_exec` presence/hash check."
            )
            next_required_action = {
                "tool_names": ["ssh_exec"],
                "required_fields": sorted(required_arguments),
                "required_arguments": required_arguments,
                "notes": [
                    "Binary/key files should be verified with metadata or hashes, not decoded as text.",
                    "A successful `test -s ... && sha256sum ...` check is valid proof for this file.",
                ],
            }
        else:
            error = (
                "Cannot complete the task while a raw `ssh_exec` remote file mutation still needs meaningful verification. "
                "Read back the changed file with `ssh_file_read`, or redo the edit with `ssh_file_patch` / "
                "`ssh_file_replace_between` so the harness can verify the readback hash."
            )
            next_required_action = {
                "tool_names": ["ssh_file_read", "ssh_file_patch", "ssh_file_replace_between"],
                "required_fields": sorted(required_arguments),
                "required_arguments": required_arguments,
                "notes": [
                    "A grep-only positive match is not enough for replacement tasks.",
                    "Verify that the replacement exists and the old target is gone.",
                ],
            }
    if path_hint:
        error += f" Suspected path(s): {path_hint}."
    if first_directory_check:
        error += f" Directory check: {first_directory_check.get('path')}."
    if required_arguments:
        tool_name = "ssh_file_read" if "path" in required_arguments else "ssh_exec"
        verifier_call = tool_name + "(" + ", ".join(
            f"{key}={required_arguments[key]!r}"
            for key in ("target", "host", "user", "path", "command")
            if key in required_arguments
        ) + ")"
        error += f" Next required verifier: `{verifier_call}`."
    return {
        "error": error,
        "next_required_action": next_required_action,
    }


def _remote_mutation_directory_checks(requirement: dict[str, Any]) -> list[dict[str, str]]:
    raw_checks = requirement.get("directory_empty_checks")
    if not isinstance(raw_checks, list):
        return []
    checks: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_checks:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip().rstrip("/")
        if not path or path in seen:
            continue
        seen.add(path)
        glob = str(item.get("glob") or "").strip()
        checks.append({"path": path, "glob": glob})
    return checks


def _write_session_resume_action(
    state: LoopState,
    failure: dict[str, Any] | None,
) -> dict[str, Any] | None:
    session = state.write_session
    if session is None or str(session.status or "").strip().lower() == "complete":
        return None

    # If the session already has all sections and no next section, suggest finalizing.
    is_finalizable = (
        session.write_sections_completed
        and not str(session.write_next_section or "").strip()
        and str(session.status or "open").strip().lower() in {"open", "verifying"}
    )
    if is_finalizable:
        return {
            "tool_name": "finalize_write_session",
            "required_fields": [],
            "required_arguments": {},
            "optional_fields": [],
            "notes": ["The write session is ready to finalize. Call finalize_write_session to promote the staged file."],
        }

    section_name = str(
        (failure or {}).get("recommended_section_name")
        or session.write_next_section
        or session.write_current_section
        or "imports"
    ).strip() or "imports"
    required_arguments = {
        "path": str((failure or {}).get("target_path") or session.write_target_path or "").strip(),
        "write_session_id": str(session.write_session_id or ""),
        "section_name": section_name,
    }
    notes = ["Provide non-empty `content` for this section."]
    if session.write_sections_completed and not session.write_next_section:
        notes.append("Omit `next_section_name` on the final chunk so the session can finalize after verification.")
    else:
        notes.append("Set `next_section_name` only if another section still needs to be written after this one.")
    if failure:
        missing = failure.get("required_fields")
        if isinstance(missing, list) and missing:
            notes.append(
                "Last schema failure was missing: "
                + ", ".join(str(field) for field in missing if str(field).strip())
            )
    return {
        "tool_name": "file_write",
        "required_fields": ["path", "content", "write_session_id", "section_name"],
        "required_arguments": required_arguments,
        "optional_fields": ["next_section_name"],
        "notes": notes,
    }


def _is_weather_lookup_task(state: LoopState) -> bool:
    task_text = str(getattr(getattr(state, "run_brief", None), "original_task", "") or "").strip().lower()
    if not task_text:
        return False
    return any(marker in task_text for marker in ("weather", "forecast", "temperature"))


def _has_specific_weather_answer(message: str) -> bool:
    text = " ".join(str(message or "").strip().lower().split())
    if not text:
        return False

    explicit_unavailable_markers = (
        "could not verify the exact",
        "couldn't verify the exact",
        "unable to verify the exact",
        "exact weather could not be verified",
        "exact current weather could not be verified",
        "exact temperature could not be verified",
        "i could not verify the exact",
    )
    if any(marker in text for marker in explicit_unavailable_markers):
        return True

    temperature_markers = ("°f", "°c", " fahrenheit", " celsius", " degree", " degrees")
    weather_markers = (
        "temperature",
        "temp",
        "forecast",
        "high",
        "low",
        "today",
        "currently",
        "weather",
        "sunny",
        "cloudy",
        "clear",
        "rain",
        "showers",
        "storm",
        "snow",
        "windy",
        "humid",
        "overcast",
        "drizzle",
        "thunder",
    )
    has_temperature = any(marker in text for marker in temperature_markers)
    has_weather_detail = any(marker in text for marker in weather_markers)
    return has_temperature and has_weather_detail


def _looks_like_weather_search_meta_completion(message: str) -> bool:
    text = " ".join(str(message or "").strip().lower().split())
    if not text:
        return False
    if _has_specific_weather_answer(text):
        return False
    meta_markers = (
        "web search completed",
        "search completed",
        "found ",
        "returned ",
    )
    return any(marker in text for marker in meta_markers) and "result" in text


def _unresolved_missing_input_file(state: LoopState) -> dict | None:
    scratchpad = getattr(state, "scratchpad", {})
    if not isinstance(scratchpad, dict):
        return None
    blocker = scratchpad.get("_unresolved_missing_input_file")
    if isinstance(blocker, dict) and str(blocker.get("path") or "").strip():
        return blocker
    return None


def _candidate_multi_objective_texts(state: LoopState) -> list[str]:
    texts: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in texts:
            texts.append(text)

    add(getattr(state.run_brief, "original_task", ""))
    add(getattr(state.run_brief, "current_phase_objective", ""))
    add(getattr(state.working_memory, "current_goal", ""))
    scratchpad = getattr(state, "scratchpad", {})
    if isinstance(scratchpad, dict):
        for key in ("_task_transaction", "_last_task_handoff"):
            payload = scratchpad.get(key)
            if not isinstance(payload, dict):
                continue
            for field in ("raw_task", "effective_task", "current_goal", "user_goal"):
                add(payload.get(field))
    return texts


def _clean_objective_title(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip()).strip(" \"'")
    return text[:260]


def _extract_multi_objectives(text: str) -> list[str]:
    raw_text = str(text or "")
    if not raw_text.strip():
        return []
    raw_text = re.sub(
        r"\s+([-*]\s+(?:critical|high|medium|low|p[0-3])\s*:)",
        r"\n\1",
        raw_text,
        flags=re.IGNORECASE,
    )
    has_issue_context = _ISSUE_LIST_CONTEXT_RE.search(raw_text) is not None
    objectives: list[str] = []
    current: list[str] = []
    severity_count = 0
    in_code_block = False

    def flush() -> None:
        if not current:
            return
        title = _clean_objective_title(" ".join(current))
        current.clear()
        if title and title not in objectives:
            objectives.append(title)

    for raw_line in raw_text.splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith("```"):
            flush()
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        match = _ISSUE_BULLET_RE.match(line)
        if match:
            body = str(match.group("body") or "").strip()
            is_severity_item = _SEVERITY_PREFIX_RE.match(body) is not None
            if not has_issue_context and not is_severity_item:
                flush()
                continue
            flush()
            if is_severity_item:
                severity_count += 1
            current.append(body)
            continue
        if current and line.startswith((" ", "\t")) and line.strip():
            current.append(line.strip())
        elif current and not line.strip():
            flush()
    flush()
    if len(objectives) < 2:
        return []
    if not has_issue_context and severity_count < 2:
        return []
    return objectives


def _ensure_multi_objective_ledger(state: LoopState) -> dict[str, Any] | None:
    scratchpad = getattr(state, "scratchpad", {})
    if not isinstance(scratchpad, dict):
        return None
    existing = scratchpad.get(_MULTI_OBJECTIVE_LEDGER_KEY)
    if isinstance(existing, dict) and isinstance(existing.get("objectives"), list):
        return existing

    best_text = ""
    best_objectives: list[str] = []
    for text in _candidate_multi_objective_texts(state):
        objectives = _extract_multi_objectives(text)
        if len(objectives) > len(best_objectives):
            best_text = text
            best_objectives = objectives
    if len(best_objectives) < 2:
        return None

    ledger = {
        "status": "active",
        "parent_goal": _clean_objective_title(best_text)[:500],
        "objectives": [
            {
                "objective_id": f"O{index}",
                "title": title,
                "status": "pending",
                "evidence": [],
            }
            for index, title in enumerate(best_objectives, start=1)
        ],
    }
    scratchpad[_MULTI_OBJECTIVE_LEDGER_KEY] = ledger
    return ledger


def _objective_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in _OBJECTIVE_TOKEN_RE.findall(str(text or "").lower()):
        token = raw.strip("()")
        if len(token) < 4 or token in _OBJECTIVE_STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _objective_matches_text(objective: dict[str, Any], text: str) -> bool:
    title = str(objective.get("title") or "").strip().lower()
    haystack = str(text or "").strip().lower()
    if not title or not haystack:
        return False
    if title in haystack:
        return True
    title_tokens = _objective_tokens(title)
    if not title_tokens:
        return False
    matched = title_tokens.intersection(_objective_tokens(haystack))
    required = max(2, min(4, len(title_tokens) // 3 or 1))
    return len(matched) >= required


def _resolved_followup_objective_id(state: LoopState) -> str:
    scratchpad = getattr(state, "scratchpad", {})
    if not isinstance(scratchpad, dict):
        return ""
    resolved = scratchpad.get("_resolved_followup")
    if not isinstance(resolved, dict):
        return ""
    try:
        index = int(resolved.get("option_index") or 0)
    except (TypeError, ValueError):
        return ""
    return f"O{index}" if index > 0 else ""


def _mark_objective_done(objective: dict[str, Any], message: str) -> bool:
    if str(objective.get("status") or "").strip().lower() == "done":
        return False
    objective["status"] = "done"
    evidence = str(message or "").strip()
    if evidence:
        current = objective.get("evidence")
        if not isinstance(current, list):
            current = []
        current.append(evidence[:240])
        objective["evidence"] = current[-4:]
    return True


def _multi_objective_completion_block(
    state: LoopState,
    *,
    message: str,
    verifier_verdict: dict[str, Any] | None,
) -> dict[str, Any] | None:
    ledger = _ensure_multi_objective_ledger(state)
    if not ledger:
        return None
    objectives = [item for item in ledger.get("objectives", []) if isinstance(item, dict)]
    if not objectives:
        return None

    resolved_objective_id = _resolved_followup_objective_id(state)
    completed_now: list[str] = []
    for objective in objectives:
        if (
            resolved_objective_id
            and str(objective.get("objective_id") or "") == resolved_objective_id
        ) or _objective_matches_text(objective, message):
            if _mark_objective_done(objective, message):
                completed_now.append(str(objective.get("objective_id") or ""))

    remaining = [
        {
            "objective_id": str(item.get("objective_id") or ""),
            "title": str(item.get("title") or ""),
        }
        for item in objectives
        if str(item.get("status") or "").strip().lower() != "done"
    ]
    if not remaining:
        ledger["status"] = "done"
        return None

    return {
        "completed_now": completed_now,
        "remaining_objectives": remaining,
        "ledger": ledger,
        "last_verifier_verdict": verifier_verdict,
    }


def _open_plan_subtasks(state: LoopState) -> list[dict[str, Any]]:
    plan = state.active_plan or state.draft_plan
    ledger = getattr(state, "subtask_ledger", None)
    if plan is None or ledger is None:
        return []
    step_ids = {
        str(getattr(step, "step_id", "") or "").strip()
        for step in plan.iter_steps()
        if str(getattr(step, "step_id", "") or "").strip()
    }
    if not step_ids:
        return []
    open_items: list[dict[str, Any]] = []
    for task in getattr(ledger, "subtasks", []) or []:
        subtask_id = str(getattr(task, "subtask_id", "") or "").strip()
        if subtask_id not in step_ids:
            continue
        status = str(getattr(task, "status", "") or "").strip().lower()
        if status in {"done", "abandoned"}:
            continue
        open_items.append(
            {
                "subtask_id": subtask_id,
                "title": str(getattr(task, "title", "") or ""),
                "goal": str(getattr(task, "goal", "") or ""),
                "status": status or "pending",
                "acceptance": list(getattr(task, "acceptance", []) or []),
                "evidence": list(getattr(task, "evidence", []) or [])[-3:],
                "blockers": list(getattr(task, "blockers", []) or [])[-3:],
                "next_action": getattr(task, "next_action", None),
                "attempts": int(getattr(task, "attempts", 0) or 0),
            }
        )
    return open_items


def _plan_subtask_completion_block(
    state: LoopState,
    *,
    verifier_verdict: dict[str, Any] | None,
) -> dict[str, Any] | None:
    open_items = _open_plan_subtasks(state)
    if not open_items:
        return None
    first = open_items[0]
    status = str(first.get("status") or "").lower()
    blocked = status in {"blocked", "failed"}
    if blocked:
        next_required_action = {
            "tool_names": ["escalate_to_bigger_model", "ask_human", "task_fail"],
            "notes": [
                "Escalate to a bigger model if stronger debugging or planning is needed.",
                "Ask the human if progress requires missing information, approval, credentials, or an ambiguous choice.",
                "Fail the task only if the blocker is terminal.",
            ],
        }
    else:
        next_required_action = {
            "tool_names": ["loop_status"],
            "notes": [
                "Continue the active plan subtask with concrete tool evidence.",
                "Do not call task_complete until all plan subtasks are done or explicitly abandoned/failed.",
            ],
        }
    return {
        "open_plan_subtasks": open_items,
        "next_required_subtask": first,
        "next_required_action": next_required_action,
        "last_verifier_verdict": verifier_verdict,
    }


async def task_complete(message: str, state: LoopState, harness: Any) -> dict:
    if state.plan_execution_mode and state.active_step_id:
        return fail(
            "Cannot call `task_complete` while staged execution is active. Use `step_complete` for the active step.",
            metadata={
                "reason": "task_complete_blocked_in_staged_execution",
                "active_step_id": state.active_step_id,
                "active_step_run_id": state.active_step_run_id,
            },
        )
    verifier_verdict = _normalized_verifier_verdict(state)
    ledger_service = getattr(harness, "subtask_ledger", None)
    if ledger_service is not None:
        try:
            ledger_service.import_plan_if_needed(replace_synthetic_root=True)
        except Exception:
            pass
    remote_requirement = _remote_mutation_verification_requirement(state)
    if remote_requirement is not None:
        block_payload = _remote_mutation_block_payload(remote_requirement)
        return fail(
            str(block_payload["error"]),
            metadata={
                "reason": "remote_mutation_requires_verification",
                "remote_mutation_requirement": remote_requirement,
                "next_required_action": block_payload["next_required_action"],
                "last_verifier_verdict": verifier_verdict,
            },
        )
    missing_input = _unresolved_missing_input_file(state)
    if missing_input is not None:
        path = str(missing_input.get("path") or "").strip()
        return fail(
            f"Cannot complete the task because required input file `{path}` was not found.",
            metadata={
                "reason": "missing_required_input_file",
                "missing_input_file": missing_input,
                "next_required_action": {
                    "tool_names": ["file_read", "ask_human", "task_fail"],
                    "notes": [
                        "Read the correct input file if the path was a typo.",
                        "Ask the user for the correct path if no intended file is available.",
                        "Do not infer the missing file contents from memory or directory listings.",
                    ],
                },
                "last_verifier_verdict": verifier_verdict,
            },
        )
    session = state.write_session
    if session is not None and str(session.status or "").strip().lower() != "complete":
        # P2: auto-finalize if the session is clearly ready but was never finalized.
        from ..graph.write_session_outcomes import _attempt_write_session_finalize
        from ..tools.fs_sessions import _write_session_can_finalize

        is_finalizable = (
            not str(session.write_next_section or "").strip()
            and session.write_sections_completed
            and str(session.status or "open").strip().lower() in {"open", "verifying"}
            and _write_session_can_finalize(session)
        )
        if is_finalizable:
            finalized, _ = await _attempt_write_session_finalize(harness, session)
            if finalized:
                session = state.write_session

        if str(session.status or "").strip().lower() != "complete":
            # Fix 3 (RCA 8b79ca76): Track consecutive task_complete rejections
            # for the same write session. Auto-abandon orphan sessions (no
            # sections written, target file already exists) to prevent deadlock.
            scratchpad = state.scratchpad
            blocker_key = f"write_session:{session.write_session_id}"
            last_blocker = scratchpad.get("_task_complete_last_blocker")
            if last_blocker == blocker_key:
                scratchpad["_task_complete_blocker_count"] = scratchpad.get("_task_complete_blocker_count", 0) + 1
            else:
                scratchpad["_task_complete_last_blocker"] = blocker_key
                scratchpad["_task_complete_blocker_count"] = 1

            if scratchpad.get("_task_complete_blocker_count", 0) >= 2:
                has_no_sections = not session.write_sections_completed
                from ..tools.fs import _resolve
                try:
                    target = _resolve(session.write_target_path, getattr(state, "cwd", None))
                    file_exists = target.exists() and target.is_file() and target.stat().st_size > 0
                except Exception:
                    file_exists = False

                if has_no_sections and file_exists:
                    record_write_session_event(
                        state,
                        event="session_abandoned",
                        session=session,
                        details={
                            "reason": "auto_abandoned_orphan_session",
                            "rejection_count": scratchpad["_task_complete_blocker_count"],
                        },
                    )
                    state.write_session = None
                    scratchpad.pop("_task_complete_last_blocker", None)
                    scratchpad.pop("_task_complete_blocker_count", None)
                    return await task_complete(message, state, harness)

            record_write_session_event(
                state,
                event="task_complete_blocked",
                session=session,
                details={"reason": "session_incomplete"},
            )
            session_status = str(session.status or "open").strip() or "open"
            error = (
                "Cannot complete the task while Write Session "
                f"`{session.write_session_id}` for `{session.write_target_path}` is still {session_status}."
            )
            next_section = str(session.write_next_section or "").strip()
            if next_section:
                error += f" Next required section: `{next_section}`."
            elif session.write_pending_finalize:
                error += " The staged file still needs verification and finalization."
            else:
                error += " The staged file has not been finalized to the target path yet."
            failure = _write_session_schema_failure(state)
            return fail(
                error,
                metadata={
                    "write_session": session.to_dict(),
                    "next_required_tool": _write_session_resume_action(state, failure),
                    "last_verifier_verdict": verifier_verdict,
                    "acceptance_checklist": state.acceptance_checklist(),
                },
            )
    verifier_failure_satisfies_diagnostic = (
        verifier_verdict
        and str(verifier_verdict.get("verdict", "")).strip() not in {"", "pass"}
        and diagnostic_failure_completion_allowed(state, message=message, verifier=verifier_verdict)
    )
    if (
        verifier_verdict
        and _verifier_requires_human_approval(verifier_verdict)
        and not state.acceptance_waived
    ):
        error = "Cannot complete the task until the latest verifier check is approved or rerun with approval."
        verifier_summary = _verifier_failure_summary(verifier_verdict)
        if verifier_summary:
            error = f"{error} Latest verifier: {verifier_summary}."
        return fail(
            error,
            metadata={
                "last_verifier_verdict": verifier_verdict,
                "acceptance_checklist": state.acceptance_checklist(),
                "approval_required": True,
            },
        )
    if (
        verifier_verdict
        and str(verifier_verdict.get("verdict", "")).strip() not in {"", "pass"}
        and not state.acceptance_waived
        and not verifier_failure_satisfies_diagnostic
    ):
        error = "Cannot complete the task while the latest verifier verdict is still failing."
        verifier_summary = _verifier_failure_summary(verifier_verdict)
        if verifier_summary:
            error = f"{error} Latest verifier: {verifier_summary}."
        return fail(
            error,
            metadata={
                "last_verifier_verdict": verifier_verdict,
                "acceptance_checklist": state.acceptance_checklist(),
            },
        )
    plan_subtask_block = _plan_subtask_completion_block(state, verifier_verdict=verifier_verdict)
    if plan_subtask_block is not None:
        next_subtask = plan_subtask_block["next_required_subtask"]
        return fail(
            "Cannot complete the task while plan subtasks are still open. "
            f"Next required subtask: {next_subtask.get('subtask_id')} - {next_subtask.get('title')}.",
            metadata={
                "reason": "plan_subtasks_incomplete",
                **plan_subtask_block,
            },
        )
    if not state.acceptance_ready() and not verifier_failure_satisfies_diagnostic:
        checklist = state.acceptance_checklist()
        pending = [item["criterion"] for item in checklist if not item["satisfied"]]
        return fail(
            "Cannot complete the task until acceptance criteria are satisfied or waived.",
            metadata={
                "pending_acceptance_criteria": pending,
                "acceptance_checklist": checklist,
                "last_verifier_verdict": _normalized_verifier_verdict(state),
            },
        )
    if _is_weather_lookup_task(state) and _looks_like_weather_search_meta_completion(message):
        return fail(
            "Task is not complete yet: the user asked for the weather, but your completion message only reports that a search ran. "
            "Provide the actual weather answer with attribution, or explicitly say that the exact weather could not be verified from the evidence you fetched. "
            "Do not finish with only result counts or source lists.",
            metadata={
                "reason": "lookup_answer_missing",
                "lookup_kind": "weather",
                "next_required_action": {
                    "tool_name": "web_fetch",
                    "notes": [
                        "Fetch a returned search result by `result_id` instead of inventing a forecast URL.",
                        "If fetches still fail, answer explicitly that the exact weather could not be verified from the available evidence.",
                    ],
                },
            },
        )
    objective_block = _multi_objective_completion_block(
        state,
        message=message,
        verifier_verdict=verifier_verdict,
    )
    if objective_block is not None:
        remaining = objective_block["remaining_objectives"]
        first_remaining = str(remaining[0].get("title") or "the next open objective")
        completed_now = objective_block["completed_now"]
        if completed_now:
            error = (
                "Marked the current subobjective complete, but the parent task still has open objectives. "
                f"Next open objective: {first_remaining}"
            )
        else:
            error = (
                "Cannot complete the parent task because it still has open objectives and the completion message "
                f"did not match a specific open objective. Next open objective: {first_remaining}"
            )
        return fail(
            error,
            metadata={
                "reason": "multi_objective_incomplete",
                "completed_objectives": completed_now,
                "remaining_objectives": remaining,
                "multi_objective_ledger": objective_block["ledger"],
                "last_verifier_verdict": objective_block["last_verifier_verdict"],
            },
        )
    state.scratchpad["_task_complete"] = True
    state.scratchpad["_task_complete_message"] = message
    state.touch()
    return ok({"status": "complete", "message": message})


async def step_complete(message: str, state: LoopState, harness: Any) -> dict:
    if not state.plan_execution_mode or not state.active_step_id:
        return fail(
            "`step_complete` is only available while staged execution has an active step.",
            metadata={"reason": "staged_execution_inactive"},
        )
    remote_requirement = _remote_mutation_verification_requirement(state)
    if remote_requirement is not None:
        block_payload = _remote_mutation_block_payload(remote_requirement)
        return fail(
            str(block_payload["error"]).replace("task", "step", 1),
            metadata={
                "reason": "remote_mutation_requires_verification",
                "remote_mutation_requirement": remote_requirement,
                "next_required_action": block_payload["next_required_action"],
            },
        )
    session = state.write_session
    if session is not None and str(session.status or "").strip().lower() != "complete":
        return fail(
            "Cannot complete the step while an active write session is incomplete.",
            metadata={
                "reason": "write_session_incomplete",
                "write_session": session.to_dict(),
            },
        )
    state.scratchpad["_step_complete_requested"] = True
    state.scratchpad["_step_complete_message"] = str(message or "")
    state.scratchpad.pop("_step_failed_requested", None)
    state.scratchpad.pop("_step_failed_message", None)
    state.touch()
    log = getattr(harness, "log", None)
    if log is not None and callable(getattr(log, "info", None)):
        log.info(
            "staged_step_completion_requested step_id=%s step_run_id=%s",
            state.active_step_id,
            state.active_step_run_id,
        )
    return ok(
        {
            "status": "step_completion_requested",
            "message": message,
            "step_id": state.active_step_id,
            "step_run_id": state.active_step_run_id,
        }
    )


async def step_fail(message: str, state: LoopState, harness: Any) -> dict:
    if not state.plan_execution_mode or not state.active_step_id:
        return fail(
            "`step_fail` is only available while staged execution has an active step.",
            metadata={"reason": "staged_execution_inactive"},
        )
    state.scratchpad["_step_failed_requested"] = True
    state.scratchpad["_step_failed_message"] = str(message or "")
    state.scratchpad.pop("_step_complete_requested", None)
    state.scratchpad.pop("_step_complete_message", None)
    state.recent_errors.append(str(message or "Step failed."))
    state.touch()
    return ok(
        {
            "status": "step_failed_requested",
            "message": message,
            "step_id": state.active_step_id,
            "step_run_id": state.active_step_run_id,
        }
    )


async def finalize_write_session(state: LoopState, harness: Any) -> dict:
    session = state.write_session
    if not session:
        return fail("No active write session to finalize.")
    if str(session.status or "").strip().lower() == "complete":
        target_path = str(getattr(session, "write_target_path", "") or "").strip()
        message = "Write session is already complete."
        if target_path:
            message += f" Promoted file: `{target_path}`."
        return ok({"status": "already_finalized", "message": message})

    from ..graph.write_session_outcomes import _attempt_write_session_finalize
    success, detail = await _attempt_write_session_finalize(harness, session)
    if success:
        return ok({"status": "finalized", "message": detail})
    return fail(f"Unable to finalize write session: {detail}")


async def task_fail(message: str, state: LoopState) -> dict:
    session = state.write_session
    if session is not None and str(session.status or "").strip().lower() != "complete":
        record_write_session_event(
            state,
            event="session_abandoned",
            session=session,
            details={"reason": "task_fail"},
        )
    state.scratchpad["_task_failed"] = True
    state.scratchpad["_task_failed_message"] = message
    state.recent_errors.append(message)
    state.touch()
    return ok({"status": "failed", "message": message})


async def ask_human(question: str, state: LoopState) -> dict:
    state.scratchpad["_ask_human"] = True
    state.scratchpad["_ask_human_question"] = question
    state.touch()
    return ok({"status": "human_input_required", "question": question})


def _subtask_ledger_status(state: LoopState) -> dict[str, Any] | None:
    ledger = getattr(state, "subtask_ledger", None)
    if ledger is None:
        return None
    subtasks = []
    for task in getattr(ledger, "subtasks", []) or []:
        subtasks.append(
            {
                "subtask_id": str(getattr(task, "subtask_id", "") or ""),
                "title": str(getattr(task, "title", "") or ""),
                "goal": str(getattr(task, "goal", "") or ""),
                "status": str(getattr(task, "status", "") or ""),
                "acceptance": list(getattr(task, "acceptance", []) or []),
                "evidence": list(getattr(task, "evidence", []) or [])[-3:],
                "blockers": list(getattr(task, "blockers", []) or [])[-3:],
                "next_action": getattr(task, "next_action", None),
                "attempts": int(getattr(task, "attempts", 0) or 0),
            }
        )
    return {
        "task_id": getattr(ledger, "task_id", None),
        "active_subtask_id": getattr(ledger, "active_subtask_id", None),
        "active_subtask": next(
            (item for item in subtasks if item["subtask_id"] == getattr(ledger, "active_subtask_id", None)),
            None,
        ),
        "subtasks": subtasks[-12:],
        "done_subtask_ids": [item["subtask_id"] for item in subtasks if item["status"] == "done"],
        "pending_subtask_ids": [
            item["subtask_id"] for item in subtasks if item["status"] in {"pending", "active", "blocked"}
        ],
    }


async def loop_status(state: LoopState) -> dict:
    max_steps = state.scratchpad.get("_max_steps")
    try:
        max_steps_int = int(max_steps) if max_steps is not None else 0
    except (TypeError, ValueError):
        max_steps_int = 0

    if max_steps_int > 0:
        progress_pct = min(1.0, max(0.0, state.step_count / max_steps_int))
    else:
        progress_pct = 0.0

    verifier_verdict = _normalized_verifier_verdict(state)
    acceptance_checklist = state.acceptance_checklist()
    contract_phase = state.contract_phase()
    write_session_failure = _write_session_schema_failure(state)
    write_session_payload = state.write_session.to_dict() if state.write_session else None
    write_session_events = recent_write_session_events(state, limit=10)
    next_required_tool = _write_session_resume_action(state, write_session_failure)
    if write_session_payload is not None and write_session_failure is not None:
        write_session_payload = dict(write_session_payload)
        write_session_payload["last_schema_failure"] = write_session_failure
    if write_session_payload is not None and next_required_tool is not None:
        write_session_payload = dict(write_session_payload)
        write_session_payload["resume_action"] = next_required_tool

    # Fix 3: Emit a persistent, top-level reminder while a session is open.
    # The model must include write_session_id on every file_write/file_patch/ast_patch call
    # to the session target; omitting it will now be rejected (see fs.py Fix 1).
    write_session_warning: str | None = None
    active_ws = state.write_session
    if active_ws is not None and str(getattr(active_ws, "status", "") or "").strip().lower() not in {"complete"}:
        _ws_id_hint = str(getattr(active_ws, "write_session_id", "") or "").strip()
        _ws_path_hint = str(getattr(active_ws, "write_target_path", "") or "").strip()
        _ws_next_hint = str(getattr(active_ws, "write_next_section", "") or "").strip() or "imports"
        if _ws_id_hint and _ws_path_hint:
            write_session_warning = (
                f"Write Session `{_ws_id_hint}` is open for `{_ws_path_hint}`. "
                f"All file_write / file_patch / ast_patch calls to that path MUST include "
                f"`write_session_id='{_ws_id_hint}'` and `section_name='{_ws_next_hint}'`. "
                f"A bare file_write without write_session_id will be rejected and will NOT "
                f"advance the session. task_complete will be blocked until the session is finalized."
            )

    return ok(
        {
            "phase": state.current_phase,
            "contract_phase": contract_phase,
            "step_count": state.step_count,
            "token_usage": state.token_usage,
            "elapsed_seconds": state.elapsed_seconds,
            "recent_errors": state.recent_errors[-5:],
            "cwd": state.cwd,
            "active_tool_profiles": state.active_tool_profiles,
            "plan_execution_mode": state.plan_execution_mode,
            "active_step_id": state.active_step_id,
            "active_step_run_id": state.active_step_run_id,
            "step_verification_result": state.step_verification_result,
            "max_steps": max_steps_int or None,
            "progress_pct": round(progress_pct, 4),
            "acceptance_ready": state.acceptance_ready(),
            "acceptance_waived": state.acceptance_waived,
            "acceptance_checklist": acceptance_checklist,
            "pending_acceptance_criteria": [item["criterion"] for item in acceptance_checklist if not item["satisfied"]],
            "subtask_ledger": _subtask_ledger_status(state),
            "last_verifier_verdict": verifier_verdict,
            "last_failure_class": state.last_failure_class,
            "files_changed_this_cycle": state.files_changed_this_cycle,
            "system_repair_cycle_id": state.repair_cycle_id,
            "stagnation_counters": state.stagnation_counters,
            "next_required_tool": next_required_tool,
            "write_session": write_session_payload,
            "write_session_warning": write_session_warning,
            "write_session_events": write_session_events,
            "loop_guard": build_loop_guard_status(state),
        }
    )
