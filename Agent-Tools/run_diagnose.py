#!/usr/bin/env python3
"""run_diagnose — produce a structured diagnosis of a SmallCTL run.

This is the heavier cousin of logwatch.py. It prints a narrative report
suitable for starting an agentic debug loop.

Examples:
  python Agent-Tools/run_diagnose.py
  python Agent-Tools/run_diagnose.py 6d6c87f1
  python Agent-Tools/run_diagnose.py --save
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from agent_tools_lib import (
    Colors,
    colorize,
    error_records,
    event_counter,
    extract_trace_id,
    format_duration,
    format_record_summary,
    iter_records,
    load_summaries,
    load_task_summaries,
    resolve_run_dir,
    run_duration_seconds,
    warning_records,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose a SmallCTL run.")
    parser.add_argument("run", nargs="?", default="latest", help="Run dir, run id, 'latest', or 'latest-N'")
    parser.add_argument("--logs-dir", help="Custom logs directory")
    parser.add_argument("--save", action="store_true", help="Write diagnosis to <run_dir>/diagnosis.json")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    return parser.parse_args()


def _has_write_overwrite_guard_failures(failed_dispatches: list[dict[str, Any]]) -> bool:
    for rec in failed_dispatches:
        data = rec.get("data") or {}
        if not isinstance(data, dict):
            continue
        text = json.dumps(data, default=str, ensure_ascii=False)
        if "chunked_write_overwrite_new_section_after_progress" in text:
            return True
        if "Refusing to overwrite the entire staged file for a new chunk section" in text:
            return True
    return False



def _file_patch_target_loop_count(failed_dispatches: list[dict[str, Any]]) -> int:
    count = 0
    needles = (
        "patch_target_not_found",
        "target text not found",
        "target_text_not_found",
        "exact text",
        "repair_cycle_read_required",
        "fresh_file_read_required_before_patch",
        "reading the target file before patching",
        "old target text is gone",
        "patch already landed",
        "already matches",
        "no changes needed",
        "repeat_sensitive_patch_already_applied",
    )
    for rec in failed_dispatches:
        data = rec.get("data") or {}
        if not isinstance(data, dict):
            continue
        tool_name = str(data.get("tool_name") or "").strip()
        if tool_name != "file_patch":
            continue
        text = json.dumps(data, default=str, ensure_ascii=False).lower()
        if any(needle in text for needle in needles):
            count += 1
    return count


def _has_chat_terminal_repetition_stall(records: list[dict[str, Any]], session: dict[str, Any]) -> bool:
    """Detect a chat-mode task that ended with a degenerate loop and no tools.

    This is distinct from loop-mode terminal tool stalls and from general
    model degeneration, because the root cause is usually that the runtime
    exposed only terminal tools (task_complete/task_fail) for an implementation
    task, leaving the model unable to make progress.
    """
    last_task_id = str(session.get("latest_task_id") or "").strip()
    if not last_task_id:
        return False
    saw_terminal_only = False
    saw_degenerate = False
    saw_chat_completed = False
    tool_call_count = 0
    for rec in records:
        tid = extract_trace_id(rec) or ""
        if not (":" + last_task_id + ":" in tid):
            continue
        event = str(rec.get("event") or "")
        data = rec.get("data") or {}
        if event == "chat_tool_selection" and isinstance(data, dict):
            reason = str(data.get("reason") or "").strip()
            if reason in {"non_lookup_chat_terminal_only", "smalltalk_terminal_only"}:
                saw_terminal_only = True
        if event == "model_output_degenerate_loop_exhausted":
            saw_degenerate = True
        if event == "dispatch_tools_start":
            tool_call_count += 1
        if event == "task_finalize":
            if isinstance(data, dict):
                result = data.get("result", {})
                if str(result.get("status") or "").strip().lower() == "chat_completed":
                    saw_chat_completed = True
    return saw_terminal_only and saw_degenerate and saw_chat_completed and tool_call_count == 0


def _has_ask_human_resume_terminal_stall(records: list[dict[str, Any]]) -> bool:
    saw_interrupt_resume = False
    saw_terminal_only_after_resume = False
    saw_stall_after_resume = False
    for rec in records:
        event = str(rec.get("event") or "")
        data = rec.get("data") or {}
        if event == "interrupt_resume":
            saw_interrupt_resume = True
            continue
        if not saw_interrupt_resume:
            continue
        if event == "chat_tool_selection" and isinstance(data, dict):
            reason = str(data.get("reason") or "").strip()
            if reason in {"non_lookup_chat_terminal_only", "smalltalk_terminal_only"}:
                saw_terminal_only_after_resume = True
        if event in {"action_stall", "model_output_degenerate_loop_exhausted"}:
            saw_stall_after_resume = True
    return saw_interrupt_resume and saw_terminal_only_after_resume and saw_stall_after_resume


def _classify_failure(
    events: Counter[str],
    errors: list[dict[str, Any]],
    session: dict[str, Any],
    failed_dispatches: list[dict[str, Any]],
    harness_records: list[dict[str, Any]],
) -> str:
    overall = session.get("overall_objective_status")
    final = session.get("final_task_status")
    deliverable_verified = session.get("deliverable_verified")
    if overall in {"complete", "completed"} and deliverable_verified is True and not errors:
        return "success"
    if overall in {"complete", "completed"} and deliverable_verified is True and errors:
        return "success_with_errors"
    if _has_chat_terminal_repetition_stall(harness_records, session):
        return "chat_terminal_repetition_stall"
    if _has_write_overwrite_guard_failures(failed_dispatches):
        return "write_session_overwrite_guard_loop"
    if _file_patch_target_loop_count(failed_dispatches) >= 3:
        return "file_patch_target_not_found_loop"
    if _has_ask_human_resume_terminal_stall(harness_records):
        return "ask_human_resume_terminal_tool_stall"
    if events.get("model_output_degenerate_loop_exhausted"):
        return "model_degeneration"
    if events.get("action_stall") or events.get("no_tool_recovery"):
        return "model_tool_loop_stall"
    if events.get("dispatch_tools_error") or events.get("initialize_run_error"):
        return "runtime_exception"
    if events.get("tool_blocked_not_exposed"):
        return "policy_block"
    if events.get("fama_tool_call_blocked"):
        return "fama_block"
    if events.get("recovery_failure_event_recorded"):
        return "recovery_failure"
    if overall == "incomplete" and session.get("deliverable_verified") is False:
        return "incomplete_unverified"
    return "unknown"


def _diagnose(run_dir: Path) -> dict[str, Any]:
    summaries = load_summaries(run_dir)
    session = summaries.get("session_summary", {})
    task_summary = summaries.get("task_summary", {})
    task_summaries = load_task_summaries(run_dir)

    harness_records = list(iter_records(run_dir, "harness"))
    tools_records = list(iter_records(run_dir, "tools"))
    events = event_counter(harness_records)
    errors = error_records(harness_records)
    warnings = warning_records(harness_records)

    dispatch_completes = [r for r in tools_records if r.get("event") == "dispatch_complete"]
    failed_dispatches = [r for r in dispatch_completes if (r.get("data") or {}).get("success") is False]
    tool_failures_by_name: Counter[str] = Counter()
    for rec in failed_dispatches:
        name = (rec.get("data") or {}).get("tool_name", "unknown")
        tool_failures_by_name[name] += 1

    last_error_trace_ids = [extract_trace_id(e) for e in errors[-5:]]

    diagnosis = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "duration_seconds": run_duration_seconds(harness_records),
        "session_summary": session,
        "task_summary": task_summary,
        "failure_classification": _classify_failure(events, errors, session, failed_dispatches, harness_records),
        "event_counts": dict(events),
        "error_record_count": len(errors),
        "warning_record_count": len(warnings),
        "recent_errors": [format_record_summary(e) for e in errors[-10:]],
        "last_error_trace_ids": [t for t in last_error_trace_ids if t],
        "tool_failure_count": len(failed_dispatches),
        "tool_failures_by_name": dict(tool_failures_by_name),
        "task_count": len(task_summaries),
        "task_statuses": [{"task_id": t.get("task_id"), "status": t.get("status"), "reason": t.get("reason")} for t in task_summaries],
        "recommended_next_steps": _recommend_next_steps(events, errors, session, failed_dispatches, harness_records),
    }
    return diagnosis


def _recommend_next_steps(
    events: Counter[str],
    errors: list[dict[str, Any]],
    session: dict[str, Any],
    failed_dispatches: list[dict[str, Any]],
    harness_records: list[dict[str, Any]],
) -> list[str]:
    steps: list[str] = []
    if errors and (tid := extract_trace_id(errors[-1])):
        steps.append(f"Trace the most recent error: python3 Agent-Tools/trace_call.py --run <run> {tid}")
    if events.get("model_output_degenerate_loop_exhausted"):
        steps.append("Investigate model stream degeneration; check model_output.log for repeated phrases.")
    if _has_write_overwrite_guard_failures(failed_dispatches):
        steps.append("Write-session overwrite guard loop detected; trace failed file_write calls and force a current file_read followed by file_patch/ast_patch or same-section repair.")
    patch_loop_count = _file_patch_target_loop_count(failed_dispatches)
    if patch_loop_count >= 3:
        steps.append(f"Repeated file_patch target mismatch loop detected ({patch_loop_count} failures); force a non-cached file_read of the target before allowing another patch and verify using exact live text.")
    if _has_ask_human_resume_terminal_stall(harness_records):
        steps.append("ask_human resume fell into terminal-only tool exposure; preserve original task context and expose file mutation tools after affirmative replies.")
    if _has_chat_terminal_repetition_stall(harness_records, session):
        steps.append("Chat-mode terminal-only tool exposure caused a repetition loop. Route implementation follow-ups to loop mode and expose file mutation tools.")
    if events.get("action_stall") or events.get("no_tool_recovery"):
        steps.append("Model is struggling to emit valid tool calls; inspect recent prompts and tool schemas.")
    if events.get("tool_blocked_not_exposed"):
        steps.append("Model called a tool not exposed this turn; check chat_tool_selection / phase / profile filtering.")
    if events.get("fama_tool_call_blocked"):
        steps.append("FAMA blocked a tool call; review fama signals and required verifier fingerprints.")
    if failed_dispatches:
        steps.append(f"{len(failed_dispatches)} tool dispatches failed; check tools.jsonl for error details.")
    if session.get("deliverable_verified") is False and session.get("overall_objective_status") == "incomplete":
        progress = session.get("challenge_progress") if isinstance(session, dict) else None
        if isinstance(progress, dict) and progress.get("code_change_count") and not progress.get("verified_after_last_change"):
            command = str(progress.get("last_verifier_command") or "").strip()
            suffix = f" Last known verifier: `{command}`." if command else ""
            steps.append(f"Latest code changes were not verified before termination.{suffix}")
        else:
            steps.append("Deliverable not verified; confirm required artifacts exist and verifiers ran.")
    if not steps:
        steps.append("No obvious failure pattern; review harness.log manually or compare with a baseline run.")
    return steps


def _render_text(diagnosis: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(colorize(f"Diagnosis for {diagnosis['run_name']}", Colors.BOLD + Colors.CYAN))
    lines.append(f"Run directory: {diagnosis['run_dir']}")
    lines.append(f"Duration: {format_duration(diagnosis.get('duration_seconds'))}")
    lines.append("")
    lines.append(colorize("Failure classification", Colors.BOLD + Colors.BLUE))
    lines.append(f"  {diagnosis['failure_classification']}")

    session = diagnosis["session_summary"]
    task_summary = diagnosis["task_summary"]
    lines.append("")
    lines.append(colorize("Outcome", Colors.BOLD + Colors.BLUE))
    lines.append(f"  overall_objective_status: {session.get('overall_objective_status', 'n/a')}")
    lines.append(f"  final_task_status: {task_summary.get('final_task_status', 'n/a')}")
    lines.append(f"  deliverable_verified: {session.get('deliverable_verified', 'n/a')}")

    lines.append("")
    lines.append(colorize("Tasks", Colors.BOLD + Colors.BLUE))
    for t in diagnosis["task_statuses"]:
        lines.append(f"  {t['task_id']}: {t['status']}  {t.get('reason') or ''}")

    lines.append("")
    lines.append(colorize("Errors", Colors.BOLD + Colors.BLUE))
    lines.append(f"  error records: {diagnosis['error_record_count']}")
    lines.append(f"  warning records: {diagnosis['warning_record_count']}")
    lines.append(f"  failed tool dispatches: {diagnosis['tool_failure_count']}")
    if diagnosis["tool_failures_by_name"]:
        lines.append("  failures by tool:")
        for name, count in diagnosis["tool_failures_by_name"].items():
            lines.append(f"    {count:>3}  {name}")
    if diagnosis["recent_errors"]:
        lines.append("  recent errors:")
        for err in diagnosis["recent_errors"]:
            lines.append("    " + colorize(err, Colors.RED))

    lines.append("")
    lines.append(colorize("Recommended next steps", Colors.BOLD + Colors.GREEN))
    for step in diagnosis["recommended_next_steps"]:
        lines.append(f"  - {step}")

    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    logs_dir = Path(args.logs_dir) if args.logs_dir else None
    try:
        run_dir = resolve_run_dir(args.run, logs_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(colorize(str(exc), Colors.RED), file=sys.stderr)
        return 1

    diagnosis = _diagnose(run_dir)

    if args.json:
        print(json.dumps(diagnosis, indent=2, default=str))
    else:
        print(_render_text(diagnosis))

    if args.save:
        path = run_dir / "diagnosis.json"
        path.write_text(json.dumps(diagnosis, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nSaved diagnosis to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
