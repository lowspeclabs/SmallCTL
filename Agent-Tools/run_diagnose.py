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
    detect_apt_deb822_guard_misfire,
    detect_primary_blockers,
    error_records,
    event_counter,
    extract_trace_id,
    file_patch_target_loop_count,
    format_duration,
    format_record_summary,
    get_run_objective,
    has_ask_human_resume_terminal_stall,
    has_chat_terminal_repetition_stall,
    has_continue_prompt_budget_loop,
    has_patch_first_policy_loop,
    has_stderr_signature_circuit_breaker,
    has_strong_environment_blocker,
    has_tool_call_protocol_mismatch,
    has_write_overwrite_guard_failures,
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
    return has_write_overwrite_guard_failures(failed_dispatches)


def _file_patch_target_loop_count(failed_dispatches: list[dict[str, Any]]) -> int:
    return file_patch_target_loop_count(failed_dispatches)


def _has_chat_terminal_repetition_stall(records: list[dict[str, Any]], session: dict[str, Any]) -> bool:
    return has_chat_terminal_repetition_stall(records, session)


def _has_ask_human_resume_terminal_stall(records: list[dict[str, Any]]) -> bool:
    return has_ask_human_resume_terminal_stall(records)


def _has_patch_first_policy_loop(failed_dispatches: list[dict[str, Any]]) -> bool:
    return has_patch_first_policy_loop(failed_dispatches)


def _classify_failure(
    events: Counter[str],
    errors: list[dict[str, Any]],
    session: dict[str, Any],
    failed_dispatches: list[dict[str, Any]],
    harness_records: list[dict[str, Any]],
    *,
    model_output_records: list[dict[str, Any]] | None = None,
    tools_records: list[dict[str, Any]] | None = None,
    chat_records: list[dict[str, Any]] | None = None,
) -> str:
    overall = session.get("overall_objective_status")
    deliverable_verified = session.get("deliverable_verified")
    completed = overall in {"complete", "completed", "chat_completed", "chat_success"}
    incomplete_ids = session.get("incomplete_task_ids")
    has_incomplete_tasks = bool(incomplete_ids) if isinstance(incomplete_ids, list) else False

    primary_blockers = detect_primary_blockers(harness_records, failed_dispatches)

    if overall in {"complete", "completed"} and deliverable_verified is True and not errors:
        return "success"
    if overall in {"complete", "completed"} and deliverable_verified is True and errors:
        return "success_with_errors"
    if completed and not has_incomplete_tasks:
        return "success_with_errors" if errors else "success"
    if _has_chat_terminal_repetition_stall(harness_records, session):
        return "chat_terminal_repetition_stall"
    if _has_write_overwrite_guard_failures(failed_dispatches):
        return "write_session_overwrite_guard_loop"
    if _has_patch_first_policy_loop(failed_dispatches):
        return "patch_first_policy_loop"
    if _file_patch_target_loop_count(failed_dispatches) >= 3:
        return "file_patch_target_not_found_loop"
    if _has_ask_human_resume_terminal_stall(harness_records):
        return "ask_human_resume_terminal_tool_stall"
    if has_strong_environment_blocker(primary_blockers):
        return "environment_blocker"
    if has_stderr_signature_circuit_breaker(harness_records):
        return "harness_circuit_breaker_false_positive"
    if detect_apt_deb822_guard_misfire(harness_records):
        return "guard_misfire"
    if events.get("tool_blocked_not_exposed"):
        return "policy_block"
    if events.get("fama_tool_call_blocked"):
        return "fama_block"
    if events.get("model_output_degenerate_loop_exhausted"):
        return "model_degeneration"
    if events.get("reasoning_only_stream_exhausted") or events.get("model_stream_halt_exhausted"):
        return "model_stream_stall"
    if events.get("action_stall") or events.get("no_tool_recovery"):
        return "model_tool_loop_stall"
    if events.get("dispatch_tools_error") or events.get("initialize_run_error"):
        return "runtime_exception"
    if has_continue_prompt_budget_loop(
        harness_records, chat_records or [], threshold=2
    ):
        return "continue_prompt_budget_loop"
    if has_tool_call_protocol_mismatch(
        model_output_records or [], tools_records or []
    ):
        return "tool_call_protocol_mismatch"
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
    model_output_records = list(iter_records(run_dir, "model_output"))
    chat_records = list(iter_records(run_dir, "chat"))
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
    primary_blockers = detect_primary_blockers(harness_records, failed_dispatches)
    apt_deb822_misfires = detect_apt_deb822_guard_misfire(harness_records)

    diagnosis = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "duration_seconds": run_duration_seconds(harness_records),
        "objective": get_run_objective(session, task_summary, task_summaries),
        "session_summary": session,
        "task_summary": task_summary,
        "failure_classification": _classify_failure(events, errors, session, failed_dispatches, harness_records, model_output_records=model_output_records, tools_records=tools_records, chat_records=chat_records),
        "primary_blockers": primary_blockers,
        "apt_deb822_guard_misfires": [format_record_summary(r) for r in apt_deb822_misfires],
        "event_counts": dict(events),
        "error_record_count": len(errors),
        "warning_record_count": len(warnings),
        "recent_errors": [format_record_summary(e) for e in errors[-10:]],
        "last_error_trace_ids": [t for t in last_error_trace_ids if t],
        "tool_failure_count": len(failed_dispatches),
        "tool_failures_by_name": dict(tool_failures_by_name),
        "task_count": len(task_summaries),
        "task_statuses": [{"task_id": t.get("task_id"), "status": t.get("status"), "reason": t.get("reason")} for t in task_summaries],
        "recommended_next_steps": _recommend_next_steps(events, errors, session, failed_dispatches, harness_records, model_output_records=model_output_records, tools_records=tools_records, chat_records=chat_records),
    }
    return diagnosis


def _recommend_next_steps(
    events: Counter[str],
    errors: list[dict[str, Any]],
    session: dict[str, Any],
    failed_dispatches: list[dict[str, Any]],
    harness_records: list[dict[str, Any]],
    *,
    model_output_records: list[dict[str, Any]] | None = None,
    tools_records: list[dict[str, Any]] | None = None,
    chat_records: list[dict[str, Any]] | None = None,
) -> list[str]:
    steps: list[str] = []
    primary_blockers = detect_primary_blockers(harness_records, failed_dispatches)
    if primary_blockers:
        top = primary_blockers[0]
        steps.append(
            f"Primary objective blocker: {top['pattern']} ({top['count']}x). "
            "Resolve the environmental dependency before treating this as a model/harness failure."
        )
    if errors and (tid := extract_trace_id(errors[-1])):
        steps.append(f"Trace the most recent error: python3 Agent-Tools/trace_call.py --run <run> {tid}")
    if events.get("model_output_degenerate_loop_exhausted"):
        steps.append("Investigate model stream degeneration; check model_output.log for repeated phrases.")
    if has_stderr_signature_circuit_breaker(harness_records):
        steps.append("Harness stderr-signature circuit breaker tripped on repeated identical stderr; use a different repair strategy instead of retrying the same command.")
    if detect_apt_deb822_guard_misfire(harness_records):
        steps.append("apt_deb822 preflight guard blocked after validator already passed; review guard state and whether the block is stale.")
    if _has_write_overwrite_guard_failures(failed_dispatches):
        steps.append("Write-session overwrite guard loop detected; trace failed file_write calls and force a current file_read followed by file_patch/ast_patch or same-section repair.")
    if _has_patch_first_policy_loop(failed_dispatches):
        steps.append("Patch-first policy repeatedly blocked full rewrites; use file_patch/ast_patch or pass replace_strategy='overwrite' for an intentional full rewrite.")
    patch_loop_count = _file_patch_target_loop_count(failed_dispatches)
    if patch_loop_count >= 3:
        steps.append(f"Repeated file_patch target mismatch loop detected ({patch_loop_count} failures); force a non-cached file_read of the target before allowing another patch and verify using exact live text.")
    if _has_ask_human_resume_terminal_stall(harness_records):
        steps.append("ask_human resume fell into terminal-only tool exposure; preserve original task context and expose file mutation tools after affirmative replies.")
    if _has_chat_terminal_repetition_stall(harness_records, session):
        steps.append("Chat-mode terminal-only tool exposure caused a repetition loop. Route implementation follow-ups to loop mode and expose file mutation tools.")
    if has_continue_prompt_budget_loop(harness_records, chat_records or [], threshold=2):
        steps.append("Continue/proceed loop repeatedly overflowed the prompt budget; reset context (new task or fresh run) instead of appending more 'continue' messages.")
    if has_tool_call_protocol_mismatch(model_output_records or [], tools_records or []):
        steps.append("Tool-call protocol mismatch detected; review reasoning-channel tool-call recovery and ensure the model emits proper JSON/native tool calls rather than markup fragments.")
    if events.get("action_stall") or events.get("no_tool_recovery"):
        steps.append("Model is struggling to emit valid tool calls; inspect recent prompts and tool schemas.")
    if events.get("reasoning_only_stream_exhausted") or events.get("model_stream_halt_exhausted"):
        steps.append("Model stream stalled in reasoning without producing an assistant answer or tool call; inspect prompt pressure and stream-halt recovery events.")
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
    objective = diagnosis.get("objective")
    if objective:
        lines.append(f"Objective: {objective}")
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

    blockers = diagnosis.get("primary_blockers") or []
    lines.append("")
    lines.append(colorize("Primary blockers", Colors.BOLD + Colors.BLUE))
    if blockers:
        for b in blockers:
            lines.append(f"  {b['pattern']}: {b['count']}x  sample: {b['sample'][:120]}")
    else:
        lines.append("  none detected")

    apt_misfires = diagnosis.get("apt_deb822_guard_misfires") or []
    if apt_misfires:
        lines.append("")
        lines.append(colorize("apt_deb822 guard misfires", Colors.BOLD + Colors.YELLOW))
        for m in apt_misfires[:5]:
            lines.append("  " + colorize(m, Colors.YELLOW))

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
