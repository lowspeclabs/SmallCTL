#!/usr/bin/env python3
"""runscan — batch health check over recent SmallCTL runs.

Useful when you have many logs and want to know which recent runs failed,
what failure patterns are trending, and which runs deserve closer inspection.

Examples:
  python3 Agent-Tools/runscan.py
  python3 Agent-Tools/runscan.py --last 50
  python3 Agent-Tools/runscan.py --last 100 --failures-only
  python3 Agent-Tools/runscan.py --last 20 --json
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
    detect_background_state_changing_shell,
    detect_primary_blockers,
    discover_runs,
    error_records,
    event_counter,
    file_patch_target_loop_count,
    format_duration,
    get_run_objective,
    has_ask_human_resume_terminal_stall,
    has_chat_terminal_repetition_stall,
    has_continue_prompt_budget_loop,
    has_fama_ssh_transport_circuit_breaker,
    has_patch_first_policy_loop,
    has_prompt_budget_overflow,
    has_stderr_signature_circuit_breaker,
    has_strong_environment_blocker,
    has_tool_call_protocol_mismatch,
    has_write_overwrite_guard_failures,
    iter_records,
    load_summaries,
    load_task_summaries,
    run_duration_seconds,
    warning_records,
)


FAILURE_CLASS_LABELS = {
    "success": "success",
    "success_with_errors": "success_with_errors",
    "environment_blocker": "environment_blocker",
    "harness_circuit_breaker_false_positive": "harness_circuit_breaker_false_positive",
    "guard_misfire": "guard_misfire",
    "prompt_budget_overflow": "prompt_budget_overflow",
    "background_state_change_unverified": "background_state_change_unverified",
    "ssh_transport_circuit_breaker_false_positive": "ssh_transport_circuit_breaker_false_positive",
    "model_degeneration": "model_degeneration",
    "model_stream_stall": "model_stream_stall",
    "model_tool_loop_stall": "model_tool_loop_stall",
    "file_patch_target_not_found_loop": "file_patch_target_not_found_loop",
    "patch_first_policy_loop": "patch_first_policy_loop",
    "runtime_exception": "runtime_exception",
    "policy_block": "policy_block",
    "fama_block": "fama_block",
    "chat_failure": "chat_failure",
    "chat_terminal_repetition_stall": "chat_terminal_repetition_stall",
    "write_session_overwrite_guard_loop": "write_session_overwrite_guard_loop",
    "ask_human_resume_terminal_tool_stall": "ask_human_resume_terminal_tool_stall",
    "continue_prompt_budget_loop": "continue_prompt_budget_loop",
    "tool_call_protocol_mismatch": "tool_call_protocol_mismatch",
    "cancelled_remote_verification_loop": "cancelled_remote_verification_loop",
    "recovery_failure": "recovery_failure",
    "incomplete_unverified": "incomplete_unverified",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan recent SmallCTL runs for failures.")
    parser.add_argument("--last", type=int, default=30, help="Number of recent runs to scan (default: 30)")
    parser.add_argument("--logs-dir", help="Custom logs directory")
    parser.add_argument("--failures-only", action="store_true", help="Only show runs with errors or incomplete status")
    parser.add_argument("--same-objective", dest="same_objective", help="Only include runs whose objective contains TEXT")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    return parser.parse_args()


def _has_write_overwrite_guard_failures(failed_dispatches: list[dict[str, Any]]) -> bool:
    return has_write_overwrite_guard_failures(failed_dispatches)



def _file_patch_target_loop_count(failed_dispatches: list[dict[str, Any]]) -> int:
    return file_patch_target_loop_count(failed_dispatches)


def _has_ask_human_resume_terminal_stall(records: list[dict[str, Any]]) -> bool:
    return has_ask_human_resume_terminal_stall(records)


def _has_chat_terminal_repetition_stall(records: list[dict[str, Any]], session: dict[str, Any]) -> bool:
    return has_chat_terminal_repetition_stall(records, session)


def _has_patch_first_policy_loop(failed_dispatches: list[dict[str, Any]]) -> bool:
    return has_patch_first_policy_loop(failed_dispatches)


def _has_cancelled_remote_verification_loop(
    session: dict[str, Any], failed_dispatches: list[dict[str, Any]]
) -> bool:
    if str(session.get("final_task_status") or "").strip().lower() not in {"cancelled", "interrupted"}:
        return False
    return any(
        str((record.get("data") or {}).get("tool_name") or "") == "task_complete"
        and "remote" in str((record.get("data") or {}).get("error") or "").lower()
        and "verification" in str((record.get("data") or {}).get("error") or "").lower()
        for record in failed_dispatches
    )


def _classify_run(
    events: Counter[str],
    errors: list[dict[str, Any]],
    session: dict[str, Any],
    task_summary: dict[str, Any],
    failed_dispatches: list[dict[str, Any]],
    harness_records: list[dict[str, Any]],
    *,
    model_output_records: list[dict[str, Any]] | None = None,
    tools_records: list[dict[str, Any]] | None = None,
    chat_records: list[dict[str, Any]] | None = None,
) -> str:
    overall = session.get("overall_objective_status")
    final = task_summary.get("final_task_status")
    deliverable_verified = session.get("deliverable_verified")
    completed = overall in {"complete", "completed", "chat_completed", "chat_success"}
    incomplete_ids = session.get("incomplete_task_ids")
    has_incomplete_tasks = bool(incomplete_ids) if isinstance(incomplete_ids, list) else False
    primary_blockers = detect_primary_blockers(harness_records, failed_dispatches)
    background_blockers = detect_background_state_changing_shell(tools_records or [])

    if overall in {"complete", "completed"} and deliverable_verified is True and not errors:
        return "success"
    if overall in {"complete", "completed"} and deliverable_verified is True and errors:
        return "success_with_errors"
    if overall in {"chat_completed", "chat_success"} and not errors:
        return "success"
    if completed and not has_incomplete_tasks:
        return "success_with_errors" if errors else "success"
    if _has_cancelled_remote_verification_loop(session, failed_dispatches):
        return "cancelled_remote_verification_loop"
    if final in {"chat_failed", "chat_action_blocked"}:
        return "chat_failure"
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
    if has_prompt_budget_overflow(session, task_summary, harness_records, chat_records or []):
        return "prompt_budget_overflow"
    if background_blockers:
        return "background_state_change_unverified"
    if has_strong_environment_blocker(primary_blockers):
        return "environment_blocker"
    if has_stderr_signature_circuit_breaker(harness_records):
        return "harness_circuit_breaker_false_positive"
    if has_fama_ssh_transport_circuit_breaker(harness_records):
        return "ssh_transport_circuit_breaker_false_positive"
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
    if has_continue_prompt_budget_loop(harness_records, chat_records or [], threshold=2):
        return "continue_prompt_budget_loop"
    if has_tool_call_protocol_mismatch(model_output_records or [], []):
        return "tool_call_protocol_mismatch"
    if events.get("recovery_failure_event_recorded"):
        return "recovery_failure"
    if overall == "incomplete" and session.get("deliverable_verified") is False:
        return "incomplete_unverified"
    return "unknown"


def _summarize_run(run_dir: Path) -> dict[str, Any]:
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

    dispatch_starts = [r for r in tools_records if r.get("event") == "dispatch_start"]
    failed_dispatches = [r for r in tools_records if r.get("event") == "dispatch_complete" and (r.get("data") or {}).get("success") is False]

    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "duration_seconds": run_duration_seconds(harness_records),
        "objective": get_run_objective(session, task_summary, task_summaries),
        "overall_objective_status": session.get("overall_objective_status"),
        "final_task_status": task_summary.get("final_task_status"),
        "deliverable_verified": session.get("deliverable_verified"),
        "task_count": len(task_summaries),
        "incomplete_task_count": len(session.get("incomplete_task_ids", [])),
        "dispatch_count": len(dispatch_starts),
        "failed_dispatch_count": len(failed_dispatches),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "classification": _classify_run(events, errors, session, task_summary, failed_dispatches, harness_records, model_output_records=model_output_records, tools_records=tools_records, chat_records=chat_records),
        "top_events": dict(events.most_common(5)),
    }


def _status_color(status: str | None) -> str:
    if status == "success":
        return Colors.GREEN
    if status == "success_with_errors":
        return Colors.YELLOW
    if status in {
        "model_degeneration",
        "model_stream_stall",
        "runtime_exception",
        "policy_block",
        "fama_block",
        "recovery_failure",
        "chat_failure",
        "chat_terminal_repetition_stall",
        "write_session_overwrite_guard_loop",
        "file_patch_target_not_found_loop",
        "patch_first_policy_loop",
        "ask_human_resume_terminal_tool_stall",
        "environment_blocker",
        "harness_circuit_breaker_false_positive",
        "guard_misfire",
        "prompt_budget_overflow",
        "background_state_change_unverified",
        "ssh_transport_circuit_breaker_false_positive",
        "continue_prompt_budget_loop",
        "tool_call_protocol_mismatch",
    }:
        return Colors.RED
    if status in {"model_tool_loop_stall", "incomplete_unverified", "unknown"}:
        return Colors.YELLOW
    return Colors.RESET


def main() -> int:
    args = _parse_args()
    logs_dir = Path(args.logs_dir) if args.logs_dir else None
    try:
        base = logs_dir or Path(__file__).resolve().parent.parent / "logs"
        runs = discover_runs(base)[: args.last]
    except (FileNotFoundError, ValueError) as exc:
        print(colorize(str(exc), Colors.RED), file=sys.stderr)
        return 1

    if not runs:
        print(colorize("No runs found.", Colors.YELLOW), file=sys.stderr)
        return 1

    summaries = [_summarize_run(r) for r in runs]
    if args.same_objective:
        query = args.same_objective.lower()
        summaries = [s for s in summaries if query in (s.get("objective") or "").lower()]
    if args.failures_only:
        summaries = [s for s in summaries if s["classification"] != "success" or s["error_count"] > 0]

    if args.json:
        print(json.dumps(summaries, indent=2, default=str))
        return 0

    lines: list[str] = []
    lines.append(colorize(f"Recent runs ({len(summaries)} of {args.last})", Colors.BOLD + Colors.CYAN))
    lines.append("")

    # Header
    header = f"{'run':<18} {'class':<22} {'status':<12} {'dur':<8} {'tasks':>5} {'errs':>5} {'warns':>5} {'disp':>5} {'faildisp':>8}"
    lines.append(colorize(header, Colors.BOLD))
    lines.append("-" * len(header))

    for s in summaries:
        name = s["run_name"].split("-")[0]
        cls = s["classification"]
        status = s["final_task_status"] or s["overall_objective_status"] or "n/a"
        row = (
            f"{name:<18} "
            f"{cls:<22} "
            f"{str(status):<12} "
            f"{format_duration(s['duration_seconds']):<8} "
            f"{s['task_count']:>5} "
            f"{s['error_count']:>5} "
            f"{s['warning_count']:>5} "
            f"{s['dispatch_count']:>5} "
            f"{s['failed_dispatch_count']:>8}"
        )
        lines.append(colorize(row, _status_color(cls)))

    # Failure pattern summary
    class_counter = Counter(s["classification"] for s in summaries)
    lines.append("")
    lines.append(colorize("Failure pattern counts", Colors.BOLD + Colors.BLUE))
    for cls, count in class_counter.most_common():
        lines.append(f"  {count:>4}  {cls}")

    if args.same_objective:
        lines.append("")
        lines.append(colorize(f"Filtered to objective containing: {args.same_objective}", Colors.YELLOW))

    lines.append("")
    lines.append(colorize("Next steps", Colors.BOLD + Colors.GREEN))
    lines.append("  - Inspect a run: python3 Agent-Tools/logwatch.py <run_id>")
    lines.append("  - Diagnose a run: python3 Agent-Tools/run_diagnose.py <run_id>")
    lines.append("  - Trace the latest error: python3 Agent-Tools/trace_call.py --last-error")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
