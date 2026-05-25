#!/usr/bin/env python3
"""
AHO Baseline Challenge Runner
==============================
Loads challenges from challenges.text, runs each via smallctl CLI,
tracks runtime / guards / errors / ambiguities, and writes a summary
report (default: 4-b-baseline-2.md).

Features:
- Self-monitoring with live status output for LLM observers
- Structured logging to aho/run_baseline.log
- Per-challenge and overall time tracking
- Backend smoke test before baseline run
- Interactive or non-interactive provider selection

Usage:
    cd /path/to/Harness-Redo
    python aho/run_baseline.py --yes --skip-ambiguity-prompt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .log_review import LogReview, review_logs

DEFAULT_CHALLENGES_PATH = Path(__file__).parent / "challenges.text"
DEFAULT_SMALLCTL = Path(__file__).parent.parent / ".venv" / "bin" / "smallctl"
DEFAULT_TIMEOUT = 300  # 5 minutes per challenge

DEFAULT_ENDPOINT = "http://192.168.1.9:8080"
DEFAULT_MODEL = "qwen3.5-4b"
DEFAULT_PROVIDER_PROFILE = "llamacpp"

# Ensure unbuffered stdout for live LLM monitoring
sys.stdout.reconfigure(line_buffering=True)

SCRIPT_LOG_PATH = Path(__file__).parent / "run_baseline.log"


class ScriptLogger:
    """Structured logger that writes to aho/run_baseline.log and prints status lines."""

    def __init__(self, path: Path = SCRIPT_LOG_PATH) -> None:
        self.path = path
        self._lock = threading.Lock()
        # Clear old log at start
        self.path.write_text("", encoding="utf-8")

    def _write(self, level: str, event: str, message: str, **kwargs: Any) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "event": event,
            "message": message,
            "data": kwargs,
        }
        line = json.dumps(entry, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        # Also emit human-readable status to stdout for live monitoring
        status = f"[{entry['timestamp']}] [{level}] {event}: {message}"
        if kwargs:
            extra = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            status += f" | {extra}"
        print(status, flush=True)

    def info(self, event: str, message: str, **kwargs: Any) -> None:
        self._write("INFO", event, message, **kwargs)

    def warn(self, event: str, message: str, **kwargs: Any) -> None:
        self._write("WARN", event, message, **kwargs)

    def error(self, event: str, message: str, **kwargs: Any) -> None:
        self._write("ERROR", event, message, **kwargs)

    def status(self, message: str, **kwargs: Any) -> None:
        self._write("STATUS", "runner_status", message, **kwargs)


LOGGER = ScriptLogger()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_challenges(text: str) -> tuple[list[dict], list[dict], dict[str, str]]:
    """Parse challenges.text into sysadmin and coding challenge dicts."""
    lines = text.splitlines()

    # Extract credentials from first lines
    creds: dict[str, str] = {}
    for line in lines[:4]:
        m = re.match(r"host:\s*(.+)", line, re.I)
        if m:
            creds["host_line"] = m.group(1).strip()
        m = re.match(r'password:\s*"([^"]+)"', line)
        if m:
            creds["password"] = m.group(1).strip()

    # Find sections
    sysadmin_start = None
    coding_start = None
    for i, line in enumerate(lines):
        if re.match(r"^\d+\.\s+\*\*", line):
            if sysadmin_start is None:
                sysadmin_start = i
        if "coding challanges" in line.lower() or "coding challenges" in line.lower():
            coding_start = i + 1
            break

    if sysadmin_start is None:
        sysadmin_start = 0
    if coding_start is None:
        coding_start = len(lines)

    def _extract(start_idx: int, end_idx: int) -> list[dict]:
        challenges: list[dict] = []
        current: dict | None = None
        for i in range(start_idx, end_idx):
            line = lines[i]
            # Format A: coding challenges have title + description on same line
            #   1. **Easy** — Build a self-contained Python script...
            m = re.match(r"^(\d+)\.\s+\*\*([^*]+)\*\*\s*[-—]\s*(.+)", line)
            if m:
                if current:
                    challenges.append(current)
                current = {
                    "id": f"challenge-{m.group(1).zfill(2)}",
                    "number": int(m.group(1)),
                    "difficulty": m.group(2).strip(),
                    "description": m.group(3).strip(),
                    "raw_lines": [line],
                }
                continue
            # Format B: sysadmin challenges have title on one line, description on next
            #   1. **Easy — `/etc/hosts` review and safe edit**
            #      SSH to the host, inspect `/etc/hosts`...
            m = re.match(r"^(\d+)\.\s+\*\*([^*]+)\*\*\s*$", line)
            if m:
                if current:
                    challenges.append(current)
                current = {
                    "id": f"challenge-{m.group(1).zfill(2)}",
                    "number": int(m.group(1)),
                    "difficulty": m.group(2).strip(),
                    "description": "",
                    "raw_lines": [line],
                }
                continue
            elif current is not None:
                current["raw_lines"].append(line)
                current["description"] += " " + line.strip()
        if current:
            challenges.append(current)
        return challenges

    sysadmin = _extract(sysadmin_start, coding_start)
    coding = _extract(coding_start, len(lines))

    return sysadmin, coding, creds


def build_task_prompt(challenge: dict, creds: dict | None = None, is_sysadmin: bool = False) -> str:
    """Assemble the full task prompt injected with creds and evaluation rules."""
    body = challenge["description"]

    parts = [body]

    if is_sysadmin and creds:
        host_line = creds.get("host_line", "root@192.168.1.89")
        password = creds.get("password", "Temp@Pass")
        # Derive user/host from host_line
        user = "root"
        host = "192.168.1.89"
        if "@" in host_line:
            user, host = host_line.split("@", 1)
        parts.append(
            f"\nRemote credentials for this sysadmin challenge are available directly here: "
            f"target={host_line}, user={user}, host={host}, password={password}. "
            f"Use these values directly in ssh_exec or ssh file tools; do not spend time searching for the challenge file."
        )

    parts.append(
        "\nHarness baseline evaluation rules: finish the requested task end to end. "
        "If this is a coding task, create the requested file, run it, run or add the embedded unittest suite, "
        "patch/debug failures before finishing, and explicitly report the verification command and result. "
        "If this is a sysadmin task, use the provided SSH target and password from aho/challanges.text, "
        "write the requested local report artifact, and summarize what changed. "
        "Call task_complete only after the required artifact exists and verification is done."
    )

    return "\n".join(parts)


def discover_log_dirs(stdout: str) -> dict[str, Path | None]:
    """Parse smallctl stdout for both logging_ready and log_dir_finalized paths."""
    reported: Path | None = None
    actual: Path | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            status = obj.get("status")
            run_log_dir = obj.get("run_log_dir")
            if status == "logging_ready" and run_log_dir:
                p = Path(run_log_dir)
                if p.exists():
                    reported = p
            if status == "log_dir_finalized" and run_log_dir:
                p = Path(run_log_dir)
                if p.exists():
                    actual = p
        except Exception:
            continue

    if actual is None and reported is not None:
        # fallback: if reported exists but no finalized event, use reported
        actual = reported

    if actual is None:
        # Try timestamp-suffix search under logs/ or .venv/logs/
        for base in (Path("logs"), Path(".venv/logs")):
            if base.exists():
                candidates = sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                for c in candidates:
                    if c.is_dir():
                        actual = c
                        break
            if actual is not None:
                break

    return {"reported_log_dir": reported, "actual_log_dir": actual}


def _stream_reader(pipe, lines: list[str]) -> None:
    """Thread target: read lines from a pipe and store them."""
    try:
        for line in iter(pipe.readline, ""):
            lines.append(line)
    except Exception:
        pass
    finally:
        pipe.close()


def run_challenge(
    challenge: dict,
    smallctl: Path,
    args: argparse.Namespace,
    creds: dict | None = None,
    is_sysadmin: bool = False,
    current: int = 0,
    total: int = 0,
) -> dict[str, Any]:
    """Run a single challenge via smallctl CLI with live status reporting. Returns result dict."""
    task = build_task_prompt(challenge, creds, is_sysadmin)
    cmd: list[str] = [str(smallctl), "--task", task]

    if args.endpoint:
        cmd += ["--endpoint", args.endpoint]
    if args.model:
        cmd += ["--model", args.model]
    if args.provider_profile:
        cmd += ["--provider-profile", args.provider_profile]
    if args.api_key:
        cmd += ["--api-key", args.api_key]
    if args.max_prompt_tokens is not None:
        cmd += ["--max-prompt-tokens", str(args.max_prompt_tokens)]
    if args.context_limit is not None:
        cmd += ["--context-limit", str(args.context_limit)]
    if args.tool_profiles:
        cmd += ["--tool-profiles", args.tool_profiles]
    if args.run_mode:
        cmd += ["--run-mode", args.run_mode]
    if args.phase:
        cmd += ["--phase", args.phase]
    if args.config:
        cmd += ["--config", args.config]
    if args.fresh_run:
        cmd += ["--fresh-run"]
    if args.hide_thinking:
        cmd += ["--hide-thinking"]
    if args.debug:
        cmd += ["--debug"]

    LOGGER.status(
        f"Starting challenge {current}/{total}: {challenge['id']} ({challenge['difficulty']})",
        challenge_id=challenge["id"],
        category="sysadmin" if is_sysadmin else "coding",
        timeout_sec=args.timeout,
    )

    t0 = time.monotonic()
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    proc = None

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(Path(smallctl).parent.parent),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        LOGGER.error("challenge_launch_failed", str(exc), challenge_id=challenge["id"])
        return {
            "challenge_id": challenge["id"],
            "status": "crashed",
            "elapsed_sec": round(elapsed, 2),
            "stdout": "",
            "stderr": "",
            "returncode": -1,
            "log_review": {},
        }

    # Start reader threads
    stdout_thread = threading.Thread(target=_stream_reader, args=(proc.stdout, stdout_lines))
    stderr_thread = threading.Thread(target=_stream_reader, args=(proc.stderr, stderr_lines))
    stdout_thread.start()
    stderr_thread.start()

    # Status reporter thread
    stop_status = threading.Event()
    status_thread: threading.Thread | None = None

    def _status_reporter() -> None:
        interval = 30  # seconds
        while not stop_status.wait(interval):
            elapsed = time.monotonic() - t0
            LOGGER.status(
                f"Challenge {current}/{total} still running: {challenge['id']} — elapsed {elapsed:.1f}s",
                challenge_id=challenge["id"],
                elapsed_sec=round(elapsed, 1),
                pid=proc.pid if proc else None,
            )

    status_thread = threading.Thread(target=_status_reporter, daemon=True)
    status_thread.start()

    try:
        return_code = proc.wait(timeout=args.timeout)
        stop_status.set()
    except subprocess.TimeoutExpired:
        stop_status.set()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        elapsed = time.monotonic() - t0
        LOGGER.status(
            f"Challenge {current}/{total} TIMED OUT: {challenge['id']} after {elapsed:.1f}s",
            challenge_id=challenge["id"],
            elapsed_sec=round(elapsed, 1),
        )
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)
        log_dirs = discover_log_dirs(stdout)
        actual_log_dir = log_dirs.get("actual_log_dir")
        log_review = review_logs(actual_log_dir) if actual_log_dir else LogReview()
        return {
            "challenge_id": challenge["id"],
            "status": "timeout",
            "elapsed_sec": round(elapsed, 2),
            "stdout_preview": stdout[-2000:] if len(stdout) > 2000 else stdout,
            "stderr_preview": stderr[-1000:] if len(stderr) > 1000 else stderr,
            "returncode": -1,
            "log_dir": str(actual_log_dir) if actual_log_dir else None,
            "reported_log_dir": str(log_dirs.get("reported_log_dir")) if log_dirs.get("reported_log_dir") else None,
            "actual_log_dir": str(actual_log_dir) if actual_log_dir else None,
            "log_review": log_review,
            "timed_out_pid": proc.pid if proc else None,
            "category": "sysadmin" if is_sysadmin else "coding",
        }

    stop_status.set()
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

    elapsed = time.monotonic() - t0
    stdout = "".join(stdout_lines)
    stderr = "".join(stderr_lines)

    # Try to parse final JSON result from stdout (smallctl prints JSON result last)
    # smallctl uses indent=2 for the final result, so it may span multiple lines.
    final_result: dict | None = None
    idx = stdout.rfind("{")
    while idx != -1:
        try:
            candidate = json.loads(stdout[idx:])
            if isinstance(candidate, dict):
                final_result = candidate
                break
        except Exception:
            pass
        idx = stdout.rfind("{", 0, idx)

    log_dirs = discover_log_dirs(stdout)
    actual_log_dir = log_dirs.get("actual_log_dir")
    log_review = review_logs(actual_log_dir) if actual_log_dir else LogReview()

    status = "unknown"
    if final_result:
        status = final_result.get("status", "unknown")
    elif return_code != 0:
        status = "crashed"
    else:
        status = "completed"

    LOGGER.status(
        f"Challenge {current}/{total} finished: {challenge['id']} — status={status}, elapsed={elapsed:.1f}s, returncode={return_code}",
        challenge_id=challenge["id"],
        status=status,
        elapsed_sec=round(elapsed, 2),
        returncode=return_code,
        ask_human_count=len(log_review.ask_human_calls),
        guard_event_count=len(log_review.guard_events),
        tool_failure_count=len(log_review.tool_failures),
    )

    return {
        "challenge_id": challenge["id"],
        "status": status,
        "elapsed_sec": round(elapsed, 2),
        "returncode": return_code,
        "stdout_preview": stdout[-2000:] if len(stdout) > 2000 else stdout,
        "stderr_preview": stderr[-1000:] if len(stderr) > 1000 else stderr,
        "final_result": final_result,
        "log_dir": str(actual_log_dir) if actual_log_dir else None,
        "reported_log_dir": str(log_dirs.get("reported_log_dir")) if log_dirs.get("reported_log_dir") else None,
        "actual_log_dir": str(actual_log_dir) if actual_log_dir else None,
        "log_review": log_review,
        "category": "sysadmin" if is_sysadmin else "coding",
    }


def ask_user_for_clarification(questions: list[str]) -> str | None:
    """Prompt user for clarification when model escalates ambiguity."""
    print("\n" + "-" * 40)
    print("MODEL ESCALATED AMBIGUITY")
    for q in questions:
        print(f"  Q: {q}")
    print("-" * 40)
    try:
        answer = input("Enter clarification (or press Enter to skip / fail): ").strip()
        return answer if answer else None
    except (EOFError, KeyboardInterrupt):
        return None


def _derive_final_class(r: dict) -> str:
    """Derive granular final classification from result and log review."""
    status = r.get("status", "unknown")
    log_review = r.get("log_review")
    if isinstance(log_review, LogReview):
        step_count = log_review.step_count
        ask_human_count = len(log_review.ask_human_calls)
        guard_events = log_review.guard_events
        tool_failures = log_review.tool_failures
        stall_classification = log_review.stall_classification
        backend = log_review.backend
        verified = log_review.verified
    else:
        log = log_review or {}
        step_count = log.get("step_count", 0)
        ask_human_count = len(log.get("ask_human_calls", []))
        guard_events = log.get("guard_events", [])
        tool_failures = log.get("tool_failures", [])
        stall_classification = None
        backend = False
        verified = None

    final_result = r.get("final_result")
    challenge_progress = {}
    if isinstance(final_result, dict):
        raw_progress = final_result.get("challenge_progress")
        if isinstance(raw_progress, dict):
            challenge_progress = raw_progress

    if not challenge_progress:
        log_dir = r.get("log_dir")
        if log_dir:
            summary_path = Path(str(log_dir)) / "task_summary.json"
            if summary_path and summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    challenge_progress = summary.get("challenge_progress", {})
                except Exception:
                    pass

    if status == "completed":
        if verified or challenge_progress.get("verified_after_last_change"):
            return "verified_completed"
        return "completed"

    if backend:
        return "backend_failed"

    if status == "timeout":
        if stall_classification:
            return stall_classification
        if step_count == 0 and not tool_failures and not ask_human_count:
            return "timeout_no_first_tool"
        # Repeated tool loops dominating tail
        if guard_events and len(guard_events) >= 3:
            return "tool_loop_timeout"
        if step_count > 0 and (tool_failures or ask_human_count):
            return "timeout_after_progress"
        return "timeout_after_progress"

    if status == "needs_human":
        return "needs_human"

    if status in ("failed", "crashed"):
        return "failed"

    return "failed"


def generate_report(results: list[dict], path: Path) -> None:
    """Write findings to markdown."""
    for r in results:
        if "final_class" not in r:
            r["final_class"] = _derive_final_class(r)

    lines = [
        "# AHO Baseline Run Report\n",
        f"**Generated:** {_now()}\n",
        f"**Total challenges:** {len(results)}\n",
        "\n## Summary\n",
        "| Challenge | Category | FinalClass | Verified | Backend | Elapsed (s) | Guards | AskHuman | ToolFails | LogDir |",
        "|-----------|----------|------------|----------|---------|-------------|--------|----------|-----------|--------|",
    ]

    for r in results:
        log_review = r.get("log_review")
        if isinstance(log_review, LogReview):
            guards = len(log_review.guard_events)
            ask_human = len(log_review.ask_human_calls)
            tool_fails = len(log_review.tool_failures)
            verified = "true" if log_review.verified else "false"
            backend = "true" if log_review.backend else "false"
        else:
            log = log_review or {}
            guards = len(log.get("guard_events", []))
            ask_human = len(log.get("ask_human_calls", []))
            tool_fails = len(log.get("tool_failures", []))
            verified = "n/a"
            backend = "false"

        category = r.get("category", "")
        final_class = r.get("final_class", r["status"])
        log_dir_str = r.get("actual_log_dir") or r.get("log_dir") or "N/A"
        reported = r.get("reported_log_dir")
        if reported and reported != log_dir_str:
            log_dir_str += f" (reported: {reported})"

        cid = f"{category}/{r['challenge_id']}" if category else r["challenge_id"]
        lines.append(
            f"| {cid} | {category} | {final_class} | {verified} | {backend} | {r['elapsed_sec']} | "
            f"{guards} | {ask_human} | {tool_fails} | {log_dir_str} |"
        )

    scores = {
        "completed": sum(1 for r in results if r["status"] == "completed"),
        "verified_completed": sum(1 for r in results if r.get("final_class") == "verified_completed"),
        "needs_human": sum(1 for r in results if r["status"] == "needs_human"),
        "backend_failed": sum(1 for r in results if r.get("final_class") == "backend_failed"),
        "timeout_after_progress": sum(1 for r in results if r.get("final_class") == "timeout_after_progress"),
        "timeout_no_first_tool": sum(1 for r in results if r.get("final_class") == "timeout_no_first_tool"),
        "tool_loop_timeout": sum(1 for r in results if r.get("final_class") == "tool_loop_timeout"),
        "unverified_completion_warning": sum(1 for r in results if r.get("final_class") == "completed" and not r.get("verified")),
    }

    lines += [
        "\n",
        f"**Completed:** {scores['completed']} / {len(results)}\n",
        f"**Verified Completed:** {scores['verified_completed']} / {len(results)}\n",
        f"**Needs Human:** {scores['needs_human']} / {len(results)}\n",
        f"**Backend Failed:** {scores['backend_failed']} / {len(results)}\n",
        f"**Timeout (after progress):** {scores['timeout_after_progress']} / {len(results)}\n",
        f"**Timeout (no first tool):** {scores['timeout_no_first_tool']} / {len(results)}\n",
        f"**Tool Loop Timeout:** {scores['tool_loop_timeout']} / {len(results)}\n",
        f"**Unverified Completion Warning:** {scores['unverified_completion_warning']} / {len(results)}\n",
        "\n## Per-Challenge Details\n",
    ]

    for r in results:
        cid = r["challenge_id"]
        category = r.get("category", "")
        heading = f"{category}/{cid}" if category else cid
        log_review = r.get("log_review")
        if isinstance(log_review, LogReview):
            ask_human_calls = log_review.ask_human_calls
            guard_events = log_review.guard_events
            tool_failures = log_review.tool_failures
            errors = log_review.errors
            ambiguity_hints = log_review.ambiguity_hints
        else:
            log = log_review or {}
            ask_human_calls = log.get("ask_human_calls", [])
            guard_events = log.get("guard_events", [])
            tool_failures = log.get("tool_failures", [])
            errors = log.get("errors", [])
            ambiguity_hints = log.get("ambiguity_hints", [])

        lines += [
            f"\n### {heading}\n",
            f"- **Status:** {r['status']}\n",
            f"- **FinalClass:** {r.get('final_class', r['status'])}\n",
            f"- **Elapsed:** {r['elapsed_sec']}s\n",
            f"- **Return code:** {r['returncode']}\n",
            f"- **Log dir:** {r.get('actual_log_dir') or r.get('log_dir') or 'N/A'}\n",
        ]
        if r.get("reported_log_dir") and r.get("reported_log_dir") != r.get("actual_log_dir"):
            lines.append(f"- **Reported log dir (stale):** {r['reported_log_dir']}\n")
        if ask_human_calls:
            lines.append("- **Ambiguities (ask_human):**\n")
            for ah in ask_human_calls:
                lines.append(f"  - `{ah['question']}`\n")
        if guard_events:
            lines.append("- **Guard events:**\n")
            for g in guard_events:
                lines.append(f"  - `{g['event']}`: {g.get('message', '')}\n")
        if tool_failures:
            lines.append("- **Tool failures:**\n")
            for tf in tool_failures:
                lines.append(f"  - `{tf['tool_name']}`: {tf.get('error', 'unknown')}\n")
        if errors:
            lines.append("- **Errors:**\n")
            for e in errors:
                lines.append(f"  - {e.get('message', '')}\n")
        if ambiguity_hints:
            lines.append("- **Ambiguity hints:**\n")
            for ah in ambiguity_hints:
                lines.append(f"  - {ah}\n")
        challenge_progress = {}
        final_result = r.get("final_result")
        if isinstance(final_result, dict):
            raw_progress = final_result.get("challenge_progress")
            if isinstance(raw_progress, dict):
                challenge_progress = raw_progress
        if not challenge_progress:
            summary_path = None
            log_dir = r.get("log_dir")
            if log_dir:
                summary_path = Path(str(log_dir)) / "task_summary.json"
            if summary_path and summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    raw_progress = summary.get("challenge_progress")
                    if isinstance(raw_progress, dict):
                        challenge_progress = raw_progress
                except Exception:
                    challenge_progress = {}
        if challenge_progress:
            lines.append("- **Challenge state:**\n")
            lines.append(f"  - category: `{challenge_progress.get('task_category', '')}`\n")
            lines.append(f"  - phase: `{challenge_progress.get('phase', '')}`\n")
            lines.append(f"  - code changes: `{challenge_progress.get('code_change_count', 0)}`\n")
            lines.append(f"  - last verifier: `{challenge_progress.get('last_verifier_verdict', '')}` `{challenge_progress.get('last_verifier_command', '')}`\n")
            lines.append(f"  - verified after last change: `{challenge_progress.get('verified_after_last_change', False)}`\n")
            lines.append(f"  - redundant verifier count: `{challenge_progress.get('redundant_verifier_count', 0)}`\n")
        if r.get("stderr_preview"):
            lines.append("- **Stderr preview:**\n")
            lines.append(f"  ```\n{textwrap.indent(r['stderr_preview'], '  ')}\n  ```\n")
        if r.get("stdout_preview"):
            lines.append("- **Stdout preview (tail):**\n")
            preview = r["stdout_preview"]
            lines.append(f"  ```\n{textwrap.indent(preview, '  ')}\n  ```\n")

    path.write_text("".join(lines), encoding="utf-8")
    LOGGER.status(f"Report saved to {path}")


def smoke_test_backend(endpoint: str, model: str, api_key: str = "") -> bool:
    """Quickly probe the backend to confirm it is up and responding."""
    LOGGER.status("Running backend smoke test...")
    chat_url = endpoint.rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Say exactly: smoke test ok"}],
        "max_tokens": 10,
        "temperature": 0.0,
    }).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(chat_url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            choices = body.get("choices", [])
            if not choices:
                LOGGER.error("smoke_test", "No choices in response")
                return False
            choice = choices[0]
            content = choice.get("message", {}).get("content", "")
            reasoning = choice.get("message", {}).get("reasoning_content", "")
            if content.strip() or reasoning.strip():
                preview = (content.strip() or reasoning.strip())[:60]
                LOGGER.status(f"Smoke test PASSED — backend responded: {preview}")
                return True
            else:
                LOGGER.warn("smoke_test", "Empty response content")
                return False
    except urllib.error.HTTPError as exc:
        LOGGER.error("smoke_test", f"HTTP {exc.code}: {exc.reason}")
        try:
            LOGGER.error("smoke_test", exc.read().decode("utf-8")[:200])
        except Exception:
            pass
        return False
    except Exception as exc:
        LOGGER.error("smoke_test", f"{type(exc).__name__}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="AHO Baseline Challenge Runner")
    parser.add_argument("--challenges", type=Path, default=DEFAULT_CHALLENGES_PATH)
    parser.add_argument("--smallctl", type=Path, default=DEFAULT_SMALLCTL)
    parser.add_argument("--endpoint", default=os.getenv("SMALLCTL_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--model", default=os.getenv("SMALLCTL_MODEL", DEFAULT_MODEL))
    parser.add_argument("--provider-profile", default=os.getenv("SMALLCTL_PROVIDER_PROFILE", DEFAULT_PROVIDER_PROFILE))
    parser.add_argument("--api-key", default=os.getenv("SMALLCTL_API_KEY", ""))
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--context-limit", type=int, default=None)
    parser.add_argument("--tool-profiles", default="core,data,mutate,network")
    parser.add_argument("--run-mode", default="loop")
    parser.add_argument("--phase", default="execute")
    parser.add_argument("--config", default="")
    parser.add_argument("--fresh-run", action="store_true")
    parser.add_argument("--hide-thinking", action="store_true", default=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Per-challenge timeout in seconds")
    parser.add_argument("--report", type=Path, default=Path("4-b-baseline-2.md"))
    parser.add_argument("--max-challenges", type=int, default=None, help="Limit total challenges for testing")
    parser.add_argument("--start-at", type=int, default=1, help="Start at challenge N (1-indexed)")
    parser.add_argument("--skip-ambiguity-prompt", action="store_true", help="Do not pause for ambiguity clarification")
    parser.add_argument("--skip-smoke-test", action="store_true", help="Skip backend smoke test")
    parser.add_argument("--yes", action="store_true", help="Skip interactive provider prompt and accept defaults")
    args = parser.parse_args()

    if not args.smallctl.exists():
        print(f"ERROR: smallctl not found at {args.smallctl}")
        return 1
    if not args.challenges.exists():
        print(f"ERROR: challenges file not found at {args.challenges}")
        return 1

    # Interactive provider prompt
    print("\n" + "=" * 50)
    print("AHO Baseline Challenge Runner")
    print("=" * 50)
    print(f"Default provider settings:")
    print(f"  Endpoint:        {args.endpoint}")
    print(f"  Model:           {args.model}")
    print(f"  Provider profile: {args.provider_profile}")
    print("")
    if args.yes:
        print("Non-interactive mode (--yes): accepting defaults.")
    else:
        try:
            new_endpoint = input(f"Enter endpoint (or Enter to keep '{args.endpoint}'): ").strip()
            if new_endpoint:
                args.endpoint = new_endpoint
            new_model = input(f"Enter model (or Enter to keep '{args.model}'): ").strip()
            if new_model:
                args.model = new_model
            new_profile = input(f"Enter provider profile (or Enter to keep '{args.provider_profile}'): ").strip()
            if new_profile:
                args.provider_profile = new_profile
        except (EOFError, KeyboardInterrupt):
            print("\nUsing defaults.")
    print("=" * 50 + "\n")

    if not args.skip_smoke_test:
        backend_ok = smoke_test_backend(args.endpoint, args.model, args.api_key)
        if not backend_ok:
            try:
                cont = input("Backend smoke test failed. Continue anyway? [y/N]: ").strip().lower()
                if cont not in ("y", "yes"):
                    print("Aborted.")
                    return 1
            except (EOFError, KeyboardInterrupt):
                print("Aborted.")
                return 1
    else:
        print("[SMOKE TEST] Skipped (--skip-smoke-test)")

    raw = args.challenges.read_text(encoding="utf-8")
    sysadmin, coding, creds = parse_challenges(raw)

    total_challenges = len(sysadmin) + len(coding)
    LOGGER.info("baseline_start", "Starting baseline run", total_challenges=total_challenges, sysadmin=len(sysadmin), coding=len(coding))

    all_results: list[dict] = []
    challenge_counter = 0
    overall_t0 = time.monotonic()

    for ch in sysadmin + coding:
        challenge_counter += 1
        if challenge_counter < args.start_at:
            continue
        if args.max_challenges is not None and len(all_results) >= args.max_challenges:
            break

        is_sysadmin = challenge_counter <= len(sysadmin)
        result = run_challenge(
            ch, args.smallctl, args,
            creds=creds, is_sysadmin=is_sysadmin,
            current=challenge_counter, total=total_challenges,
        )
        all_results.append(result)

        overall_elapsed = time.monotonic() - overall_t0
        completed = sum(1 for r in all_results if r["status"] == "completed")
        failed = sum(1 for r in all_results if r["status"] in ("failed", "crashed", "timeout"))
        LOGGER.status(
            f"Overall progress: {len(all_results)}/{total_challenges} done — {completed} completed, {failed} failed — total elapsed {overall_elapsed:.1f}s",
            completed=completed,
            failed=failed,
            total_elapsed_sec=round(overall_elapsed, 1),
            challenges_run=len(all_results),
        )

        # Escalate ambiguities
        if not args.skip_ambiguity_prompt:
            log_review = result.get("log_review")
            questions = [ah["question"] for ah in (log_review.ask_human_calls if isinstance(log_review, LogReview) else log_review.get("ask_human_calls", []))]
            if questions:
                clarification = ask_user_for_clarification(questions)
                if clarification:
                    LOGGER.status(f"Retrying {ch['id']} with user clarification...")
                    ch_retry = dict(ch)
                    ch_retry["description"] += f"\n\nClarification from user: {clarification}"
                    result2 = run_challenge(
                        ch_retry, args.smallctl, args,
                        creds=creds, is_sysadmin=is_sysadmin,
                        current=challenge_counter, total=total_challenges,
                    )
                    all_results[-1] = result2  # replace with retry result

    overall_elapsed = time.monotonic() - overall_t0
    LOGGER.info("baseline_complete", "All challenges finished", total_elapsed_sec=round(overall_elapsed, 1), challenges_run=len(all_results))
    generate_report(all_results, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
