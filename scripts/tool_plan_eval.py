from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from subprocess import PIPE
from typing import Any

import yaml


MODES = ("loop", "tool_plan")

RECOVERY_METRIC_KEYS = (
    "tool_plan_invocations",
    "tool_plan_parse_failures",
    "tool_plan_steps_requested",
    "tool_plan_steps_executed",
    "tool_plan_step_failures",
    "tool_plan_unsafe_steps_blocked",
    "tool_plan_wrong_path_count",
    "tool_plan_repeated_read_count",
    "tool_plan_fallback_count",
    "tool_plan_evidence_before_patch_count",
    "tool_plan_observation_tokens",
    "tool_plan_planner_tokens",
    "tool_plan_solver_tokens",
    "tool_plan_total_tokens",
    "tool_plan_planner_valid",
    "tool_plan_planner_step_count",
    "tool_plan_planner_tools",
    "tool_plan_repair_attempts",
    "tool_plan_worker_steps_requested",
    "tool_plan_worker_steps_executed",
    "tool_plan_worker_step_failures",
    "tool_plan_worker_success_rate",
    "tool_plan_worker_missing_record_count",
    "tool_plan_worker_duplicate_read_count",
    "tool_plan_worker_artifact_yield_count",
    "tool_plan_worker_tool_failure_classes",
    "tool_plan_refine_verdict",
    "model_stream_halt_count",
)

MEASURED_USAGE_KEYS = ("token_usage",)

LATENCY_KEYS = (
    "planner_latency_sec",
    "worker_latency_sec",
    "solver_latency_sec",
    "tool_execution_duration_sec",
    "overhead_preparation_duration_sec",
)

PROMPT_SHAPE_KEYS = (
    "planner_has_rewoo_plan_state",
    "planner_excludes_tool_observations",
    "solver_has_rewoo_evidence",
    "solver_excludes_generic_warm_summaries",
    "solver_has_tool_plan_evidence_ids",
)

TOKEN_SAVINGS_GATE_TAGS = ("repo_analysis", "log_investigation")
TOKEN_SAVINGS_GATE_PCT = -20.0
DEFAULT_PROGRESS_INTERVAL_SEC = 30.0
DIAGNOSTIC_TAIL_CHARS = 4000
CHILD_PROGRESS_LINE_LIMIT = 1200

CHILD_PROGRESS_MARKERS = (
    "build_registry",
    "graph node started",
    "graph node completed",
    "model_call_start",
    "model_call_end",
    "model_call_cancelled",
    "thinking:",
    "model_thinking",
    "chunk chat stream chunk",
    "tool_calls",
    "dispatch_start",
    "dispatch_complete",
    "tool dispatch started",
    "tool dispatch finished",
    "prepare_prompt_start",
    "prepare_prompt_end",
    "prompt_budget",
    "prompt_shape",
    "tool_plan",
    "planner",
    "solver",
    "refiner",
    "task_interrupted",
    "task_finalize",
    "ERROR",
    "WARNING",
    "Traceback",
    "Exception",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run loop vs ToolPlan comparison tasks.")
    parser.add_argument("--tasks", default="evals/tool_plan/tasks.jsonl", help="Task file or directory (.jsonl, .yaml, .yml).")
    parser.add_argument("--output", default=".smallctl/artifacts/tool_plan_eval_results.jsonl", help="JSONL output path.")
    parser.add_argument("--report", default=".smallctl/artifacts/tool_plan_eval_report.json", help="Side-by-side report path.")
    parser.add_argument(
        "--markdown-report",
        default=None,
        help="Optional markdown summary path. Defaults to the JSON report path with a .md suffix.",
    )
    parser.add_argument("--mode", choices=[*MODES, "both"], default="both", help="Runtime mode to run.")
    parser.add_argument("--timeout-sec", type=int, default=900, help="Per-task timeout.")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the full task set this many times.")
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Run only the task with this id. May be passed more than once.",
    )
    parser.add_argument("--max-tasks", type=int, default=None, help="Run only the first N loaded tasks after filtering.")
    parser.add_argument(
        "--rewoo-frames",
        action="store_true",
        help="Enable ReWOO planner/solver/refiner lane frames for tool_plan eval runs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args(argv)

    tasks = _filter_tasks(_load_tasks(Path(args.tasks)), task_ids=args.task_id, max_tasks=args.max_tasks)
    modes = list(MODES if args.mode == "both" else (args.mode,))
    repeat = max(1, int(args.repeat))
    commands = [
        _command_for(task["task"], mode=mode, rewoo_frames=args.rewoo_frames)
        for _repeat_index in range(repeat)
        for task in tasks
        for mode in modes
    ]
    _log_progress(
        f"loaded {len(tasks)} task(s); running {len(commands)} child process(es) "
        f"with timeout={max(1, int(args.timeout_sec))}s"
    )
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    run_index = 0
    for repeat_index in range(1, repeat + 1):
        for task in tasks:
            for mode in modes:
                run_index += 1
                _log_progress(
                    f"starting {run_index}/{len(commands)} repeat={repeat_index}/{repeat} task={task.get('id')} mode={mode}"
                )
                result = _run_task(
                    task,
                    mode=mode,
                    timeout_sec=max(1, int(args.timeout_sec)),
                    rewoo_frames=args.rewoo_frames,
                    repeat_index=repeat_index,
                    repeat_total=repeat,
                )
                results.append(result)
                with output_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=True, sort_keys=True) + "\n")
                    handle.flush()
                print(json.dumps(_summary(result), ensure_ascii=True, sort_keys=True))
                _emit_result_progress(result)
                _emit_error_output(result)

    if args.mode == "both":
        report = _build_report(results)
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        markdown_report_path = Path(args.markdown_report) if args.markdown_report else report_path.with_suffix(".md")
        markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_report_path.write_text(_build_markdown_report(report), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True, sort_keys=True))
        if report.get("prompt_shape_failures"):
            _log_progress(f"prompt shape failures: {json.dumps(report['prompt_shape_failures'], sort_keys=True)}")
        _log_progress(f"wrote report to {report_path}")
        _log_progress(f"wrote markdown report to {markdown_report_path}")
        return _report_exit_code(report)
    return 0


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def _add_task(task: dict[str, Any], source: str, line_number: int | None = None) -> None:
        task = dict(task)
        task_text = str(task.get("task") or "").strip()
        if not task_text:
            loc = f"{source}:{line_number}" if line_number else source
            raise SystemExit(f"{loc}: missing task")
        task["task"] = task_text
        task.setdefault("id", f"task-{len(tasks) + 1}")
        tid = str(task["id"])
        if tid in seen_ids:
            return
        seen_ids.add(tid)
        tasks.append(task)

    def _load_file(file_path: Path) -> None:
        if file_path.suffix.lower() == ".jsonl":
            for line_number, raw in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{file_path}:{line_number}: invalid JSON: {exc}") from exc
                if isinstance(payload, dict):
                    _add_task(payload, str(file_path), line_number)
        elif file_path.suffix.lower() in (".yaml", ".yml"):
            try:
                data = yaml.safe_load(file_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise SystemExit(f"{file_path}: invalid YAML: {exc}") from exc
            if isinstance(data, dict):
                _add_task(data, str(file_path))
            elif isinstance(data, list):
                for idx, item in enumerate(data, start=1):
                    if isinstance(item, dict):
                        _add_task(item, str(file_path), idx)
        else:
            raise SystemExit(f"Unsupported task file format: {file_path}")

    if path.is_dir():
        files = sorted(
            p for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in (".jsonl", ".yaml", ".yml")
        )
        for file_path in files:
            _load_file(file_path)
    else:
        _load_file(path)

    if not tasks:
        raise SystemExit(f"No tasks found in {path}")
    return tasks


def _filter_tasks(
    tasks: list[dict[str, Any]],
    *,
    task_ids: list[str] | None = None,
    max_tasks: int | None = None,
) -> list[dict[str, Any]]:
    selected = list(tasks)
    ids = [str(task_id).strip() for task_id in (task_ids or []) if str(task_id).strip()]
    if ids:
        wanted = set(ids)
        selected = [task for task in selected if str(task.get("id") or "") in wanted]
        if not selected:
            raise SystemExit(f"No tasks matched --task-id: {', '.join(ids)}")
    if max_tasks is not None:
        max_count = int(max_tasks)
        if max_count <= 0:
            raise SystemExit("--max-tasks must be greater than 0")
        selected = selected[:max_count]
    if not selected:
        raise SystemExit("No tasks selected")
    return selected


def _command_for(task: str, *, mode: str, rewoo_frames: bool = False) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "smallctl.main",
        "--run-mode",
        mode,
        "--fresh-run",
        "--task",
        task,
    ]
    if mode == "tool_plan" and rewoo_frames:
        command.extend(
            [
                "--rewoo-planner-frame",
                "--rewoo-solver-frame",
                "--rewoo-refiner-frame",
            ]
        )
    return command


def _run_task(
    task: dict[str, Any],
    *,
    mode: str,
    timeout_sec: int,
    rewoo_frames: bool = False,
    repeat_index: int = 1,
    repeat_total: int = 1,
) -> dict[str, Any]:
    command = _command_for(str(task["task"]), mode=mode, rewoo_frames=rewoo_frames)
    env = dict(os.environ)
    repo_src = str((Path(__file__).resolve().parents[1] / "src").resolve())
    env["PYTHONPATH"] = repo_src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    started = time.perf_counter()
    stdout = ""
    stderr = ""
    returncode = 127
    timed_out = False
    progress_interval = _progress_interval_sec()
    _log_progress(f"command task={task.get('id')} mode={mode}: {_shell_join(command)}")
    try:
        process = subprocess.Popen(
            command,
            stdout=PIPE,
            stderr=PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
    except OSError as exc:
        stderr = str(exc)
    else:
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        stop_tailers = threading.Event()
        tail_threads: list[threading.Thread] = []
        tailed_run_dirs: set[str] = set()
        tail_lock = threading.Lock()

        def _maybe_start_log_tailers(line: str) -> None:
            run_dir = _run_log_dir_from_line(line)
            if not run_dir:
                return
            with tail_lock:
                if run_dir in tailed_run_dirs:
                    return
                tailed_run_dirs.add(run_dir)
            _log_progress(f"tailing child logs from {run_dir}")
            for log_name in ("harness.log", "model_output.log", "tools.log"):
                tail_threads.append(
                    _start_log_file_tailer(
                        Path(run_dir) / log_name,
                        stop_event=stop_tailers,
                        task_id=str(task.get("id") or ""),
                        mode=mode,
                    )
                )

        stdout_thread = _start_stream_reader(
            process.stdout,
            chunks=stdout_chunks,
            stream_name="stdout",
            task_id=str(task.get("id") or ""),
            mode=mode,
            on_line=_maybe_start_log_tailers,
        )
        stderr_thread = _start_stream_reader(
            process.stderr,
            chunks=stderr_chunks,
            stream_name="stderr",
            task_id=str(task.get("id") or ""),
            mode=mode,
            on_line=_maybe_start_log_tailers,
        )
        next_progress_at = progress_interval
        while True:
            elapsed = time.perf_counter() - started
            remaining = timeout_sec - elapsed
            if remaining <= 0:
                timed_out = True
                process.kill()
                process.wait()
                returncode = 124
                _log_progress(
                    f"timeout task={task.get('id')} mode={mode} after {round(elapsed, 1)}s"
                )
                break
            returncode = process.poll()
            if returncode is not None:
                process.wait()
                break
            if elapsed >= next_progress_at:
                _log_progress(
                    f"still running task={task.get('id')} mode={mode} "
                    f"elapsed={round(elapsed, 1)}s timeout={timeout_sec}s"
                )
                next_progress_at += progress_interval
            time.sleep(min(0.2, max(0.01, remaining)))
        stop_tailers.set()
        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)
        for thread in tail_threads:
            thread.join(timeout=2)
        stdout = "".join(stdout_chunks)
        stderr = "".join(stderr_chunks)
        if returncode is None:
            returncode = process.returncode
            if returncode is None:
                returncode = 127
    duration = round(time.perf_counter() - started, 3)
    final_json = _last_json_object(stdout)
    metrics = _extract_metrics(final_json)
    prompt_shape = _prompt_shape_assertions(stdout)
    success = _is_success(final_json, task.get("tags", []))
    return {
        "task_id": task.get("id"),
        "tags": task.get("tags", []),
        "mode": mode,
        "repeat_index": repeat_index,
        "repeat_total": repeat_total,
        "task": task["task"],
        "duration_sec": duration,
        "returncode": returncode,
        "timed_out": timed_out,
        "final_json": final_json,
        "final_success": success,
        "expectations": task.get("expectations") if isinstance(task.get("expectations"), dict) else None,
        **metrics,
        "prompt_shape": prompt_shape,
        "stdout": stdout[-12000:],
        "stderr": stderr[-12000:],
    }


def _progress_interval_sec() -> float:
    raw = os.environ.get("TOOL_PLAN_EVAL_PROGRESS_SEC")
    if raw is None:
        return DEFAULT_PROGRESS_INTERVAL_SEC
    try:
        return max(1.0, float(raw))
    except ValueError:
        return DEFAULT_PROGRESS_INTERVAL_SEC


def _log_progress(message: str) -> None:
    print(f"[tool-plan-eval] {message}", file=sys.stderr, flush=True)


def _start_stream_reader(
    stream: Any,
    *,
    chunks: list[str],
    stream_name: str,
    task_id: str,
    mode: str,
    on_line: Any | None = None,
) -> threading.Thread:
    def _read() -> None:
        if stream is None:
            return
        try:
            for line in stream:
                chunks.append(line)
                if on_line is not None:
                    on_line(line)
                _emit_child_line(task_id=task_id, mode=mode, stream_name=stream_name, line=line)
        finally:
            try:
                stream.close()
            except Exception:
                pass

    thread = threading.Thread(target=_read, daemon=True)
    thread.start()
    return thread


def _start_log_file_tailer(
    path: Path,
    *,
    stop_event: threading.Event,
    task_id: str,
    mode: str,
) -> threading.Thread:
    def _tail() -> None:
        deadline = time.monotonic() + 10.0
        while not stop_event.is_set() and not path.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        if stop_event.is_set() or not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                while not stop_event.is_set():
                    line = handle.readline()
                    if line:
                        _emit_child_line(
                            task_id=task_id,
                            mode=mode,
                            stream_name=f"log/{path.name}",
                            line=line,
                        )
                    else:
                        time.sleep(0.2)
        except OSError as exc:
            _log_progress(f"{task_id} {mode} log/{path.name}: unable to tail log: {exc}")

    thread = threading.Thread(target=_tail, daemon=True)
    thread.start()
    return thread


def _emit_child_line(*, task_id: str, mode: str, stream_name: str, line: str) -> None:
    rendered = _render_child_progress_line(line)
    if rendered is None:
        return
    _log_progress(f"{task_id} {mode} {stream_name}: {rendered}")


def _render_child_progress_line(line: str) -> str | None:
    text = str(line or "").strip()
    if not text:
        return None
    if not _is_interesting_child_line(text):
        return None
    compact = _compact_child_log_line(text)
    return _truncate_line(compact, CHILD_PROGRESS_LINE_LIMIT)


def _is_interesting_child_line(text: str) -> bool:
    return any(marker in text for marker in CHILD_PROGRESS_MARKERS)


def _compact_child_log_line(text: str) -> str:
    payload = _json_payload_from_log_line(text)
    if not isinstance(payload, dict):
        return text
    event = _event_name_from_log_line(text)
    if event == "chunk":
        chunk = payload.get("chunk")
        if isinstance(chunk, dict):
            choices = chunk.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                delta = first_choice.get("delta") if isinstance(first_choice, dict) else None
                if isinstance(delta, dict) and delta.get("tool_calls"):
                    return f"model_chunk {_summarize_tool_call_delta(delta.get('tool_calls'))}"
        return text
    if event in {"dispatch_start", "dispatch_complete"}:
        tool_name = payload.get("tool_name")
        args = payload.get("arguments")
        success = payload.get("success")
        tier = payload.get("tier")
        parts = [event]
        if tool_name:
            parts.append(f"tool={tool_name}")
        if success is not None:
            parts.append(f"success={success}")
        if tier:
            parts.append(f"tier={tier}")
        if args:
            parts.append(f"args={_truncate_line(json.dumps(args, sort_keys=True, ensure_ascii=True), 500)}")
        return " ".join(parts)
    if event in {
        "model_call_start",
        "model_call_end",
        "model_call_cancelled",
        "prepare_prompt_start",
        "prepare_prompt_end",
        "task_interrupted",
        "task_finalize",
    }:
        fields = {
            key: payload.get(key)
            for key in ("node", "elapsed_sec", "timeout_sec", "status", "result_status", "reason", "task_id")
            if key in payload
        }
        return f"{event} {json.dumps(fields, sort_keys=True, ensure_ascii=True)}"
    if event == "prompt_budget":
        fields = {
            key: payload.get(key)
            for key in ("estimated_prompt_tokens", "message_count", "sections")
            if key in payload
        }
        return f"{event} {json.dumps(fields, sort_keys=True, ensure_ascii=True)}"
    if event in {"thinking", "model_thinking"}:
        thinking = payload.get("thinking_text") or payload.get("text")
        if thinking:
            return f"{event}: {thinking}"
    return text


def _summarize_tool_call_delta(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list):
        return "tool_call_delta"
    parts: list[str] = []
    for call in tool_calls[:3]:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        function = function if isinstance(function, dict) else {}
        name = function.get("name")
        args = function.get("arguments")
        item = "tool_call"
        if name:
            item += f"={name}"
        if args:
            item += f" args_fragment={_truncate_line(str(args), 300)}"
        parts.append(item)
    return "; ".join(parts) if parts else "tool_call_delta"


def _json_payload_from_log_line(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        value = json.loads(text[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _run_log_dir_from_line(line: str) -> str | None:
    payload = _json_payload_from_log_line(str(line or ""))
    if not isinstance(payload, dict):
        return None
    value = payload.get("run_log_dir")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _event_name_from_log_line(text: str) -> str:
    prefix = text.split("{", 1)[0].strip()
    parts = prefix.split()
    if len(parts) >= 2:
        return parts[1]
    return ""


def _truncate_line(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _shell_join(command: list[str]) -> str:
    return " ".join(_shell_quote(part) for part in command)


def _shell_quote(value: str) -> str:
    if not value:
        return "''"
    safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_@%+=:,./-"
    if all(char in safe for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _emit_result_progress(result: dict[str, Any]) -> None:
    _log_progress(
        f"finished task={result.get('task_id')} mode={result.get('mode')} "
        f"returncode={result.get('returncode')} timed_out={result.get('timed_out')} "
        f"duration={result.get('duration_sec')}s final_success={result.get('final_success')}"
    )


def _emit_error_output(result: dict[str, Any]) -> None:
    stderr = str(result.get("stderr") or "")
    stdout = str(result.get("stdout") or "")
    failed = bool(result.get("timed_out")) or int(result.get("returncode") or 0) != 0
    if not failed and not _stderr_has_error(stderr):
        return
    label = (
        f"task={result.get('task_id')} mode={result.get('mode')} "
        f"returncode={result.get('returncode')} timed_out={result.get('timed_out')}"
    )
    if stderr.strip():
        _log_progress(f"stderr tail for {label}:\n{stderr[-DIAGNOSTIC_TAIL_CHARS:]}")
    elif stdout.strip():
        _log_progress(f"stdout tail for {label}:\n{stdout[-DIAGNOSTIC_TAIL_CHARS:]}")
    else:
        _log_progress(f"no captured stdout/stderr for {label}")


def _stderr_has_error(stderr: str) -> bool:
    lowered = stderr.lower()
    return "error" in lowered or "traceback" in lowered or "exception" in lowered


def _text_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _extract_metrics(final_json: dict[str, Any] | None) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    recovery = (final_json or {}).get("recovery_metrics") or {}
    if isinstance(recovery, dict):
        for key in RECOVERY_METRIC_KEYS:
            metrics[key] = recovery.get(key)
    metrics["token_usage"] = _extract_token_usage(final_json)
    metrics["latency_metrics"] = _extract_latency_metrics(final_json)
    return metrics


def _extract_token_usage(final_json: dict[str, Any] | None) -> int | None:
    if not isinstance(final_json, dict):
        return None
    value = final_json.get("token_usage")
    return _coerce_optional_int(value)


def _extract_latency_metrics(final_json: dict[str, Any] | None) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {key: None for key in LATENCY_KEYS}
    if not isinstance(final_json, dict):
        return metrics
    raw = final_json.get("latency_metrics")
    if not isinstance(raw, dict):
        return metrics
    for key in LATENCY_KEYS:
        value = raw.get(key)
        metrics[key] = _coerce_optional_float(value)
    return metrics


def _coerce_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _prompt_shape_assertions(stdout: str) -> dict[str, bool | None]:
    text = str(stdout or "")
    has_planner = "You are the ToolPlan planner" in text
    has_solver = "You are the ToolPlan solver" in text
    return {
        "planner_has_rewoo_plan_state": ("REWOO PLAN STATE" in text) if has_planner else None,
        "planner_excludes_tool_observations": ("TOOL PLAN OBSERVATIONS" not in _planner_slice(text)) if has_planner else None,
        "solver_has_rewoo_evidence": ("REWOO EVIDENCE" in text) if has_solver else None,
        "solver_excludes_generic_warm_summaries": (
            "RETRIEVED SUMMARIES" not in _solver_slice(text)
            and "EPISODIC SUMMARIES" not in _solver_slice(text)
        ) if has_solver else None,
        "solver_has_tool_plan_evidence_ids": bool(_solver_has_tool_plan_evidence_id(_solver_slice(text))) if has_solver else None,
    }


def _planner_slice(text: str) -> str:
    start = text.find("You are the ToolPlan planner")
    if start < 0:
        return ""
    end = text.find("You are the ToolPlan solver", start)
    return text[start:] if end < 0 else text[start:end]


def _solver_slice(text: str) -> str:
    start = text.find("You are the ToolPlan solver")
    if start < 0:
        return ""
    return text[start:]


def _solver_has_tool_plan_evidence_id(text: str) -> bool:
    return "TP-E" in text or "TP:" in text


def _is_success(final_json: dict[str, Any] | None, tags: list[str]) -> bool:
    if not isinstance(final_json, dict):
        return False
    status = str(final_json.get("status") or "").strip().lower()
    if status in ("completed", "success", "ok"):
        return True
    if "wrong_path" in tags:
        return status == "stopped" or bool(final_json.get("recovery_metrics", {}).get("tool_plan_fallback_count"))
    return False


def _final_text(final_json: dict[str, Any] | None) -> str:
    if not isinstance(final_json, dict):
        return ""
    parts: list[str] = []
    for key in ("reason", "message", "output", "answer", "summary"):
        value = final_json.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
        elif isinstance(value, dict):
            for nested_key in ("message", "question", "reason", "content"):
                nested = value.get(nested_key)
                if isinstance(nested, str) and nested.strip():
                    parts.append(nested.strip())
    return "\n".join(parts).strip()


def _regex_count(pattern: str, text: str) -> int:
    if not text:
        return 0
    return len(re.findall(pattern, text))


def _collect_report_measurements(result: dict[str, Any]) -> dict[str, Any]:
    final_json = result.get("final_json") if isinstance(result.get("final_json"), dict) else {}
    metrics = _extract_metrics(final_json)
    latency_metrics = metrics.pop("latency_metrics", {})
    return {
        "token_usage": metrics.get("token_usage"),
        "latency_metrics": latency_metrics,
        **metrics,
    }


def _collect_recovery_metric(final_json: dict[str, Any] | None, key: str) -> Any:
    if not isinstance(final_json, dict):
        return None
    recovery = final_json.get("recovery_metrics")
    if not isinstance(recovery, dict):
        return None
    return recovery.get(key)


def _task_expectations(task: dict[str, Any]) -> dict[str, Any]:
    expectations = task.get("expectations")
    return expectations if isinstance(expectations, dict) else {}


def _score_expectations(
    *,
    task: dict[str, Any],
    final_json: dict[str, Any] | None,
    text: str,
    evidence_ids: list[str],
    artifact_ids: list[str],
) -> dict[str, Any]:
    expectations = _task_expectations(task)
    required_files = [str(path) for path in expectations.get("required_files", []) if str(path).strip()]
    required_terms = [str(term) for term in expectations.get("required_terms", []) if str(term).strip()]
    success_statuses = {str(status).strip().lower() for status in expectations.get("success_statuses", []) if str(status).strip()}
    grounding_required = bool(expectations.get("grounding_required", False))
    unsafe_fallback_expected = bool(expectations.get("unsafe_fallback_expected", False))

    missing_files = [path for path in required_files if path not in text]
    missing_terms = [term for term in required_terms if term not in text]
    evidence_ref_count = len(evidence_ids)
    artifact_ref_count = len(artifact_ids)
    grounded = bool(evidence_ref_count or artifact_ref_count or (required_files and not missing_files) or (required_terms and not missing_terms))
    if grounding_required:
        grounded = bool(evidence_ref_count or artifact_ref_count)

    status = str((final_json or {}).get("status") or "").strip().lower()
    success = status in success_statuses if success_statuses else _is_success(final_json, list(task.get("tags", [])))
    if unsafe_fallback_expected:
        success = success or status == "stopped" or bool(_collect_recovery_metric(final_json, "tool_plan_fallback_count"))
    if grounding_required and not grounded:
        success = False

    unbacked_claim_count = len(missing_files) + len(missing_terms)
    if grounding_required and not grounded:
        unbacked_claim_count = max(1, unbacked_claim_count)

    return {
        "expectations": expectations or None,
        "expectation_success": success,
        "expectation_grounded": grounded if expectations else None,
        "expectation_unsafe_fallback": unsafe_fallback_expected if expectations else None,
        "expectation_missing_files": missing_files or None,
        "expectation_missing_terms": missing_terms or None,
        "solver_grounded": grounded,
        "solver_evidence_refs_count": evidence_ref_count,
        "solver_uses_observation_artifacts": artifact_ref_count > 0,
        "solver_unbacked_claim_count": unbacked_claim_count,
        "solver_mutation_after_evidence": bool((_collect_recovery_metric(final_json, "tool_plan_evidence_before_patch_count") or 0)),
        "solver_refine_verdict": _collect_recovery_metric(final_json, "tool_plan_refine_verdict"),
        "solver_artifact_refs": artifact_ids or None,
        "solver_evidence_refs": evidence_ids or None,
    }


def _last_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    last: dict[str, Any] | None = None
    last_end = -1
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        absolute_end = index + end
        if isinstance(value, dict) and absolute_end >= last_end:
            last = value
            last_end = absolute_end
    return last


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    final_json = result.get("final_json")
    status = final_json.get("status") if isinstance(final_json, dict) else None
    measurements = _collect_report_measurements(result)
    latency = measurements.get("latency_metrics") if isinstance(measurements.get("latency_metrics"), dict) else {}
    summary: dict[str, Any] = {
        "task_id": result.get("task_id"),
        "mode": result.get("mode"),
        "status": status,
        "returncode": result.get("returncode"),
        "duration_sec": result.get("duration_sec"),
        "timed_out": result.get("timed_out"),
        "final_success": result.get("final_success"),
        "token_usage": measurements.get("token_usage"),
        "latency_metrics": latency,
    }
    for key in RECOVERY_METRIC_KEYS:
        if key in result:
            summary[key] = result[key]
    return summary


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _pct_delta(base: float | None, other: float | None) -> float | None:
    if base is None or other is None or base == 0:
        return None
    return round(((other - base) / base) * 100.0, 3)


def _delta(base: float | None, other: float | None) -> float | None:
    if base is None or other is None:
        return None
    return round(other - base, 3)


def _truthy_rate(comparisons: list[dict[str, Any]], key: str) -> float | None:
    values = [1.0 if bool(comparison.get(key)) else 0.0 for comparison in comparisons if comparison.get(key) is not None]
    return _mean(values)


def _numeric_rate(comparisons: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for comparison in comparisons:
        value = comparison.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return _mean(values)


def _abort_indicator(comparison: dict[str, Any]) -> bool:
    return bool(
        comparison.get("abort_count")
        or comparison.get("loop_guard_count")
        or comparison.get("tool_plan_timed_out")
    )


def _token_savings_gate(comparisons: list[dict[str, Any]]) -> bool | None:
    scoped = [
        comparison for comparison in comparisons
        if any(tag in TOKEN_SAVINGS_GATE_TAGS for tag in comparison.get("tags", []))
        and comparison.get("token_delta_pct") is not None
    ]
    if not scoped:
        return None
    return all(float(comparison["token_delta_pct"]) <= TOKEN_SAVINGS_GATE_PCT for comparison in scoped)


def _build_comparison(result_pair: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    loop_result = result_pair["loop"]
    tp_result = result_pair["tool_plan"]
    loop_final = loop_result.get("final_json") if isinstance(loop_result.get("final_json"), dict) else {}
    tp_final = tp_result.get("final_json") if isinstance(tp_result.get("final_json"), dict) else {}
    loop_metrics = _extract_metrics(loop_final)
    tp_metrics = _extract_metrics(tp_final)
    loop_latency = loop_metrics.get("latency_metrics") if isinstance(loop_metrics.get("latency_metrics"), dict) else {}
    tp_latency = tp_metrics.get("latency_metrics") if isinstance(tp_metrics.get("latency_metrics"), dict) else {}
    task_tags = list(task.get("tags", []) or [])
    task_expectations = _task_expectations(task)
    final_text = _final_text(tp_final)
    evidence_ids = _extract_evidence_ids(tp_final, final_text)
    artifact_ids = _extract_artifact_refs(tp_final, final_text)
    expectations_score = _score_expectations(
        task=task,
        final_json=tp_final,
        text=final_text,
        evidence_ids=evidence_ids,
        artifact_ids=artifact_ids,
    )
    comparison: dict[str, Any] = {
        "task_id": result_pair["task_id"],
        "run_id": result_pair.get("run_id") or result_pair["task_id"],
        "repeat_index": int(task.get("repeat_index") or 1),
        "repeat_total": int(task.get("repeat_total") or 1),
        "tags": task_tags,
        "task": task.get("task", ""),
        "expectations": task_expectations or None,
        "loop_success": loop_result.get("final_success"),
        "tool_plan_success": tp_result.get("final_success"),
        "loop_status": loop_final.get("status") if isinstance(loop_final, dict) else None,
        "tool_plan_status": tp_final.get("status") if isinstance(tp_final, dict) else None,
        "loop_returncode": loop_result.get("returncode"),
        "tool_plan_returncode": tp_result.get("returncode"),
        "loop_timed_out": loop_result.get("timed_out"),
        "tool_plan_timed_out": tp_result.get("timed_out"),
        "loop_duration_sec": loop_result.get("duration_sec"),
        "tool_plan_duration_sec": tp_result.get("duration_sec"),
        "duration_delta_sec": _delta(_coerce_optional_float(loop_result.get("duration_sec")), _coerce_optional_float(tp_result.get("duration_sec"))),
        "duration_delta_pct": _pct_delta(_coerce_optional_float(loop_result.get("duration_sec")), _coerce_optional_float(tp_result.get("duration_sec"))),
        "loop_tokens": loop_metrics.get("token_usage"),
        "tool_plan_tokens": tp_metrics.get("token_usage"),
        "token_delta": _delta(_coerce_optional_float(loop_metrics.get("token_usage")), _coerce_optional_float(tp_metrics.get("token_usage"))),
        "token_delta_pct": _pct_delta(_coerce_optional_float(loop_metrics.get("token_usage")), _coerce_optional_float(tp_metrics.get("token_usage"))),
        "loop_latency_sec": loop_latency.get("tool_execution_duration_sec") if isinstance(loop_latency, dict) else None,
        "tool_plan_latency_sec": tp_latency.get("tool_execution_duration_sec") if isinstance(tp_latency, dict) else None,
        "loop_planner_latency_sec": loop_latency.get("planner_latency_sec") if isinstance(loop_latency, dict) else None,
        "tool_plan_planner_latency_sec": tp_latency.get("planner_latency_sec") if isinstance(tp_latency, dict) else None,
        "loop_worker_latency_sec": loop_latency.get("worker_latency_sec") if isinstance(loop_latency, dict) else None,
        "tool_plan_worker_latency_sec": tp_latency.get("worker_latency_sec") if isinstance(tp_latency, dict) else None,
        "loop_solver_latency_sec": loop_latency.get("solver_latency_sec") if isinstance(loop_latency, dict) else None,
        "tool_plan_solver_latency_sec": tp_latency.get("solver_latency_sec") if isinstance(tp_latency, dict) else None,
        "loop_tool_execution_duration_sec": loop_latency.get("tool_execution_duration_sec") if isinstance(loop_latency, dict) else None,
        "tool_plan_tool_execution_duration_sec": tp_latency.get("tool_execution_duration_sec") if isinstance(tp_latency, dict) else None,
        "loop_overhead_preparation_duration_sec": loop_latency.get("overhead_preparation_duration_sec") if isinstance(loop_latency, dict) else None,
        "tool_plan_overhead_preparation_duration_sec": tp_latency.get("overhead_preparation_duration_sec") if isinstance(tp_latency, dict) else None,
        "planner_valid": bool(tp_metrics.get("tool_plan_planner_valid")),
        "planner_parse_failures": tp_metrics.get("tool_plan_parse_failures"),
        "tool_plan_parse_failures": tp_metrics.get("tool_plan_parse_failures"),
        "planner_repair_attempts": tp_metrics.get("tool_plan_repair_attempts"),
        "planner_unsafe_steps_blocked": tp_metrics.get("tool_plan_unsafe_steps_blocked"),
        "tool_plan_unsafe_steps_blocked": tp_metrics.get("tool_plan_unsafe_steps_blocked"),
        "planner_wrong_path_count": tp_metrics.get("tool_plan_wrong_path_count"),
        "tool_plan_wrong_path_count": tp_metrics.get("tool_plan_wrong_path_count"),
        "planner_step_count": tp_metrics.get("tool_plan_planner_step_count") or tp_metrics.get("tool_plan_steps_requested"),
        "planner_tools": tp_metrics.get("tool_plan_planner_tools"),
        "worker_steps_requested": tp_metrics.get("tool_plan_worker_steps_requested"),
        "worker_steps_executed": tp_metrics.get("tool_plan_worker_steps_executed"),
        "worker_step_failures": tp_metrics.get("tool_plan_worker_step_failures"),
        "worker_success_rate": tp_metrics.get("tool_plan_worker_success_rate"),
        "worker_missing_record_count": tp_metrics.get("tool_plan_worker_missing_record_count"),
        "worker_duplicate_read_count": tp_metrics.get("tool_plan_worker_duplicate_read_count"),
        "worker_artifact_yield_count": tp_metrics.get("tool_plan_worker_artifact_yield_count"),
        "worker_tool_failure_classes": tp_metrics.get("tool_plan_worker_tool_failure_classes"),
        "solver_grounded": expectations_score["solver_grounded"],
        "solver_evidence_refs_count": expectations_score["solver_evidence_refs_count"],
        "solver_uses_observation_artifacts": expectations_score["solver_uses_observation_artifacts"],
        "solver_unbacked_claim_count": expectations_score["solver_unbacked_claim_count"],
        "solver_refine_verdict": expectations_score["solver_refine_verdict"],
        "solver_mutation_after_evidence": expectations_score["solver_mutation_after_evidence"],
        "solver_evidence_refs": expectations_score["solver_evidence_refs"],
        "solver_artifact_refs": expectations_score["solver_artifact_refs"],
        "expectation_success": expectations_score["expectation_success"],
        "expectation_grounded": expectations_score["expectation_grounded"],
        "expectation_missing_files": expectations_score["expectation_missing_files"],
        "expectation_missing_terms": expectations_score["expectation_missing_terms"],
        "fallback_to_loop_count": tp_metrics.get("tool_plan_fallback_count"),
        "tool_plan_fallback_count": tp_metrics.get("tool_plan_fallback_count"),
        "tool_plan_total_tokens": tp_metrics.get("tool_plan_total_tokens"),
        "tool_plan_steps_requested": tp_metrics.get("tool_plan_steps_requested"),
        "tool_plan_steps_executed": tp_metrics.get("tool_plan_steps_executed"),
        "abort_count": 1 if str(tp_final.get("status") or "").strip().lower() in {"cancelled", "aborted"} else 0,
        "loop_guard_count": _loop_guard_count(tp_final),
        "repeated_read_count": tp_metrics.get("tool_plan_repeated_read_count"),
        "tool_plan_repeated_read_count": tp_metrics.get("tool_plan_repeated_read_count"),
        "planner_repair_loop_count": tp_metrics.get("tool_plan_repair_attempts"),
        "model_stream_halt_count": tp_metrics.get("model_stream_halt_count"),
        "prompt_shape": _prompt_shape_for_comparison(tp_result),
    }
    comparison["latency_delta_sec"] = _delta(_coerce_optional_float(comparison["loop_latency_sec"]), _coerce_optional_float(comparison["tool_plan_latency_sec"]))
    comparison["latency_delta_pct"] = _pct_delta(_coerce_optional_float(comparison["loop_latency_sec"]), _coerce_optional_float(comparison["tool_plan_latency_sec"]))
    comparison["tool_execution_duration_delta_sec"] = _delta(
        _coerce_optional_float(comparison["loop_tool_execution_duration_sec"]),
        _coerce_optional_float(comparison["tool_plan_tool_execution_duration_sec"]),
    )
    comparison["tool_execution_duration_delta_pct"] = _pct_delta(
        _coerce_optional_float(comparison["loop_tool_execution_duration_sec"]),
        _coerce_optional_float(comparison["tool_plan_tool_execution_duration_sec"]),
    )
    return comparison


def _prompt_shape_for_comparison(result: dict[str, Any]) -> dict[str, bool | None] | None:
    prompt_shape = result.get("prompt_shape")
    if not isinstance(prompt_shape, dict):
        return None
    return {key: prompt_shape.get(key) for key in PROMPT_SHAPE_KEYS}


def _extract_evidence_ids(final_json: dict[str, Any] | None, text: str) -> list[str]:
    evidence_ids = []
    if isinstance(final_json, dict):
        for key in ("evidence_ids", "evidence", "citations"):
            value = final_json.get(key)
            if isinstance(value, list):
                evidence_ids.extend(str(item) for item in value if str(item).startswith("TP-E"))
            elif isinstance(value, str) and "TP-E" in value:
                evidence_ids.extend(re.findall(r"TP-E[0-9A-Za-z:-]+", value))
    evidence_ids.extend(re.findall(r"TP-E[0-9A-Za-z:-]+", text))
    return sorted({item for item in evidence_ids if item})


def _extract_artifact_refs(final_json: dict[str, Any] | None, text: str) -> list[str]:
    refs: set[str] = set()
    if isinstance(final_json, dict):
        for key in ("artifact_ids", "artifact_refs", "artifacts"):
            value = final_json.get(key)
            if isinstance(value, list):
                refs.update(str(item) for item in value if str(item).strip())
            elif isinstance(value, str) and value.strip():
                refs.add(value.strip())
    refs.update(re.findall(r"\bartifact(?:_id)?[:=]\s*([A-Za-z0-9_.:-]+)", text, flags=re.IGNORECASE))
    refs.update(re.findall(r"\bA[0-9]{4,}\b", text))
    return sorted(refs)


def _loop_guard_count(final_json: dict[str, Any] | None) -> int | None:
    if not isinstance(final_json, dict):
        return None
    recovery = final_json.get("recovery_metrics")
    if not isinstance(recovery, dict):
        return None
    direct = _coerce_optional_int(recovery.get("loop_guard_count"))
    if direct is not None:
        return direct
    total = 0
    for key, value in recovery.items():
        if "loop_guard" not in str(key):
            continue
        bucket = value
        if isinstance(bucket, dict):
            total += sum(_coerce_optional_int(item) or 0 for item in bucket.values())
        else:
            total += _coerce_optional_int(bucket) or 0
    failure_events = recovery.get("failure_events_by_class")
    if isinstance(failure_events, dict):
        total += sum(
            _coerce_optional_int(count) or 0
            for name, count in failure_events.items()
            if "loop_guard" in str(name)
        )
    return total or None


def _normalize_comparison_values(comparison: dict[str, Any]) -> None:
    for key in ("loop_tokens", "tool_plan_tokens", "token_delta", "token_delta_pct", "duration_delta_sec", "duration_delta_pct", "latency_delta_sec", "latency_delta_pct"):
        if comparison.get(key) is None:
            continue
        try:
            comparison[key] = round(float(comparison[key]), 3)
        except (TypeError, ValueError):
            comparison[key] = None


def _series_stats(values: list[float], *, worst_mode: str = "max") -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "worst": None}
    worst = max(values)
    if worst_mode == "min":
        worst = min(values)
    elif worst_mode == "abs":
        worst = max(values, key=lambda item: abs(item))
    return {
        "mean": _mean(values),
        "median": round(float(statistics.median(values)), 3),
        "worst": round(float(worst), 3),
    }


def _runtime_abort_indicator(result: dict[str, Any]) -> bool:
    final_json = result.get("final_json") if isinstance(result.get("final_json"), dict) else {}
    status = str(final_json.get("status") or "").strip().lower()
    return bool(
        result.get("timed_out")
        or int(result.get("returncode") or 0) not in {0, 127}
        or status in {"cancelled", "aborted", "stopped", "failed"}
    )


def _build_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    repeat_count = max((int(r.get("repeat_total") or 1) for r in results), default=1)
    by_task: dict[str, dict[str, Any]] = {}
    task_lookup: dict[str, dict[str, Any]] = {}
    for r in results:
        tid = str(r.get("task_id") or "unknown")
        repeat_index = int(r.get("repeat_index") or 1)
        group_key = f"{tid}::repeat-{repeat_index}"
        by_task.setdefault(group_key, {"tags": r.get("tags", []), "task": r.get("task", ""), "expectations": r.get("expectations"), "repeat_index": repeat_index, "repeat_total": int(r.get("repeat_total") or 1)})
        by_task[group_key][r["mode"]] = r
        task_lookup[group_key] = {
            "id": tid,
            "run_id": group_key,
            "repeat_index": repeat_index,
            "repeat_total": int(r.get("repeat_total") or 1),
            "tags": r.get("tags", []),
            "task": r.get("task", ""),
            "expectations": r.get("expectations"),
        }

    comparisons: list[dict[str, Any]] = []
    for group_key, data in by_task.items():
        loop_result = data.get("loop")
        tp_result = data.get("tool_plan")
        if loop_result is None or tp_result is None:
            continue
        task_meta = task_lookup[group_key]
        comparison = _build_comparison(
            {"task_id": task_meta["id"], "run_id": group_key, "loop": loop_result, "tool_plan": tp_result},
            task_meta,
        )
        _normalize_comparison_values(comparison)
        comparisons.append(comparison)

    prompt_shape_failures = [
        {
            "task_id": c["task_id"],
            "failed": [
                key for key, value in (c.get("prompt_shape") or {}).items()
                if value is False
            ],
        }
        for c in comparisons
        if any(value is False for value in (c.get("prompt_shape") or {}).values())
    ]
    wrong_path_tasks = [c for c in comparisons if "wrong_path" in c.get("tags", [])]
    wrong_path_fallback_ok = all(
        (c.get("tool_plan_fallback_count") or 0) > 0 or not c.get("tool_plan_success")
        for c in wrong_path_tasks
    )

    loop_tokens = [float(c["loop_tokens"]) for c in comparisons if c.get("loop_tokens") is not None]
    tp_tokens = [float(c["tool_plan_tokens"]) for c in comparisons if c.get("tool_plan_tokens") is not None]
    token_delta_values = [float(c["token_delta"]) for c in comparisons if c.get("token_delta") is not None]
    token_delta_pct_values = [float(c["token_delta_pct"]) for c in comparisons if c.get("token_delta_pct") is not None]
    loop_latencies = [float(c["loop_latency_sec"]) for c in comparisons if c.get("loop_latency_sec") is not None]
    tp_latencies = [float(c["tool_plan_latency_sec"]) for c in comparisons if c.get("tool_plan_latency_sec") is not None]
    latency_delta_values = [float(c["latency_delta_sec"]) for c in comparisons if c.get("latency_delta_sec") is not None]
    latency_delta_pct_values = [float(c["latency_delta_pct"]) for c in comparisons if c.get("latency_delta_pct") is not None]
    loop_durations = [float(c["loop_duration_sec"]) for c in comparisons if c.get("loop_duration_sec") is not None]
    tp_durations = [float(c["tool_plan_duration_sec"]) for c in comparisons if c.get("tool_plan_duration_sec") is not None]
    duration_delta_values = [float(c["duration_delta_sec"]) for c in comparisons if c.get("duration_delta_sec") is not None]

    by_tag: dict[str, dict[str, Any]] = {}
    for tag in sorted({tag for comparison in comparisons for tag in comparison.get("tags", [])}):
        tagged = [comparison for comparison in comparisons if tag in comparison.get("tags", [])]
        tagged_loop_results = [result for result in results if result.get("mode") == "loop" and tag in (result.get("tags") or [])]
        tagged_tp_results = [result for result in results if result.get("mode") == "tool_plan" and tag in (result.get("tags") or [])]
        tag_loop_tokens = [float(r["token_usage"]) for r in tagged_loop_results if r.get("token_usage") is not None]
        tag_tp_tokens = [float(r["token_usage"]) for r in tagged_tp_results if r.get("token_usage") is not None]
        tag_loop_duration = [float(r["duration_sec"]) for r in tagged_loop_results if r.get("duration_sec") is not None]
        tag_tp_duration = [float(r["duration_sec"]) for r in tagged_tp_results if r.get("duration_sec") is not None]
        tag_abort_loop_rate = _mean([1.0 if _runtime_abort_indicator(r) else 0.0 for r in tagged_loop_results]) if tagged_loop_results else None
        tag_abort_tool_plan_rate = _mean([1.0 if _runtime_abort_indicator(r) else 0.0 for r in tagged_tp_results]) if tagged_tp_results else None
        by_tag[tag] = {
            "total_comparisons": len(tagged),
            "repeat_count": max((int(c.get("repeat_index") or 1) for c in tagged), default=1),
            "tool_plan_wins": sum(1 for c in tagged if c["tool_plan_success"] and not c["loop_success"]),
            "loop_wins": sum(1 for c in tagged if c["loop_success"] and not c["tool_plan_success"]),
            "both_pass": sum(1 for c in tagged if c["loop_success"] and c["tool_plan_success"]),
            "both_fail": sum(1 for c in tagged if not c["loop_success"] and not c["tool_plan_success"]),
            "token_delta_total": sum(float(c["token_delta"]) for c in tagged if c.get("token_delta") is not None) if any(c.get("token_delta") is not None for c in tagged) else None,
            "token_delta_pct_mean": _mean([float(c["token_delta_pct"]) for c in tagged if c.get("token_delta_pct") is not None]),
            "latency_delta_total_sec": sum(float(c["latency_delta_sec"]) for c in tagged if c.get("latency_delta_sec") is not None) if any(c.get("latency_delta_sec") is not None for c in tagged) else None,
            "latency_delta_pct_mean": _mean([float(c["latency_delta_pct"]) for c in tagged if c.get("latency_delta_pct") is not None]),
            "loop_duration_sec_mean": _mean(tag_loop_duration),
            "loop_duration_sec_median": _series_stats(tag_loop_duration)["median"],
            "loop_duration_sec_worst": _series_stats(tag_loop_duration)["worst"],
            "tool_plan_duration_sec_mean": _mean(tag_tp_duration),
            "tool_plan_duration_sec_median": _series_stats(tag_tp_duration)["median"],
            "tool_plan_duration_sec_worst": _series_stats(tag_tp_duration)["worst"],
            "loop_tokens_mean": _mean(tag_loop_tokens),
            "loop_tokens_median": _series_stats(tag_loop_tokens)["median"],
            "loop_tokens_worst": _series_stats(tag_loop_tokens)["worst"],
            "tool_plan_tokens_mean": _mean(tag_tp_tokens),
            "tool_plan_tokens_median": _series_stats(tag_tp_tokens)["median"],
            "tool_plan_tokens_worst": _series_stats(tag_tp_tokens)["worst"],
            "planner_valid_rate": _truthy_rate(tagged, "planner_valid"),
            "worker_success_rate_mean": _numeric_rate(tagged, "worker_success_rate"),
            "solver_grounded_rate": _truthy_rate(tagged, "solver_grounded"),
            "abort_loop_rate": _mean([1.0 if _abort_indicator(c) else 0.0 for c in tagged]) if tagged else None,
            "loop_abort_rate": tag_abort_loop_rate,
            "tool_plan_abort_rate": tag_abort_tool_plan_rate,
            "token_savings_gate": _token_savings_gate(tagged) if tag in TOKEN_SAVINGS_GATE_TAGS else None,
        }

    loop_results = [r for r in results if r.get("mode") == "loop"]
    tp_results = [r for r in results if r.get("mode") == "tool_plan"]
    loop_abort_rate = _mean([1.0 if _runtime_abort_indicator(r) else 0.0 for r in loop_results]) if loop_results else None
    tp_abort_rate = _mean([1.0 if _runtime_abort_indicator(r) else 0.0 for r in tp_results]) if tp_results else None

    loop_duration_stats = _series_stats(loop_durations)
    tp_duration_stats = _series_stats(tp_durations)
    token_delta_stats = _series_stats(token_delta_values, worst_mode="abs")
    token_delta_pct_stats = _series_stats(token_delta_pct_values, worst_mode="abs")
    latency_delta_stats = _series_stats(latency_delta_values, worst_mode="abs")
    latency_delta_pct_stats = _series_stats(latency_delta_pct_values, worst_mode="abs")
    duration_delta_stats = _series_stats(duration_delta_values, worst_mode="abs")

    summary = {
        "total_comparisons": len(comparisons),
        "repeat_count": repeat_count,
        "tool_plan_wins": sum(1 for c in comparisons if c["tool_plan_success"] and not c["loop_success"]),
        "loop_wins": sum(1 for c in comparisons if c["loop_success"] and not c["tool_plan_success"]),
        "both_pass": sum(1 for c in comparisons if c["loop_success"] and c["tool_plan_success"]),
        "both_fail": sum(1 for c in comparisons if not c["loop_success"] and not c["tool_plan_success"]),
        "token_delta_total": sum(token_delta_values) if token_delta_values else None,
        "token_delta_pct_mean": token_delta_pct_stats["mean"],
        "token_delta_pct_median": token_delta_pct_stats["median"],
        "token_delta_pct_worst": token_delta_pct_stats["worst"],
        "token_delta_mean": token_delta_stats["mean"],
        "token_delta_median": token_delta_stats["median"],
        "token_delta_worst": token_delta_stats["worst"],
        "latency_delta_total_sec": sum(latency_delta_values) if latency_delta_values else None,
        "latency_delta_pct_mean": latency_delta_pct_stats["mean"],
        "latency_delta_pct_median": latency_delta_pct_stats["median"],
        "latency_delta_pct_worst": latency_delta_pct_stats["worst"],
        "latency_delta_mean": latency_delta_stats["mean"],
        "latency_delta_median": latency_delta_stats["median"],
        "latency_delta_worst": latency_delta_stats["worst"],
        "duration_delta_total_sec": sum(duration_delta_values) if duration_delta_values else None,
        "duration_delta_mean": duration_delta_stats["mean"],
        "duration_delta_median": duration_delta_stats["median"],
        "duration_delta_worst": duration_delta_stats["worst"],
        "planner_valid_rate": _truthy_rate(comparisons, "planner_valid"),
        "worker_success_rate_mean": _numeric_rate(comparisons, "worker_success_rate"),
        "solver_grounded_rate": _truthy_rate(comparisons, "solver_grounded"),
        "abort_loop_rate": tp_abort_rate,
        "loop_abort_rate": loop_abort_rate,
        "tool_plan_abort_rate": tp_abort_rate,
        "loop_tokens_total": sum(loop_tokens) if loop_tokens else None,
        "tool_plan_tokens_total": sum(tp_tokens) if tp_tokens else None,
        "loop_duration_sec_mean": loop_duration_stats["mean"],
        "loop_duration_sec_median": loop_duration_stats["median"],
        "loop_duration_sec_worst": loop_duration_stats["worst"],
        "tool_plan_duration_sec_mean": tp_duration_stats["mean"],
        "tool_plan_duration_sec_median": tp_duration_stats["median"],
        "tool_plan_duration_sec_worst": tp_duration_stats["worst"],
    }

    summary["token_savings"] = summary["token_delta_total"]
    summary["wrong_path_fallback_ok"] = wrong_path_fallback_ok
    summary["wrong_path_tasks"] = len(wrong_path_tasks)
    summary["planner_valid_gate"] = bool(summary["planner_valid_rate"] is not None and summary["planner_valid_rate"] >= 0.9)
    summary["worker_success_gate"] = bool(summary["worker_success_rate_mean"] is not None and summary["worker_success_rate_mean"] >= 0.85)
    summary["solver_grounded_gate"] = bool(summary["solver_grounded_rate"] is not None and summary["solver_grounded_rate"] >= 0.85)
    summary["tool_plan_abort_not_worse"] = bool(
        summary["tool_plan_abort_rate"] is not None
        and summary["loop_abort_rate"] is not None
        and summary["tool_plan_abort_rate"] <= summary["loop_abort_rate"]
    )
    summary["tool_plan_latency_not_worse"] = bool(
        summary["tool_plan_duration_sec_median"] is not None
        and summary["loop_duration_sec_median"] is not None
        and summary["tool_plan_duration_sec_median"] <= summary["loop_duration_sec_median"] * 1.25
    )
    token_gate = _token_savings_gate(comparisons)
    summary["read_heavy_token_savings_gate"] = token_gate
    summary["read_heavy_token_savings_threshold_pct"] = TOKEN_SAVINGS_GATE_PCT
    summary["decision"] = _decision_from_summary(summary)
    summary["decision_reason"] = _decision_reason_from_summary(summary)

    return {
        "summary": summary,
        "by_tag": by_tag,
        "total_comparisons": summary["total_comparisons"],
        "tool_plan_wins": summary["tool_plan_wins"],
        "loop_wins": summary["loop_wins"],
        "both_pass": summary["both_pass"],
        "both_fail": summary["both_fail"],
        "token_savings": summary["token_savings"],
        "wrong_path_fallback_ok": wrong_path_fallback_ok,
        "wrong_path_tasks": len(wrong_path_tasks),
        "prompt_shape_failures": prompt_shape_failures,
        "comparisons": comparisons,
        "decision": summary["decision"],
        "decision_reason": summary["decision_reason"],
    }


def _decision_from_summary(summary: dict[str, Any]) -> str:
    if not summary.get("wrong_path_fallback_ok", True):
        return "pause"
    if not summary.get("planner_valid_gate"):
        return "pause"
    if not summary.get("worker_success_gate"):
        return "narrow rollout"
    if not summary.get("solver_grounded_gate"):
        return "narrow rollout"
    if not summary.get("tool_plan_abort_not_worse"):
        return "narrow rollout"
    if not summary.get("tool_plan_latency_not_worse"):
        return "narrow rollout"
    if summary.get("read_heavy_token_savings_gate") is False:
        return "narrow rollout"
    return "continue"


def _decision_reason_from_summary(summary: dict[str, Any]) -> str:
    parts: list[str] = []
    parts.append(f"planner_valid_rate={summary.get('planner_valid_rate')}")
    parts.append(f"worker_success_rate_mean={summary.get('worker_success_rate_mean')}")
    parts.append(f"solver_grounded_rate={summary.get('solver_grounded_rate')}")
    parts.append(f"tool_plan_abort_rate={summary.get('tool_plan_abort_rate')}")
    parts.append(f"loop_abort_rate={summary.get('loop_abort_rate')}")
    parts.append(f"tool_plan_duration_sec_median={summary.get('tool_plan_duration_sec_median')}")
    parts.append(f"loop_duration_sec_median={summary.get('loop_duration_sec_median')}")
    parts.append(f"read_heavy_token_savings_gate={summary.get('read_heavy_token_savings_gate')}")
    return "; ".join(parts)


def _build_markdown_report(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    comparisons = report.get("comparisons") if isinstance(report.get("comparisons"), list) else []
    by_tag = report.get("by_tag") if isinstance(report.get("by_tag"), dict) else {}
    prompt_shape_failures = report.get("prompt_shape_failures") if isinstance(report.get("prompt_shape_failures"), list) else []

    lines: list[str] = []
    lines.append("# ToolPlan Eval Report")
    lines.append("")
    lines.append(f"Decision: **{summary.get('decision', 'unknown')}**")
    if summary.get("decision_reason"):
        lines.append("")
        lines.append(str(summary["decision_reason"]))
    lines.append("")
    lines.append("## Summary")
    for key in (
        "total_comparisons",
        "repeat_count",
        "tool_plan_wins",
        "loop_wins",
        "both_pass",
        "both_fail",
        "planner_valid_rate",
        "worker_success_rate_mean",
        "solver_grounded_rate",
        "token_delta_total",
        "token_delta_mean",
        "token_delta_median",
        "token_delta_worst",
        "duration_delta_total_sec",
        "duration_delta_mean",
        "duration_delta_median",
        "duration_delta_worst",
        "latency_delta_total_sec",
        "latency_delta_mean",
        "latency_delta_median",
        "latency_delta_worst",
        "tool_plan_abort_rate",
        "loop_abort_rate",
        "wrong_path_tasks",
        "wrong_path_fallback_ok",
        "read_heavy_token_savings_gate",
    ):
        lines.append(f"- `{key}`: {summary.get(key)}")
    lines.append("")
    lines.append("## Gates")
    for key, label in (
        ("planner_valid_gate", "Planner validity >= 90%"),
        ("worker_success_gate", "Worker success >= 85%"),
        ("solver_grounded_gate", "Solver grounded >= 85%"),
        ("tool_plan_abort_not_worse", "ToolPlan abort rate no worse than loop"),
        ("tool_plan_latency_not_worse", "ToolPlan latency within 1.25x loop median"),
        ("read_heavy_token_savings_gate", "Read-heavy token savings >= 20% when measurable"),
        ("wrong_path_fallback_ok", "Wrong-path fallback behavior"),
    ):
        value = summary.get(key)
        status = "N/A" if value is None else ("PASS" if value else "FAIL")
        lines.append(f"- {label}: {status}")
    lines.append("")
    if by_tag:
        lines.append("## By Tag")
        for tag in sorted(by_tag):
            stats = by_tag[tag]
            lines.append(f"### `{tag}`")
            for key in (
                "total_comparisons",
                "repeat_count",
                "tool_plan_wins",
                "loop_wins",
                "planner_valid_rate",
                "worker_success_rate_mean",
                "solver_grounded_rate",
                "token_delta_total",
                "duration_delta_total_sec",
                "tool_plan_abort_rate",
                "loop_abort_rate",
            ):
                lines.append(f"- `{key}`: {stats.get(key)}")
            lines.append("")
    if prompt_shape_failures:
        lines.append("## Prompt Shape Failures")
        for entry in prompt_shape_failures:
            failed = ", ".join(entry.get("failed", [])) if isinstance(entry, dict) else ""
            lines.append(f"- `{entry.get('task_id')}`: {failed}")
        lines.append("")
    if comparisons:
        lines.append("## Top Failures")
        failing = [c for c in comparisons if not c.get("tool_plan_success") or not c.get("expectation_success") or not c.get("solver_grounded")]
        if not failing:
            failing = comparisons[:5]
        for comparison in failing[:5]:
            lines.append(f"- `{comparison.get('task_id')}` repeat={comparison.get('repeat_index')} mode pair")
            lines.append(f"  - task: {comparison.get('task')}")
            lines.append(f"  - loop_success: {comparison.get('loop_success')}")
            lines.append(f"  - tool_plan_success: {comparison.get('tool_plan_success')}")
            lines.append(f"  - expectation_success: {comparison.get('expectation_success')}")
            lines.append(f"  - solver_grounded: {comparison.get('solver_grounded')}")
    return "\n".join(lines).rstrip() + "\n"


def _report_exit_code(report: dict[str, Any]) -> int:
    if not report.get("wrong_path_fallback_ok", True):
        return 1
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    if summary and summary.get("decision") not in (None, "continue"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
