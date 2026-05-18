from __future__ import annotations

import argparse
import json
import os
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
)

PROMPT_SHAPE_KEYS = (
    "planner_has_rewoo_plan_state",
    "planner_excludes_tool_observations",
    "solver_has_rewoo_evidence",
    "solver_excludes_generic_warm_summaries",
    "solver_has_tool_plan_evidence_ids",
)

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
    parser.add_argument("--mode", choices=[*MODES, "both"], default="both", help="Runtime mode to run.")
    parser.add_argument("--timeout-sec", type=int, default=900, help="Per-task timeout.")
    parser.add_argument(
        "--rewoo-frames",
        action="store_true",
        help="Enable ReWOO planner/solver/refiner lane frames for tool_plan eval runs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args(argv)

    tasks = _load_tasks(Path(args.tasks))
    modes = list(MODES if args.mode == "both" else (args.mode,))
    commands = [_command_for(task["task"], mode=mode, rewoo_frames=args.rewoo_frames) for task in tasks for mode in modes]
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
    for task in tasks:
        for mode in modes:
            run_index += 1
            _log_progress(
                f"starting {run_index}/{len(commands)} task={task.get('id')} mode={mode}"
            )
            result = _run_task(
                task,
                mode=mode,
                timeout_sec=max(1, int(args.timeout_sec)),
                rewoo_frames=args.rewoo_frames,
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
        print(json.dumps(report, ensure_ascii=True, sort_keys=True))
        if report.get("prompt_shape_failures"):
            _log_progress(f"prompt shape failures: {json.dumps(report['prompt_shape_failures'], sort_keys=True)}")
        _log_progress(f"wrote report to {report_path}")
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


def _run_task(task: dict[str, Any], *, mode: str, timeout_sec: int, rewoo_frames: bool = False) -> dict[str, Any]:
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
        "task": task["task"],
        "duration_sec": duration,
        "returncode": returncode,
        "timed_out": timed_out,
        "final_json": final_json,
        "final_success": success,
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
            metrics[key] = recovery.get(key, 0)
    return metrics


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


def _last_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    last: dict[str, Any] | None = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            last = value
    return last


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    final_json = result.get("final_json")
    status = final_json.get("status") if isinstance(final_json, dict) else None
    summary: dict[str, Any] = {
        "task_id": result.get("task_id"),
        "mode": result.get("mode"),
        "status": status,
        "returncode": result.get("returncode"),
        "duration_sec": result.get("duration_sec"),
        "timed_out": result.get("timed_out"),
        "final_success": result.get("final_success"),
    }
    for key in RECOVERY_METRIC_KEYS:
        if key in result:
            summary[key] = result[key]
    return summary


def _build_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = {}
    for r in results:
        tid = str(r.get("task_id") or "unknown")
        by_task.setdefault(tid, {"tags": r.get("tags", []), "task": r.get("task", "")})
        by_task[tid][r["mode"]] = r

    comparisons: list[dict[str, Any]] = []
    for tid, data in by_task.items():
        loop_result = data.get("loop")
        tp_result = data.get("tool_plan")
        if loop_result is None or tp_result is None:
            continue
        comparison: dict[str, Any] = {
            "task_id": tid,
            "tags": data["tags"],
            "task": data["task"],
            "loop_duration_sec": loop_result.get("duration_sec"),
            "tool_plan_duration_sec": tp_result.get("duration_sec"),
            "loop_success": loop_result.get("final_success"),
            "tool_plan_success": tp_result.get("final_success"),
            "loop_tokens": loop_result.get("tool_plan_total_tokens", 0),
            "tool_plan_tokens": tp_result.get("tool_plan_total_tokens", 0),
        }
        for key in RECOVERY_METRIC_KEYS:
            comparison[key] = tp_result.get(key, 0)
        prompt_shape = tp_result.get("prompt_shape")
        if isinstance(prompt_shape, dict):
            comparison["prompt_shape"] = {
                key: prompt_shape.get(key)
                for key in PROMPT_SHAPE_KEYS
            }
        comparisons.append(comparison)

    total = len(comparisons)
    tp_wins = sum(1 for c in comparisons if c["tool_plan_success"] and not c["loop_success"])
    loop_wins = sum(1 for c in comparisons if c["loop_success"] and not c["tool_plan_success"])
    both_pass = sum(1 for c in comparisons if c["loop_success"] and c["tool_plan_success"])
    both_fail = sum(1 for c in comparisons if not c["loop_success"] and not c["tool_plan_success"])
    token_savings = sum(c["loop_tokens"] - c["tool_plan_tokens"] for c in comparisons)
    wrong_path_tasks = [c for c in comparisons if "wrong_path" in c.get("tags", [])]
    wrong_path_fallback_ok = all(
        c["tool_plan_fallback_count"] > 0 or not c["tool_plan_success"]
        for c in wrong_path_tasks
    )
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

    return {
        "total_comparisons": total,
        "tool_plan_wins": tp_wins,
        "loop_wins": loop_wins,
        "both_pass": both_pass,
        "both_fail": both_fail,
        "token_savings": token_savings,
        "wrong_path_fallback_ok": wrong_path_fallback_ok,
        "wrong_path_tasks": len(wrong_path_tasks),
        "prompt_shape_failures": prompt_shape_failures,
        "comparisons": comparisons,
    }


def _report_exit_code(report: dict[str, Any]) -> int:
    if not report.get("wrong_path_fallback_ok", True):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
