from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


MODES = ("loop", "tool_plan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run loop vs ToolPlan comparison tasks.")
    parser.add_argument("--tasks", default="evals/tool_plan/tasks.jsonl", help="JSONL task file.")
    parser.add_argument("--output", default=".smallctl/artifacts/tool_plan_eval_results.jsonl", help="JSONL output path.")
    parser.add_argument("--mode", choices=[*MODES, "both"], default="both", help="Runtime mode to run.")
    parser.add_argument("--timeout-sec", type=int, default=900, help="Per-task timeout.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args(argv)

    tasks = _load_tasks(Path(args.tasks))
    modes = list(MODES if args.mode == "both" else (args.mode,))
    commands = [_command_for(task["task"], mode=mode) for task in tasks for mode in modes]
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for task in tasks:
            for mode in modes:
                result = _run_task(task, mode=mode, timeout_sec=max(1, int(args.timeout_sec)))
                handle.write(json.dumps(result, ensure_ascii=True, sort_keys=True) + "\n")
                handle.flush()
                print(json.dumps(_summary(result), ensure_ascii=True, sort_keys=True))
    return 0


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        task = str(payload.get("task") or "").strip()
        if not task:
            raise SystemExit(f"{path}:{line_number}: missing task")
        payload["task"] = task
        payload.setdefault("id", f"task-{line_number}")
        tasks.append(payload)
    if not tasks:
        raise SystemExit(f"No tasks found in {path}")
    return tasks


def _command_for(task: str, *, mode: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "smallctl.main",
        "--run-mode",
        mode,
        "--fresh-run",
        "--task",
        task,
    ]


def _run_task(task: dict[str, Any], *, mode: str, timeout_sec: int) -> dict[str, Any]:
    command = _command_for(str(task["task"]), mode=mode)
    env = dict(os.environ)
    repo_src = str((Path(__file__).resolve().parents[1] / "src").resolve())
    env["PYTHONPATH"] = repo_src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
        )
        timed_out = False
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        returncode = 124
    duration = round(time.perf_counter() - started, 3)
    return {
        "task_id": task.get("id"),
        "tags": task.get("tags", []),
        "mode": mode,
        "task": task["task"],
        "duration_sec": duration,
        "returncode": returncode,
        "timed_out": timed_out,
        "final_json": _last_json_object(stdout),
        "stdout": stdout[-12000:],
        "stderr": stderr[-12000:],
    }


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
    return {
        "task_id": result.get("task_id"),
        "mode": result.get("mode"),
        "status": status,
        "returncode": result.get("returncode"),
        "duration_sec": result.get("duration_sec"),
        "timed_out": result.get("timed_out"),
    }


if __name__ == "__main__":
    raise SystemExit(main())

