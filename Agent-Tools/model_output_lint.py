#!/usr/bin/env python3
"""model_output_lint — scan model_output.jsonl for malformed tool calls, repetition, and
thinking-text leakage.

Examples:
  python3 Agent-Tools/model_output_lint.py latest
  python3 Agent-Tools/model_output_lint.py 4b54c65e --repeated-phrases
  python3 Agent-Tools/model_output_lint.py latest --json
  python3 Agent-Tools/model_output_lint.py latest --summary
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from agent_tools_lib import (
    Colors,
    colorize,
    extract_trace_id,
    iter_records,
    resolve_run_dir,
)


TOOL_CALL_PATTERN = re.compile(
    r"\b(file_read|file_write|file_patch|shell_exec|ssh_exec|"
    + r"ssh_file_read|ssh_file_write|ssh_file_patch|web_fetch|web_search|"
    + r"task_complete|task_fail|dir_list|artifact_read|artifact_print)\s*\("
)

THINKING_TAGS_PATTERN = re.compile(r"(?:</?think(?:ing)?>|\[thinking\]|\[assistant\])")

# Gemma/Qwen-style control tokens that leak into the assistant channel.
CONTROL_TOKEN_PATTERN = re.compile(r"^\s*\|>[a-z_]+<\|\s*$", re.IGNORECASE)

# Tool-call wrappers that appear in reasoning/assistant text but were not
# dispatched as proper native/JSON tool calls.
TOOL_CALL_WRAPPER_PATTERN = re.compile(
    r"<[/|]?\s*tool_call\b|<\|tool_call\||</tool_call\|>"
    r"|<function\s*=|<call\b",
    re.IGNORECASE,
)


def _has_recorded_native_tool_calls(data: dict[str, Any]) -> bool:
    """Return True when the model_output record itself carries native tool_calls."""
    for key in ("tool_calls", "function_calls"):
        value = data.get(key)
        if isinstance(value, list) and value:
            return True
    return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lint model_output.jsonl for common issues."
    )
    parser.add_argument(
        "run",
        nargs="?",
        default="latest",
        help="Run dir, run id, 'latest', or 'latest-N'",
    )
    parser.add_argument("--logs-dir", help="Custom logs directory")
    parser.add_argument(
        "--repeated-phrases",
        action="store_true",
        help="Highlight repeated-phrase degeneration",
    )
    parser.add_argument(
        "--summary", action="store_true", help="Summary only (default if no flags)"
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    return parser.parse_args()


def _lint(run_dir: Path) -> dict[str, Any]:
    records = list(iter_records(run_dir, "model_output"))
    tools_records = list(iter_records(run_dir, "tools"))

    # A model_output record may carry native tool_calls, or the actual calls may
    # be recorded in the tools channel under the same trace_id.
    native_tool_traces: set[str] = set()
    for rec in tools_records:
        if rec.get("event") in {"dispatch_start", "dispatch_complete"}:
            tid = extract_trace_id(rec)
            if tid:
                native_tool_traces.add(tid)

    results: dict[str, Any] = {
        "total_records": len(records),
        "degenerate_loops": [],
        "thinking_tags_in_assistant": [],
        "thinking_tags_in_thinking": [],
        "missing_tool_calls": [],
        "tool_call_syntax_suspected": [],
        "control_token_fragments": [],
        "reasoning_tool_call_wrappers": [],
        "empty_outputs": [],
        "repeated_phrases": [],
    }

    for rec in records:
        event = rec.get("event", "")
        data = rec.get("data") or {}
        tid = extract_trace_id(rec) or ""
        text = str(data.get("assistant_text") or "")
        thinking_text = str(data.get("thinking_text") or "")

        has_native_tool_call = _has_recorded_native_tool_calls(data) or tid in native_tool_traces

        if (
            event == "model_output_degenerate_loop"
            or event == "model_output_degenerate_loop_exhausted"
        ):
            phrase = data.get("repeated_phrase") or ""
            results["degenerate_loops"].append(
                {
                    "timestamp": rec.get("timestamp"),
                    "trace_id": tid,
                    "repeated_phrase": phrase,
                    "repeat_count": data.get("repeat_count"),
                }
            )

        if THINKING_TAGS_PATTERN.search(text):
            results["thinking_tags_in_assistant"].append(
                {
                    "timestamp": rec.get("timestamp"),
                    "trace_id": tid,
                    "length": len(text),
                }
            )

        if THINKING_TAGS_PATTERN.search(thinking_text):
            results["thinking_tags_in_thinking"].append(
                {
                    "timestamp": rec.get("timestamp"),
                    "trace_id": tid,
                    "length": len(thinking_text),
                }
            )

        if text.strip() and not TOOL_CALL_PATTERN.search(text):
            if (
                len(text) > 20
                and not event.startswith("model_token")
                and event != "model_thinking"
                and not has_native_tool_call
            ):
                results["missing_tool_calls"].append(
                    {
                        "timestamp": rec.get("timestamp"),
                        "trace_id": tid,
                        "text_preview": text[:120],
                    }
                )

        if text.strip() and TOOL_CALL_PATTERN.search(text):
            results["tool_call_syntax_suspected"].append(
                {
                    "timestamp": rec.get("timestamp"),
                    "trace_id": tid,
                    "text_preview": text[:200],
                }
            )

        if CONTROL_TOKEN_PATTERN.match(text):
            results["control_token_fragments"].append(
                {
                    "timestamp": rec.get("timestamp"),
                    "trace_id": tid,
                    "text_preview": text[:80],
                }
            )

        combined_text = f"{text}\n{thinking_text}"
        if TOOL_CALL_WRAPPER_PATTERN.search(combined_text) and not has_native_tool_call:
            results["reasoning_tool_call_wrappers"].append(
                {
                    "timestamp": rec.get("timestamp"),
                    "trace_id": tid,
                    "text_preview": thinking_text[:200] or text[:200],
                }
            )

        if not text.strip() and event == "model_output":
            results["empty_outputs"].append(
                {
                    "timestamp": rec.get("timestamp"),
                    "trace_id": tid,
                }
            )

    # Repeated-phrase detection across consecutive assistant texts
    prev_text = ""
    for rec in records:
        text = str(rec.get("data", {}).get("assistant_text") or "").strip()
        if text and text == prev_text and len(text) > 20:
            results["repeated_phrases"].append(
                {
                    "timestamp": rec.get("timestamp"),
                    "trace_id": extract_trace_id(rec) or "",
                    "text_preview": text[:120],
                }
            )
        if text:
            prev_text = text

    return results


def _render_text(
    run_dir: Path, results: dict[str, Any], show_repeated: bool, show_summary: bool
) -> str:
    lines: list[str] = []
    lines.append(
        colorize(f"Model output lint for {run_dir.name}", Colors.BOLD + Colors.CYAN)
    )
    lines.append(f"Run directory: {run_dir}")

    total = results["total_records"]
    lines.append("")
    lines.append(colorize("Summary", Colors.BOLD + Colors.BLUE))
    lines.append(f"  total records:                    {total}")
    lines.append(
        f"  degenerate loops:                 {len(results['degenerate_loops'])}"
    )
    lines.append(
        f"  thinking tags in assistant text:  {len(results['thinking_tags_in_assistant'])}"
    )
    lines.append(
        f"  thinking tags in thinking text:   {len(results['thinking_tags_in_thinking'])}"
    )
    lines.append(
        f"  suspicious (no tool call syntax): {len(results['missing_tool_calls'])}"
    )
    lines.append(
        f"  tool call syntax detected:        {len(results['tool_call_syntax_suspected'])}"
    )
    lines.append(
        f"  control token fragments:          {len(results['control_token_fragments'])}"
    )
    lines.append(
        f"  reasoning tool-call wrappers:     {len(results['reasoning_tool_call_wrappers'])}"
    )
    lines.append(f"  empty outputs:                    {len(results['empty_outputs'])}")

    if results["degenerate_loops"]:
        lines.append("")
        lines.append(colorize("Degenerate loop events", Colors.BOLD + Colors.RED))
        for d in results["degenerate_loops"][:10]:
            lines.append(
                f"  {d['timestamp']}  phrase={d['repeated_phrase'][:80]}  count={d['repeat_count']}  [{d['trace_id']}]"
            )
        if len(results["degenerate_loops"]) > 10:
            lines.append(f"  ... and {len(results['degenerate_loops']) - 10} more")

    if results["thinking_tags_in_assistant"]:
        lines.append("")
        lines.append(
            colorize(
                "Thinking tags leaked into assistant text", Colors.BOLD + Colors.YELLOW
            )
        )
        for t in results["thinking_tags_in_assistant"][:5]:
            lines.append(f"  {t['timestamp']}  [{t['trace_id']}]")

    if results["thinking_tags_in_thinking"]:
        lines.append("")
        lines.append(
            colorize(
                "Thinking tags leaked into thinking text", Colors.BOLD + Colors.YELLOW
            )
        )
        for t in results["thinking_tags_in_thinking"][:5]:
            lines.append(f"  {t['timestamp']}  [{t['trace_id']}]")

    if results["control_token_fragments"]:
        lines.append("")
        lines.append(
            colorize("Control-token fragments in assistant text", Colors.BOLD + Colors.YELLOW)
        )
        for c in results["control_token_fragments"][:5]:
            lines.append(f"  {c['timestamp']}  {c['text_preview'][:60]}  [{c['trace_id']}]")

    if results["reasoning_tool_call_wrappers"]:
        lines.append("")
        lines.append(
            colorize(
                "Tool-call wrappers in reasoning/assistant text (no dispatch)",
                Colors.BOLD + Colors.YELLOW,
            )
        )
        for r in results["reasoning_tool_call_wrappers"][:5]:
            lines.append(f"  {r['timestamp']}  {r['text_preview'][:80]}  [{r['trace_id']}]")

    if show_repeated and results["repeated_phrases"]:
        lines.append("")
        lines.append(
            colorize("Repeated identical assistant texts", Colors.BOLD + Colors.YELLOW)
        )
        for r in results["repeated_phrases"][:10]:
            lines.append(
                f"  {r['timestamp']}  {r['text_preview'][:80]}  [{r['trace_id']}]"
            )

    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    logs_dir = Path(args.logs_dir) if args.logs_dir else None
    try:
        run_dir = resolve_run_dir(args.run, logs_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(colorize(str(exc), Colors.RED), file=sys.stderr)
        return 1

    results = _lint(run_dir)
    show_summary = args.summary or not args.repeated_phrases

    if args.json:
        print(
            json.dumps(
                {"run_dir": str(run_dir), "run_name": run_dir.name, **results},
                indent=2,
                default=str,
            )
        )
        return 0

    print(
        _render_text(
            run_dir,
            results,
            show_repeated=args.repeated_phrases,
            show_summary=show_summary,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
