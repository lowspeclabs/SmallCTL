#!/usr/bin/env python3
"""promptdiff — compare prompt-state frame snapshots between runs or steps.

Examples:
  python3 Agent-Tools/promptdiff.py latest-1 latest
  python3 Agent-Tools/promptdiff.py 4b54c65e 2e2d6b5f --step 5
  python3 Agent-Tools/promptdiff.py latest-1 latest --lane turn_bundles
  python3 Agent-Tools/promptdiff.py latest --step 3 --step 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agent_tools_lib import (
    Colors,
    colorize,
    iter_records,
    resolve_run_dir,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare prompt-state frames between SmallCTL runs or steps.")
    parser.add_argument("left", help="Left run dir, run id, or 'latest-N'")
    parser.add_argument("right", nargs="?", help="Right run dir, run id, or 'latest-N' (default: same as left)")
    parser.add_argument("--logs-dir", help="Custom logs directory")
    parser.add_argument("--step", type=int, action="append", help="Step number(s) to compare; repeat for left/right")
    parser.add_argument("--lane", help="Only compare this lane (e.g. turn_bundles, artifact_snippets)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    return parser.parse_args()


def _frame_records(run_dir: Path) -> list[dict[str, Any]]:
    return [r for r in iter_records(run_dir, "harness") if r.get("event") == "prompt_state_frame_compiled"]


def _pick_frame(records: list[dict[str, Any]], step: int | None, task_id: str | None = None) -> dict[str, Any] | None:
    candidates = records
    if task_id:
        candidates = [r for r in candidates if f":{task_id}:" in (r.get("trace_id") or "")]
    if step is None:
        return candidates[-1] if candidates else None
    for rec in candidates:
        data = rec.get("data") or {}
        if data.get("step") == step:
            return rec
        tid = rec.get("trace_id") or ""
        if f":step-{step}:" in tid:
            return rec
    return None


def _lane_counts(frame: dict[str, Any]) -> dict[str, int]:
    data = frame.get("data") or {}
    return dict(data.get("included_lane_counts") or {})


def _selected_ids(frame: dict[str, Any], lane: str | None = None) -> dict[str, list[str]]:
    data = frame.get("data") or {}
    keys = {
        "selected_artifact_ids",
        "selected_experience_ids",
        "selected_turn_bundle_ids",
        "selected_brief_ids",
        "selected_summary_ids",
    }
    result: dict[str, list[str]] = {}
    for key in keys:
        values = data.get(key) or []
        name = key.replace("selected_", "").replace("_ids", "")
        if lane and name != lane:
            continue
        result[name] = [str(v) for v in values]
    return result


def _frame_summary(frame: dict[str, Any]) -> dict[str, Any]:
    data = frame.get("data") or {}
    return {
        "timestamp": frame.get("timestamp"),
        "trace_id": frame.get("trace_id"),
        "active_phase": data.get("active_phase"),
        "active_intent": data.get("active_intent"),
        "coding_profile_enabled": data.get("coding_profile_enabled"),
        "lane_counts": _lane_counts(frame),
        "selected_ids": _selected_ids(frame),
    }


def _diff_lane_counts(left: dict[str, int], right: dict[str, int]) -> dict[str, tuple[int, int]]:
    lanes = set(left) | set(right)
    return {lane: (left.get(lane, 0), right.get(lane, 0)) for lane in lanes if left.get(lane, 0) != right.get(lane, 0)}


def _diff_selected(left: dict[str, list[str]], right: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for lane in set(left) | set(right):
        lset = set(left.get(lane, []))
        rset = set(right.get(lane, []))
        if lset != rset:
            result[lane] = {
                "left_only": sorted(lset - rset),
                "right_only": sorted(rset - lset),
                "common": sorted(lset & rset),
            }
    return result


def _render_text(
    left_dir: Path,
    right_dir: Path,
    left_frame: dict[str, Any],
    right_frame: dict[str, Any],
    lane_filter: str | None,
) -> str:
    left_summary = _frame_summary(left_frame)
    right_summary = _frame_summary(right_frame)
    lines: list[str] = []
    lines.append(colorize("Prompt frame diff", Colors.BOLD + Colors.CYAN))
    lines.append(f"  left:  {left_dir.name}  {left_summary['timestamp']}  {left_summary['trace_id']}")
    lines.append(f"  right: {right_dir.name}  {right_summary['timestamp']}  {right_summary['trace_id']}")

    lines.append("")
    lines.append(colorize("Frame context", Colors.BOLD + Colors.BLUE))
    if left_summary["active_phase"] != right_summary["active_phase"]:
        lines.append(f"  active_phase: {left_summary['active_phase']} -> {right_summary['active_phase']}")
    if left_summary["active_intent"] != right_summary["active_intent"]:
        lines.append(f"  active_intent: {left_summary['active_intent']} -> {right_summary['active_intent']}")
    if left_summary["coding_profile_enabled"] != right_summary["coding_profile_enabled"]:
        lines.append(f"  coding_profile_enabled: {left_summary['coding_profile_enabled']} -> {right_summary['coding_profile_enabled']}")

    lane_diff = _diff_lane_counts(left_summary["lane_counts"], right_summary["lane_counts"])
    if lane_filter:
        lane_diff = {k: v for k, v in lane_diff.items() if k == lane_filter}
    lines.append("")
    lines.append(colorize("Lane count deltas", Colors.BOLD + Colors.BLUE))
    if lane_diff:
        for lane, (lc, rc) in sorted(lane_diff.items()):
            sign = "+" if rc - lc >= 0 else ""
            lines.append(f"  {lane}: {lc} -> {rc} ({sign}{rc - lc})")
    else:
        lines.append("  no lane count differences")

    selected_diff = _diff_selected(left_summary["selected_ids"], right_summary["selected_ids"])
    if lane_filter:
        selected_diff = {k: v for k, v in selected_diff.items() if k == lane_filter}
    lines.append("")
    lines.append(colorize("Selected item deltas", Colors.BOLD + Colors.BLUE))
    if selected_diff:
        for lane, delta in sorted(selected_diff.items()):
            lines.append(f"  {lane}:")
            if delta["left_only"]:
                lines.append(f"    -> left only ({len(delta['left_only'])}): {', '.join(delta['left_only'][:5])}{'...' if len(delta['left_only']) > 5 else ''}")
            if delta["right_only"]:
                lines.append(f"    -> right only ({len(delta['right_only'])}): {', '.join(delta['right_only'][:5])}{'...' if len(delta['right_only']) > 5 else ''}")
    else:
        lines.append("  no selected item differences")

    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    logs_dir = Path(args.logs_dir) if args.logs_dir else None
    try:
        left_dir = resolve_run_dir(args.left, logs_dir)
        right_dir = resolve_run_dir(args.right or args.left, logs_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(colorize(str(exc), Colors.RED), file=sys.stderr)
        return 1

    left_frames = _frame_records(left_dir)
    right_frames = _frame_records(right_dir)
    if not left_frames or not right_frames:
        print(colorize("No prompt_state_frame_compiled records found in one or both runs.", Colors.RED), file=sys.stderr)
        return 1

    steps = args.step or []
    left_step = steps[0] if len(steps) >= 1 else None
    right_step = steps[1] if len(steps) >= 2 else (steps[0] if len(steps) == 1 else None)

    left_frame = _pick_frame(left_frames, left_step)
    right_frame = _pick_frame(right_frames, right_step)
    if left_frame is None or right_frame is None:
        print(colorize("Could not find matching prompt-state frame for the requested step(s).", Colors.RED), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({
            "left": _frame_summary(left_frame),
            "right": _frame_summary(right_frame),
            "lane_count_deltas": _diff_lane_counts(_lane_counts(left_frame), _lane_counts(right_frame)),
            "selected_deltas": _diff_selected(_selected_ids(left_frame), _selected_ids(right_frame)),
        }, indent=2, default=str))
        return 0

    print(_render_text(left_dir, right_dir, left_frame, right_frame, args.lane))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
