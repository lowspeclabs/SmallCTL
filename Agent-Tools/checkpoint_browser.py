#!/usr/bin/env python3
"""checkpoint_browser — list and compare harness checkpoints.

Checkpoints are JSONL records with event=checkpoint or stored in
.smallctl-langgraph-checkpoints.json / .smallctl-checkpoint.json.

Examples:
  python3 Agent-Tools/checkpoint_browser.py latest
  python3 Agent-Tools/checkpoint_browser.py latest --diff step-3 step-7
  python3 Agent-Tools/checkpoint_browser.py latest --list
  python3 Agent-Tools/checkpoint_browser.py latest --json
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

CHECKPOINT_FILE_NAMES = (
    ".smallctl-langgraph-checkpoints.json",
    ".smallctl-checkpoint.json",
)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse harness checkpoints for a SmallCTL run.")
    parser.add_argument("run", nargs="?", default="latest", help="Run dir, run id, 'latest', or 'latest-N'")
    parser.add_argument("--logs-dir", help="Custom logs directory")
    parser.add_argument("--list", action="store_true", help="List available checkpoints (default)")
    parser.add_argument("--diff", nargs=2, metavar=("A", "B"), help="Compare two checkpoints by key or index")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    return parser.parse_args()


def _find_checkpoint_file(run_dir: Path) -> Path | None:
    for name in CHECKPOINT_FILE_NAMES:
        path = run_dir / name
        if path.exists():
            return path
    parent = run_dir.parent
    for name in CHECKPOINT_FILE_NAMES:
        path = parent / name
        if path.exists():
            return path
    return None


def _load_checkpoints(run_dir: Path) -> list[dict[str, Any]]:
    ckpt_file = _find_checkpoint_file(run_dir)
    if ckpt_file:
        try:
            data = json.loads(ckpt_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return [{"key": "root", **data}]
            if isinstance(data, list):
                return [{"index": i, **item} for i, item in enumerate(data)]
        except (OSError, json.JSONDecodeError):
            pass
    # Fallback: scan harness records
    return [r for r in iter_records(run_dir, "harness") if r.get("event") == "checkpoint"]


def _checkpoint_key(ckpt: dict[str, Any]) -> str:
    return str(ckpt.get("key") or ckpt.get("configurable", {}).get("checkpoint_id") or ckpt.get("checkpoint_id") or ckpt.get("index", "?"))


def _render_summary(ckpt: dict[str, Any], index: int, verbose: bool = False) -> str:
    key = _checkpoint_key(ckpt)
    ts = ckpt.get("timestamp") or ckpt.get("created_at") or ""
    data = ckpt.get("data") or ckpt.get("state") or ckpt.get("checkpoint") or {}

    parts = [f"#{index}  key={key}"]
    if ts:
        parts.append(f"  ts={ts}")
    if isinstance(data, dict):
        phase = data.get("current_phase") or data.get("active_phase", "")
        step = data.get("step") or ""
        mode = data.get("run_mode") or ""
        if phase:
            parts.append(f"  phase={phase}")
        if step:
            parts.append(f"  step={step}")
        if mode:
            parts.append(f"  mode={mode}")

    return "".join(parts)


def _diff_checkpoints(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_data = left.get("data") or left.get("state") or left.get("checkpoint") or {}
    right_data = right.get("data") or right.get("state") or right.get("checkpoint") or {}

    def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
        result: dict[str, Any] = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(_flatten(v, key))
            else:
                result[key] = v
        return result

    flat_left = _flatten(left_data)
    flat_right = _flatten(right_data)

    all_keys = set(flat_left) | set(flat_right)
    diffs: dict[str, Any] = {"changed": {}, "left_only": {}, "right_only": {}}
    for key in sorted(all_keys):
        lv = flat_left.get(key)
        rv = flat_right.get(key)
        if key in flat_left and key not in flat_right:
            diffs["left_only"][key] = lv
        elif key not in flat_left and key in flat_right:
            diffs["right_only"][key] = rv
        elif lv != rv:
            diffs["changed"][key] = (lv, rv)

    return diffs


def _render_text(
    run_dir: Path,
    checkpoints: list[dict[str, Any]],
    diff_pair: tuple[str, str] | None,
) -> str:
    lines: list[str] = []
    lines.append(colorize(f"Checkpoints for {run_dir.name}", Colors.BOLD + Colors.CYAN))
    lines.append(f"Run directory: {run_dir}")

    if not checkpoints:
        lines.append("")
        lines.append(colorize("No checkpoints found.", Colors.YELLOW))
        return "\n".join(lines)

    if diff_pair:
        indices: list[int] = []
        for spec in diff_pair:
            if spec.startswith("step-"):
                matches = [i for i, c in enumerate(checkpoints) if f":{spec}" in str(c)]
                indices.append(matches[0] if matches else int(spec.split("-")[1]))
            else:
                indices.append(int(spec))
        left = checkpoints[indices[0]]
        right = checkpoints[indices[1]]
        diffs = _diff_checkpoints(left, right)
        lines.append("")
        lines.append(colorize(f"Diff #{indices[0]} vs #{indices[1]}", Colors.BOLD + Colors.BLUE))
        if diffs["changed"]:
            lines.append("  Changed:")
            for key, (lv, rv) in list(diffs["changed"].items())[:30]:
                lines.append(f"    {key}: {str(lv)[:80]} -> {str(rv)[:80]}")
            if len(diffs["changed"]) > 30:
                lines.append(f"    ... and {len(diffs['changed']) - 30} more")
        if diffs["left_only"]:
            lines.append(f"  Left only ({len(diffs['left_only'])}): {', '.join(list(diffs['left_only'])[:10])}")
        if diffs["right_only"]:
            lines.append(f"  Right only ({len(diffs['right_only'])}): {', '.join(list(diffs['right_only'])[:10])}")
        return "\n".join(lines)

    lines.append("")
    lines.append(colorize(f"{len(checkpoints)} checkpoint(s)", Colors.BOLD + Colors.BLUE))
    for i, ckpt in enumerate(checkpoints):
        lines.append("  " + _render_summary(ckpt, i))

    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    logs_dir = Path(args.logs_dir) if args.logs_dir else None
    try:
        run_dir = resolve_run_dir(args.run, logs_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(colorize(str(exc), Colors.RED), file=sys.stderr)
        return 1

    checkpoints = _load_checkpoints(run_dir)

    if args.json:
        print(json.dumps({
            "run_dir": str(run_dir),
            "checkpoints": [{
                "key": _checkpoint_key(c),
                "timestamp": c.get("timestamp") or c.get("created_at"),
                "data": c.get("data") or c.get("state") or c.get("checkpoint"),
            } for c in checkpoints],
        }, indent=2, default=str))
        return 0

    print(_render_text(run_dir, checkpoints, diff_pair=args.diff))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
