#!/usr/bin/env python3
"""fama_inspector — inspect FAMA signals, mitigations, and exposure for a run.

Examples:
  python3 Agent-Tools/fama_inspector.py latest
  python3 Agent-Tools/fama_inspector.py 4b54c65e --signals
  python3 Agent-Tools/fama_inspector.py latest --mitigations --json
  python3 Agent-Tools/fama_inspector.py latest --exposure
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
    extract_trace_id,
    iter_records,
    resolve_run_dir,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect FAMA activity for a SmallCTL run.")
    parser.add_argument("run", nargs="?", default="latest", help="Run dir, run id, 'latest', or 'latest-N'")
    parser.add_argument("--logs-dir", help="Custom logs directory")
    parser.add_argument("--signals", action="store_true", help="Show FAMA signal details")
    parser.add_argument("--mitigations", action="store_true", help="Show active mitigation details")
    parser.add_argument("--exposure", action="store_true", help="Show FAMA tool-exposure decisions")
    parser.add_argument("--summary", action="store_true", help="Show summary counts (default if no section flags)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    return parser.parse_args()


def _fama_records(run_dir: Path) -> list[dict[str, Any]]:
    fama_events = {
        "fama_signal_detected",
        "fama_signal_suppressed",
        "fama_signal_to_mitigation",
        "fama_mitigation_activated",
        "fama_mitigation_expired",
        "fama_mitigation_ttl",
        "fama_capsule_rendered",
        "fama_tool_exposure_applied",
        "fama_tool_call_blocked",
    }
    return [r for r in iter_records(run_dir, "harness") if r.get("event") in fama_events]


def _extract_signals(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for rec in records:
        if rec.get("event") != "fama_signal_detected":
            continue
        data = rec.get("data") or {}
        signals.append({
            "trace_id": extract_trace_id(rec),
            "timestamp": rec.get("timestamp"),
            "kind": data.get("kind"),
            "severity": data.get("severity"),
            "source": data.get("source"),
            "tool_name": data.get("tool_name"),
            "failure_class": data.get("failure_class"),
            "step": data.get("step"),
        })
    return signals


def _extract_mitigations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mitigations: list[dict[str, Any]] = []
    for rec in records:
        event = rec.get("event")
        if event not in {"fama_mitigation_activated", "fama_mitigation_expired", "fama_mitigation_ttl"}:
            continue
        data = rec.get("data") or {}
        mitigations.append({
            "trace_id": extract_trace_id(rec),
            "timestamp": rec.get("timestamp"),
            "event": event,
            "name": data.get("mitigation") or data.get("name"),
            "reason": data.get("reason"),
            "expires_after_step": data.get("expires_after_step"),
        })
    return mitigations


def _extract_exposure(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exposure: list[dict[str, Any]] = []
    for rec in records:
        if rec.get("event") != "fama_tool_exposure_applied":
            continue
        data = rec.get("data") or {}
        exposure.append({
            "trace_id": extract_trace_id(rec),
            "timestamp": rec.get("timestamp"),
            "mode": data.get("mode"),
            "hidden_tools": data.get("hidden_tools") or [],
            "hidden_tool_reasons": data.get("hidden_tool_reasons") or {},
            "active_mitigations": data.get("active_mitigations") or [],
        })
    return exposure


def _summarize(signals: list[dict], mitigations: list[dict], exposure: list[dict]) -> dict[str, Any]:
    return {
        "signal_count": len(signals),
        "mitigation_activations": len([m for m in mitigations if m["event"] == "fama_mitigation_activated"]),
        "mitigation_expirations": len([m for m in mitigations if m["event"] == "fama_mitigation_expired"]),
        "exposure_decisions": len(exposure),
        "signals_by_kind": dict(Counter(str(s.get("kind")) for s in signals)),
        "signals_by_failure_class": dict(Counter(str(s.get("failure_class")) for s in signals if s.get("failure_class"))),
        "mitigations_by_name": dict(Counter(str(m.get("name")) for m in mitigations if m.get("name"))),
        "hidden_tools": dict(Counter(
            tool for exp in exposure for tool in exp.get("hidden_tools") or []
        )),
    }


def _render_text(
    run_dir: Path,
    signals: list[dict],
    mitigations: list[dict],
    exposure: list[dict],
    show_signals: bool,
    show_mitigations: bool,
    show_exposure: bool,
    show_summary: bool,
) -> str:
    lines: list[str] = []
    lines.append(colorize(f"FAMA inspection for {run_dir.name}", Colors.BOLD + Colors.CYAN))
    lines.append(f"Run directory: {run_dir}")
    lines.append("")

    summary = _summarize(signals, mitigations, exposure)
    lines.append(colorize("Summary", Colors.BOLD + Colors.BLUE))
    lines.append(f"  signals:            {summary['signal_count']}")
    lines.append(f"  mitigations active: {summary['mitigation_activations']} (expired {summary['mitigation_expirations']})")
    lines.append(f"  exposure decisions: {summary['exposure_decisions']}")

    if summary["signals_by_kind"]:
        lines.append("")
        lines.append(colorize("Signals by kind", Colors.BOLD + Colors.BLUE))
        for kind, count in Counter(summary["signals_by_kind"]).most_common():
            lines.append(f"  {count:>3}  {kind}")

    if summary["signals_by_failure_class"]:
        lines.append("")
        lines.append(colorize("Signals by failure class", Colors.BOLD + Colors.BLUE))
        for fc, count in Counter(summary["signals_by_failure_class"]).most_common():
            lines.append(f"  {count:>3}  {fc}")

    if summary["mitigations_by_name"]:
        lines.append("")
        lines.append(colorize("Mitigations by name", Colors.BOLD + Colors.BLUE))
        for name, count in Counter(summary["mitigations_by_name"]).most_common():
            lines.append(f"  {count:>3}  {name}")

    if show_signals:
        lines.append("")
        lines.append(colorize(f"Signals ({len(signals)})", Colors.BOLD + Colors.BLUE))
        for s in signals:
            tid = s.get("trace_id") or ""
            lines.append(
                f"  {s.get('timestamp')}  {colorize(str(s.get('kind')), Colors.YELLOW)}  "
                f"severity={s.get('severity')}  source={s.get('source')}  "
                f"failure_class={s.get('failure_class') or '-'}  [{tid}]"
            )

    if show_mitigations:
        lines.append("")
        lines.append(colorize(f"Mitigations ({len(mitigations)})", Colors.BOLD + Colors.BLUE))
        for m in mitigations:
            tid = m.get("trace_id") or ""
            event_color = Colors.GREEN if m["event"] == "fama_mitigation_activated" else Colors.DIM
            lines.append(
                f"  {colorize(m['event'], event_color)}  {m.get('name') or '-'}  "
                f"step_exp={m.get('expires_after_step') or '-'}  [{tid}]"
            )
            if m.get("reason"):
                lines.append(f"       reason: {m['reason'][:160]}")

    if show_exposure:
        lines.append("")
        lines.append(colorize(f"Tool exposure decisions ({len(exposure)})", Colors.BOLD + Colors.BLUE))
        for exp in exposure:
            tid = exp.get("trace_id") or ""
            hidden = exp.get("hidden_tools") or []
            reasons = exp.get("hidden_tool_reasons") or {}
            lines.append(f"  mode={exp.get('mode')}  hidden={len(hidden)}  [{tid}]")
            for tool in hidden[:10]:
                reason = reasons.get(tool, ["?"])[0] if isinstance(reasons.get(tool), list) else reasons.get(tool, "?")
                lines.append(f"       {tool}  ({reason})")
            if len(hidden) > 10:
                lines.append(f"       ... and {len(hidden) - 10} more")

    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    logs_dir = Path(args.logs_dir) if args.logs_dir else None
    try:
        run_dir = resolve_run_dir(args.run, logs_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(colorize(str(exc), Colors.RED), file=sys.stderr)
        return 1

    records = _fama_records(run_dir)
    signals = _extract_signals(records)
    mitigations = _extract_mitigations(records)
    exposure = _extract_exposure(records)

    show_summary = args.summary or not any((args.signals, args.mitigations, args.exposure))

    if args.json:
        print(json.dumps({
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "summary": _summarize(signals, mitigations, exposure),
            "signals": signals,
            "mitigations": mitigations,
            "exposure": exposure,
        }, indent=2, default=str))
        return 0

    print(_render_text(
        run_dir,
        signals,
        mitigations,
        exposure,
        show_signals=args.signals,
        show_mitigations=args.mitigations,
        show_exposure=args.exposure,
        show_summary=show_summary,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
