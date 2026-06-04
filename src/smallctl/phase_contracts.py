from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from .phase_contracts_support import (
    _file_has_symbol,
    _node_defines_name,
)

_QUALITY_SCORES = {
    "none": 0,
    "setup_or_probe": 0,
    "syntax": 1,
    "import": 2,
    "execution": 2,
    "behavioral": 3,
    "integration": 4,
    "e2e": 5,
}


def phase_contract_for_state(state: Any) -> dict[str, Any] | None:
    """Return the phase contract for the current session only.

    Never falls back to a globally-persisted file from another session.
    """
    scratchpad = getattr(state, "scratchpad", {})
    if isinstance(scratchpad, dict) and isinstance(scratchpad.get("_phase_contract"), dict):
        return scratchpad["_phase_contract"]
    return _infer_phase_contract(state)


def _infer_phase_contract(state: Any) -> dict[str, Any] | None:
    task_text = _task_text(state)
    active_number = _active_phase_number(task_text)
    if active_number <= 0:
        return None
    cwd = Path(str(getattr(state, "cwd", "") or "."))
    spec_path = _phase_spec_path(cwd, task_text)
    phases = _phases_from_spec(spec_path) if spec_path is not None else {}
    if not phases:
        phases = {f"phase_{active_number}": {"title": f"Phase {active_number}", "status": "active"}}
    active_phase = f"phase_{active_number}"
    if active_phase not in phases:
        phases[active_phase] = {"title": f"Phase {active_number}", "status": "active"}
    expected_files = _inferred_expected_files(cwd, spec_path, task_text)
    checks = _inferred_checks(expected_files)
    for phase_id, phase in phases.items():
        phase.setdefault("status", "active" if phase_id == active_phase else "pending")
        phase.setdefault("expected_files", expected_files)
        phase.setdefault("required_symbols", [])
        phase.setdefault("checks", checks)
        phase.setdefault("promotion", {"required_quality": "behavioral"})
    phases[active_phase]["status"] = "active"
    return {
        "version": 1,
        "source": "inferred",
        "active_phase": active_phase,
        "phases": phases,
    }


def _task_text(state: Any) -> str:
    run_brief = getattr(state, "run_brief", None)
    wm = getattr(state, "working_memory", None)
    parts = [
        str(getattr(run_brief, "original_task", "") or ""),
        str(getattr(run_brief, "task_contract", "") or ""),
        " ".join(str(item or "") for item in getattr(run_brief, "acceptance_criteria", []) or []),
        str(getattr(wm, "current_goal", "") or ""),
    ]
    return "\n".join(part for part in parts if part)


def _active_phase_number(text: str) -> int:
    matches = re.findall(r"\bphase\s*([0-9]+)\b", str(text or ""), flags=re.IGNORECASE)
    if not matches:
        return 0
    try:
        return max(1, int(matches[-1]))
    except (TypeError, ValueError):
        return 0


def _phase_spec_path(cwd: Path, task_text: str) -> Path | None:
    path_matches = re.findall(r"(?:\./)?[\w./-]*(?:spec|roadmap|plan)[\w./-]*\.md", task_text, flags=re.IGNORECASE)
    for raw in path_matches:
        candidate = cwd / raw.lstrip("./")
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _phases_from_spec(spec_path: Path | None) -> dict[str, dict[str, Any]]:
    if spec_path is None:
        return {}
    try:
        text = spec_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    phases: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        match = re.match(r"^\s{0,3}#{1,4}\s+Phase\s+([0-9]+)\s*:?\s*(.*)$", line, flags=re.IGNORECASE)
        if not match:
            match = re.match(r"^\s*(?:[-*]|\d+[.)])\s+Phase\s+([0-9]+)\s*:?\s*(.*)$", line, flags=re.IGNORECASE)
        if not match:
            continue
        phase_number = int(match.group(1))
        title_suffix = str(match.group(2) or "").strip(" -")
        title = f"Phase {phase_number}" + (f": {title_suffix}" if title_suffix else "")
        phases[f"phase_{phase_number}"] = {"title": title, "status": "pending"}
    return phases


def _inferred_expected_files(cwd: Path, spec_path: Path | None, task_text: str) -> list[str]:
    explicit = re.findall(r"(?:\./)?[\w./-]+\.py", task_text)
    files = [path.lstrip("./") for path in explicit]
    if spec_path is not None:
        try:
            spec_text = spec_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            spec_text = ""
        files.extend(path.lstrip("./") for path in re.findall(r"(?:\./)?[\w./-]+\.py", spec_text))
        if not files:
            for candidate in sorted(spec_path.parent.glob("*.py")):
                try:
                    files.append(str(candidate.relative_to(cwd)))
                except ValueError:
                    files.append(str(candidate))
    seen: set[str] = set()
    unique: list[str] = []
    for item in files:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique[:8]


def _inferred_checks(expected_files: list[str]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for path in expected_files[:3]:
        if not path.endswith(".py"):
            continue
        module_path = Path(path)
        module_name = module_path.stem.replace("-", "_")
        python_path = str(module_path.parent) if str(module_path.parent) not in {"", "."} else "."
        checks.append(
            {
                "id": f"{module_name}_behavior_smoke",
                "quality": "behavioral",
                "command": f"PYTHONPATH={python_path} python3 -c \"import {module_name}; assert {module_name}\"",
            }
        )
    return checks


def phase_contract_status(
    state: Any,
    *,
    verifier_verdict: dict[str, Any] | None = None,
    verifier_quality: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    contract = phase_contract_for_state(state)
    if not contract:
        return None
    phase_id, phase = _active_phase(contract)
    if not phase_id or not isinstance(phase, dict):
        return {"status": "blocked", "reason": "phase_contract_missing_active_phase", "contract": _contract_preview(contract)}
    cwd = Path(str(getattr(state, "cwd", "") or "."))
    missing_files = _missing_expected_files(cwd, phase)
    missing_symbols = _missing_required_symbols(cwd, phase)
    promotion = phase.get("promotion") if isinstance(phase.get("promotion"), dict) else {}
    required_quality = _required_quality(promotion)
    suggested_verifier = _suggested_verifier(phase, required_quality)
    actual_quality = verifier_quality if isinstance(verifier_quality, dict) else {"score": 0, "label": "none"}
    verifier = verifier_verdict if isinstance(verifier_verdict, dict) else {}
    verifier_passed = str(verifier.get("verdict") or "").strip().lower() == "pass"

    reason = "phase_contract_satisfied"
    status = "passed"
    if missing_files:
        status = "blocked"
        reason = "phase_contract_missing_expected_files"
    elif missing_symbols:
        status = "blocked"
        reason = "phase_contract_missing_required_symbols"
    elif not verifier_passed:
        status = "blocked"
        reason = "phase_contract_verifier_not_passing"
    elif int(actual_quality.get("score") or 0) < int(required_quality.get("score") or 0):
        status = "blocked"
        reason = "phase_contract_verifier_quality_too_low"

    return {
        "status": status,
        "reason": reason,
        "active_phase": phase_id,
        "title": str(phase.get("title") or ""),
        "expected_files": _string_list(phase.get("expected_files")),
        "missing_files": missing_files,
        "required_symbols": _string_list(phase.get("required_symbols")),
        "missing_symbols": missing_symbols,
        "checks": _checks_preview(phase),
        "suggested_verifier": suggested_verifier,
        "verifier_quality": actual_quality,
        "required_verifier_quality": required_quality,
    }


def phase_contract_completion_block(
    state: Any,
    *,
    verifier_verdict: dict[str, Any] | None,
    verifier_quality: dict[str, Any] | None,
) -> dict[str, Any] | None:
    contract = phase_contract_for_state(state)
    if not contract:
        return None
    # Only enforce explicit phase contracts; inferred contracts are hints, not requirements.
    if str(contract.get("source") or "").strip() == "inferred":
        return None
    status = phase_contract_status(
        state,
        verifier_verdict=verifier_verdict,
        verifier_quality=verifier_quality,
    )
    if not status or status.get("status") == "passed":
        return None
    planning_mode = bool(getattr(state, "planning_mode_enabled", False))
    notes = _next_action_notes(status)
    if planning_mode:
        notes = [
            *notes,
            "Planning mode cannot execute the verifier directly; call `request_validation_execution` with the suggested command instead of `run` or `shell_exec`.",
        ]
    action = {
        "tool_name": "request_validation_execution" if planning_mode else "shell_exec",
        "notes": notes,
    }
    suggested = status.get("suggested_verifier")
    if isinstance(suggested, dict) and str(suggested.get("command") or "").strip():
        action["required_arguments"] = {"command": str(suggested["command"]).strip()}
        action["check_id"] = str(suggested.get("id") or "")
    return {
        "reason": status.get("reason") or "phase_contract_blocked",
        "phase_contract": status,
        "last_verifier_verdict": verifier_verdict if isinstance(verifier_verdict, dict) else None,
        "next_required_action": action,
    }


def _active_phase(contract: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    phases = contract.get("phases")
    if not isinstance(phases, dict):
        return "", None
    active = str(contract.get("active_phase") or "").strip()
    if active and isinstance(phases.get(active), dict):
        return active, phases[active]
    for phase_id, phase in phases.items():
        if isinstance(phase, dict) and str(phase.get("status") or "").strip().lower() == "active":
            return str(phase_id), phase
    return "", None


def _required_quality(promotion: dict[str, Any]) -> dict[str, Any]:
    label = str(promotion.get("required_quality") or "behavioral").strip().lower() or "behavioral"
    return {"score": _QUALITY_SCORES.get(label, 3), "label": label}


def _missing_expected_files(cwd: Path, phase: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for item in _string_list(phase.get("expected_files")):
        if not (cwd / item).exists():
            missing.append(item)
    return missing


def _missing_required_symbols(cwd: Path, phase: dict[str, Any]) -> list[str]:
    symbols = _string_list(phase.get("required_symbols"))
    if not symbols:
        return []
    files = [cwd / item for item in _string_list(phase.get("expected_files")) if item.endswith(".py")]
    files = [path for path in files if path.exists()]
    if not files:
        return symbols
    missing: list[str] = []
    for symbol in symbols:
        if not any(_file_has_symbol(path, symbol) for path in files):
            missing.append(symbol)
    return missing


def _checks_preview(phase: dict[str, Any]) -> list[dict[str, Any]]:
    checks = phase.get("checks")
    if not isinstance(checks, list):
        return []
    preview: list[dict[str, Any]] = []
    for check in checks[:8]:
        if not isinstance(check, dict):
            continue
        preview.append(
            {
                "id": str(check.get("id") or ""),
                "quality": str(check.get("quality") or ""),
                "command": str(check.get("command") or ""),
            }
        )
    return preview


def _suggested_verifier(phase: dict[str, Any], required_quality: dict[str, Any]) -> dict[str, Any] | None:
    checks = phase.get("checks")
    if not isinstance(checks, list):
        return None
    minimum = int(required_quality.get("score") or 0)
    fallback: dict[str, Any] | None = None
    for check in checks:
        if not isinstance(check, dict):
            continue
        command = str(check.get("command") or "").strip()
        if not command:
            continue
        quality_label = str(check.get("quality") or "").strip().lower()
        score = _QUALITY_SCORES.get(quality_label, 0)
        candidate = {
            "id": str(check.get("id") or ""),
            "quality": quality_label or "none",
            "score": score,
            "command": command,
        }
        if fallback is None:
            fallback = candidate
        if score >= minimum:
            return candidate
    return fallback


def _contract_preview(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": contract.get("version"),
        "active_phase": contract.get("active_phase"),
        "phase_ids": list((contract.get("phases") or {}).keys()) if isinstance(contract.get("phases"), dict) else [],
    }


def _next_action_notes(status: dict[str, Any]) -> list[str]:
    reason = str(status.get("reason") or "")
    suggested = status.get("suggested_verifier")
    suffix = ""
    if isinstance(suggested, dict) and str(suggested.get("command") or "").strip():
        check_id = str(suggested.get("id") or "phase check").strip() or "phase check"
        suffix = f" Suggested verifier `{check_id}` is provided in required_arguments.command."
    if reason == "phase_contract_missing_expected_files":
        return ["Create or restore the contract's missing expected files before promoting the phase." + suffix]
    if reason == "phase_contract_missing_required_symbols":
        return ["Implement the missing required symbols, then rerun the phase verifier." + suffix]
    if reason == "phase_contract_verifier_quality_too_low":
        return ["Run a verifier matching the contract's required quality before task_complete." + suffix]
    return ["Run the active phase contract checks and fix the first failing gate before task_complete." + suffix]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
