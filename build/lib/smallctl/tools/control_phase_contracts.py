from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def normalize_phase_contract_payload(contract: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(contract)
    phases = normalized.get("phases")
    if not isinstance(phases, dict):
        return normalized
    normalized_phases: dict[Any, Any] = {}
    for phase_id, raw_phase in phases.items():
        if not isinstance(raw_phase, dict):
            normalized_phases[phase_id] = raw_phase
            continue
        phase = dict(raw_phase)
        if "title" not in phase and isinstance(phase.get("name"), str):
            phase["title"] = phase.get("name")
        promotion = phase.get("promotion")
        if isinstance(promotion, str):
            phase["promotion"] = normalize_phase_promotion(promotion)
        checks = phase.get("checks")
        if isinstance(checks, list):
            expected_files = [str(item or "").strip() for item in phase.get("expected_files") or [] if str(item or "").strip()]
            phase["checks"] = [
                normalize_phase_check(check, index=index, expected_files=expected_files)
                for index, check in enumerate(checks, start=1)
            ]
        normalized_phases[phase_id] = phase
    normalized["phases"] = normalized_phases
    return normalized


def normalize_phase_promotion(value: str) -> dict[str, Any]:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "all_checks_pass", "all_checks", "pass", "passed"}:
        return {"required_quality": "behavioral"}
    if normalized in {"syntax", "import", "execution", "behavioral", "integration", "e2e"}:
        return {"required_quality": normalized}
    return {"required_quality": "behavioral", "criteria": value}


def normalize_phase_check(check: Any, *, index: int, expected_files: list[str] | None = None) -> Any:
    if not isinstance(check, str):
        return check
    command = normalize_phase_check_command(check.strip(), expected_files=expected_files or [])
    if not command:
        return {"id": f"check_{index}", "quality": "none", "command": ""}
    return {
        "id": phase_check_id_from_command(command, index=index),
        "quality": phase_check_quality_from_command(command),
        "command": command,
    }


def normalize_phase_check_command(command: str, *, expected_files: list[str]) -> str:
    normalized = str(command or "").strip()
    if not normalized or re.search(r"(?:^|&&|;)\s*cd\s+", normalized):
        return normalized
    for path in expected_files:
        filename = Path(path).name
        if not filename or "/" in filename:
            continue
        replacement = str(path).lstrip("./")
        normalized = re.sub(
            rf"(?<![\w./-]){re.escape(filename)}(?![\w./-])",
            replacement,
            normalized,
        )
    return normalized


def phase_check_id_from_command(command: str, *, index: int) -> str:
    tokens = re.findall(r"[A-Za-z0-9_]+", command)
    meaningful = [token.lower() for token in tokens if token.lower() not in {"cd", "python", "python3", "m"}]
    stem = "_".join(meaningful[:4]).strip("_") or f"check_{index}"
    return stem[:80]


def phase_check_quality_from_command(command: str) -> str:
    normalized = " ".join(str(command or "").strip().lower().split())
    padded = f" {normalized} "
    if any(pattern in padded for pattern in (" pytest ", " -m pytest ", " unittest ", " -m unittest ")):
        return "behavioral"
    if " -m py_compile " in padded or " py_compile " in padded:
        return "syntax"
    if " -c " in padded:
        if "assert " in normalized or ".move" in normalized or ".spawn" in normalized or ".handle_" in normalized:
            return "behavioral"
        if "import " in normalized:
            return "import"
    if re.search(r"(?:^|[\s/])test_[\w.-]+\.py\b", normalized):
        return "behavioral"
    if " --smoke" in padded or " smoke" in padded:
        return "behavioral"
    return "execution"


def phase_contract_validation_error(contract: dict[str, Any]) -> str:
    phases = contract.get("phases")
    if not isinstance(phases, dict) or not phases:
        return "Phase contract must include a non-empty `phases` object."
    active_phase = str(contract.get("active_phase") or "").strip()
    active_count = 0
    for phase_id, phase in phases.items():
        if not str(phase_id or "").strip():
            return "Phase contract phase IDs must be non-empty strings."
        if not isinstance(phase, dict):
            return f"Phase `{phase_id}` must be an object."
        if str(phase.get("status") or "").strip().lower() == "active":
            active_count += 1
        for key in ("expected_files", "required_symbols", "checks"):
            value = phase.get(key)
            if value is not None and not isinstance(value, list):
                return f"Phase `{phase_id}` field `{key}` must be a list when provided."
        promotion = phase.get("promotion")
        if promotion is not None and not isinstance(promotion, dict):
            return f"Phase `{phase_id}` field `promotion` must be an object when provided."
    if active_phase and active_phase not in phases:
        return f"Active phase `{active_phase}` is not present in `phases`."
    if not active_phase and active_count != 1:
        return "Phase contract must set `active_phase` or mark exactly one phase as active."
    return ""
