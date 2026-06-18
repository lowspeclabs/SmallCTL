from __future__ import annotations

import hashlib
from typing import Any

from ..docker_retry_normalization import (
    docker_failure_is_registry_resolution,
    docker_retry_family,
    docker_retry_key,
    extract_docker_command_target,
)
from ..models.tool_result import ToolEnvelope


def _update_acceptance_ledger(state: Any, *, verdict: str) -> None:
    criteria = []
    if hasattr(state, "active_acceptance_criteria"):
        try:
            criteria = list(state.active_acceptance_criteria())
        except Exception:
            criteria = []
    if not criteria:
        return
    ledger = state.acceptance_ledger if isinstance(getattr(state, "acceptance_ledger", None), dict) else {}
    if verdict == "pass":
        for criterion in criteria:
            ledger[criterion] = "passed"
    elif verdict == "needs_human":
        for criterion in criteria:
            ledger.setdefault(criterion, "pending")
    state.acceptance_ledger = ledger


def _record_docker_retry_state(
    state: Any,
    *,
    command: str,
    failure_class: str,
    verdict: str,
) -> dict[str, Any]:
    if verdict != "fail" or not docker_failure_is_registry_resolution(failure_class):
        return {}

    parsed = extract_docker_command_target(command)
    if parsed is None:
        return {}
    command_kind, image_ref = parsed
    retry_key = docker_retry_key(command, failure_class)
    retry_family = docker_retry_family(command)
    if not retry_key or not retry_family:
        return {}

    scratchpad = state.scratchpad if isinstance(getattr(state, "scratchpad", None), dict) else {}
    state.scratchpad = scratchpad

    retry_counts = scratchpad.setdefault("_docker_registry_retry_counts", {})
    if not isinstance(retry_counts, dict):
        retry_counts = {}
        scratchpad["_docker_registry_retry_counts"] = retry_counts
    retry_count = int(retry_counts.get(retry_key, 0) or 0) + 1
    retry_counts[retry_key] = retry_count

    family_counts = scratchpad.setdefault("_docker_registry_family_counts", {})
    if not isinstance(family_counts, dict):
        family_counts = {}
        scratchpad["_docker_registry_family_counts"] = family_counts
    family_count = int(family_counts.get(retry_family, 0) or 0) + 1
    family_counts[retry_family] = family_count

    exhausted_families = scratchpad.setdefault("_docker_registry_exhausted_families", [])
    if not isinstance(exhausted_families, list):
        exhausted_families = []
        scratchpad["_docker_registry_exhausted_families"] = exhausted_families
    if family_count >= 4 and retry_family not in exhausted_families:
        exhausted_families.append(retry_family)

    scratchpad["_last_docker_registry_retry_key"] = retry_key
    scratchpad["_last_docker_registry_retry_family"] = retry_family

    return {
        "docker_command_kind": command_kind,
        "docker_image_ref": image_ref,
        "docker_retry_family": retry_family,
        "docker_retry_key": retry_key,
        "docker_retry_count": retry_count,
        "docker_retry_family_count": family_count,
    }


def _update_repair_cycle_state(
    state: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    command: str,
    target: str,
    verdict: str,
    failure_class: str,
    docker_retry: dict[str, Any] | None = None,
) -> None:
    if verdict == "pass":
        if getattr(state, "acceptance_ready", None) and state.acceptance_ready():
            state.scratchpad["_contract_phase"] = "execute"
        elif getattr(state, "repair_cycle_id", ""):
            state.scratchpad["_contract_phase"] = "verify"
        return

    if verdict == "needs_human":
        return

    docker_retry = docker_retry if isinstance(docker_retry, dict) else {}
    semantic_target = str(docker_retry.get("docker_retry_family") or target)
    semantic_failure = str(docker_retry.get("docker_retry_key") or "")
    semantic_command = str(docker_retry.get("docker_retry_family") or command)

    signature_seed = "|".join(
        [
            str(getattr(state, "thread_id", "") or ""),
            tool_name,
            semantic_command or command,
            semantic_target or target,
            semantic_failure or failure_class,
            str(result.error or ""),
        ]
    )
    repair_cycle_id = f"repair-{hashlib.sha1(signature_seed.encode('utf-8')).hexdigest()[:8]}"
    if getattr(state, "repair_cycle_id", "") != repair_cycle_id:
        state.repair_cycle_id = repair_cycle_id
        state.scratchpad["_repair_cycle_reads"] = []
        state.files_changed_this_cycle = []
    # Increment repair step count every time we enter repair
    repair_steps = int(state.scratchpad.get("_repair_step_count", 0) or 0) + 1
    state.scratchpad["_repair_step_count"] = repair_steps
    max_repair = int(state.scratchpad.get("_max_repair_steps", 3) or 3)
    if repair_steps >= max_repair:
        state.acceptance_waived = True
    # Flag for auto-escalation after 2 repair cycles on same target
    if repair_steps >= 2:
        state.scratchpad["_repair_cycle_escalation_ready"] = True
    state.scratchpad["_contract_phase"] = "repair"

    counters = state.stagnation_counters if isinstance(getattr(state, "stagnation_counters", None), dict) else {}
    failure_signature = semantic_failure or f"{tool_name}|{command}|{target}|{failure_class}|{result.error or ''}"
    last_failure_signature = str(state.scratchpad.get("_repair_last_failure_signature", "") or "")
    if last_failure_signature == failure_signature:
        counters["no_progress"] = int(counters.get("no_progress", 0)) + 1
    state.scratchpad["_repair_last_failure_signature"] = failure_signature

    command_fingerprint = hashlib.sha1(f"{tool_name}|{semantic_command or command}|{semantic_target or target}".encode("utf-8")).hexdigest()
    last_command_fingerprint = str(state.scratchpad.get("_repair_last_command_fingerprint", "") or "")
    if last_command_fingerprint == command_fingerprint:
        counters["repeat_command"] = int(counters.get("repeat_command", 0)) + 1
    state.scratchpad["_repair_last_command_fingerprint"] = command_fingerprint
    state.stagnation_counters = counters
