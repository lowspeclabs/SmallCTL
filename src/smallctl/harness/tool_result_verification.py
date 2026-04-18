from __future__ import annotations

import hashlib
from typing import Any

from ..models.tool_result import ToolEnvelope


def _store_verifier_verdict(
    state: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if tool_name not in {"shell_exec", "ssh_exec"}:
        return None
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    output = result.output if isinstance(result.output, dict) else {}
    if not output:
        metadata_output = metadata.get("output")
        if isinstance(metadata_output, dict):
            output = metadata_output
    args = arguments if isinstance(arguments, dict) else {}
    command = str(
        args.get("command")
        or metadata.get("command")
        or metadata.get("target")
        or ""
    ).strip()
    target = command
    if tool_name == "ssh_exec":
        host = str(args.get("host") or metadata.get("host") or "").strip()
        remote_command = str(args.get("command") or metadata.get("command") or "").strip()
        if host and remote_command:
            target = f"{host} :: {remote_command}"
        elif host:
            target = host
        elif remote_command:
            target = remote_command

    exit_code = output.get("exit_code") if isinstance(output, dict) else None
    stdout = _snip_text(output.get("stdout") if isinstance(output, dict) else "", limit=400)
    stderr = _snip_text(output.get("stderr") if isinstance(output, dict) else "", limit=400)
    status = str(result.status or metadata.get("status") or "").strip()
    if status == "needs_human":
        verdict = "needs_human"
    elif result.success and (exit_code in (0, None)):
        verdict = "pass"
    else:
        verdict = "fail"
    failure_class = _classify_execution_failure(result.error or stderr or stdout)
    acceptance_delta = {
        "status": "satisfied" if verdict == "pass" else "blocked",
        "notes": ["execution succeeded"] if verdict == "pass" else [str(result.error or status or "execution failed")],
    }
    normalized = {
        "tool": tool_name,
        "target": target,
        "command": command,
        "exit_code": exit_code,
        "key_stdout": stdout,
        "key_stderr": stderr,
        "verdict": verdict,
        "failure_mode": failure_class,
        "acceptance_delta": acceptance_delta,
    }
    state.last_verifier_verdict = normalized
    state.scratchpad["_last_verifier_verdict"] = normalized
    state.last_failure_class = failure_class
    state.scratchpad["_last_failure_class"] = failure_class
    _update_acceptance_ledger(state, verdict=verdict)
    _update_repair_cycle_state(
        state,
        tool_name=tool_name,
        result=result,
        command=command,
        target=target,
        verdict=verdict,
        failure_class=failure_class,
    )
    return normalized


def _annotate_verifier_artifact(
    artifact: Any,
    *,
    verifier_verdict: dict[str, Any],
) -> None:
    if not hasattr(artifact, "metadata") or not isinstance(artifact.metadata, dict):
        artifact.metadata = {}
    metadata = artifact.metadata
    verdict = str(verifier_verdict.get("verdict") or "").strip()
    target = str(verifier_verdict.get("target") or verifier_verdict.get("command") or "").strip()
    command = str(verifier_verdict.get("command") or "").strip()
    metadata["verifier_verdict"] = verdict
    metadata["verifier_target"] = target
    metadata["verifier_command"] = command
    metadata["verifier_exit_code"] = verifier_verdict.get("exit_code")
    metadata["verifier_stdout"] = verifier_verdict.get("key_stdout")
    metadata["verifier_stderr"] = verifier_verdict.get("key_stderr")
    if target and (not getattr(artifact, "source", "")):
        artifact.source = target
    status_label = "SUCCESS" if verdict == "pass" else "FAILURE"
    summary = f"{artifact.tool_name or artifact.kind} {status_label}"
    if command:
        summary = f"{summary}: {command}"
    elif target:
        summary = f"{summary}: {target}"
    artifact.summary = summary[:160]


def _snip_text(value: Any, *, limit: int = 400) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _classify_execution_failure(text: str) -> str:
    lowered = str(text or "").lower()
    if not lowered:
        return ""
    if "syntaxerror" in lowered or "parseerror" in lowered:
        return "syntax"
    if "importerror" in lowered or "modulenotfounderror" in lowered:
        return "import"
    if "no such file" in lowered or "not found" in lowered:
        return "path"
    if "timed out" in lowered or "timeout" in lowered or "connection timed out" in lowered:
        return "environment"
    if "permission denied" in lowered or "password" in lowered or "sudo" in lowered:
        return "environment"
    if "assert" in lowered or "failed" in lowered or "traceback" in lowered:
        return "test"
    return "logic"


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


def _update_repair_cycle_state(
    state: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    command: str,
    target: str,
    verdict: str,
    failure_class: str,
) -> None:
    if verdict == "pass":
        if getattr(state, "acceptance_ready", None) and state.acceptance_ready():
            state.scratchpad["_contract_phase"] = "execute"
        elif getattr(state, "repair_cycle_id", ""):
            state.scratchpad["_contract_phase"] = "verify"
        return

    if verdict == "needs_human":
        return

    signature_seed = "|".join(
        [
            str(getattr(state, "thread_id", "") or ""),
            tool_name,
            command,
            target,
            failure_class,
            str(result.error or ""),
        ]
    )
    repair_cycle_id = f"repair-{hashlib.sha1(signature_seed.encode('utf-8')).hexdigest()[:8]}"
    if getattr(state, "repair_cycle_id", "") != repair_cycle_id:
        state.repair_cycle_id = repair_cycle_id
        state.scratchpad["_repair_cycle_reads"] = []
        state.files_changed_this_cycle = []
    state.scratchpad["_contract_phase"] = "repair"

    counters = state.stagnation_counters if isinstance(getattr(state, "stagnation_counters", None), dict) else {}
    failure_signature = f"{tool_name}|{command}|{target}|{failure_class}|{result.error or ''}"
    last_failure_signature = str(state.scratchpad.get("_repair_last_failure_signature", "") or "")
    if last_failure_signature == failure_signature:
        counters["no_progress"] = int(counters.get("no_progress", 0)) + 1
    state.scratchpad["_repair_last_failure_signature"] = failure_signature

    command_fingerprint = hashlib.sha1(f"{tool_name}|{command}|{target}".encode("utf-8")).hexdigest()
    last_command_fingerprint = str(state.scratchpad.get("_repair_last_command_fingerprint", "") or "")
    if last_command_fingerprint == command_fingerprint:
        counters["repeat_command"] = int(counters.get("repeat_command", 0)) + 1
    state.scratchpad["_repair_last_command_fingerprint"] = command_fingerprint
    state.stagnation_counters = counters
