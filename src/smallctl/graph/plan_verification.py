from __future__ import annotations

import asyncio
import py_compile
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..state import PlanStep, StepEvidenceArtifact, StepVerificationResult, StepVerifierSpec, json_safe_value

VerifierFn = Callable[[Any, StepVerifierSpec, PlanStep], Awaitable[dict[str, Any]]]


async def verify_artifact_exists(harness: Any, spec: StepVerifierSpec, step: PlanStep) -> dict[str, Any]:
    artifact_id = str(spec.args.get("artifact_id") or spec.args.get("ref") or "").strip()
    artifacts = getattr(harness.state, "artifacts", {})
    passed = bool(artifact_id and isinstance(artifacts, dict) and artifact_id in artifacts)
    return {"kind": spec.kind, "passed": passed, "artifact_id": artifact_id}


async def verify_file_exists(harness: Any, spec: StepVerifierSpec, step: PlanStep) -> dict[str, Any]:
    path = _workspace_path(harness, spec)
    return {"kind": spec.kind, "passed": path.exists(), "path": str(path)}


async def verify_file_changed_since_step_start(harness: Any, spec: StepVerifierSpec, step: PlanStep) -> dict[str, Any]:
    path = _workspace_path(harness, spec)
    baseline = getattr(harness.state, "scratchpad", {}).get("_staged_step_file_baseline")
    baseline_mtime = 0.0
    if isinstance(baseline, dict):
        baseline_mtime = float(baseline.get(str(path), 0.0) or 0.0)
    try:
        current_mtime = path.stat().st_mtime
    except OSError:
        current_mtime = 0.0
    return {
        "kind": spec.kind,
        "passed": current_mtime > baseline_mtime,
        "path": str(path),
        "baseline_mtime": baseline_mtime,
        "current_mtime": current_mtime,
    }


async def verify_file_count(harness: Any, spec: StepVerifierSpec, step: PlanStep) -> dict[str, Any]:
    root = _workspace_path(harness, spec, key="path")
    pattern = str(spec.args.get("pattern") or "*")
    min_count = int(spec.args.get("min", 1) or 1)
    count = len(list(root.glob(pattern))) if root.exists() else 0
    return {"kind": spec.kind, "passed": count >= min_count, "path": str(root), "pattern": pattern, "count": count}


async def verify_syntax_ok(harness: Any, spec: StepVerifierSpec, step: PlanStep) -> dict[str, Any]:
    path = _workspace_path(harness, spec)
    if path.suffix != ".py":
        return {"kind": spec.kind, "passed": path.exists(), "path": str(path), "note": "non-python existence check"}
    try:
        py_compile.compile(str(path), doraise=True)
    except Exception as exc:
        return {"kind": spec.kind, "passed": False, "path": str(path), "error": str(exc)}
    return {"kind": spec.kind, "passed": True, "path": str(path)}


async def verify_last_command_passed(harness: Any, spec: StepVerifierSpec, step: PlanStep) -> dict[str, Any]:
    records = getattr(harness.state, "tool_execution_records", {})
    step_run_id = str(getattr(harness.state, "active_step_run_id", "") or "")
    for record in reversed(list(records.values())) if isinstance(records, dict) else []:
        if not isinstance(record, dict):
            continue
        if step_run_id and str(record.get("step_run_id") or "") != step_run_id:
            continue
        if str(record.get("tool_name") or "") not in {"shell_exec", "ssh_exec"}:
            continue
        result = record.get("result") if isinstance(record.get("result"), dict) else {}
        output = result.get("output") if isinstance(result, dict) else {}
        exit_code = _extract_exit_code(output)
        return {"kind": spec.kind, "passed": exit_code == 0, "operation_id": record.get("operation_id"), "exit_code": exit_code}
    return {"kind": spec.kind, "passed": False, "error": "no command record found"}


async def verify_remote_readback_contains(harness: Any, spec: StepVerifierSpec, step: PlanStep) -> dict[str, Any]:
    needle = str(spec.args.get("text") or spec.args.get("contains") or "").strip()
    if not needle:
        return {"kind": spec.kind, "passed": False, "error": "missing contains text"}
    records = getattr(harness.state, "tool_execution_records", {})
    step_run_id = str(getattr(harness.state, "active_step_run_id", "") or "")
    for record in reversed(list(records.values())) if isinstance(records, dict) else []:
        if not isinstance(record, dict):
            continue
        if step_run_id and str(record.get("step_run_id") or "") != step_run_id:
            continue
        payload = json_safe_value(record.get("result"))
        if needle in str(payload):
            return {"kind": spec.kind, "passed": True, "operation_id": record.get("operation_id")}
    return {"kind": spec.kind, "passed": False, "error": "text not found in step tool records"}


async def verify_custom_command(harness: Any, spec: StepVerifierSpec, step: PlanStep) -> dict[str, Any]:
    command = str(spec.args.get("command") or "").strip()
    if not command:
        return {"kind": spec.kind, "passed": False, "error": "missing command"}
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=str(getattr(harness.state, "cwd", "") or "."),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=max(1, int(spec.timeout_sec or 30)))
    except asyncio.TimeoutError:
        proc.kill()
        return {"kind": spec.kind, "passed": False, "error": "timeout", "command": command}
    return {
        "kind": spec.kind,
        "passed": proc.returncode == 0,
        "command": command,
        "exit_code": proc.returncode,
        "stdout": stdout.decode("utf-8", errors="replace")[:1000],
        "stderr": stderr.decode("utf-8", errors="replace")[:1000],
    }


VERIFIER_REGISTRY: dict[str, VerifierFn] = {
    "artifact_exists": verify_artifact_exists,
    "file_exists": verify_file_exists,
    "file_changed": verify_file_changed_since_step_start,
    "file_count": verify_file_count,
    "syntax_ok": verify_syntax_ok,
    "last_command_passed": verify_last_command_passed,
    "remote_readback_contains": verify_remote_readback_contains,
    "custom_command": verify_custom_command,
}


class StepCompletionGate:
    async def verify_step(self, harness: Any, step: PlanStep) -> StepVerificationResult:
        verifier_results: list[dict[str, Any]] = []
        failed_criteria: list[str] = []
        for output in step.outputs_expected:
            if not output.required:
                continue
            if output.kind == "file":
                path = Path(getattr(harness.state, "cwd", ".")) / output.ref
                passed = path.exists()
                result = {"kind": "expected_output:file", "passed": passed, "path": str(path)}
            elif output.kind == "artifact":
                artifacts = getattr(harness.state, "artifacts", {})
                passed = bool(output.ref and isinstance(artifacts, dict) and output.ref in artifacts)
                result = {"kind": "expected_output:artifact", "passed": passed, "artifact_id": output.ref}
            else:
                result = {"kind": f"expected_output:{output.kind}", "passed": True, "ref": output.ref}
            verifier_results.append(result)
            if not result["passed"]:
                failed_criteria.append(output.description or output.ref or output.kind)

        for spec in step.verifiers:
            verifier = VERIFIER_REGISTRY.get(spec.kind)
            if verifier is None:
                result = {"kind": spec.kind, "passed": False, "error": "unknown verifier"}
            else:
                try:
                    result = await asyncio.wait_for(
                        verifier(harness, spec, step),
                        timeout=max(1, int(spec.timeout_sec or 30)) + 1,
                    )
                except Exception as exc:
                    result = {"kind": spec.kind, "passed": False, "error": str(exc)}
            result["required"] = bool(spec.required)
            verifier_results.append(result)
            if spec.required and not bool(result.get("passed")):
                failed_criteria.append(spec.kind)

        passed = not failed_criteria
        result = StepVerificationResult(
            step_id=step.step_id,
            step_run_id=str(getattr(harness.state, "active_step_run_id", "") or ""),
            passed=passed,
            failed_criteria=failed_criteria,
            verifier_results=verifier_results,
        )
        harness.state.step_verification_result = result
        return result


def compact_step_evidence(
    harness: Any,
    step: PlanStep,
    verification_result: StepVerificationResult,
) -> StepEvidenceArtifact:
    state = harness.state
    step_run_id = verification_result.step_run_id or str(getattr(state, "active_step_run_id", "") or "")
    records = getattr(state, "tool_execution_records", {})
    operation_ids: list[str] = []
    artifact_ids: list[str] = []
    if isinstance(records, dict):
        for operation_id, record in records.items():
            if not isinstance(record, dict) or str(record.get("step_run_id") or "") != step_run_id:
                continue
            operation_ids.append(str(operation_id))
            artifact_id = str(record.get("artifact_id") or "").strip()
            if artifact_id:
                artifact_ids.append(artifact_id)
    artifacts = getattr(state, "artifacts", {})
    if isinstance(artifacts, dict):
        for artifact_id, artifact in artifacts.items():
            metadata = getattr(artifact, "metadata", {}) or {}
            if isinstance(metadata, dict) and str(metadata.get("step_run_id") or "") == step_run_id:
                artifact_ids.append(str(artifact_id))
    files_touched = list(dict.fromkeys(getattr(state, "files_changed_this_cycle", []) or []))
    message = str(state.scratchpad.get("_step_complete_message") or "").strip()
    if not message:
        message = f"Completed {step.step_id}: {step.title}"
    return StepEvidenceArtifact(
        step_id=step.step_id,
        step_run_id=step_run_id,
        attempt=int(step.retry_count or 0) + 1,
        summary=message[:1000],
        artifact_ids=list(dict.fromkeys(artifact_ids)),
        files_touched=files_touched,
        decisions=[],
        verifier_results=verification_result.verifier_results,
        tool_operation_ids=list(dict.fromkeys(operation_ids)),
    )


def _workspace_path(harness: Any, spec: StepVerifierSpec, *, key: str = "path") -> Path:
    raw = str(spec.args.get(key) or spec.args.get("ref") or "").strip()
    path = Path(raw)
    if path.is_absolute():
        return path
    return Path(getattr(harness.state, "cwd", ".") or ".") / path


def _extract_exit_code(output: Any) -> int | None:
    if isinstance(output, dict):
        for key in ("exit_code", "returncode", "return_code"):
            if key in output:
                try:
                    return int(output[key])
                except (TypeError, ValueError):
                    return None
    return None
