from __future__ import annotations

import json
import logging
import shlex
import hashlib
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ..guards import is_small_model_name
from ..models.conversation import ConversationMessage
from ..models.tool_result import ToolEnvelope
from ..normalization import dedupe_keep_tail
from ..redaction import redact_sensitive_data
from ..state import json_safe_value

if TYPE_CHECKING:
    from ..harness import Harness

logger = logging.getLogger("smallctl.harness.tool_results")

class ToolResultService:
    def __init__(self, harness: Harness):
        self.harness = harness

    def _is_small_model(self) -> bool:
        scratchpad = getattr(self.harness.state, "scratchpad", {})
        if isinstance(scratchpad, dict) and "_model_is_small" in scratchpad:
            return bool(scratchpad.get("_model_is_small"))
        model_name = getattr(getattr(self.harness, "client", None), "model", None)
        if not model_name and isinstance(scratchpad, dict):
            model_name = scratchpad.get("_model_name")
        return is_small_model_name(str(model_name or ""))

    async def record_result(
        self,
        *,
        tool_name: str,
        tool_call_id: str | None,
        result: ToolEnvelope,
        arguments: dict[str, Any] | None = None,
    ) -> ConversationMessage:
        from ..context.policy import estimate_text_tokens

        if isinstance(result.metadata, dict) and result.metadata:
            result.metadata = redact_sensitive_data(result.metadata)

        # Ensure arguments are preserved in metadata for loop detection
        if arguments is not None and "arguments" not in result.metadata:
            result.metadata["arguments"] = redact_sensitive_data(arguments)

        if result.metadata.get("cache_hit"):
            artifact_id = str(result.metadata.get("artifact_id", "")).strip()
            artifact = self.harness.state.artifacts.get(artifact_id)
            if artifact is None:
                from ..context import format_reused_artifact_message # Avoid circular import if possible
                compact_content = json.dumps(json_safe_value(result.to_dict()), ensure_ascii=True)
            else:
                from ..context import format_reused_artifact_message
                self.harness.state.retrieval_cache = [artifact.artifact_id]
                compact_content = format_reused_artifact_message(artifact)
                self.harness._runlog(
                    "artifact_reused",
                    "tool result satisfied from cached artifact",
                    artifact_id=artifact.artifact_id,
                    tool_name=tool_name,
                    source=artifact.source,
                )
            return ConversationMessage(
                role="tool",
                name=tool_name,
                tool_call_id=tool_call_id,
                content=compact_content,
                metadata={"artifact_id": artifact_id, "cache_hit": True},
            )

        artifact = None
        # Optimization: Do NOT create new artifacts for tool snapshots of existing artifacts.
        if tool_name == "artifact_read" and result.success and result.metadata.get("artifact_id"):
             artifact_id = str(result.metadata["artifact_id"])
             artifact = self.harness.state.artifacts.get(artifact_id)
             if artifact is not None and isinstance(result.output, str):
                  artifact.preview_text = result.output[:self.harness.artifact_store.policy.preview_char_limit]
        
        if artifact is None:
             artifact = self.harness.artifact_store.persist_tool_result(
                 tool_name=tool_name,
                 result=result,
                 session_id=str(getattr(self.harness.state, "thread_id", "") or ""),
                 tool_call_id=str(tool_call_id or ""),
             )
        
        # Automatic summarization for large outputs
        if result.success and result.output and artifact:
            out_str = str(result.output)
            tokens = estimate_text_tokens(out_str)
            if tokens > self.harness.context_policy.artifact_summarization_threshold and self.harness.summarizer_client:
                self.harness._runlog(
                    "context_summarize_request",
                    "requesting summarization for large tool result",
                    tool_name=tool_name,
                    tokens=tokens,
                )
                try:
                    distilled = await self.harness.summarizer.summarize_artifact_async(
                        client=self.harness.summarizer_client,
                        artifact_id=artifact.artifact_id,
                        content=out_str,
                        label=artifact.source or tool_name
                    )
                    if distilled:
                        artifact.summary = f"Distilled: {distilled}"
                        artifact.preview_text = distilled[:self.harness.artifact_store.policy.preview_char_limit]
                        artifact.metadata["summarized"] = True
                except Exception as exc:
                    logger.warning("Automatic context summarization failed: %s", exc)

        if artifact:
            self.harness.state.artifacts[artifact.artifact_id] = artifact
            self.harness.state.retrieval_cache = [artifact.artifact_id]
            if tool_name in {"shell_exec", "ssh_exec"}:
                _consolidate_shell_attempt_family(
                    state=self.harness.state,
                    artifact_id=artifact.artifact_id,
                    result=result,
                    tool_name=tool_name,
                )
            verifier_verdict = _store_verifier_verdict(
                self.harness.state,
                tool_name=tool_name,
                result=result,
                arguments=arguments,
            )
            if artifact and isinstance(verifier_verdict, dict) and verifier_verdict:
                _annotate_verifier_artifact(artifact, verifier_verdict=verifier_verdict)
            if tool_name == "file_read" and result.success:
                cache_key = _file_read_cache_key(self.harness.state.cwd, result.metadata)
                if cache_key:
                    cache = self.harness.state.scratchpad.setdefault("file_read_cache", {})
                    if isinstance(cache, dict):
                        cache[cache_key] = artifact.artifact_id

            if tool_name in {"plan_set", "plan_step_update", "plan_request_execution", "plan_export"}:
                playbook_artifact_id = str(result.metadata.get("artifact_id", "") or "").strip()
                if playbook_artifact_id:
                    self.harness.state.plan_artifact_id = playbook_artifact_id
                    self.harness.state.plan_resolved = True
                    self.harness.state.retrieval_cache = [playbook_artifact_id]

            if tool_name == "memory_update" and result.success:
                section = str(result.metadata.get("section", "")).strip().lower()
                if section == "plan":
                    self.harness.state.plan_artifact_id = artifact.artifact_id
                    self.harness.state.plan_resolved = True
            elif tool_name == "artifact_read" and result.success:
                artifact_id = str(result.metadata.get("artifact_id", "")).strip()
                if artifact_id:
                    if artifact_id == self.harness.state.plan_artifact_id:
                        self.harness.state.plan_resolved = True
                    elif (
                        not self.harness.state.plan_artifact_id
                        and artifact.tool_name == "memory_update"
                        and str(artifact.metadata.get("section", "")).strip().lower() == "plan"
                    ):
                        self.harness.state.plan_artifact_id = artifact_id
                        self.harness.state.plan_resolved = True
                    if result.metadata.get("truncated"):
                        suppressed = self.harness.state.scratchpad.get("suppressed_truncated_artifact_ids", [])
                        if isinstance(suppressed, list):
                            self.harness.state.scratchpad["suppressed_truncated_artifact_ids"] = dedupe_keep_tail(
                                suppressed + [artifact_id],
                                limit=12,
                            )
                        else:
                            self.harness.state.scratchpad["suppressed_truncated_artifact_ids"] = [artifact_id]

            if tool_name != "shell_exec" and not result.metadata.get("skip_auto_fact_record"):
                fact_label = artifact.summary or tool_name
                self.harness.state.working_memory.known_facts = dedupe_keep_tail(
                    self.harness.state.working_memory.known_facts + [f"{tool_name}: {fact_label}"],
                    limit=12,
                )
            # Success clears consecutive error count
            self.harness.state.recent_errors = []
            
        if not result.success and result.error:
            if not result.metadata.get("hallucination") and not result.metadata.get("approval_denied"):
                self.harness.state.working_memory.failures = dedupe_keep_tail(
                    self.harness.state.working_memory.failures + [f"{tool_name}: {result.error}"],
                    limit=8,
                )
                self.harness.state.recent_errors.append(f"{tool_name}: {result.error}")
        
        request_text = self.harness.state.run_brief.original_task or self.harness._current_user_task()
        compact_full_file = self._is_small_model()
        preview_chars = max(180, int(self.harness.context_policy.tool_result_inline_token_limit * 2))
        compact_content = (
            self.harness.artifact_store.compact_tool_message(
                artifact,
                result,
                request_text=request_text,
                inline_full_file=not compact_full_file,
                full_file_preview_chars=preview_chars if compact_full_file else None,
            )
            if artifact
            else str(result.output)
        )
        if tool_name in {"plan_set", "plan_step_update", "plan_request_execution", "plan_export"}:
            playbook_artifact_id = str(result.metadata.get("artifact_id", "") or "").strip()
            if playbook_artifact_id:
                compact_content = (
                    f"Plan playbook captured as Artifact {playbook_artifact_id}.\n\n{compact_content}"
                )

        self.harness._runlog(
            "artifact_created",
            "tool result processed",
            artifact_id=artifact.artifact_id if artifact else None,
            tool_name=tool_name,
            source=artifact.source if artifact else tool_name,
            size_bytes=artifact.size_bytes if artifact else 0,
            inline=bool(artifact and artifact.inline_content is not None),
        )

        msg = ConversationMessage(
            role="tool",
            name=tool_name,
            tool_call_id=tool_call_id,
            content=compact_content,
            metadata={"artifact_id": artifact.artifact_id} if artifact else {},
        )

        # Record tool history fingerprint
        args_str = json.dumps(result.metadata.get("arguments", {}), sort_keys=True)
        outcome = "success" if result.success else f"error:{result.error}"
        fingerprint = f"{tool_name}|{args_str}|{outcome}"
        self.harness.state.append_tool_history(fingerprint)

        return msg

    def maybe_reuse_file_read(self, *, tool_name: str, args: dict[str, Any]) -> ToolEnvelope | None:
        if tool_name != "file_read":
            return None
        cache = self.harness.state.scratchpad.get("file_read_cache")
        if not isinstance(cache, dict):
            return None
        cache_key = _file_read_cache_key(self.harness.state.cwd, args)
        if not cache_key:
            return None
        artifact_id = cache.get(cache_key)
        if not isinstance(artifact_id, str) or not artifact_id:
            return None
        artifact = self.harness.state.artifacts.get(artifact_id)
        if artifact is None:
            return None
        self.harness._runlog(
            "tool_cache_hit",
            "reusing prior file_read result",
            tool_name=tool_name,
            artifact_id=artifact_id,
            path=artifact.source,
        )
        return ToolEnvelope(
            success=True,
            output={
                "status": "cached",
                "artifact_id": artifact_id,
                "path": artifact.source,
                "summary": artifact.summary,
            },
            metadata={
                "cache_hit": True,
                "artifact_id": artifact_id,
                "path": artifact.source,
                "tool_name": tool_name,
            },
        )

    def compact_oversized_tool_messages(self, *, soft_limit: int) -> bool:
        from ..context.policy import estimate_text_tokens
        limit_threshold = 150 # _TOOL_MSG_COMPACT_THRESHOLD

        compacted_any = False
        for message in reversed(self.harness.state.recent_messages):
            if message.role != "tool":
                continue
            content = message.content or ""
            if estimate_text_tokens(content) <= limit_threshold:
                continue
            artifact_id = message.metadata.get("artifact_id") if message.metadata else None
            if not isinstance(artifact_id, str) or not artifact_id:
                char_cap = limit_threshold * 4
                if len(content) > char_cap:
                    message.content = content[:char_cap] + " [truncated]"
                    compacted_any = True
                continue
            artifact = self.harness.state.artifacts.get(artifact_id)
            if artifact is None:
                continue
            dummy_result = ToolEnvelope(
                success=True,
                output=None,
                error=None,
                metadata=dict(artifact.metadata or {}),
            )
            compact = self.harness.artifact_store.compact_tool_message(
                artifact,
                dummy_result,
                request_text=self.harness._current_user_task(),
            )
            if estimate_text_tokens(compact) < estimate_text_tokens(content):
                self.harness._runlog(
                    "budget_policy",
                    "compacted oversized tool message to artifact reference",
                    artifact_id=artifact_id,
                    original_tokens=estimate_text_tokens(content),
                    compacted_tokens=estimate_text_tokens(compact),
                )
                message.content = compact
                compacted_any = True
        return compacted_any


def _file_read_cache_key(cwd: str, payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    path = payload.get("path")
    if not isinstance(path, str) or not path.strip():
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(cwd) / candidate
    try:
        resolved = str(candidate.resolve())
    except Exception:
        resolved = str(candidate)
    start_line = payload.get("requested_start_line", payload.get("start_line"))
    end_line = payload.get("requested_end_line", payload.get("end_line"))
    max_bytes = payload.get("max_bytes", 100_000)
    return f"{resolved}|{start_line}|{end_line}|{max_bytes}"


def _consolidate_shell_attempt_family(
    *,
    state: Any, # LoopState
    artifact_id: str,
    result: ToolEnvelope,
    tool_name: str,
) -> None:
    arguments = result.metadata.get("arguments") if isinstance(result.metadata, dict) else None
    command = ""
    if isinstance(arguments, dict):
        command = str(arguments.get("command") or "").strip()
    if not command and isinstance(result.metadata, dict):
        command = str(result.metadata.get("command") or "").strip()
    if not command and tool_name == "ssh_exec":
         # Fallback to host for ssh_exec if no command is specified (e.g. just raw connection test)
         command = str(arguments.get("host") or result.metadata.get("host") or "").strip()
    
    if not command:
        return

    family_key = _shell_attempt_family_key(command)
    if not family_key:
        return

    family_state = state.scratchpad.setdefault("_shell_attempt_families", {})
    if not isinstance(family_state, dict):
        family_state = {}
        state.scratchpad["_shell_attempt_families"] = family_state

    record = family_state.get(family_key)
    if not isinstance(record, dict):
        record = {
            "tool_name": tool_name,
            "members": [],
            "canonical_artifact_id": None,
            "resolved": False,
        }
        family_state[family_key] = record

    members = record.get("members")
    if not isinstance(members, list):
        members = []
        record["members"] = members

    is_diagnostic = _shell_attempt_is_diagnostic(command)
    root = _shell_command_root(command)
    canonical_artifact_id = record.get("canonical_artifact_id")
    canonical_artifact_id = canonical_artifact_id if isinstance(canonical_artifact_id, str) and canonical_artifact_id else None

    artifact = state.artifacts.get(artifact_id)
    if artifact is None:
        return

    artifact.metadata["attempt_family"] = family_key
    if root:
        artifact.metadata["attempt_family_root"] = root

    if canonical_artifact_id:
        artifact.metadata["attempt_status"] = "redundant"
        artifact.metadata["superseded_by"] = canonical_artifact_id
        artifact.metadata["canonical_attempt_artifact_id"] = canonical_artifact_id
        members.append(artifact_id)
        return

    previous_members = [member_id for member_id in members if member_id != artifact_id]
    members.append(artifact_id)

    if result.success and not is_diagnostic:
        record["resolved"] = True
        record["canonical_artifact_id"] = artifact_id
        artifact.metadata["attempt_status"] = "canonical"
        artifact.metadata["canonical_attempt_artifact_id"] = artifact_id
        for member_id in previous_members:
            _mark_artifact_superseded(
                state=state,
                artifact_id=member_id,
                superseded_by=artifact_id,
                family_key=family_key,
                reason="resolved_by_success",
            )
        return

    artifact.metadata["attempt_status"] = "diagnostic" if is_diagnostic and result.success else "failed"
    for member_id in previous_members:
        _mark_artifact_superseded(
            state=state,
            artifact_id=member_id,
            superseded_by=artifact_id,
            family_key=family_key,
            reason="replaced_by_new_attempt",
        )


def _mark_artifact_superseded(
    *,
    state: Any,
    artifact_id: str,
    superseded_by: str,
    family_key: str,
    reason: str,
) -> None:
    artifact = state.artifacts.get(artifact_id)
    if artifact is None:
        return
    artifact.metadata["attempt_family"] = family_key
    artifact.metadata["superseded_by"] = superseded_by
    artifact.metadata["attempt_status"] = "superseded"
    artifact.metadata["superseded_reason"] = reason


def _shell_attempt_family_key(command: str) -> str | None:
    root = _shell_command_root(command)
    if not root:
        return None
    return f"shell_exec:{root}"


def _shell_attempt_is_diagnostic(command: str) -> bool:
    tokens = _shell_tokens(command)
    if not tokens:
        return False
    first = tokens[0].lower()
    wrapper_tokens = {"bash", "sh", "zsh", "dash", "ksh", "pwsh", "powershell", "cmd", "cmd.exe", "sudo"}
    if first in wrapper_tokens:
        inner = _shell_unwrap_command(tokens)
        return _shell_attempt_is_diagnostic(inner)
    lowered = [token.lower() for token in tokens]
    return any(token in {"-h", "--help", "/?"} for token in lowered) or "help" in lowered[1:]


def _shell_command_root(command: str) -> str | None:
    tokens = _shell_tokens(command)
    if not tokens:
        return None
    first = tokens[0].lower()
    wrapper_tokens = {"bash", "sh", "zsh", "dash", "ksh", "pwsh", "powershell", "cmd", "cmd.exe", "sudo"}
    if first in wrapper_tokens:
        inner = _shell_unwrap_command(tokens)
        if inner == command:
            return first
        return _shell_command_root(inner)
    for token in tokens:
        if token.startswith("-"):
            continue
        if "=" in token and token.split("=", 1)[0].isidentifier():
            continue
        return token.lower()
    return first


def _shell_unwrap_command(tokens: list[str]) -> str:
    if not tokens:
        return ""
    first = tokens[0].lower()
    if first == "sudo":
        inner_tokens = tokens[1:]
        while inner_tokens and inner_tokens[0].startswith("-"):
            inner_tokens = inner_tokens[1:]
        return " ".join(inner_tokens)
    if len(tokens) < 2:
        return tokens[0]
    if tokens[1] in {"-c", "-lc", "/c", "-Command", "-command"}:
        return " ".join(tokens[2:])
    return " ".join(tokens[1:])


def _shell_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


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
