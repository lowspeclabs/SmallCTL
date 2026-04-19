from __future__ import annotations

import uuid
import json
import logging
import hashlib
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ..state import (
    ExperienceMemory,
    _coerce_experience_memory,
    align_memory_entries,
    clip_string_list,
    clip_text_value,
    json_safe_value,
)
from ..normalization import dedupe_keep_tail
from ..models.tool_result import ToolEnvelope
from ..task_targets import extract_task_target_paths
from ..memory.taxonomy import (
    SCHEMA_VALIDATION_ERROR,
    ZERO_ARG_TOOL_ARG_LEAK,
    normalize_failure_mode,
)
from ..memory_namespace import infer_memory_namespace
from ..redaction import redact_sensitive_text
from .task_intent import (
    completion_next_action,
    derive_task_contract,
    extract_intent_state,
    infer_entity_tags,
    infer_environment_tags,
    next_action_for_task,
)
from .task_classifier import classify_task_mode
from .tool_message_compaction import trim_recent_messages_window

if TYPE_CHECKING:
    from ..harness import Harness

logger = logging.getLogger("smallctl.harness.memory")

_CODE_TARGET_SUFFIXES = {".py", ".js", ".ts", ".tsx", ".json", ".yaml", ".yml", ".toml", ".sh", ".md"}
_COMPLEX_WRITE_FEATURE_GROUPS: tuple[tuple[str, int, tuple[str, ...]], ...] = (
    ("tests", 2, ("unittest", "unit test", "test suite", "tests covering", "tests.", "with tests")),
    ("debug_trace", 1, ("--debug", "debug mode", "trace output", "step-by-step trace", "trace mode")),
    ("validation", 2, ("validate", "validation", "malformed", "invalid input", "malformed input")),
    ("dependency_logic", 2, ("dependency", "dependencies", "topological", "cycle", "cycles")),
    ("edge_cases", 1, ("edge case", "edge cases")),
    ("bugfix_loop", 2, ("bug you encounter and fix", "bug you encounter", "encounter and fix during development")),
)


def _tool_pattern_keys(tool_name: str, arguments: dict[str, Any]) -> list[str]:
    if tool_name == "ssh_exec":
        ordered_pairs = (
            ("host", "host"),
            ("user", "user"),
            ("password", "auth"),
            ("command", "command"),
        )
        return [label for key, label in ordered_pairs if arguments.get(key) not in (None, "")]
    return list(arguments.keys())


def assess_write_task_complexity(
    task: str,
    *,
    cwd: str | None = None,
    threshold: int = 5,
) -> dict[str, Any]:
    text = str(task or "").strip()
    lowered = text.lower()
    target_paths = extract_task_target_paths(text)
    code_targets = [
        path
        for path in target_paths
        if Path(path).suffix.lower() in _CODE_TARGET_SUFFIXES
    ]
    score = 0
    reasons: list[str] = []

    creation_markers = ("build", "create", "make", "write", "generate", "implement", "produce")
    if any(marker in lowered for marker in creation_markers):
        score += 1
        reasons.append("task requests file creation or implementation")
    if code_targets:
        score += 1
        reasons.append("task targets a source file")
    if "script" in lowered or any(Path(path).suffix.lower() == ".py" for path in code_targets):
        score += 1
        reasons.append("task describes a script-shaped implementation")

    if cwd and code_targets:
        missing_targets: list[str] = []
        base = Path(cwd)
        for path in code_targets:
            candidate = Path(path)
            resolved = candidate if candidate.is_absolute() else base / candidate
            if not resolved.exists():
                missing_targets.append(path)
        if missing_targets:
            score += 2
            reasons.append("task creates a new target file")

    feature_hits = 0
    for label, weight, keywords in _COMPLEX_WRITE_FEATURE_GROUPS:
        if any(keyword in lowered for keyword in keywords):
            score += weight
            feature_hits += 1
            reasons.append(f"task requires {label.replace('_', ' ')}")

    if feature_hits >= 3:
        score += 2
        reasons.append("task combines multiple independent requirements")

    force_chunk_targets = list(code_targets if code_targets else target_paths) if score >= threshold else []
    return {
        "task": text,
        "target_paths": target_paths,
        "code_targets": code_targets,
        "risk_score": score,
        "threshold": threshold,
        "reasons": reasons,
        "force_chunk_mode_targets": force_chunk_targets,
    }

class MemoryService:
    def __init__(self, harness: Harness):
        self.harness = harness

    def prime_write_policy(self, task: str) -> None:
        threshold = 5
        config = getattr(self.harness, "config", None)
        if config is not None:
            try:
                threshold = max(1, int(getattr(config, "task_risk_chunk_mode_score_threshold", threshold)))
            except (TypeError, ValueError):
                threshold = 5
        analysis = assess_write_task_complexity(
            task,
            cwd=getattr(self.harness.state, "cwd", None),
            threshold=threshold,
        )
        scratchpad = self.harness.state.scratchpad
        scratchpad["_write_task_complexity"] = analysis
        targets = analysis.get("force_chunk_mode_targets") or []
        if targets:
            scratchpad["_force_chunk_mode_targets"] = list(targets)
        else:
            scratchpad.pop("_force_chunk_mode_targets", None)

    def update_working_memory(self, recent_messages_limit: int) -> None:
        if not self.harness.state.run_brief.current_phase_objective:
            self.harness.state.run_brief.current_phase_objective = (
                self.harness.state.working_memory.current_goal
                or self.harness.state.run_brief.original_task
                or self.harness.state.current_phase
            )
        if not self.harness.state.working_memory.current_goal:
            self.harness.state.working_memory.current_goal = (
                self.harness.state.run_brief.current_phase_objective or self.harness.state.run_brief.original_task
            )
        if self.harness.state.active_plan is not None or self.harness.state.draft_plan is not None:
            self.harness.state.sync_plan_mirror()
        self._refresh_active_intent()
        scratchpad = self.harness.state.scratchpad
        previous_contract_phase = str(scratchpad.get("_last_contract_phase_seen", "") or "")
        current_contract_phase = str(self.harness.state.contract_phase() or "")
        if previous_contract_phase and current_contract_phase and previous_contract_phase != current_contract_phase:
            self._emit_context_invalidation(
                reason="phase_advanced",
                details={
                    "from_phase": previous_contract_phase,
                    "to_phase": current_contract_phase,
                    "state_change": f"Phase advanced: {previous_contract_phase} -> {current_contract_phase}",
                },
            )
        scratchpad["_last_contract_phase_seen"] = current_contract_phase

        previous_environment = str(scratchpad.get("_last_environment_fingerprint", "") or "")
        current_environment = self._context_environment_fingerprint()
        if previous_environment and current_environment and previous_environment != current_environment:
            self._emit_context_invalidation(
                reason="environment_changed",
                details={
                    "from_environment": previous_environment,
                    "to_environment": current_environment,
                    "state_change": "Execution environment changed",
                },
            )
        scratchpad["_last_environment_fingerprint"] = current_environment

        previous_write_target = str(scratchpad.get("_last_write_session_target", "") or "")
        current_write_target = ""
        if self.harness.state.write_session is not None:
            current_write_target = str(self.harness.state.write_session.write_target_path or "").strip()
        if previous_write_target and previous_write_target != current_write_target:
            self._emit_context_invalidation(
                reason="write_session_target_changed",
                paths=[previous_write_target, current_write_target],
                details={
                    "from_target": previous_write_target,
                    "to_target": current_write_target,
                    "state_change": "Active write-session target changed",
                },
            )
        scratchpad["_last_write_session_target"] = current_write_target

        self.harness.state.prune_stale_meta(limit=self.harness.context_policy.memory_staleness_step_limit)
        self.harness.state.align_meta_to_content()
        
        self.harness.state.working_memory.open_questions = clip_string_list(
            self.harness.state.working_memory.open_questions,
            limit=4,
            item_char_limit=240,
        )[0]
        self.harness.state.working_memory.plan = clip_string_list(
            self.harness.state.working_memory.plan,
            limit=10,
            item_char_limit=400,
        )[0]
        self.harness.state.working_memory.decisions = clip_string_list(
            self.harness.state.working_memory.decisions,
            limit=10,
            item_char_limit=400,
        )[0]
        self.harness.state.working_memory.known_facts = clip_string_list(
            self.harness.state.working_memory.known_facts,
            limit=12,
            item_char_limit=320,
        )[0]
        self.harness.state.working_memory.failures = clip_string_list(
            self.harness.state.working_memory.failures,
            limit=8,
            item_char_limit=320,
        )[0]
        
        original_task = self.harness.state.run_brief.original_task or ""
        has_recent_tool_evidence = any(message.role == "tool" for message in self.harness.state.recent_messages[-4:])
        has_known_facts = bool(self.harness.state.working_memory.known_facts)
        task_guidance = next_action_for_task(self.harness, original_task)

        if self.harness.state.current_phase == "verify" or has_recent_tool_evidence or has_known_facts:
            next_action = completion_next_action()
            if self.harness.state.working_memory.next_actions and self.harness.state.working_memory.next_actions[-1] == task_guidance:
                self.harness.state.working_memory.next_actions.pop()
        else:
            next_action = task_guidance
            
        next_action, _ = clip_text_value(next_action, limit=240)
        self.harness.state.working_memory.next_actions = clip_string_list(
            dedupe_keep_tail(self.harness.state.working_memory.next_actions + [next_action], limit=6),
            limit=6,
            item_char_limit=240,
        )[0]
        self.harness.state.working_memory.next_action_meta = align_memory_entries(
            self.harness.state.working_memory.next_actions,
            self.harness.state.working_memory.next_action_meta,
            current_step=self.harness.state.step_count,
            current_phase=self.harness.state.current_phase,
            confidence=0.7,
        )

        self.harness.state.recent_messages = trim_recent_messages_window(
            self.harness.state.recent_messages,
            limit=recent_messages_limit,
        )

        compact_oversized = getattr(self.harness, "_compact_oversized_tool_messages", None)
        if callable(compact_oversized):
            soft_limit = (
                self.harness.context_policy.soft_prompt_token_limit
                or self.harness.context_policy.max_prompt_tokens
                or 0
            )
            compact_oversized(soft_limit=soft_limit)

    def _refresh_active_intent(self) -> None:
        task = self.harness.state.run_brief.original_task or self.harness._current_user_task()
        self.prime_write_policy(task)
        self.harness.state.task_mode = classify_task_mode(task)
        primary, secondary, tags = extract_intent_state(self.harness, task)
        self.harness.state.active_intent = primary
        self.harness.state.secondary_intents = secondary
        self.harness.state.intent_tags = tags

    def record_experience(
        self,
        *,
        tool_name: str,
        result: ToolEnvelope,
        evidence_refs: list[str] | None = None,
        notes: str = "",
        source: str = "observed",
    ) -> ExperienceMemory:
        failure_mode = self._normalize_failure_mode(result.error, tool_name=tool_name, success=result.success)
        outcome = "success" if result.success else "failure"
        confidence = 0.85 if result.success else 0.60
        if tool_name == "task_complete" and result.success:
            confidence = 0.95
        if tool_name == "artifact_print" and result.success:
            confidence = 0.55
        
        op_notes = notes
        if not op_notes:
            if result.success:
                if tool_name == "task_complete":
                    op_notes = "Task finished successfully. Call task_complete when objectives met."
                elif tool_name == "artifact_print":
                    op_notes = (
                        "Displayed artifact contents for inspection. "
                        "Treat this as already-shown evidence, not a reusable progress step."
                    )
                else:
                    args_meta = result.metadata.get("arguments") or {}
                    if not args_meta:
                        op_notes = f"Successfully called {tool_name} with no arguments."
                    else:
                        op_notes = f"Successfully called {tool_name}. Key pattern: {_tool_pattern_keys(tool_name, args_meta)}."
            else:
                if failure_mode == ZERO_ARG_TOOL_ARG_LEAK:
                    op_notes = f"Do not send placeholder arguments to {tool_name}. Call it empty."
                elif failure_mode == SCHEMA_VALIDATION_ERROR:
                    op_notes = f"Argument mismatch for {tool_name}: {result.error}"
                else:
                    op_notes = str(result.error or result.output or "").strip()
        op_notes = redact_sensitive_text(op_notes)

        correction_ids = [
            memory.memory_id
            for memory in self.harness.state.warm_experiences
            if memory.intent == self.harness.state.active_intent and memory.tool_name == tool_name and memory.outcome == "failure"
        ]
        task_mode = str(getattr(self.harness.state, "task_mode", "") or "").strip().lower()
        if not task_mode:
            task_mode = classify_task_mode(self.harness.state.run_brief.original_task or "")
        intent_tags = list(self.harness.state.intent_tags)
        environment_tags = infer_environment_tags(self.harness)
        entity_tags = infer_entity_tags(self.harness.state.run_brief.original_task)
        namespace = infer_memory_namespace(
            task_mode=task_mode,
            tool_name=tool_name,
            intent=self.harness.state.active_intent,
            intent_tags=intent_tags,
            environment_tags=environment_tags,
            entity_tags=entity_tags,
            notes=op_notes,
            original_task=self.harness.state.run_brief.original_task,
        )
        memory = ExperienceMemory(
            memory_id=f"mem-{uuid.uuid4().hex[:10]}",
            tier="warm",
            source=source,
            run_id=self.harness.state.thread_id,
            phase=self.harness.state.current_phase,
            intent=self.harness.state.active_intent,
            namespace=namespace,
            intent_tags=intent_tags,
            environment_tags=environment_tags,
            entity_tags=entity_tags,
            action_type="tool_call",
            tool_name=tool_name,
            arguments_fingerprint=self._argument_fingerprint(result.metadata.get("arguments")),
            outcome=outcome,
            failure_mode=failure_mode,
            confidence=confidence,
            notes=op_notes,
            evidence_refs=evidence_refs or [],
            supersedes=correction_ids if result.success else [],
        )
        stored = self.harness.state.upsert_experience(memory)
        self.harness.warm_memory_store.upsert(stored)
        self._reinforce_retrieved_experiences(tool_name=tool_name, success=result.success)
        if stored.confidence >= 0.9 or stored.reuse_count >= 3:
            promoted = _coerce_experience_memory(json_safe_value(stored))
            promoted.tier = "cold"
            self.harness.cold_memory_store.upsert(promoted)
        return stored

    def _normalize_failure_mode(self, error: Any, *, tool_name: str, success: bool) -> str:
        return normalize_failure_mode(error, tool_name=tool_name, success=success)

    def _reinforce_retrieved_experiences(self, *, tool_name: str, success: bool) -> None:
        if not self.harness.state.retrieved_experience_ids:
            return
        for memory in self.harness.state.warm_experiences:
            if memory.memory_id in self.harness.state.retrieved_experience_ids and memory.tool_name == tool_name:
                if memory.tool_name == "artifact_print":
                    continue
                self.harness.state.reinforce_experience(memory.memory_id, success=success)
                self.harness.warm_memory_store.upsert(memory)
                if memory.confidence >= 0.9 or memory.reuse_count >= 3:
                     promoted = _coerce_experience_memory(json_safe_value(memory))
                     promoted.tier = "cold"
                     self.harness.cold_memory_store.upsert(promoted)

    def record_terminal_experience(self, result: dict[str, Any]) -> None:
        status = str(result.get("status", "") or "")
        if not status:
            return
        success = status in {"completed", "chat_completed"}
        reason = str(result.get("reason", "") or result.get("message", "") or status)
        from ..models.tool_result import ToolEnvelope
        payload = ToolEnvelope(
            success=success,
            output={"status": status, "message": reason},
            error=None if success else reason,
            metadata={"status": status},
        )
        self.record_experience(
            tool_name="task_complete" if success else "task_fail",
            result=payload,
            notes=reason,
        )

    def _argument_fingerprint(self, arguments: Any) -> str:
        payload = json.dumps(json_safe_value(arguments or {}), sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def derive_task_contract(self, task: str) -> str:
        return derive_task_contract(task)

    def _emit_context_invalidation(
        self,
        *,
        reason: str,
        paths: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        event = self.harness.state.invalidate_context(
            reason=reason,
            paths=paths,
            details=details,
        )
        runlog = getattr(self.harness, "_runlog", None)
        if callable(runlog):
            runlog(
                "context_invalidated",
                "context invalidation applied",
                reason=event.get("reason", reason),
                paths=event.get("paths", []),
                invalidated_fact_count=event.get("invalidated_fact_count", 0),
                invalidated_memory_count=event.get("invalidated_memory_count", 0),
                invalidated_facts=event.get("invalidated_facts", []),
                invalidated_memory_ids=event.get("invalidated_memory_ids", []),
                invalidated_turn_bundle_count=event.get("invalidated_turn_bundle_count", 0),
                invalidated_turn_bundle_ids=event.get("invalidated_turn_bundle_ids", []),
                invalidated_brief_count=event.get("invalidated_brief_count", 0),
                invalidated_brief_ids=event.get("invalidated_brief_ids", []),
                invalidated_summary_count=event.get("invalidated_summary_count", 0),
                invalidated_summary_ids=event.get("invalidated_summary_ids", []),
                invalidated_artifact_count=event.get("invalidated_artifact_count", 0),
                invalidated_artifact_ids=event.get("invalidated_artifact_ids", []),
                invalidated_observation_count=event.get("invalidated_observation_count", 0),
                invalidated_observation_ids=event.get("invalidated_observation_ids", []),
                details=event.get("details", {}),
            )

    def _context_environment_fingerprint(self) -> str:
        phase = str(self.harness.state.current_phase or "").strip().lower()
        cwd = str(self.harness.state.cwd or "").strip().lower()
        task_mode = str(self.harness.state.task_mode or "").strip().lower()
        ssh_targets = self.harness.state.scratchpad.get("_session_ssh_targets")
        ssh_hosts: list[str] = []
        if isinstance(ssh_targets, dict):
            ssh_hosts = sorted(str(host).strip().lower() for host in ssh_targets.keys() if str(host).strip())
        return "|".join([phase, task_mode, cwd, ",".join(ssh_hosts)])
