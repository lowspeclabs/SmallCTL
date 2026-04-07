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
        self.harness.state.run_brief.current_phase_objective = self.harness.state.current_phase
        if not self.harness.state.working_memory.current_goal:
            self.harness.state.working_memory.current_goal = (
                self.harness.state.run_brief.current_phase_objective or self.harness.state.run_brief.original_task
            )
        if self.harness.state.active_plan is not None or self.harness.state.draft_plan is not None:
            self.harness.state.sync_plan_mirror()
        self._refresh_active_intent()
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
        task_guidance = self._next_action_for_task(original_task)

        if self.harness.state.current_phase == "verify" or has_recent_tool_evidence or has_known_facts:
            next_action = self._completion_next_action()
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

        from . import _trim_recent_messages_window
        self.harness.state.recent_messages = _trim_recent_messages_window(
            self.harness.state.recent_messages,
            limit=recent_messages_limit,
        )

    def _refresh_active_intent(self) -> None:
        task = self.harness.state.run_brief.original_task or self.harness._current_user_task()
        self.prime_write_policy(task)
        primary, secondary, tags = self._extract_intent_state(task)
        self.harness.state.active_intent = primary
        self.harness.state.secondary_intents = secondary
        self.harness.state.intent_tags = tags

    def _extract_intent_state(self, task: str) -> tuple[str, list[str], list[str]]:
        text = (task or "").lower()
        secondary: list[str] = []
        tags: list[str] = []
        requested_tool = self._infer_requested_tool_name(task)
        
        primary = "general_task"
        if requested_tool:
            primary = f"use_{requested_tool}"
            secondary.append("complete_validation_task")
            tags.append(requested_tool)
            if requested_tool in {"scratch_list", "cwd_get", "loop_status"}:
                secondary.append("call_zero_arg_tool")
            if requested_tool in {"task_complete", "task_fail", "ask_human"}:
                secondary.append("control_tool")
        elif any(token in text for token in {"inspect", "read", "grep", "find", "search", "list"}):
            primary = "inspect_repo"
            secondary.append("read_artifacts")
        elif any(token in text for token in {"write", "edit", "patch", "create", "update", "diff"}):
            primary = "write_file"
            secondary.append("mutate_repo")
        elif "contract" in text or "plan" in text:
            primary = "plan_execution"
            secondary.append("complete_validation_task")
        
        if self.harness.state.working_memory.failures:
            secondary.append("recover_from_validation_error")
            
        tags.extend(self._infer_environment_tags())
        tags.extend(self._infer_entity_tags(task))
        tags.extend([t for t in self.harness.state.working_memory.next_actions[-2:] if " " not in t][:2])
        
        return primary, clip_string_list(secondary, limit=3, item_char_limit=48)[0], clip_string_list(tags, limit=6, item_char_limit=64)[0]

    def _infer_environment_tags(self) -> list[str]:
        tags = [self.harness.provider_profile, self.harness.state.current_phase]
        cwd = self.harness.state.cwd.lower()
        if "localhost" in cwd:
            tags.append("localhost")
        if "scripts" in cwd:
            tags.append("scripts")
        return tags

    def _infer_entity_tags(self, task: str) -> list[str]:
        text = (task or "").lower()
        tags = []
        if "ansible" in text:
            tags.append("ansible")
        if "python" in text or ".py" in text:
            tags.append("python")
        if "bash" in text or ".sh" in text:
            tags.append("bash")
        return tags

    def _infer_requested_tool_name(self, task: str) -> str:
        text = (task or "").lower()
        creation_markers = (
            "build",
            "create",
            "make",
            "write",
            "generate",
            "implement",
            "save",
            "produce",
        )
        if (
            "script" in text
            and any(marker in text for marker in creation_markers)
        ) or any(ext in text for ext in (".py", ".sh", ".bash", ".ps1")):
            return "write_file"
        memory_markers = (
            "save this in memory",
            "save memory",
            "remember this",
            "store this in memory",
            "store this",
            "note this",
            "pin this",
            "persist this",
            "keep this in memory",
            "write this down",
        )
        if any(marker in text for marker in memory_markers):
            return "memory_update"
        if "read_file" in text or "cat" in text:
            return "read_file"
        if "write_file" in text:
            return "write_file"
        if hasattr(self.harness, "_looks_like_shell_request") and self.harness._looks_like_shell_request(task):
             return "shell_exec"
        return ""

    def _extract_memory_payload(self, task: str) -> str:
        text = (task or "").strip()
        if not text:
            return ""

        lowered = text.lower()
        prefixes = (
            "save this in memory",
            "store this in memory",
            "keep this in memory",
            "remember this",
            "note this",
            "pin this",
            "persist this",
            "write this down",
            "save memory",
            "store this",
            "remember",
            "note",
            "pin",
            "persist",
            "keep",
            "write down",
            "save",
            "store",
        )
        for prefix in prefixes:
            start = lowered.find(prefix)
            if start == -1:
                continue
            remainder = text[start + len(prefix):].strip()
            remainder = remainder.lstrip(":-— ")
            if remainder:
                return remainder
        return ""

    def _memory_fact_hint(self, task: str) -> str:
        payload = self._extract_memory_payload(task)
        if not payload:
            return ""
        clipped, _ = clip_text_value(payload, limit=180)
        return clipped

    def _completion_next_action(self) -> str:
        return "Decide whether the current evidence is sufficient; call task_complete when it is."

    def _next_action_for_task(self, task: str) -> str:
        memory_fact_hint = self._memory_fact_hint(task)
        if memory_fact_hint:
            return (
                "Call `memory_update(section='known_facts', content="
                f"{json.dumps(memory_fact_hint, ensure_ascii=True)})` to persist the fact."
            )
        return f"{self.harness.state.current_phase}: gather the next missing fact for {clip_text_value(task, limit=40)[0]}"

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
        
        op_notes = notes
        if not op_notes:
            if result.success:
                if tool_name == "task_complete":
                    op_notes = "Task finished successfully. Call task_complete when objectives met."
                else:
                    args_meta = result.metadata.get("arguments") or {}
                    if not args_meta:
                        op_notes = f"Successfully called {tool_name} with no arguments."
                    else:
                        op_notes = f"Successfully called {tool_name}. Key pattern: {list(args_meta.keys())}."
            else:
                if failure_mode == ZERO_ARG_TOOL_ARG_LEAK:
                    op_notes = f"Do not send placeholder arguments to {tool_name}. Call it empty."
                elif failure_mode == SCHEMA_VALIDATION_ERROR:
                    op_notes = f"Argument mismatch for {tool_name}: {result.error}"
                else:
                    op_notes = str(result.error or result.output or "").strip()

        correction_ids = [
            memory.memory_id
            for memory in self.harness.state.warm_experiences
            if memory.intent == self.harness.state.active_intent and memory.tool_name == tool_name and memory.outcome == "failure"
        ]
        memory = ExperienceMemory(
            memory_id=f"mem-{uuid.uuid4().hex[:10]}",
            tier="warm",
            source=source,
            run_id=self.harness.state.thread_id,
            phase=self.harness.state.current_phase,
            intent=self.harness.state.active_intent,
            intent_tags=list(self.harness.state.intent_tags),
            environment_tags=self._infer_environment_tags(),
            entity_tags=self._infer_entity_tags(self.harness.state.run_brief.original_task),
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
        lowered = (task or "").lower()
        memory_fact_hint = self._memory_fact_hint(task)
        if memory_fact_hint:
            return f"memory_update known_facts: {memory_fact_hint}"
        if "contract" in lowered or "plan" in lowered:
            return "high_fidelity"
        return "general"
