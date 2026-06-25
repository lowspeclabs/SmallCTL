from __future__ import annotations

import hashlib
from time import time
from typing import Any

from ..recovery_metrics import increment_metric
from ..recovery_schema import FailureEvent, ReflectionMemory, Subtask, SubtaskLedger
from ..logging_utils import runlog

_SEVERITY_ORDER = {"info": 0, "warning": 1, "recoverable": 2, "hard": 3}
_DEFAULT_MAX_ITEMS = 5

TEMPLATES = {
    "repeated_action": {
        "lesson": "The previous action repeated without producing new evidence.",
        "avoid": "Do not call the same tool with the same arguments again unless checking changed state.",
        "next": "Use existing evidence to take the next smallest different action.",
    },
    "wrong_path": {
        "lesson": "The chosen path was not verified or was relative to the wrong cwd/host.",
        "avoid": "Do not guess absolute paths or mix local and remote paths.",
        "next": "Run a narrow directory/file check in the correct scope, then retry with the verified path.",
    },
    "empty_write": {
        "lesson": "The write attempt produced empty or near-empty content.",
        "avoid": "Do not assume the write succeeded just because the tool returned.",
        "next": "Verify file size/content, then patch or retry the smallest write target.",
    },
    "verifier_failed": {
        "lesson": "The task is not complete because verification failed.",
        "avoid": "Do not call task_complete while the verifier is failing.",
        "next": "Read the failing output and patch one narrow cause, then rerun the smallest check.",
    },
    "test_failed": {
        "lesson": "The task is not complete because tests failed.",
        "avoid": "Do not treat a failing test run as acceptance evidence.",
        "next": "Patch one narrow test failure cause, then rerun the smallest relevant test.",
    },
    "tool_schema_invalid": {
        "lesson": "The previous tool call failed argument validation.",
        "avoid": "Do not repeat the same malformed tool arguments.",
        "next": "Use the tool schema and emit one minimal valid call.",
    },
    "write_session_stall": {
        "lesson": "The write session stopped progressing during verification, repair, or finalization.",
        "avoid": "Do not replay the same chunk or finalization attempt without new staged-file evidence.",
        "next": "Inspect the staged file and verifier output, then make the smallest local repair before retrying.",
    },
    "tool_plan_invalid": {
        "lesson": "The ToolPlan evidence planner did not produce a safe bounded read-only plan.",
        "avoid": "Do not dispatch unsafe or malformed evidence steps.",
        "next": "Fallback to normal loop or retry with fewer targeted read/search steps.",
    },
    "tool_plan_unsafe": {
        "lesson": "The ToolPlan evidence planner proposed a step outside the read-only safety policy.",
        "avoid": "Do not dispatch ToolPlan steps with unsafe paths, disabled web access, or non-read-only tools.",
        "next": "Fallback to normal loop or gather evidence with workspace-relative read/search steps only.",
    },
}


class ReflexionService:
    def __init__(self, harness: Any) -> None:
        self.harness = harness

    def maybe_create_reflection(
        self,
        failure: FailureEvent,
        ledger: SubtaskLedger | None = None,
    ) -> ReflectionMemory | None:
        config = getattr(self.harness, "config", None)
        if not bool(getattr(config, "reflexion_enabled", True)):
            return None
        min_severity = str(getattr(config, "reflexion_min_failure_severity", "warning") or "warning")
        if _SEVERITY_ORDER.get(failure.severity, 1) < _SEVERITY_ORDER.get(min_severity, 1):
            return None
        template = TEMPLATES.get(failure.failure_class, _generic_template(failure))
        state = getattr(self.harness, "state", None)
        if state is None:
            return None
        memory = getattr(state, "reflexion_memory", None)
        if not isinstance(memory, list):
            memory = []
            state.reflexion_memory = memory

        evidence_summary = _evidence_summary(failure)
        dedupe_key = _reflection_key(
            failure_class=failure.failure_class,
            subtask_id=failure.subtask_id,
            evidence_summary=evidence_summary,
        )
        for reflection in memory:
            if not isinstance(reflection, ReflectionMemory):
                continue
            if reflection.metadata.get("dedupe_key") != dedupe_key:
                continue
            reflection.score = float(reflection.score or 1.0) + 0.25
            reflection.timestamp = time()
            reflection.evidence_summary = evidence_summary
            increment_metric(state, "reflections_reinforced")
            return reflection

        reflection = ReflectionMemory(
            reflection_id="R-" + dedupe_key[:12],
            timestamp=time(),
            task_id=_task_id(state),
            failure_class=failure.failure_class,
            subtask_id=failure.subtask_id,
            lesson=str(template["lesson"]),
            avoid=str(template["avoid"]),
            next_safe_action=failure.suggested_next_action or str(template["next"]),
            evidence_summary=evidence_summary,
            metadata={"dedupe_key": dedupe_key, "source_event_id": failure.event_id},
        )
        memory.append(reflection)
        increment_metric(state, "reflections_created")
        limit = max(1, int(getattr(config, "reflexion_max_items", _DEFAULT_MAX_ITEMS) or _DEFAULT_MAX_ITEMS))
        memory.sort(key=lambda item: (float(getattr(item, "score", 0.0) or 0.0), float(getattr(item, "timestamp", 0.0) or 0.0)))
        del memory[:-limit]
        runlog(
        self.harness,
        "reflexion_created",
        "Reflexion memory updated",
        reflection_id=reflection.reflection_id,
        failure_class=failure.failure_class,
        source_event_id=failure.event_id,
    )
        return reflection

    def select_for_prompt(
        self,
        *,
        task_text: str,
        active_subtask: Subtask | None,
        limit: int,
    ) -> list[ReflectionMemory]:
        del task_text
        state = getattr(self.harness, "state", None)
        memory = getattr(state, "reflexion_memory", None) if state is not None else None
        if not isinstance(memory, list):
            return []
        active_id = str(getattr(active_subtask, "subtask_id", "") or "").strip()
        candidates = [
            item
            for item in memory
            if isinstance(item, ReflectionMemory)
            and (not active_id or not item.subtask_id or item.subtask_id == active_id)
        ]
        candidates.sort(
            key=lambda item: (
                float(item.score or 0.0),
                float(item.timestamp or 0.0),
            ),
            reverse=True,
        )
        return candidates[: max(0, int(limit or 0))]

    def record_used(self, reflection_ids: list[str]) -> None:
        wanted = {str(item).strip() for item in reflection_ids if str(item).strip()}
        if not wanted:
            return
        state = getattr(self.harness, "state", None)
        memory = getattr(state, "reflexion_memory", None) if state is not None else None
        if not isinstance(memory, list):
            return
        for reflection in memory:
            if isinstance(reflection, ReflectionMemory) and reflection.reflection_id in wanted:
                reflection.used_count = int(reflection.used_count or 0) + 1


def _generic_template(failure: FailureEvent) -> dict[str, str]:
    return {
        "lesson": f"The previous action hit {failure.failure_class}.",
        "avoid": "Do not repeat the same failing action without new evidence.",
        "next": failure.suggested_next_action or "Use the evidence to take the next smallest safe action.",
    }


def _evidence_summary(failure: FailureEvent) -> str:
    text = " | ".join(str(item).strip() for item in failure.evidence if str(item).strip())
    if not text:
        text = failure.message
    return text[:240]


def _reflection_key(*, failure_class: str, subtask_id: str | None, evidence_summary: str) -> str:
    normalized_evidence = " ".join(evidence_summary.lower().split())
    raw = f"{failure_class}|{subtask_id or ''}|{normalized_evidence}"
    return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()


def _task_id(state: Any) -> str | None:
    run_brief = getattr(state, "run_brief", None)
    task = str(getattr(run_brief, "original_task", "") or "").strip()
    if not task:
        return None
    return "task-" + hashlib.sha1(task.encode("utf-8", errors="replace")).hexdigest()[:12]
