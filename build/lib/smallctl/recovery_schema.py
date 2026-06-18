from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any, Literal


FailureSeverity = Literal["info", "warning", "recoverable", "hard"]
SubtaskStatus = Literal["pending", "active", "blocked", "done", "failed", "abandoned"]


@dataclass
class FailureEvent:
    event_id: str
    timestamp: float
    failure_class: str
    severity: FailureSeverity
    source: str
    message: str
    evidence: list[str] = field(default_factory=list)
    fama_kind: str | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    operation_id: str | None = None
    subtask_id: str | None = None
    suggested_next_action: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflectionMemory:
    reflection_id: str
    timestamp: float
    task_id: str | None
    failure_class: str
    subtask_id: str | None
    lesson: str
    avoid: str
    next_safe_action: str
    evidence_summary: str
    score: float = 1.0
    used_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Subtask:
    subtask_id: str
    title: str
    goal: str
    status: SubtaskStatus = "pending"
    acceptance: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    next_action: str | None = None
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)
    attempts: int = 0
    failure_classes: list[str] = field(default_factory=list)


@dataclass
class SubtaskLedger:
    task_id: str | None
    subtasks: list[Subtask] = field(default_factory=list)
    active_subtask_id: str | None = None

    def active(self) -> Subtask | None:
        for task in self.subtasks:
            if task.subtask_id == self.active_subtask_id:
                return task
        return None
