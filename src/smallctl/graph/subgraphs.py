from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from .checkpoint import create_graph_checkpointer

from ..context import ArtifactStore, ChildRunRequest, ChildRunResult
from ..state import LoopState, json_safe_value, ArtifactRecord, _coerce_artifact_record

if TYPE_CHECKING:
    from ..harness import Harness


class ChildSubgraphPayload(TypedDict, total=False):
    child_state: dict[str, Any]
    child_result: dict[str, Any] | None


class ChildSubgraphRunner:
    async def execute(
        self,
        *,
        parent: "Harness",
        request: ChildRunRequest,
        harness_factory: Callable[..., "Harness"] | None = None,
    ) -> ChildRunResult:
        child_state = parent.subtask_runner.create_child_state(
            parent_state=parent.state,
            request=request,
        )
        return await self._run_langgraph(
            parent=parent,
            child_state=child_state,
            request=request,
            harness_factory=harness_factory,
        )

    async def _run_langgraph(
        self,
        *,
        parent: "Harness",
        child_state: LoopState,
        request: ChildRunRequest,
        harness_factory: Callable[..., "Harness"] | None = None,
    ) -> ChildRunResult:
        compiled = self._build_compiled_subgraph(
            parent=parent,
            request=request,
            harness_factory=harness_factory,
        )
        values = await compiled.ainvoke(
            {"child_state": child_state.to_dict()},
            {
                "configurable": {
                    "thread_id": child_state.thread_id or uuid.uuid4().hex,
                }
            },
        )
        if not isinstance(values, dict):
            raise RuntimeError("Child subgraph ended without a terminal payload.")
        return self._inflate_child_result(values.get("child_result"))

    def _build_compiled_subgraph(
        self,
        *,
        parent: "Harness",
        request: ChildRunRequest,
        harness_factory: Callable[..., "Harness"] | None = None,
    ):
        builder = StateGraph(ChildSubgraphPayload)
        builder.add_node(
            "execute_child_task",
            self._make_execute_child_task_node(
                parent=parent,
                request=request,
                harness_factory=harness_factory,
            ),
        )
        builder.add_edge(START, "execute_child_task")
        builder.add_edge("execute_child_task", END)
        return builder.compile(checkpointer=self._get_checkpointer(parent))

    def _make_execute_child_task_node(
        self,
        *,
        parent: "Harness",
        request: ChildRunRequest,
        harness_factory: Callable[..., "Harness"] | None = None,
    ):
        async def execute_child_task_node(payload: ChildSubgraphPayload) -> ChildSubgraphPayload:
            raw_child_state = payload.get("child_state")
            child_state = LoopState.from_dict(raw_child_state if isinstance(raw_child_state, dict) else {})
            result = await self._execute_child_task(
                parent=parent,
                child_state=child_state,
                request=request,
                harness_factory=harness_factory,
            )
            return {"child_result": self._serialize_child_result(result)}

        return execute_child_task_node

    @staticmethod
    def _get_checkpointer(parent: "Harness"):
        saver = getattr(parent, "_child_graph_checkpointer", None)
        if saver is None:
            saver = create_graph_checkpointer(
                backend=getattr(parent, "graph_checkpointer", "memory"),
                path=getattr(parent, "graph_checkpoint_path", None),
            )
            setattr(parent, "_child_graph_checkpointer", saver)
        return saver

    @staticmethod
    async def _execute_child_task(
        *,
        parent: "Harness",
        child_state: LoopState,
        request: ChildRunRequest,
        harness_factory: Callable[..., "Harness"] | None = None,
    ) -> ChildRunResult:
        # Avoid artifact ID collisions by starting child IDs after parent's current IDs
        artifact_start_index = parent.artifact_store._next_index
        child = parent._create_child_harness(
            request=request,
            harness_factory=harness_factory,
            artifact_start_index=artifact_start_index,
        )
        child.state = child_state
        child.state.recent_message_limit = child.context_policy.recent_message_limit
        artifact_base_dir = Path(child.state.cwd).resolve() / ".smallctl" / "artifacts"
        child.artifact_store = ArtifactStore(
            base_dir=artifact_base_dir, 
            run_id=child.conversation_id,
            artifact_start_index=artifact_start_index,
        )
        parent._runlog(
            "subtask_child_created",
            "child harness created",
            conversation_id=child.conversation_id,
            cwd=child.state.cwd,
            phase=child.state.current_phase,
            runtime="graph",
        )
        result = await child.run_task(request.brief)
        return parent._build_subtask_result(
            child=child,
            request=request,
            result=result,
        )

    @staticmethod
    def _serialize_child_result(result: ChildRunResult) -> dict[str, Any]:
        return {
            "status": result.status,
            "summary": result.summary,
            "artifact_ids": list(result.artifact_ids),
            "files_touched": list(result.files_touched),
            "decisions": list(result.decisions),
            "remaining_plan": list(result.remaining_plan),
            "artifacts": {aid: json_safe_value(record) for aid, record in result.artifacts.items()},
            "metadata": _coerce_child_metadata(result.metadata),
        }

    @staticmethod
    def _inflate_child_result(payload: Any) -> ChildRunResult:
        if not isinstance(payload, dict):
            raise RuntimeError("Child subgraph ended without a child result.")
        metadata = _coerce_child_metadata(payload.get("metadata"))
        return ChildRunResult(
            status=str(payload.get("status", "unknown")),
            summary=str(payload.get("summary", "")),
            artifact_ids=_coerce_string_list(payload.get("artifact_ids")),
            files_touched=_coerce_string_list(payload.get("files_touched")),
            decisions=_coerce_string_list(payload.get("decisions")),
            remaining_plan=_coerce_string_list(payload.get("remaining_plan")),
            artifacts={
                str(k): _coerce_artifact_record(v, artifact_id=str(k))
                for k, v in (payload.get("artifacts") or {}).items()
            },
            metadata=metadata,
        )


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _coerce_child_metadata(value: Any) -> dict[str, Any]:
    normalized = json_safe_value(value or {})
    return normalized if isinstance(normalized, dict) else {}


__all__ = ["ChildSubgraphRunner"]
