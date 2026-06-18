from __future__ import annotations

from ..harness.run_mode import _has_plan_execution_approval_context
from ..harness.task_classifier import looks_like_tool_plan_candidate
from ..interrupt_replies import is_interrupt_response, is_plan_approval_reply
from .deps import GraphRuntimeDeps


class AutoGraphRuntime:
    def __init__(self, deps: GraphRuntimeDeps) -> None:
        self.deps = deps

    @classmethod
    def from_harness(
        cls,
        harness: object,
        *,
        event_handler: object = None,
    ) -> "AutoGraphRuntime":
        return cls(
            GraphRuntimeDeps(
                harness=harness,
                event_handler=event_handler,
            ),
        )

    async def run(self, task: str) -> dict[str, object]:
        from .runtime import ChatGraphRuntime, LoopGraphRuntime
        from .runtime_staged import StagedExecutionRuntime
        from .runtime_specialized import IndexerGraphRuntime, PlanningGraphRuntime, ToolPlanRuntime

        harness = self.deps.harness
        # Defensive fallback: if the user is providing a short plan-approval reply
        # but the harness has lost or failed to recognize the pending interrupt,
        # still route to the planning resume path when plan-approval context exists.
        if is_plan_approval_reply(task) and _has_plan_execution_approval_context(harness):
            harness._runlog(
                "runtime_route",
                "routing task to interrupt resume (plan approval fallback)",
                interrupt_kind="plan_execute_approval",
                execution_path=self._execution_path(),
            )
            return await PlanningGraphRuntime.from_harness(
                harness,
                event_handler=self.deps.event_handler,
            ).resume(task)

        if harness.has_pending_interrupt():
            pending = harness.get_pending_interrupt() or {}
            interrupt_kind = str(pending.get("kind") or "ask_human")
            if is_interrupt_response(pending, task):
                harness._runlog(
                    "runtime_route",
                    "routing task to interrupt resume",
                    interrupt_kind=interrupt_kind,
                    execution_path=self._execution_path(),
                )
                if interrupt_kind == "plan_execute_approval":
                    return await PlanningGraphRuntime.from_harness(
                        harness,
                        event_handler=self.deps.event_handler,
                    ).resume(task)
                if self._should_resume_staged_interrupt(interrupt_kind):
                    return await StagedExecutionRuntime.from_harness(
                        harness,
                        event_handler=self.deps.event_handler,
                    ).resume(task)
                return await LoopGraphRuntime.from_harness(
                    harness,
                    event_handler=self.deps.event_handler,
                ).resume(task)
            harness._runlog(
                "runtime_route",
                "replacing pending interrupt with new task",
                interrupt_kind=interrupt_kind,
                interrupt_plan_id=str(pending.get("plan_id") or ""),
                replacement_task=str(task or "")[:200],
                execution_path=self._execution_path(),
            )
            harness.state.pending_interrupt = None
            if interrupt_kind == "plan_execute_approval":
                harness.state.planner_interrupt = None
        explicit_mode = str(getattr(getattr(harness, "config", None), "run_mode", "auto") or "auto").strip().lower()
        if explicit_mode and explicit_mode != "auto":
            harness._runlog(
                "runtime_route",
                "routing task to explicit runtime",
                mode=explicit_mode,
                execution_path=self._execution_path(),
            )
            if explicit_mode == "planning":
                return await PlanningGraphRuntime.from_harness(
                    harness,
                    event_handler=self.deps.event_handler,
                ).run(task)
            if explicit_mode == "chat":
                return await ChatGraphRuntime.from_harness(
                    harness,
                    event_handler=self.deps.event_handler,
                ).run(task)
            if explicit_mode == "indexer":
                return await IndexerGraphRuntime.from_harness(
                    harness,
                    event_handler=self.deps.event_handler,
                ).run(task)
            if explicit_mode == "tool_plan":
                return await ToolPlanRuntime.from_harness(
                    harness,
                    event_handler=self.deps.event_handler,
                ).run(task)
            return await LoopGraphRuntime.from_harness(
                harness,
                event_handler=self.deps.event_handler,
            ).run(task)
        config = getattr(harness, "config", None)
        if (
            bool(getattr(config, "tool_plan_runtime_enabled", False))
            and bool(getattr(config, "tool_plan_auto_select", False))
            and looks_like_tool_plan_candidate(task)
        ):
            harness._runlog(
                "runtime_route",
                "auto-selecting tool_plan runtime",
                mode="tool_plan",
                execution_path=self._execution_path(),
            )
            return await ToolPlanRuntime.from_harness(
                harness,
                event_handler=self.deps.event_handler,
            ).run(task)
        mode = await harness.decide_run_mode(task)
        harness._runlog(
            "runtime_route",
            "routing task to runtime",
            mode=mode,
            execution_path=self._execution_path(),
        )
        if mode == "planning":
            return await PlanningGraphRuntime.from_harness(
                harness,
                event_handler=self.deps.event_handler,
            ).run(task)
        if mode == "chat":
            return await ChatGraphRuntime.from_harness(
                harness,
                event_handler=self.deps.event_handler,
            ).run(task)
        # Auto-transition to staged execution when an approved plan exists
        plan = getattr(harness.state, "active_plan", None) or getattr(harness.state, "draft_plan", None)
        if (
            plan is not None
            and getattr(plan, "approved", False)
            and bool(getattr(getattr(harness, "config", None), "staged_execution_enabled", False))
            and not getattr(harness.state, "plan_execution_mode", False)
        ):
            harness._runlog(
                "runtime_route",
                "auto-transitioning loop runtime to staged execution",
                plan_id=getattr(plan, "plan_id", ""),
            )
            return await StagedExecutionRuntime.from_harness(
                harness,
                event_handler=self.deps.event_handler,
            ).run(plan.goal or task)
        return await LoopGraphRuntime.from_harness(
            harness,
            event_handler=self.deps.event_handler,
        ).run(task)

    def _execution_path(self) -> str:
        return "compiled"

    def _should_resume_staged_interrupt(self, interrupt_kind: str) -> bool:
        harness = self.deps.harness
        if interrupt_kind == "staged_step_blocked":
            return True
        return bool(
            getattr(harness.state, "plan_execution_mode", False)
            and getattr(getattr(harness, "config", None), "staged_execution_enabled", False)
        )
