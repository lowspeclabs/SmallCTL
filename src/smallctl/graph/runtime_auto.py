from __future__ import annotations

from ..interrupt_replies import is_interrupt_response
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
        from .runtime_specialized import PlanningGraphRuntime

        harness = self.deps.harness
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
            from .runtime_staged import StagedExecutionRuntime

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
