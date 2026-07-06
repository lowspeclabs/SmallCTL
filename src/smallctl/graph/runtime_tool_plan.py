from __future__ import annotations

from time import perf_counter
from types import SimpleNamespace
from typing import Any

from langgraph.graph import END

from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..context.rewoo_lanes import ReWOOLaneCompiler
from ..context.policy import estimate_text_tokens
from ..recovery_metrics import increment_metric, recovery_metrics
from .interpret_nodes import interpret_model_output
from .lifecycle_nodes import initialize_loop_run, prepare_loop_step, prepare_prompt, select_loop_tools
from .model_call_nodes import model_call
from .model_stream import process_model_stream
from .runtime import LoopGraphRuntime
from .runtime_base import (
    LoopGraphPayload,
    RuntimeGraphSpec,
    apply_outcomes_node,
    interpret_node,
    load_runtime_state,
    model_call_node,
    prepare_prompt_node,
    route_if_final_else,
    route_if_final_else_pending_else,
    route_if_interrupt_else_final_else_pending_else,
    serialize_runtime_state,
)
from .state import GraphRunState, inflate_graph_state
from .tool_execution_nodes import dispatch_tools, persist_tool_results
from .tool_outcomes import apply_tool_outcomes
from .tool_plan_executor import prepare_tool_plan_dispatch
from .tool_plan_observations import (
    attach_tool_plan_observation_evidence,
    build_tool_plan_observations,
    summarize_tool_plan_observations,
    render_tool_plan_observations,
)
from .tool_plan_parser import parse_tool_plan, summarize_tool_plan_json
from .tool_plan_prompts import build_tool_plan_planner_prompt, build_tool_plan_solver_system_suffix
from .tool_plan_helpers import (
    _MUTATING_SOLVER_TOOLS,
    _TERMINAL_SOLVER_TOOLS,
    _coerce_tool_plan,
    _rewoo_frame_budget,
    _rewoo_role_enabled,
    _select_no_tools,
    _select_solver_tools,
    _tool_plan_config,
    _tool_plan_observation_budget,
)
from .runtime_tool_plan_support import (
    _add_latency_metric,
    _attach_tool_plan_evidence,
    _record_tool_plan_failure,
    _record_tool_plan_planner_metadata,
    _record_tool_plan_tokens,
)
from .tool_plan_safety import validate_tool_plan
from .tool_dag import build_execution_dag
from .tool_dag_executor import dispatch_tool_dag
from .tool_dag_safety import NonParallelizableStepInDAGError, assert_parallelizable_steps
from ..harness.refine_service import RefineService
from ..harness.trajectory_recorder import TrajectoryRecorder

async def _prepare_planner_prompt(graph_state: GraphRunState, deps: Any) -> list[dict[str, Any]] | None:
    harness = deps.harness
    task = str(
        harness.state.run_brief.original_task
        or getattr(harness.state.run_brief, "effective_task", "")
        or ""
    )
    max_steps = max(1, int(_tool_plan_config(deps, "tool_plan_max_steps", 6) or 6))
    context_frame = ""
    if _rewoo_role_enabled(deps, "rewoo_planner_frame_enabled"):
        try:
            compiler = ReWOOLaneCompiler(getattr(harness, "context_policy", None))
            frame = compiler.compile(
                state=harness.state,
                role="planner",
                token_budget=_rewoo_frame_budget(deps),
            )
            context_frame = compiler.render(frame, token_budget=_rewoo_frame_budget(deps))
            harness.state.scratchpad["_rewoo_planner_drop_log"] = frame.drop_log
            if frame.drop_log:
                harness._runlog(
                    "rewoo_planner_drop_log",
                    "ReWOO planner frame dropped items",
                    drops=[{"lane": d.lane, "reason": d.reason, "count": d.dropped_count} for d in frame.drop_log],
                )
        except Exception as exc:
            harness._runlog("rewoo_planner_frame_failed", "planner frame compilation failed", error=str(exc))
            context_frame = ""
    prompt = build_tool_plan_planner_prompt(
        task=task,
        max_steps=max_steps,
        max_parallel=int(_tool_plan_config(deps, "tool_dag_max_parallel", 4) or 4),
        context_frame=context_frame,
    )
    repair_nudge = str(harness.state.scratchpad.pop("_tool_plan_repair_nudge", "") or "").strip()
    if repair_nudge:
        prompt = f"{prompt}\n\nRepair previous invalid ToolPlan output:\n{repair_nudge}\nReturn ONLY corrected JSON."
    return [{"role": "system", "content": prompt}]


async def _prepare_solver_prompt(graph_state: GraphRunState, deps: Any) -> list[dict[str, Any]] | None:
    harness = deps.harness
    if _rewoo_role_enabled(deps, "rewoo_solver_frame_enabled"):
        try:
            observations = harness.state.scratchpad.get("_tool_plan_observations")
            if not isinstance(observations, list):
                observations = []
            compiler = ReWOOLaneCompiler(getattr(harness, "context_policy", None))
            frame = compiler.compile(
                state=harness.state,
                role="solver",
                tool_plan_observations=observations,
                token_budget=_rewoo_frame_budget(deps),
            )
            frame_text = compiler.render(frame, token_budget=_rewoo_frame_budget(deps))
            harness.state.scratchpad["_rewoo_solver_drop_log"] = frame.drop_log
            if frame.drop_log:
                harness._runlog(
                    "rewoo_solver_drop_log",
                    "ReWOO solver frame dropped items",
                    drops=[{"lane": d.lane, "reason": d.reason, "count": d.dropped_count} for d in frame.drop_log],
                )
            fresh_output_limit = max(1, int(_tool_plan_config(deps, "tool_plan_solver_fresh_output_limit", 1200) or 1200))
            task = str(
                harness.state.run_brief.original_task
                or getattr(harness.state.run_brief, "effective_task", "")
                or ""
            )
            return [
                {
                    "role": "system",
                    "content": build_tool_plan_solver_system_suffix(
                        frame_text,
                        fresh_output_limit=fresh_output_limit,
                    ),
                },
                {"role": "user", "content": task or "Use the ReWOO evidence frame to decide the next action."},
            ]
        except Exception as exc:
            harness._runlog("rewoo_solver_frame_failed", "solver frame compilation failed", error=str(exc))
    else:
        observations_text = str(harness.state.scratchpad.pop("_tool_plan_observations_text", "") or "").strip()
        if observations_text:
            fresh_output_limit = max(1, int(_tool_plan_config(deps, "tool_plan_solver_fresh_output_limit", 1200) or 1200))
            harness.state.append_message(
                ConversationMessage(
                    role="system",
                    content=build_tool_plan_solver_system_suffix(
                        observations_text,
                        fresh_output_limit=fresh_output_limit,
                    ),
                    metadata={"tool_plan_observations": True},
                )
            )
    return await prepare_prompt(graph_state, deps)


class ToolPlanRuntime(LoopGraphRuntime):
    _run_mode = "tool_plan"
    _run_execution_message = "executing tool_plan runtime"
    _empty_result_message = "ToolPlan graph ended without a terminal result."

    GRAPH_SPEC = RuntimeGraphSpec(
        node_map={
            "initialize_run": "_initialize_run_node",
            "prepare_tool_plan_prompt": "_prepare_tool_plan_prompt_node",
            "planner_model_call": "_planner_model_call_node",
            "parse_and_validate_tool_plan": "_parse_and_validate_tool_plan_node",
            "prepare_tool_plan_dispatch": "_prepare_tool_plan_dispatch_node",
            "dispatch_tools": "_dispatch_tools_node",
            "persist_tool_results": "_persist_tool_results_node",
            "compress_observations": "_compress_observations_node",
            "prepare_solver_prompt": "_prepare_solver_prompt_node",
            "solver_model_call": "_solver_model_call_node",
            "interpret_solver_output": "_interpret_solver_output_node",
            "prepare_step": "_prepare_step_node",
            "prepare_prompt": "_prepare_prompt_node",
            "model_call": "_model_call_node",
            "interpret_model_output": "_interpret_model_output_node",
            "apply_tool_outcomes": "_apply_tool_outcomes_node",
            "interrupt_for_human": "_interrupt_for_human_node",
        },
        edge_map={
            "initialize_run": ("_route_after_initialize", {"prepare_tool_plan_prompt": "prepare_tool_plan_prompt", END: END}),
            "prepare_tool_plan_prompt": ("_route_after_tool_plan_prompt", {"planner_model_call": "planner_model_call", END: END}),
            "planner_model_call": ("_route_after_planner_model_call", {"parse_and_validate_tool_plan": "parse_and_validate_tool_plan", END: END}),
            "parse_and_validate_tool_plan": (
                "_route_after_parse",
                {
                    "prepare_tool_plan_prompt": "prepare_tool_plan_prompt",
                    "prepare_tool_plan_dispatch": "prepare_tool_plan_dispatch",
                    "prepare_solver_prompt": "prepare_solver_prompt",
                    "prepare_step": "prepare_step",
                    END: END,
                },
            ),
            "prepare_tool_plan_dispatch": (
                "_route_after_prepare_dispatch",
                {"dispatch_tools": "dispatch_tools", "prepare_step": "prepare_step", END: END},
            ),
            "dispatch_tools": ("_route_after_dispatch", {"persist_tool_results": "persist_tool_results", END: END}),
            "compress_observations": (
                "_route_after_compress",
                {
                    "prepare_solver_prompt": "prepare_solver_prompt",
                    "prepare_step": "prepare_step",
                    "apply_tool_outcomes": "apply_tool_outcomes",
                    END: END,
                },
            ),
            "prepare_solver_prompt": ("_route_after_solver_prompt", {"solver_model_call": "solver_model_call", END: END}),
            "solver_model_call": ("_route_after_solver_model_call", {"interpret_solver_output": "interpret_solver_output", END: END}),
            "interpret_solver_output": (
                "_route_after_interpret",
                {"dispatch_tools": "dispatch_tools", "prepare_step": "prepare_step", END: END},
            ),
            "prepare_step": (
                "_route_after_prepare_step",
                {"prepare_prompt": "prepare_prompt", "dispatch_tools": "dispatch_tools", END: END},
            ),
            "prepare_prompt": ("_route_after_prepare_prompt", {"model_call": "model_call", END: END}),
            "model_call": ("_route_after_model_call", {"interpret_model_output": "interpret_model_output", END: END}),
            "interpret_model_output": (
                "_route_after_interpret",
                {"dispatch_tools": "dispatch_tools", "prepare_step": "prepare_step", END: END},
            ),
            "apply_tool_outcomes": (
                "_route_after_apply",
                {
                    "dispatch_tools": "dispatch_tools",
                    "prepare_step": "prepare_step",
                    "interrupt_for_human": "interrupt_for_human",
                    END: END,
                },
            ),
        },
        static_edges=[
            ("persist_tool_results", "compress_observations"),
            ("interrupt_for_human", "prepare_step"),
        ],
    )

    async def run(self, task: str) -> dict[str, object]:
        result = await super().run(task)
        self._restore_tool_plan_active_subtask()
        return result

    def _restore_tool_plan_active_subtask(self) -> None:
        state = getattr(self.deps.harness, "state", None)
        ledger = getattr(state, "subtask_ledger", None)
        if ledger is None:
            return
        if ledger.active() is not None:
            return
        for subtask in getattr(ledger, "subtasks", []) or []:
            if subtask.subtask_id == "tool-plan" and getattr(subtask, "evidence", None):
                ledger.active_subtask_id = subtask.subtask_id
                return

    async def _initialize_run_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = load_runtime_state(self, payload)
        await initialize_loop_run(graph_state, self.deps, task=str(payload.get("input_task", "")))
        graph_state.loop_state.scratchpad["_tool_plan_phase"] = "planning"
        increment_metric(graph_state.loop_state, "tool_plan_invocations")
        return serialize_runtime_state(graph_state)

    async def _prepare_tool_plan_prompt_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        return await prepare_prompt_node(self, payload, _prepare_planner_prompt)

    async def _planner_model_call_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = load_runtime_state(self, payload)
        messages = graph_state.loop_state.scratchpad.pop("_compiled_prompt_messages", [])
        started = perf_counter()
        result = await process_model_stream(
            graph_state,
            self.deps,
            messages=messages,
            tools=[],
            suppress_ui_events=True,
        )
        _add_latency_metric(graph_state, "planner_latency_sec", perf_counter() - started)
        if graph_state.final_result is None:
            usage_payload = result.usage if isinstance(result.usage, dict) else {}
            if usage_payload:
                await self.deps.harness._apply_usage(usage_payload)
                _record_tool_plan_tokens(graph_state.loop_state, "tool_plan_planner_tokens", usage_payload)
            graph_state.last_usage = usage_payload
            graph_state.last_assistant_text = result.stream.assistant_text
            graph_state.last_thinking_text = result.stream.thinking_text
            graph_state.pending_tool_calls = []
        return serialize_runtime_state(graph_state)

    async def _parse_and_validate_tool_plan_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = load_runtime_state(self, payload)
        harness = self.deps.harness
        max_steps = max(1, int(_tool_plan_config(self.deps, "tool_plan_max_steps", 6) or 6))
        plan = parse_tool_plan(graph_state.last_assistant_text, max_steps=max_steps)
        if plan is None:
            increment_metric(harness.state, "tool_plan_parse_failures")
            harness._runlog(
                "tool_plan_parse_failed",
                "ToolPlan planner output could not be parsed",
                raw_output=graph_state.last_assistant_text,
                raw_thinking=graph_state.last_thinking_text,
            )
            if self._maybe_retry_planner(
                graph_state,
                "Planner output was not valid bounded ToolPlan JSON.",
            ):
                return serialize_runtime_state(graph_state)
            await self._emit_rejected_planner_summary(
                graph_state,
                "ToolPlan planner returned invalid evidence JSON; continuing in normal mode.",
            )
            self._fallback_to_loop(
                graph_state,
                "ToolPlan planner did not return valid bounded JSON.",
                failure_class="tool_plan_invalid",
            )
            return serialize_runtime_state(graph_state)
        increment_metric(harness.state, "tool_plan_steps_requested", len(plan.steps))
        safe_plan, errors = validate_tool_plan(
            plan,
            harness=harness,
            max_steps=max_steps,
            allow_web=bool(_tool_plan_config(self.deps, "tool_plan_allow_web", True)),
            allow_artifact_read=bool(_tool_plan_config(self.deps, "tool_plan_allow_artifact_read", True)),
            allow_git=bool(_tool_plan_config(self.deps, "tool_plan_allow_git", False)),
        )
        if safe_plan is None:
            increment_metric(harness.state, "tool_plan_unsafe_steps_blocked", len(plan.steps))
            if any("path must be a relative path inside the workspace" in error for error in errors):
                increment_metric(harness.state, "tool_plan_wrong_path_count")
            if self._maybe_retry_planner(
                graph_state,
                "Unsafe ToolPlan evidence steps were rejected: " + " ".join(errors),
            ):
                return serialize_runtime_state(graph_state)
            await self._emit_rejected_planner_summary(
                graph_state,
                "ToolPlan planner proposed steps that are not safe read-only evidence; continuing in normal mode.",
            )
            self._fallback_to_loop(
                graph_state,
                "ToolPlan rejected unsafe evidence steps: " + " ".join(errors),
                failure_class="tool_plan_unsafe",
            )
            harness._runlog("tool_plan_rejected", "unsafe ToolPlan rejected", errors=errors)
            return serialize_runtime_state(graph_state)
        graph_state.loop_state.scratchpad["_tool_plan"] = plan
        graph_state.loop_state.scratchpad["_tool_plan_phase"] = "dispatch"
        _record_tool_plan_planner_metadata(harness, plan)
        harness._runlog("tool_plan_accepted", "ToolPlan accepted", steps=len(plan.steps), objective=plan.objective)
        if not plan.steps:
            graph_state.loop_state.scratchpad["_tool_plan_phase"] = "solver"
            graph_state.loop_state.scratchpad["_tool_plan_observations_text"] = "No read-only evidence steps were required for this task."
            graph_state.loop_state.scratchpad["_tool_plan_observations"] = []
        return serialize_runtime_state(graph_state)

    async def _emit_rejected_planner_summary(self, graph_state: GraphRunState, heading: str) -> None:
        summary = summarize_tool_plan_json(graph_state.last_assistant_text)
        if not summary:
            return
        await self.deps.harness._emit(
            self.deps.event_handler,
            UIEvent(
                event_type=UIEventType.ASSISTANT,
                content=f"{heading}\n\n{summary}",
            ),
        )

    async def _prepare_tool_plan_dispatch_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = load_runtime_state(self, payload)
        plan = _coerce_tool_plan(graph_state.loop_state.scratchpad.get("_tool_plan"))
        if plan is None:
            self._fallback_to_loop(
                graph_state,
                "ToolPlan was not available for dispatch.",
                failure_class="tool_plan_invalid",
            )
        else:
            graph_state.loop_state.scratchpad["_tool_plan_recent_errors_start"] = len(
                getattr(graph_state.loop_state, "recent_errors", []) or []
            )
            prepare_tool_plan_dispatch(graph_state, plan)
        return serialize_runtime_state(graph_state)

    async def _dispatch_tools_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = load_runtime_state(self, payload)
        phase = str(graph_state.loop_state.scratchpad.get("_tool_plan_phase") or "")
        started = perf_counter() if phase == "dispatch" else None
        plan = _coerce_tool_plan(graph_state.loop_state.scratchpad.get("_tool_plan"))
        use_dag = phase == "dispatch" and plan is not None and bool(
            _tool_plan_config(self.deps, "tool_dag_enabled", False)
        )
        if use_dag:
            try:
                batches = build_execution_dag(plan)
                assert_parallelizable_steps(batches)
                pending_by_id = {pc.tool_call_id: pc for pc in graph_state.pending_tool_calls}
                pending_batches: list[list[Any]] = []
                for batch in batches:
                    pb = []
                    for step in batch:
                        pc = pending_by_id.get(f"toolplan:{step.id}")
                        if pc is not None:
                            pb.append(pc)
                    if pb:
                        pending_batches.append(pb)
                if plan.steps and not pending_batches:
                    raise ValueError("DAG setup produced no pending ToolPlan batches.")
                metrics = recovery_metrics(self.deps.harness.state)
                batch_sizes = [len(batch) for batch in pending_batches if batch]
                metrics["tool_plan_dag_batch_count"] = len(batch_sizes)
                metrics["tool_plan_dag_max_batch_size"] = max(batch_sizes, default=0)
                metrics["tool_plan_dag_step_count"] = sum(batch_sizes)
                graph_state.recorded_tool_call_ids = []
                graph_state.last_tool_results = []
                dag_started = perf_counter()
                records = await dispatch_tool_dag(
                    graph_state,
                    self.deps,
                    pending_batches,
                    max_parallel=int(_tool_plan_config(self.deps, "tool_dag_max_parallel", 4)),
                    timeout_sec=int(_tool_plan_config(self.deps, "tool_dag_timeout_sec", 30)),
                    preserve_result_order=bool(_tool_plan_config(self.deps, "tool_dag_preserve_result_order", True)),
                )
                actual_ms = max(0.0, (perf_counter() - dag_started) * 1000.0)
                estimated_serial_ms = 0.0
                dispatch_error_count = 0
                for record in records:
                    metadata = getattr(record.result, "metadata", {}) or {}
                    try:
                        estimated_serial_ms += float(metadata.get("dag_latency_ms", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        pass
                    if metadata.get("reason") in {"dag_dispatch_exception", "dag_timeout", "dag_exception"}:
                        dispatch_error_count += 1
                metrics["tool_plan_dag_actual_ms"] = round(actual_ms, 3)
                metrics["tool_plan_dag_estimated_serial_ms"] = round(estimated_serial_ms, 3)
                if actual_ms > 0 and estimated_serial_ms > 0:
                    metrics["tool_plan_dag_speedup"] = round(estimated_serial_ms / actual_ms, 3)
                if dispatch_error_count:
                    increment_metric(self.deps.harness.state, "tool_plan_dag_dispatch_error_count", dispatch_error_count)
                graph_state.last_tool_results = records
                graph_state.pending_tool_calls = []
                if started is not None:
                    _add_latency_metric(graph_state, "worker_latency_sec", perf_counter() - started)
                return serialize_runtime_state(graph_state)
            except Exception as exc:
                increment_metric(self.deps.harness.state, "tool_plan_dag_fallback_count")
                self.deps.harness._runlog("tool_dag_aborted", str(exc), exception_type=exc.__class__.__name__)
                payload = serialize_runtime_state(graph_state)
                # fall through to serial dispatch
        next_payload = await super()._dispatch_tools_node(payload)
        if started is not None:
            next_state = inflate_graph_state(next_payload)
            _add_latency_metric(next_state, "worker_latency_sec", perf_counter() - started)
            return serialize_runtime_state(next_state)
        return next_payload

    async def _compress_observations_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = load_runtime_state(self, payload)
        phase = str(graph_state.loop_state.scratchpad.get("_tool_plan_phase") or "")
        if phase != "dispatch":
            return serialize_runtime_state(graph_state)
        plan = _coerce_tool_plan(graph_state.loop_state.scratchpad.get("_tool_plan"))
        if plan is None:
            return serialize_runtime_state(graph_state)
        token_limit, max_chars_per_step = _tool_plan_observation_budget(self.deps)
        observations = build_tool_plan_observations(
            plan,
            list(graph_state.last_tool_results),
            token_limit=token_limit,
            max_chars_per_step=max_chars_per_step,
        )
        observation_stats = summarize_tool_plan_observations(plan, observations)
        metrics = recovery_metrics(self.deps.harness.state)
        metrics["tool_plan_worker_steps_requested"] = observation_stats.requested_steps
        metrics["tool_plan_worker_steps_executed"] = observation_stats.executed_steps
        metrics["tool_plan_worker_step_failures"] = observation_stats.failed_steps
        metrics["tool_plan_worker_success_rate"] = observation_stats.success_rate
        metrics["tool_plan_worker_missing_record_count"] = observation_stats.missing_record_count
        metrics["tool_plan_worker_duplicate_read_count"] = observation_stats.duplicate_read_count
        metrics["tool_plan_worker_artifact_yield_count"] = observation_stats.artifact_yield_count
        metrics["tool_plan_worker_tool_failure_classes"] = observation_stats.tool_failure_classes
        increment_metric(self.deps.harness.state, "tool_plan_steps_executed", len(observations))
        increment_metric(
            self.deps.harness.state,
            "tool_plan_step_failures",
            sum(1 for observation in observations if not observation.success),
        )
        repeated_read_count = sum(1 for observation in observations if observation.duplicate_of)
        if repeated_read_count:
            increment_metric(self.deps.harness.state, "tool_plan_repeated_read_count", repeated_read_count)
        success_rate = observation_stats.success_rate
        threshold = float(_tool_plan_config(self.deps, "tool_plan_failed_evidence_fallback_threshold", 0.5) or 0.5)
        if (
            success_rate is not None
            and observation_stats.requested_steps > 0
            and success_rate < threshold
        ):
            self._fallback_to_loop(
                graph_state,
                f"ToolPlan evidence gathering succeeded for only {int(success_rate * 100)}% of requested steps "
                f"({observation_stats.successful_steps}/{observation_stats.requested_steps}). "
                "Falling back to the normal loop so the model can gather the missing evidence.",
                failure_class="tool_plan_insufficient_evidence",
                suppress_worker_error_burst=True,
            )
            return serialize_runtime_state(graph_state)
        rendered = render_tool_plan_observations(plan.objective, observations)
        graph_state.loop_state.scratchpad["_tool_plan_observations_text"] = rendered
        graph_state.loop_state.scratchpad["_tool_plan_observations"] = observations
        graph_state.loop_state.scratchpad["_tool_plan_evidence_ids"] = attach_tool_plan_observation_evidence(
            graph_state.loop_state,
            objective=plan.objective,
            observations=observations,
        )
        graph_state.loop_state.scratchpad["_tool_plan_phase"] = "solver"
        metrics["tool_plan_observation_tokens"] = int(metrics.get("tool_plan_observation_tokens", 0) or 0) + estimate_text_tokens(rendered)
        _attach_tool_plan_evidence(
            SimpleNamespace(
                config=getattr(self.deps.harness, "config", None),
                state=graph_state.loop_state,
                subtask_ledger=None,
            ),
            rendered,
        )
        try:
            from ..recovery_schema import Subtask, SubtaskLedger

            ledger = getattr(graph_state.loop_state, "subtask_ledger", None)
            if ledger is None:
                ledger = SubtaskLedger(task_id=str(getattr(graph_state.loop_state, "thread_id", "") or "tool-plan"))
                graph_state.loop_state.subtask_ledger = ledger
            active = ledger.active()
            if active is None:
                active = Subtask(
                    subtask_id="tool-plan",
                    title="ToolPlan evidence",
                    goal="Gather bounded evidence",
                    status="active",
                )
                ledger.subtasks.append(active)
                ledger.active_subtask_id = active.subtask_id
            evidence = "ToolPlan observations: " + rendered[:210]
            if evidence not in active.evidence:
                active.evidence.append(evidence)
            if getattr(self.deps.harness, "state", None) is not graph_state.loop_state:
                self.deps.harness.state.subtask_ledger = ledger
        except Exception as exc:
            self.deps.harness._runlog(
                "tool_plan_subtask_ledger_failed",
                "failed to update subtask ledger from ToolPlan observations",
                error=str(exc),
                exception_type=exc.__class__.__name__,
            )
        self.deps.harness._runlog(
            "tool_plan_observations",
            "ToolPlan observations compressed",
            steps=len(observations),
            chars=len(rendered),
        )
        return serialize_runtime_state(graph_state)

    async def _prepare_solver_prompt_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        return await prepare_prompt_node(self, payload, _prepare_solver_prompt)

    async def _solver_model_call_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = load_runtime_state(self, payload)
        started = perf_counter() if str(graph_state.loop_state.scratchpad.get("_tool_plan_phase") or "") == "solver" else None
        next_payload = await model_call_node(self, payload, select_tools_fn=_select_solver_tools, model_call_fn=model_call)
        graph_state = inflate_graph_state(next_payload)
        if started is not None:
            _add_latency_metric(graph_state, "solver_latency_sec", perf_counter() - started)
        if graph_state.last_usage:
            _record_tool_plan_tokens(graph_state.loop_state, "tool_plan_solver_tokens", graph_state.last_usage)
            return serialize_runtime_state(graph_state)
        if started is not None:
            return serialize_runtime_state(graph_state)
        return next_payload

    async def _interpret_solver_output_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = load_runtime_state(self, payload)
        harness = self.deps.harness
        phase = str(graph_state.loop_state.scratchpad.get("_tool_plan_phase") or "")
        draft = str(graph_state.last_assistant_text or "").strip()
        if phase == "solver" and draft and bool(_tool_plan_config(self.deps, "solver_refine_enabled", False)):
            passes = int(graph_state.loop_state.scratchpad.get("_tool_plan_refine_passes", 0) or 0)
            max_passes = max(0, int(_tool_plan_config(self.deps, "solver_refine_max_passes", 1) or 0))
            if passes < max_passes:
                graph_state.loop_state.scratchpad["_tool_plan_refine_passes"] = passes + 1
                observations_text = str(graph_state.loop_state.scratchpad.get("_tool_plan_observations_text") or "").strip()
                context_frame = ""
                if _rewoo_role_enabled(self.deps, "rewoo_refiner_frame_enabled"):
                    try:
                        observations = graph_state.loop_state.scratchpad.get("_tool_plan_observations")
                        if not isinstance(observations, list):
                            observations = []
                        compiler = ReWOOLaneCompiler(getattr(harness, "context_policy", None))
                        frame = compiler.compile(
                            state=graph_state.loop_state,
                            role="refiner",
                            tool_plan_observations=observations,
                            token_budget=_rewoo_frame_budget(self.deps),
                        )
                        context_frame = compiler.render(frame, token_budget=_rewoo_frame_budget(self.deps))
                        graph_state.loop_state.scratchpad["_rewoo_refiner_drop_log"] = frame.drop_log
                        if frame.drop_log:
                            harness._runlog(
                                "rewoo_refiner_drop_log",
                                "ReWOO refiner frame dropped items",
                                drops=[{"lane": d.lane, "reason": d.reason, "count": d.dropped_count} for d in frame.drop_log],
                            )
                    except Exception as exc:
                        harness._runlog("rewoo_refiner_frame_failed", "refiner frame compilation failed", error=str(exc))
                        observations_text = str(graph_state.loop_state.scratchpad.get("_tool_plan_observations_text") or "").strip()
                        context_frame = ""
                refine = RefineService(harness)
                refine_result = await refine.run_bounded_refine(
                    draft=draft,
                    observations_text=observations_text,
                    context_frame=context_frame,
                )
                if refine_result is not None:
                    harness.state.scratchpad["_tool_plan_refine_verdict"] = refine_result.verdict
                    recovery_metrics(harness.state)["tool_plan_refine_verdict"] = refine_result.verdict
                    if refine_result.verdict == "block":
                        harness.state.append_message(
                            ConversationMessage(
                                role="system",
                                content="Solver draft was blocked by critique: " + "; ".join(refine_result.issues),
                                metadata={"is_recovery_nudge": True, "recovery_kind": "solver_refine_block"},
                            )
                        )
                        graph_state.last_assistant_text = ""
                        graph_state.pending_tool_calls = []
                        return serialize_runtime_state(graph_state)
                    if refine_result.verdict == "revise" and refine_result.revised_output:
                        graph_state.last_assistant_text = refine_result.revised_output
                        harness._runlog("solver_refine", "solver draft revised", issues=refine_result.issues)
        await interpret_model_output(graph_state, self.deps)
        if phase == "solver" and graph_state.pending_tool_calls:
            terminal_calls = [
                call for call in graph_state.pending_tool_calls
                if str(call.tool_name or "").strip() in _TERMINAL_SOLVER_TOOLS
            ]
            blocked_calls = [
                str(call.tool_name or "").strip()
                for call in graph_state.pending_tool_calls
                if str(call.tool_name or "").strip() not in _TERMINAL_SOLVER_TOOLS
            ]
            if blocked_calls:
                increment_metric(self.deps.harness.state, "tool_plan_solver_blocked_tool_calls")
                harness._runlog(
                    "tool_plan_solver_tool_blocked",
                    "blocked non-terminal ToolPlan solver tool call",
                    tools=blocked_calls,
                )
                graph_state.pending_tool_calls = terminal_calls
                if not terminal_calls:
                    graph_state.final_result = harness._failure(
                        "ToolPlan solver requested non-terminal tools after evidence was gathered.",
                        error_type="tool_plan_solver_tool_call",
                        details={"blocked_tools": blocked_calls},
                    )
                    return serialize_runtime_state(graph_state)
        if graph_state.pending_tool_calls:
            graph_state.loop_state.scratchpad["_tool_plan_phase"] = "normal"
            if any(str(call.tool_name or "").strip() in _MUTATING_SOLVER_TOOLS for call in graph_state.pending_tool_calls):
                increment_metric(self.deps.harness.state, "tool_plan_evidence_before_patch_count")
        return serialize_runtime_state(graph_state)

    async def _prepare_step_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = load_runtime_state(self, payload)
        graph_state.loop_state.scratchpad.pop("_tool_plan_phase", None)
        await prepare_loop_step(graph_state, self.deps)
        return serialize_runtime_state(graph_state)

    async def _after_run(self, harness: Any, result: dict[str, object]) -> None:
        if str(result.get("status") or "").strip().lower() in {"completed", "success", "ok", "stopped"}:
            recorder = TrajectoryRecorder()
            try:
                recorder.record_tool_plan_trajectory(harness, result)
            except Exception as exc:
                harness._runlog(
                    "tool_plan_trajectory_record_failed",
                    "failed to record ToolPlan trajectory",
                    error=str(exc),
                    exception_type=exc.__class__.__name__,
                )

    def _fallback_to_loop(
        self,
        graph_state: GraphRunState,
        message: str,
        *,
        failure_class: str,
        suppress_worker_error_burst: bool = False,
    ) -> None:
        graph_state.loop_state.scratchpad["_tool_plan_phase"] = "fallback"
        graph_state.pending_tool_calls = []
        if suppress_worker_error_burst:
            self._suppress_tool_plan_worker_error_burst(graph_state)
        increment_metric(self.deps.harness.state, "tool_plan_fallback_count")
        _record_tool_plan_failure(
            self.deps.harness,
            message,
            failure_class=failure_class,
        )
        self.deps.harness.state.append_message(
            ConversationMessage(
                role="system",
                content=f"{message} Continue with the normal loop runtime.",
                metadata={"is_recovery_nudge": True, "recovery_kind": "tool_plan_fallback"},
            )
        )

    def _suppress_tool_plan_worker_error_burst(self, graph_state: GraphRunState) -> None:
        """Keep failed evidence batches from tripping the normal-loop error guard."""
        state = graph_state.loop_state
        recent_errors = list(getattr(state, "recent_errors", []) or [])
        start_raw = state.scratchpad.pop("_tool_plan_recent_errors_start", None)
        try:
            start = max(0, int(start_raw))
        except (TypeError, ValueError):
            start = 0
        if start >= len(recent_errors):
            return
        suppressed = recent_errors[start:]
        state.recent_errors = recent_errors[:start]
        self.deps.harness._runlog(
            "tool_plan_worker_errors_suppressed",
            "suppressed ToolPlan worker errors before normal-loop fallback",
            suppressed_count=len(suppressed),
        )

    def _maybe_retry_planner(self, graph_state: GraphRunState, message: str) -> bool:
        max_attempts = max(0, int(_tool_plan_config(self.deps, "tool_plan_max_repair_attempts", 1) or 0))
        attempts = int(graph_state.loop_state.scratchpad.get("_tool_plan_repair_attempts", 0) or 0)
        if attempts >= max_attempts:
            return False
        graph_state.loop_state.scratchpad["_tool_plan_repair_attempts"] = attempts + 1
        increment_metric(self.deps.harness.state, "tool_plan_repair_attempts")
        graph_state.loop_state.scratchpad["_tool_plan_phase"] = "planning_repair"
        graph_state.loop_state.scratchpad["_tool_plan_repair_nudge"] = message[:1200]
        graph_state.pending_tool_calls = []
        self.deps.harness._runlog(
            "tool_plan_repair",
            "retrying ToolPlan planner with repair nudge",
            retry_count=attempts + 1,
            max_attempts=max_attempts,
        )
        return True

    @staticmethod
    def _route_after_initialize(payload: LoopGraphPayload) -> str:
        return route_if_final_else(payload, "prepare_tool_plan_prompt")

    @staticmethod
    def _route_after_tool_plan_prompt(payload: LoopGraphPayload) -> str:
        return route_if_final_else(payload, "planner_model_call")

    @staticmethod
    def _route_after_planner_model_call(payload: LoopGraphPayload) -> str:
        return route_if_final_else(payload, "parse_and_validate_tool_plan")

    @staticmethod
    def _route_after_parse(payload: LoopGraphPayload) -> str:
        graph_state = inflate_graph_state(payload)
        phase = str(graph_state.loop_state.scratchpad.get("_tool_plan_phase") or "")
        if phase == "planning_repair":
            return route_if_final_else(payload, "prepare_tool_plan_prompt")
        if phase == "solver":
            return route_if_final_else(payload, "prepare_solver_prompt")
        if phase == "dispatch":
            return "prepare_tool_plan_dispatch"
        return route_if_final_else(payload, "prepare_step")

    @staticmethod
    def _route_after_prepare_dispatch(payload: LoopGraphPayload) -> str:
        graph_state = inflate_graph_state(payload)
        phase = str(graph_state.loop_state.scratchpad.get("_tool_plan_phase") or "")
        if phase == "solver":
            return route_if_final_else(payload, "prepare_solver_prompt")
        return route_if_final_else_pending_else(payload, pending_step="dispatch_tools", fallback_step="prepare_step")

    @staticmethod
    def _route_after_dispatch(payload: LoopGraphPayload) -> str:
        return route_if_final_else(payload, "persist_tool_results")

    @staticmethod
    def _route_after_compress(payload: LoopGraphPayload) -> str:
        graph_state = inflate_graph_state(payload)
        phase = str(graph_state.loop_state.scratchpad.get("_tool_plan_phase") or "")
        if phase == "solver":
            return route_if_final_else(payload, "prepare_solver_prompt")
        if phase == "fallback":
            return route_if_final_else(payload, "prepare_step")
        return route_if_final_else(payload, "apply_tool_outcomes")

    @staticmethod
    def _route_after_solver_prompt(payload: LoopGraphPayload) -> str:
        return route_if_final_else(payload, "solver_model_call")

    @staticmethod
    def _route_after_solver_model_call(payload: LoopGraphPayload) -> str:
        return route_if_final_else(payload, "interpret_solver_output")

    @staticmethod
    def _route_after_prepare_step(payload: LoopGraphPayload) -> str:
        return route_if_final_else_pending_else(payload, pending_step="dispatch_tools", fallback_step="prepare_prompt")

    @staticmethod
    def _route_after_prepare_prompt(payload: LoopGraphPayload) -> str:
        return route_if_final_else(payload, "model_call")

    @staticmethod
    def _route_after_model_call(payload: LoopGraphPayload) -> str:
        return route_if_final_else(payload, "interpret_model_output")

    @staticmethod
    def _route_after_interpret(payload: LoopGraphPayload) -> str:
        return route_if_final_else_pending_else(payload, pending_step="dispatch_tools", fallback_step="prepare_step")

    async def _apply_tool_outcomes_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        return await apply_outcomes_node(self, payload, apply_tool_outcomes, clear_final_result_on_interrupt=True)

    @staticmethod
    def _route_after_apply(payload: LoopGraphPayload) -> str:
        return route_if_interrupt_else_final_else_pending_else(
            payload,
            interrupt_step="interrupt_for_human",
            pending_step="dispatch_tools",
            fallback_step="prepare_step",
        )
