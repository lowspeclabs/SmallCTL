from __future__ import annotations

import asyncio
import contextvars
from dataclasses import dataclass
import time
from typing import Any, Awaitable, Callable

import httpx
from langgraph.errors import NodeError, NodeTimeoutError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy

from .runtime_payloads import (  # noqa: F401
    LoopGraphPayload,
    build_runtime_payload,
    checkpoint_config,
    coerce_graph_values_payload,
    coerce_interrupt_payload,
    execute_streaming_graph,
    get_runtime_checkpointer,
    load_runtime_state,
    restore_runtime_state,
    route_if_final_else,
    route_if_final_else_pending_else,
    route_if_interrupt_else_final_else,
    route_if_interrupt_else_final_else_pending_else,
    serialize_runtime_state,
)
from .state import GraphRunState  # noqa: F401
from ..redaction import redact_sensitive_text


DEFAULT_GRAPH_NODE_TIMEOUT_SEC = 300.0
DEFAULT_GRAPH_MODEL_CALL_TIMEOUT_SEC = 600.0
DEFAULT_GRAPH_DISPATCH_TOOLS_TIMEOUT_SEC = 300.0
DEFAULT_GRAPH_RECURSION_LIMIT = 1024
DEFAULT_GRAPH_CODING_RECURSION_LIMIT = 2048


class GraphNodeTimeoutError(TimeoutError):
    """Raised when a compiled graph node exceeds its configured timeout."""

    def __init__(self, node_name: str, timeout_sec: float) -> None:
        super().__init__(f"Graph node `{node_name}` timed out after {timeout_sec:g}s")
        self.node_name = node_name
        self.timeout_sec = timeout_sec


@dataclass(frozen=True)
class RuntimeGraphSpec:
    """Configuration for building a runtime StateGraph."""

    node_map: dict[str, str]
    edge_map: dict[str, tuple[str, dict[str, str]]]
    static_edges: list[tuple[str, str]]
    entry_point: tuple[str, str] = (START, "initialize_run")


class RuntimeGraphBuilder:
    """Builds a compiled StateGraph from a runtime instance and spec."""

    def __init__(self, runtime: Any, spec: RuntimeGraphSpec) -> None:
        self.runtime = runtime
        self.spec = spec

    def build(self) -> Any:
        builder = StateGraph(LoopGraphPayload)

        for node_name, method_name in self.spec.node_map.items():
            fn = getattr(self.runtime, method_name)
            builder.add_node(
                node_name,
                self._wrap_node(self.runtime, node_name, fn),
                timeout=graph_node_timeout_policy(self.runtime.deps.harness, node_name),
                retry_policy=graph_retry_policy(self.runtime.deps.harness, node_name),
                error_handler=graph_error_handler_policy(self.runtime, node_name),
            )

        entry_source, entry_target = self.spec.entry_point
        builder.add_edge(entry_source, entry_target)

        for source, (router_name, targets) in self.spec.edge_map.items():
            router = getattr(self.runtime, router_name)
            # LangGraph routing via Command(goto=...) is stable here, while
            # chained add_conditional_edges() transitions can stall on this version.
            route_node_name = f"route__{source}"
            builder.add_node(
                route_node_name,
                self._wrap_node(self.runtime, route_node_name, self._make_route_node(router, targets)),
                timeout=graph_node_timeout_policy(self.runtime.deps.harness, route_node_name),
                retry_policy=graph_retry_policy(self.runtime.deps.harness, route_node_name),
            )
            builder.add_edge(source, route_node_name)

        for source, target in self.spec.static_edges:
            builder.add_edge(source, target)

        return builder.compile(checkpointer=get_runtime_checkpointer(self.runtime.deps.harness))

    @staticmethod
    def _wrap_node(
        runtime: Any,
        node_name: str,
        fn: Callable[[LoopGraphPayload], Awaitable[Any]],
    ) -> Callable[[LoopGraphPayload], Awaitable[Any]]:
        async def wrapped_node(payload: LoopGraphPayload) -> Any:
            harness = runtime.deps.harness
            timeout_sec = graph_node_timeout_sec(harness, node_name)
            started = time.monotonic()
            touch_graph_activity(harness)
            set_active_graph_node(harness, node_name=node_name, started=started)
            runlog(
                harness,
                f"{node_name}_start",
                "graph node started",
                node=node_name,
                timeout_sec=timeout_sec,
            )
            try:
                result = await fn(payload)
            except asyncio.CancelledError:
                elapsed = time.monotonic() - started
                touch_graph_activity(harness)
                runlog(
                    harness,
                    f"{node_name}_cancelled",
                    "graph node cancelled",
                    node=node_name,
                    elapsed_sec=round(elapsed, 3),
                )
                raise
            except Exception as exc:
                elapsed = time.monotonic() - started
                touch_graph_activity(harness)
                if type(exc).__name__ == "GraphInterrupt":
                    runlog(
                        harness,
                        f"{node_name}_pause",
                        "graph node paused for interrupt",
                        node=node_name,
                        elapsed_sec=round(elapsed, 3),
                        interrupt_type=type(exc).__name__,
                        interrupt_message=str(exc),
                    )
                    raise
                runlog(
                    harness,
                    f"{node_name}_error",
                    "graph node failed",
                    node=node_name,
                    elapsed_sec=round(elapsed, 3),
                    error_type=type(exc).__name__,
                    error_message=redact_sensitive_text(str(exc)),
                )
                raise
            elapsed = time.monotonic() - started
            touch_graph_activity(harness)
            runlog(
                harness,
                f"{node_name}_end",
                "graph node completed",
                node=node_name,
                elapsed_sec=round(elapsed, 3),
            )
            return result

        return wrapped_node

    @staticmethod
    def _make_route_node(
        router: Callable[[LoopGraphPayload], str],
        targets: dict[str, str],
    ) -> Callable[[LoopGraphPayload], Awaitable[Command]]:
        async def route_node(payload: LoopGraphPayload) -> Command:
            next_step = router(payload)
            return Command(goto=targets[next_step])

        return route_node


class CompiledGraphRuntimeBase:
    _run_mode = "loop"
    _run_execution_message = "executing loop runtime"
    _empty_result_message = "Graph loop ended without a terminal result."
    _recursion_limit = DEFAULT_GRAPH_RECURSION_LIMIT

    def __init__(self, deps: Any) -> None:
        self.deps = deps

    @classmethod
    def from_harness(
        cls,
        harness: object,
        *,
        event_handler: object = None,
    ):
        from .deps import GraphRuntimeDeps

        return cls(
            GraphRuntimeDeps(
                harness=harness,
                event_handler=event_handler,
            ),
        )

    async def run(self, task: str) -> dict[str, object]:
        harness = self.deps.harness
        self._before_run(harness)
        harness._runlog(
            "runtime_execution",
            self._run_execution_message,
            execution_path=self._execution_path(),
        )
        return await self._run_langgraph(task)

    def _before_run(self, harness: Any) -> None:
        del harness

    async def _run_langgraph(self, task: str) -> dict[str, object]:
        harness = self.deps.harness
        payload = build_runtime_payload(harness, run_mode=self._run_mode, input_task=task)
        return await self._execute_langgraph(payload)

    async def _execute_langgraph(self, payload: LoopGraphPayload | Command) -> dict[str, object]:
        harness = self.deps.harness
        recursion_limit = resolve_graph_recursion_limit(harness, default=self._recursion_limit)
        publish_graph_step_budget(harness, recursion_limit=recursion_limit)
        harness._runlog(
            "runtime_recursion_limit",
            "resolved graph recursion limit",
            recursion_limit=recursion_limit,
            task_mode=str(getattr(getattr(harness, "state", None), "task_mode", "") or ""),
        )
        return await execute_streaming_graph(
            self,
            payload,
            build_graph=self._build_compiled_graph,
            empty_result_message=self._empty_result_message,
            recursion_limit=recursion_limit,
        )

    def _build_compiled_graph(self):
        return RuntimeGraphBuilder(self, self.GRAPH_SPEC).build()

    def _execution_path(self) -> str:
        return "compiled"

    def restore(self, *, thread_id: str | None = None) -> bool:
        recursion_limit = resolve_graph_recursion_limit(self.deps.harness, default=self._recursion_limit)
        publish_graph_step_budget(self.deps.harness, recursion_limit=recursion_limit)
        return restore_runtime_state(self, thread_id=thread_id, recursion_limit=recursion_limit)


def runlog(harness: Any, event: str, message: str, **data: Any) -> None:
    logger = getattr(harness, "_runlog", None)
    if callable(logger):
        logger(event, message, **data)


def config_value(harness: Any, name: str, default: Any) -> Any:
    config = getattr(harness, "config", None)
    if config is not None and hasattr(config, name):
        return getattr(config, name)
    return getattr(harness, name, default)


def positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def is_coding_graph_task(harness: Any) -> bool:
    state = getattr(harness, "state", None)
    if state is None:
        return False
    task_mode = str(getattr(state, "task_mode", "") or "").strip().lower()
    if task_mode in {"local_execute", "debug_inspect"}:
        return True
    if getattr(state, "write_session", None) is not None:
        return True
    if getattr(state, "files_changed_this_cycle", None):
        return True
    active_intent = str(getattr(state, "active_intent", "") or "").strip().lower()
    if active_intent in {"author_write", "requested_write_file", "requested_patch_file"}:
        return True
    intent_tags = [str(tag or "").strip().lower() for tag in (getattr(state, "intent_tags", None) or [])]
    return any(tag in {"coding", "local_coding", "write_file", "patch_file"} for tag in intent_tags)


def resolve_graph_recursion_limit(harness: Any, *, default: int = DEFAULT_GRAPH_RECURSION_LIMIT) -> int:
    base_default = positive_int(default, DEFAULT_GRAPH_RECURSION_LIMIT)
    base_limit = positive_int(config_value(harness, "graph_recursion_limit", base_default), base_default)
    if not is_coding_graph_task(harness):
        return base_limit
    coding_default = max(DEFAULT_GRAPH_CODING_RECURSION_LIMIT, base_limit)
    coding_limit = positive_int(config_value(harness, "graph_coding_recursion_limit", coding_default), coding_default)
    return max(base_limit, coding_limit)


def publish_graph_step_budget(harness: Any, *, recursion_limit: int) -> None:
    state = getattr(harness, "state", None)
    publish_graph_step_budget_for_state(state, recursion_limit=recursion_limit)


def publish_graph_step_budget_for_state(state: Any, *, recursion_limit: int) -> None:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return
    step_count = positive_int(getattr(state, "step_count", 0), 0)
    scratchpad["_graph_recursion_limit"] = recursion_limit
    scratchpad["_graph_steps_remaining"] = max(0, recursion_limit - step_count)


def graph_node_timeout_sec(harness: Any, node_name: str) -> float | None:
    if "model_call" in node_name:
        default = DEFAULT_GRAPH_MODEL_CALL_TIMEOUT_SEC
        value = config_value(harness, "graph_model_call_timeout_sec", default)
    elif "dispatch_tools" in node_name:
        default = DEFAULT_GRAPH_DISPATCH_TOOLS_TIMEOUT_SEC
        value = config_value(harness, "graph_dispatch_tools_timeout_sec", default)
    else:
        default = DEFAULT_GRAPH_NODE_TIMEOUT_SEC
        value = config_value(harness, "graph_node_timeout_sec", default)
    return positive_float(value)


def graph_node_timeout_policy(harness: Any, node_name: str) -> float | None:
    if not config_value(harness, "langgraph_native_timeouts_enabled", False):
        return None
    if node_name.startswith("route__"):
        return graph_node_timeout_sec(harness, node_name)
    if node_name in {
        "dispatch_tools",
        "persist_tool_results",
        "interrupt_for_human",
    } or node_name.endswith("_tool_outcomes"):
        return None
    return graph_node_timeout_sec(harness, node_name)


# Nodes that are known to be idempotent, read-only, and prompt/retrieval-style.
# This set is intentionally explicit; no wildcard or mutation-capable node is allowed.
_RETRY_ALLOWED_NODES = frozenset({
    "prepare_prompt",
    "prepare_chat_prompt",
    "prepare_indexer_prompt",
    "prepare_tool_plan_prompt",
    "prepare_solver_prompt",
})

_RETRY_INITIAL_INTERVAL = 1.0
_RETRY_BACKOFF_FACTOR = 2.0
_RETRY_MAX_INTERVAL = 30.0
_RETRY_MAX_ATTEMPTS_CAP = 2

# Per-async-task retry failure counts so that concurrent graph runs do not share
# a global counter. The dict maps node name -> failure count for the current run.
_retry_failure_counts: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "retry_failure_counts", default=None
)


def begin_graph_retry_attempts() -> contextvars.Token[dict[str, int] | None]:
    """Start isolated retry accounting for one graph invocation."""
    return _retry_failure_counts.set({})


def end_graph_retry_attempts(token: contextvars.Token[dict[str, int] | None]) -> None:
    """Discard retry accounting after a graph invocation completes."""
    _retry_failure_counts.reset(token)


def _is_transient_error(exc: BaseException) -> bool:
    """Return True only for clearly transient transport/compute errors."""
    if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
        return False
    if type(exc).__name__ == "GraphInterrupt":
        return False
    if isinstance(exc, NodeTimeoutError):
        return False
    if isinstance(exc, ConnectionError):
        return True
    if isinstance(exc, httpx.ConnectError):
        return True
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            if exc.response.status_code >= 500:
                return True
        except Exception:
            pass
    if hasattr(exc, "status_code"):
        try:
            if int(getattr(exc, "status_code", 0)) >= 500:
                return True
        except Exception:
            pass
    return False


def _intended_retry_delay(failure_count: int) -> float:
    """Compute the intended backoff delay for a given failure count."""
    return min(
        _RETRY_INITIAL_INTERVAL * (_RETRY_BACKOFF_FACTOR ** max(0, failure_count - 1)),
        _RETRY_MAX_INTERVAL,
    )


def graph_retry_policy(harness: Any, node_name: str) -> RetryPolicy | None:
    """Return a LangGraph RetryPolicy for safe, idempotent nodes, or None."""
    if not config_value(harness, "langgraph_native_retries_enabled", False):
        return None
    if not node_name or node_name.startswith("route__"):
        return None
    if node_name not in _RETRY_ALLOWED_NODES:
        return None

    user_max = positive_int(
        config_value(harness, "langgraph_native_retry_max_attempts", _RETRY_MAX_ATTEMPTS_CAP),
        _RETRY_MAX_ATTEMPTS_CAP,
    )
    max_attempts = min(user_max, _RETRY_MAX_ATTEMPTS_CAP)
    if max_attempts < 1:
        max_attempts = _RETRY_MAX_ATTEMPTS_CAP

    def retry_on(exc: BaseException) -> bool:
        if not _is_transient_error(exc):
            return False
        counts = _retry_failure_counts.get(None)
        if counts is None:
            counts = {}
            _retry_failure_counts.set(counts)
        counts[node_name] = counts.get(node_name, 0) + 1
        failure_count = counts[node_name]
        next_attempt = failure_count + 1
        if next_attempt <= max_attempts:
            runlog(
                harness,
                "graph_retry_attempt",
                "graph node retry attempt",
                node=node_name,
                attempt=next_attempt,
                exception_type=type(exc).__name__,
                delay_sec=_intended_retry_delay(failure_count),
            )
        return True

    return RetryPolicy(
        max_attempts=max_attempts,
        initial_interval=_RETRY_INITIAL_INTERVAL,
        backoff_factor=_RETRY_BACKOFF_FACTOR,
        max_interval=_RETRY_MAX_INTERVAL,
        jitter=True,
        retry_on=retry_on,
    )


# Nodes that may receive native error handlers. This set is intentionally explicit;
# no mutation-capable, model-call, or approval node is included.
_ERROR_RECOVERY_ALLOWED_NODES = frozenset({
    "prepare_prompt",
    "prepare_chat_prompt",
    "prepare_indexer_prompt",
    "prepare_tool_plan_prompt",
    "prepare_solver_prompt",
    "interpret_model_output",
    "interpret_chat_output",
    "interpret_indexer_output",
    "interpret_planning_output",
    "interpret_solver_output",
})

_ERROR_RECOVERY_PREPARE_NODES = frozenset({
    "prepare_prompt",
    "prepare_chat_prompt",
    "prepare_indexer_prompt",
    "prepare_tool_plan_prompt",
    "prepare_solver_prompt",
})

_ERROR_RECOVERY_INTERPRET_NODES = frozenset({
    "interpret_model_output",
    "interpret_chat_output",
    "interpret_indexer_output",
    "interpret_planning_output",
    "interpret_solver_output",
})

_ERROR_RECOVERY_MAX_ATTEMPTS = 1


def _graph_error_handler_terminal_result(
    graph_state: GraphRunState,
    harness: Any,
    node_name: str,
    exc: BaseException,
) -> None:
    """Set a harness failure result on the graph state for terminal errors."""
    failure_fn = getattr(harness, "_failure", None)
    if callable(failure_fn):
        graph_state.final_result = failure_fn(
            f"Graph node `{node_name}` failed and recovery is unavailable.",
            error_type="graph_node_error",
            details={
                "node": node_name,
                "exception_type": type(exc).__name__,
            },
        )
    else:
        graph_state.final_result = {
            "status": "failed",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }


def _sanitize_error_message_for_handler(exc: BaseException) -> str:
    """Return a short, sanitized error summary with no state or secret-bearing text."""
    message = str(exc).strip() or type(exc).__name__
    # Redact credentials or request fragments before truncating for runlog storage.
    redacted = redact_sensitive_text(message)
    # Truncate to a small window; never pass raw model output, tool args, or prompts.
    return redacted[:500]


def _tool_call_repair_enabled(harness: Any) -> bool:
    return bool(getattr(getattr(harness, "config", None), "tool_call_repair_enabled", True))


def _is_interpret_error_recoverable(harness: Any, exc: BaseException) -> bool:
    """Return True only when the existing repair infrastructure can handle it."""
    # The harness's tool-call schema repair and related hardening are the existing
    # parse-repair paths. Without them, an interpret failure is terminal.
    if not _tool_call_repair_enabled(harness):
        return False
    # Do not recover timeouts, cancels, or interrupts via the error handler.
    if isinstance(exc, (asyncio.CancelledError, NodeTimeoutError)):
        return False
    if type(exc).__name__ == "GraphInterrupt":
        return False
    return True


def _prepare_node_for_runtime(runtime: Any, failed_node_name: str) -> str | None:
    """Pick the next safe preparation node for a failed allowed node."""
    # If the failed node itself is a preparation node, route back to it.
    if failed_node_name in _ERROR_RECOVERY_PREPARE_NODES:
        return failed_node_name
    # For interpret nodes, prefer a context-specific prepare node when possible.
    if failed_node_name == "interpret_solver_output":
        return "prepare_solver_prompt"
    # Otherwise find any available preparation node in the runtime spec.
    spec = getattr(runtime, "GRAPH_SPEC", None)
    node_map = getattr(spec, "node_map", {}) if spec is not None else {}
    for candidate in (
        "prepare_prompt",
        "prepare_chat_prompt",
        "prepare_indexer_prompt",
        "prepare_tool_plan_prompt",
        "prepare_solver_prompt",
    ):
        if candidate in node_map:
            return candidate
    return None


def make_graph_error_handler(runtime: Any, node_name: str):
    """Return a LangGraph node error handler closure for the given node name.

    The handler is narrow: it only attempts recovery for explicitly allowed safe
    nodes. All other failures are converted into terminal harness results.
    """
    if node_name not in _ERROR_RECOVERY_ALLOWED_NODES:
        return None

    async def handler(payload: LoopGraphPayload, error: NodeError) -> Command:
        harness = runtime.deps.harness
        original_exc = error.error
        # GraphInterrupt must never be recovered or converted to a terminal failure.
        # Re-raise it so LangGraph's interrupt machinery produces the existing
        # needs_human / resume behavior.
        if type(original_exc).__name__ == "GraphInterrupt":
            raise original_exc
        graph_state = load_runtime_state(runtime, payload)
        sanitized_message = _sanitize_error_message_for_handler(original_exc)

        # Decide whether to recover or fail.
        action = "recover"
        scratchpad = graph_state.loop_state.scratchpad
        recovery_counts = scratchpad.setdefault("_graph_error_recovery_counts", {})
        if not isinstance(recovery_counts, dict):
            recovery_counts = {}
            scratchpad["_graph_error_recovery_counts"] = recovery_counts
        recovery_count = int(recovery_counts.get(node_name, 0) or 0)
        if recovery_count >= _ERROR_RECOVERY_MAX_ATTEMPTS:
            action = "terminal"
        if node_name in _ERROR_RECOVERY_PREPARE_NODES:
            if action == "recover":
                recovery_counts[node_name] = recovery_count + 1
                scratchpad["_graph_error_recovery_hint"] = f"{node_name} failed: {sanitized_message}"
        elif node_name in _ERROR_RECOVERY_INTERPRET_NODES:
            if not _is_interpret_error_recoverable(harness, original_exc):
                action = "terminal"
            elif action == "recover":
                recovery_counts[node_name] = recovery_count + 1
                scratchpad["_graph_error_recovery_hint"] = f"{node_name} failed: {sanitized_message}"
        else:
            action = "terminal"

        runlog(
            harness,
            "graph_error_handler",
            "graph node error handler invoked",
            node=node_name,
            error_type=type(original_exc).__name__,
            error_message=sanitized_message,
            recovery_action=action,
        )

        if action == "terminal":
            _graph_error_handler_terminal_result(graph_state, harness, node_name, original_exc)
            return Command(update=serialize_runtime_state(graph_state), goto=END)

        target_node = _prepare_node_for_runtime(runtime, node_name)
        if target_node is None:
            # No safe preparation node found; treat as terminal.
            _graph_error_handler_terminal_result(graph_state, harness, node_name, original_exc)
            return Command(update=serialize_runtime_state(graph_state), goto=END)

        return Command(update=serialize_runtime_state(graph_state), goto=target_node)

    return handler


def graph_error_handler_policy(runtime: Any, node_name: str):
    """Return an error handler for allowed nodes when the feature is enabled."""
    harness = runtime.deps.harness
    if not config_value(harness, "langgraph_error_handlers_enabled", False):
        return None
    if not node_name or node_name.startswith("route__"):
        return None
    return make_graph_error_handler(runtime, node_name)


def touch_graph_activity(harness: Any) -> None:
    setattr(harness, "_last_graph_activity_monotonic", time.monotonic())


def set_active_graph_node(harness: Any, *, node_name: str | None, started: float | None) -> None:
    setattr(harness, "_active_graph_node_name", node_name)
    setattr(harness, "_active_graph_node_started_monotonic", started)


async def prepare_prompt_node(
    runtime: Any,
    payload: LoopGraphPayload,
    prepare_fn: Callable[[Any, Any], Awaitable[list[dict[str, Any]] | None]],
) -> LoopGraphPayload:
    graph_state = load_runtime_state(runtime, payload)
    recursion_limit = resolve_graph_recursion_limit(runtime.deps.harness, default=runtime._recursion_limit)
    publish_graph_step_budget_for_state(graph_state.loop_state, recursion_limit=recursion_limit)
    messages = await prepare_fn(graph_state, runtime.deps)
    next_payload = serialize_runtime_state(graph_state)
    if messages is not None:
        graph_state.loop_state.scratchpad["_compiled_prompt_messages"] = messages
        next_payload = serialize_runtime_state(graph_state)
    return next_payload


async def model_call_node(
    runtime: Any,
    payload: LoopGraphPayload,
    *,
    select_tools_fn: Callable[[Any, Any], list[dict[str, Any]]],
    model_call_fn: Callable[[Any, Any], Awaitable[None]],
) -> LoopGraphPayload:
    graph_state = load_runtime_state(runtime, payload)
    messages = graph_state.loop_state.scratchpad.pop("_compiled_prompt_messages", [])
    tools = select_tools_fn(graph_state, runtime.deps)
    await model_call_fn(graph_state, runtime.deps, messages=messages, tools=tools)
    return serialize_runtime_state(graph_state)


async def interpret_node(
    runtime: Any,
    payload: LoopGraphPayload,
    interpret_fn: Callable[[Any, Any], Awaitable[None]],
) -> LoopGraphPayload:
    graph_state = load_runtime_state(runtime, payload)
    await interpret_fn(graph_state, runtime.deps)
    return serialize_runtime_state(graph_state)


async def apply_outcomes_node(
    runtime: Any,
    payload: LoopGraphPayload,
    apply_fn: Callable[[Any, Any], Awaitable[None]],
    *,
    clear_final_result_on_interrupt: bool = False,
) -> LoopGraphPayload:
    graph_state = load_runtime_state(runtime, payload)
    await apply_fn(graph_state, runtime.deps)
    if clear_final_result_on_interrupt and graph_state.interrupt_payload is not None:
        graph_state.final_result = None
    return serialize_runtime_state(graph_state)
