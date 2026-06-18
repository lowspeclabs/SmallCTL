from __future__ import annotations

from copy import deepcopy
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
import shutil
from typing import Any

from ..models.events import UIEvent, UIEventType
from ..recovery_metrics import increment_metric, recovery_metrics
from ..state import json_safe_value
from ..tools import ToolDispatcher, build_registry
from .model_call_nodes import _conversation_tool_calls_from_pending
from .scaling_constants import LOCAL_FILE_MUTATION_TOOLS, PROMPT_VARIANTS
from .scaling_helpers import candidate_uses_only_read_only_tools, unsafe_branch_execution_reason


@dataclass(slots=True)
class ProposalCandidate:
    candidate_idx: int
    prompt_variant: str
    assistant_text: str = ""
    thinking_text: str = ""
    pending_tool_calls: list[Any] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    score: float = 0.0
    failed_criteria: list[str] = field(default_factory=list)
    isolated: bool = False


@dataclass(slots=True)
class BranchExecutionResult:
    candidate: ProposalCandidate
    state: Any
    verification: Any = None
    score_value: float = 0.0
    passed_threshold: bool = False
    error: str = ""


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for schema in tools:
        function = schema.get("function") if isinstance(schema, dict) else None
        name = str(function.get("name") or "").strip() if isinstance(function, dict) else ""
        if name:
            names.add(name)
    return names


def _resolve_snapshot_path(root: Path, raw_path: str | Path) -> Path | None:
    path = Path(raw_path)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def _snapshot_parent_dirs(root: Path, path: Path) -> list[Path]:
    dirs: list[Path] = []
    current = path.parent
    while True:
        try:
            current.relative_to(root)
        except ValueError:
            break
        dirs.append(current)
        if current == root:
            break
        current = current.parent
    return dirs


def _usage_total_tokens(usage: dict[str, Any]) -> int:
    try:
        return int((usage or {}).get("total_tokens") or 0)
    except (TypeError, ValueError):
        return 0


@dataclass(slots=True)
class CandidateStateGuard:
    active_step_run_id: str
    step_sandbox_history: list[Any]
    tool_execution_records: dict[str, Any]
    recent_messages: list[Any]
    transcript_messages: list[Any]
    scratchpad: dict[str, Any]
    files_changed_this_cycle: list[str]
    write_session: Any
    artifacts: dict[str, Any]
    pending_interrupt: dict[str, Any] | None
    step_verification_result: Any
    tool_history: list[str]
    token_usage: int
    last_completion_tokens: int

    @classmethod
    def capture(cls, state: Any) -> "CandidateStateGuard":
        return cls(
            active_step_run_id=str(getattr(state, "active_step_run_id", "") or ""),
            step_sandbox_history=deepcopy(getattr(state, "step_sandbox_history", []) or []),
            tool_execution_records=deepcopy(getattr(state, "tool_execution_records", {}) or {}),
            recent_messages=deepcopy(getattr(state, "recent_messages", []) or []),
            transcript_messages=deepcopy(getattr(state, "transcript_messages", []) or []),
            scratchpad=deepcopy(getattr(state, "scratchpad", {}) or {}),
            files_changed_this_cycle=deepcopy(getattr(state, "files_changed_this_cycle", []) or []),
            write_session=deepcopy(getattr(state, "write_session", None)),
            artifacts=deepcopy(getattr(state, "artifacts", {}) or {}),
            pending_interrupt=deepcopy(getattr(state, "pending_interrupt", None)),
            step_verification_result=deepcopy(getattr(state, "step_verification_result", None)),
            tool_history=deepcopy(getattr(state, "tool_history", []) or []),
            token_usage=int(getattr(state, "token_usage", 0) or 0),
            last_completion_tokens=int(getattr(state, "last_completion_tokens", 0) or 0),
        )

    def restore(self, state: Any, *, keep_accounting: bool = True) -> None:
        current_token_usage = int(getattr(state, "token_usage", 0) or 0)
        current_last_completion_tokens = int(getattr(state, "last_completion_tokens", 0) or 0)
        state.active_step_run_id = self.active_step_run_id
        state.step_sandbox_history = deepcopy(self.step_sandbox_history)
        state.tool_execution_records = deepcopy(self.tool_execution_records)
        state.recent_messages = deepcopy(self.recent_messages)
        state.transcript_messages = deepcopy(self.transcript_messages)
        state.scratchpad = deepcopy(self.scratchpad)
        state.files_changed_this_cycle = deepcopy(self.files_changed_this_cycle)
        state.write_session = deepcopy(self.write_session)
        state.artifacts = deepcopy(self.artifacts)
        state.pending_interrupt = deepcopy(self.pending_interrupt)
        state.step_verification_result = deepcopy(self.step_verification_result)
        state.tool_history = deepcopy(self.tool_history)
        if keep_accounting:
            state.token_usage = current_token_usage
            state.last_completion_tokens = current_last_completion_tokens
        else:
            state.token_usage = self.token_usage
            state.last_completion_tokens = self.last_completion_tokens


@dataclass(slots=True)
class FileSnapshotGuard:
    cwd: Path
    snapshots: dict[Path, bytes | None]
    directories_existed: set[Path] = field(default_factory=set)

    @classmethod
    def capture(cls, *, cwd: str | Path, paths: list[str | Path]) -> "FileSnapshotGuard":
        root = Path(cwd).resolve()
        snapshots: dict[Path, bytes | None] = {}
        directories_existed: set[Path] = {root}
        for raw_path in paths:
            path = _resolve_snapshot_path(root, raw_path)
            if path is None:
                continue
            if path in snapshots:
                continue
            for parent in _snapshot_parent_dirs(root, path):
                if parent.exists() and parent.is_dir():
                    directories_existed.add(parent)
            try:
                snapshots[path] = path.read_bytes() if path.exists() and path.is_file() else None
            except OSError:
                snapshots[path] = None
        return cls(cwd=root, snapshots=snapshots, directories_existed=directories_existed)

    def restore(self) -> None:
        for path, content in self.snapshots.items():
            if content is None:
                try:
                    if path.exists() and path.is_file():
                        path.unlink()
                except OSError:
                    continue
                continue
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            except OSError:
                continue
        self._remove_new_empty_directories()

    def _remove_new_empty_directories(self) -> None:
        candidates: set[Path] = set()
        for path in self.snapshots:
            candidates.update(_snapshot_parent_dirs(self.cwd, path))
            if path.exists() and path.is_dir():
                candidates.add(path)
        for directory in sorted(candidates, key=lambda item: len(item.parts), reverse=True):
            if directory == self.cwd or directory in self.directories_existed:
                continue
            try:
                directory.relative_to(self.cwd)
            except ValueError:
                continue
            try:
                directory.rmdir()
            except OSError:
                continue


def collect_step_snapshot_paths(step: Any) -> list[str]:
    paths: list[str] = []
    for output in getattr(step, "outputs_expected", []) or []:
        if getattr(output, "kind", "") == "file":
            ref = str(getattr(output, "ref", "") or "").strip()
            if ref:
                paths.append(ref)
    for spec in getattr(step, "verifiers", []) or []:
        if getattr(spec, "kind", "") not in {"file_exists", "file_changed", "syntax_ok", "file_count"}:
            continue
        args = getattr(spec, "args", {}) or {}
        if not isinstance(args, dict):
            continue
        ref = str(args.get("path") or args.get("ref") or "").strip()
        if ref:
            paths.append(ref)
    return list(dict.fromkeys(paths))


def collect_candidate_snapshot_paths(candidates: list[Any]) -> list[str]:
    paths: list[str] = []
    for candidate in candidates:
        for call in candidate.pending_tool_calls:
            tool_name = str(getattr(call, "tool_name", "") or "").strip()
            if tool_name not in LOCAL_FILE_MUTATION_TOOLS:
                continue
            args = getattr(call, "args", {}) or {}
            if not isinstance(args, dict):
                continue
            for key in ("path", "target_path"):
                ref = str(args.get(key) or "").strip()
                if ref:
                    paths.append(ref)
    return list(dict.fromkeys(paths))


def build_candidate_prompts(
    base_messages: list[dict[str, Any]],
    *,
    max_candidates: int,
) -> list[tuple[str, list[dict[str, Any]]]]:
    count = max(1, min(int(max_candidates or 1), len(PROMPT_VARIANTS)))
    prompts: list[tuple[str, list[dict[str, Any]]]] = []
    for variant in PROMPT_VARIANTS[:count]:
        messages = [json_safe_value(message) for message in base_messages]
        messages.append(
            {
                "role": "system",
                "content": "[TEST-TIME SCALING CANDIDATE]\n" + variant,
            }
        )
        prompts.append((variant, messages))
    return prompts


def _copy_workspace_for_candidate(root: Path, sandbox: Path) -> None:
    ignore = shutil.ignore_patterns(
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".coverage",
        "htmlcov",
        ".tox",
        ".nox",
        ".cache",
        "logs",
        ".smallctl",
        "node_modules",
        "build",
        "dist",
        "target",
        ".gradle",
    )
    shutil.copytree(root, sandbox, ignore=ignore)


def _all_fail_action(config: Any) -> str:
    action = str(getattr(config, "test_time_scaling_all_fail_action", "fallback_normal_retry") or "").strip().lower()
    if action in {"fail", "fail_step", "block"}:
        return "fail_step"
    return "fallback_normal_retry"


def _merge_branch_state(target_state: Any, branch_state: Any) -> None:
    active_plan = getattr(target_state, "active_plan", None)
    draft_plan = getattr(target_state, "draft_plan", None)
    target_state.active_step_run_id = str(getattr(branch_state, "active_step_run_id", "") or "")
    target_state.step_sandbox_history = deepcopy(getattr(branch_state, "step_sandbox_history", []) or [])
    target_state.tool_execution_records = deepcopy(getattr(branch_state, "tool_execution_records", {}) or {})
    target_state.recent_messages = deepcopy(getattr(branch_state, "recent_messages", []) or [])
    target_state.transcript_messages = deepcopy(getattr(branch_state, "transcript_messages", []) or [])
    target_state.scratchpad = deepcopy(getattr(branch_state, "scratchpad", {}) or {})
    target_state.files_changed_this_cycle = deepcopy(getattr(branch_state, "files_changed_this_cycle", []) or [])
    target_state.write_session = deepcopy(getattr(branch_state, "write_session", None))
    target_state.artifacts = deepcopy(getattr(branch_state, "artifacts", {}) or {})
    target_state.pending_interrupt = deepcopy(getattr(branch_state, "pending_interrupt", None))
    target_state.step_verification_result = deepcopy(getattr(branch_state, "step_verification_result", None))
    target_state.tool_history = deepcopy(getattr(branch_state, "tool_history", []) or [])
    target_state.active_plan = active_plan
    target_state.draft_plan = draft_plan


async def _emit_scaling_status(deps: Any, content: str, **data: Any) -> None:
    harness = deps.harness
    emit = getattr(harness, "_emit", None)
    if not callable(emit):
        return
    event_data = {
        "kind": "test_time_scaling",
        "status_activity": str(content or "").strip(),
        **json_safe_value(data),
    }
    if not isinstance(event_data, dict):
        event_data = {"kind": "test_time_scaling", "status_activity": str(content or "").strip()}
    await emit(
        deps.event_handler,
        UIEvent(
            event_type=UIEventType.SYSTEM,
            content=content,
            data=event_data,
        ),
    )


def _clone_harness_for_candidate(harness: Any) -> Any:
    branch_harness = copy(harness)
    branch_harness.state = deepcopy(harness.state)
    dispatcher = getattr(harness, "dispatcher", None)
    try:
        branch_harness.registry = build_registry(branch_harness)
        branch_harness.dispatcher = ToolDispatcher(
            registry=branch_harness.registry,
            state=branch_harness.state,
            phase=str(
                getattr(dispatcher, "phase", "")
                or getattr(harness.state, "current_phase", "execute")
                or "execute"
            ),
            run_logger=None,
        )
    except Exception:
        if dispatcher is not None:
            branch_dispatcher = copy(dispatcher)
            try:
                branch_dispatcher.state = branch_harness.state
            except Exception:
                pass
            branch_harness.dispatcher = branch_dispatcher
    return branch_harness


def _commit_selected_proposal(
    graph_state: Any,
    harness: Any,
    selected: Any,
) -> None:
    graph_state.pending_tool_calls = list(selected.pending_tool_calls)
    graph_state.last_assistant_text = selected.assistant_text
    graph_state.last_thinking_text = selected.thinking_text
    graph_state.last_usage = dict(selected.usage or {})
    conversation_tool_calls = _conversation_tool_calls_from_pending(
        graph_state.pending_tool_calls,
        thread_id=graph_state.thread_id,
        step_count=harness.state.step_count,
    )
    if selected.assistant_text or conversation_tool_calls:
        harness._record_assistant_message(
            assistant_text=selected.assistant_text,
            tool_calls=conversation_tool_calls,
            speaker=None,
            hidden_from_prompt=False,
        )
        log_conversation_state = getattr(harness, "_log_conversation_state", None)
        if callable(log_conversation_state):
            log_conversation_state("assistant_message")
        graph_state.recorded_tool_call_ids = [tc["id"] for tc in conversation_tool_calls]


def _record_branch_metrics(
    harness: Any,
    candidates: list[Any],
    selected: Any,
    *,
    failed: bool = False,
    parallel_read_only_count: int = 0,
    all_failed_action: str = "",
) -> None:
    increment_metric(harness.state, "test_time_scaling_candidates", len(candidates))
    increment_metric(
        harness.state,
        "test_time_scaling_branch_failures" if failed else "test_time_scaling_branch_successes",
    )
    metrics = recovery_metrics(harness.state)
    metrics["test_time_scaling_last"] = {
        "policy": "sequential_branch",
        "candidate_count": len(candidates),
        "read_only_candidate_count": sum(1 for candidate in candidates if candidate_uses_only_read_only_tools(candidate)),
        "read_only_branch_parallel_count": int(parallel_read_only_count),
        "selected_candidate": selected.candidate_idx,
        "selected_score": round(selected.score, 3),
        "failed_criteria": list(selected.failed_criteria),
        "failed": failed,
        "all_failed_action": all_failed_action if failed else "",
    }
    harness.state.scratchpad["_test_time_scaling_last"] = dict(metrics["test_time_scaling_last"])


def _candidate_history(
    candidates: list[ProposalCandidate],
    *,
    selected: ProposalCandidate | None = None,
) -> list[dict[str, Any]]:
    selected_idx = getattr(selected, "candidate_idx", None)
    history: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item.candidate_idx):
        tools = [str(getattr(call, "tool_name", "") or "").strip() for call in candidate.pending_tool_calls]
        tools = [tool for tool in tools if tool]
        unsafe_reason = unsafe_branch_execution_reason(candidate)
        record = {
            "candidate": candidate.candidate_idx,
            "selected": selected_idx is not None and candidate.candidate_idx == selected_idx,
            "prompt_variant": candidate.prompt_variant,
            "score": round(float(candidate.score or 0.0), 3),
            "failed_criteria": list(candidate.failed_criteria),
            "tools": tools,
            "read_only": candidate_uses_only_read_only_tools(candidate),
            "unsafe_reason": unsafe_reason,
            "isolated": bool(candidate.isolated),
            "latency_ms": round(float(candidate.latency_ms or 0.0), 3),
            "token_cost": _usage_total_tokens(candidate.usage),
        }
        history.append(record)
    return history
