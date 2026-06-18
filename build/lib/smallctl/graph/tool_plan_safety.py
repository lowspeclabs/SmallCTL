from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .tool_plan_schema import MUTATING_TOOL_PLAN_BLOCKLIST, READONLY_TOOL_PLAN_TOOLS, ToolPlan, ToolPlanStep

_LOCAL_PATH_TOOLS = {"file_read", "dir_list", "grep", "find_files", "read_log", "git_status", "git_diff"}
_REMOTE_PATH_TOOLS = {"ssh_file_read"}
_PATH_ARG_KEYS = ("path", "target_path")
_STRING_LIMITS = {
    "path": 600,
    "target_path": 600,
    "target": 600,
    "pattern": 500,
    "query": 500,
    "url": 2000,
}


def _registry_names(harness: Any) -> set[str]:
    registry = getattr(harness, "registry", None)
    names_fn = getattr(registry, "names", None) if registry is not None else None
    if not callable(names_fn):
        return set()
    try:
        return {str(name) for name in names_fn()}
    except Exception:
        return set()


def _cwd(harness: Any) -> Path:
    state = getattr(harness, "state", None)
    return Path(str(getattr(state, "cwd", ".") or ".")).resolve()


def _is_remote_looking(path: str) -> bool:
    return "://" in path or path.startswith(("ssh:", "scp:", "sftp:")) or ("@" in path and ":" in path)


def _path_within_cwd(raw_path: str, *, cwd: Path) -> bool:
    if not raw_path:
        return True
    if _is_remote_looking(raw_path):
        return False
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return False
    resolved = (cwd / candidate).resolve()
    try:
        resolved.relative_to(cwd)
    except ValueError:
        return False
    return True


def _url_is_safe(raw_url: str) -> bool:
    parsed = urlparse(raw_url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _validate_args(step: ToolPlanStep, *, harness: Any, allow_web: bool, allow_artifact_read: bool, allow_git: bool = False) -> list[str]:
    errors: list[str] = []
    for key, limit in _STRING_LIMITS.items():
        value = step.args.get(key)
        if isinstance(value, str) and len(value) > limit:
            errors.append(f"{step.id}: {key} is too long.")
    if step.tool in _LOCAL_PATH_TOOLS:
        cwd = _cwd(harness)
        for key in _PATH_ARG_KEYS:
            value = step.args.get(key)
            if value is not None and (not isinstance(value, str) or not _path_within_cwd(value, cwd=cwd)):
                errors.append(f"{step.id}: {key} must be a relative path inside the workspace.")
        if step.tool == "git_diff":
            target = step.args.get("target")
            if target is not None and (not isinstance(target, str) or not _path_within_cwd(target, cwd=cwd)):
                errors.append(f"{step.id}: target must be a relative path inside the workspace.")
    if step.tool in _REMOTE_PATH_TOOLS:
        value = step.args.get("path")
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{step.id}: ssh_file_read requires a non-empty remote path.")
    if step.tool in {"web_search", "web_fetch"} and not allow_web:
        errors.append(f"{step.id}: web tools are disabled for ToolPlan.")
    if step.tool in {"artifact_read", "artifact_grep"} and not allow_artifact_read:
        errors.append(f"{step.id}: artifact reads are disabled for ToolPlan.")
    if step.tool in {"git_status", "git_diff"} and not allow_git:
        errors.append(f"{step.id}: git tools are disabled for ToolPlan.")
    if step.tool == "web_fetch":
        has_prior_id = any(str(step.args.get(key) or "").strip() for key in ("result_id", "fetch_id"))
        raw_url = str(step.args.get("url") or "").strip()
        if raw_url and not _url_is_safe(raw_url):
            errors.append(f"{step.id}: web_fetch URL must be http(s).")
        if not raw_url and not has_prior_id:
            errors.append(f"{step.id}: web_fetch requires a url, result_id, or fetch_id.")
    return errors


def validate_tool_plan(
    plan: ToolPlan,
    *,
    harness: Any,
    max_steps: int = 6,
    allow_web: bool = True,
    allow_artifact_read: bool = True,
    allow_git: bool = False,
) -> tuple[ToolPlan | None, list[str]]:
    errors: list[str] = []
    if len(plan.steps) > max_steps:
        errors.append(f"ToolPlan requested {len(plan.steps)} steps; max is {max_steps}.")
    names = _registry_names(harness)
    valid_ids = {step.id for step in plan.steps}
    seen_ids: set[str] = set()
    for step in plan.steps:
        if step.id in seen_ids:
            errors.append(f"{step.id}: duplicate step id.")
        seen_ids.add(step.id)
        if any(dep not in valid_ids for dep in step.depends_on):
            errors.append(f"{step.id}: depends_on references an unknown step.")
        if not isinstance(step.args, dict):
            errors.append(f"{step.id}: args must be an object.")
        if step.tool in MUTATING_TOOL_PLAN_BLOCKLIST:
            errors.append(f"{step.id}: mutating tool `{step.tool}` is blocked.")
        if step.tool not in READONLY_TOOL_PLAN_TOOLS:
            errors.append(f"{step.id}: tool `{step.tool}` is not allowed in ToolPlan.")
        if names and step.tool not in names:
            errors.append(f"{step.id}: tool `{step.tool}` is not registered.")
        errors.extend(_validate_args(step, harness=harness, allow_web=allow_web, allow_artifact_read=allow_artifact_read, allow_git=allow_git))
    return (None, errors) if errors else (plan, [])
