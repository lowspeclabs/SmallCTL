from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .base import ToolSpec
from ..task_targets import primary_task_target_path


PathPart = str | int

# Tools whose primary required destination is a filesystem path. When the model
# emits one of these calls without a path, the harness can safely fall back to
# the task's primary target path rather than asking the model to retry (which
# often causes the model to dump the intended file contents into chat instead).
_PATH_BASED_FILE_TOOLS = {
    "file_write",
    "file_append",
    "file_read",
    "file_patch",
    "ast_patch",
    "file_delete",
    "ssh_file_write",
    "ssh_file_read",
    "ssh_file_patch",
    "ssh_file_replace_between",
}


@dataclass(frozen=True)
class ToolCallValidationIssue:
    path: tuple[PathPart, ...]
    kind: str
    expected: str | None = None
    actual: str | None = None
    message: str = ""


@dataclass(frozen=True)
class ToolCallRepairAction:
    kind: str
    path: tuple[PathPart, ...]
    before_preview: Any
    after_preview: Any
    message: str


@dataclass(frozen=True)
class ToolCallRepairResult:
    valid_initially: bool
    valid_after_repair: bool
    repaired: bool
    args: dict[str, Any]
    issues: list[ToolCallValidationIssue]
    actions: list[ToolCallRepairAction] = field(default_factory=list)
    stripped_extra_fields: list[str] = field(default_factory=list)
    hint: str = ""


_WRAPPER_KEYS = {"arguments", "args", "params", "parameters", "input"}
_CONTENT_FIELD_NAMES = {
    "content",
    "target_text",
    "replacement_text",
    "command",
    "query",
    "pattern",
    "message",
}
_SINGLE_ITEM_ARRAY_ALLOWLIST = {("index_write_import", ("symbols",))}
_PAIRED_RANGE_FIELDS = {
    "file_read": (("start_line",), ("end_line",)),
    "artifact_read": (("start_line",), ("end_line",)),
    "ssh_file_read": (("start_line",), ("end_line",)),
}
_DEFAULT_RANGE_WINDOW = 200
_MARKDOWN_LINK_RE = re.compile(r"^\[([^\]]+)\]\(([^)]+)\)$")

_PATCH_ARGUMENT_ALIASES = {
    "source": "target_text",
    "old_text": "target_text",
    "old": "target_text",
    "dest": "replacement_text",
    "new_text": "replacement_text",
    "new": "replacement_text",
    "replacement": "replacement_text",
}
_PATCH_ALIAS_TOOLS = {"file_patch", "ssh_file_patch", "ast_patch"}
_OPTIONAL_NONE_SENTINELS = {"", "none", "null", "nil", "n/a", "na"}
_SSH_EXEC_TOOL_NAMES = {
    "ssh_exec",
    "shell_exec",
    "ssh_file_read",
    "ssh_file_write",
    "ssh_file_patch",
    "ssh_file_replace_between",
}


def validate_tool_args(schema: dict[str, Any], args: Any) -> list[ToolCallValidationIssue]:
    return _validate_schema(schema, args, ())


def repair_tool_call_args(
    spec: ToolSpec, args: dict[str, Any], *, primary_target_path: str | None = None
) -> ToolCallRepairResult:
    if not isinstance(args, dict):
        return ToolCallRepairResult(
            valid_initially=False,
            valid_after_repair=False,
            repaired=False,
            args=args,
            issues=[
                ToolCallValidationIssue(
                    path=(),
                    kind="type",
                    expected="object",
                    actual=_type_name(args),
                    message="tool arguments must be an object",
                )
            ],
        )

    # Structural repairs are heuristic shape fixes that are always safe when the
    # corresponding alias/wrapper field is present. They run before issue-driven
    # repairs so that the rest of the catalog sees normalized field names.
    repaired_args = copy.deepcopy(args)
    actions: list[ToolCallRepairAction] = []
    _repair_patch_argument_aliases(spec, repaired_args, actions)
    _repair_ssh_exec_malformed_args(spec, repaired_args, actions)
    wrapper = _repair_wrong_object_wrapper(spec, repaired_args)
    if wrapper is not None:
        repaired_args, action = wrapper
        actions.append(action)

    _repair_missing_path_from_task_targets(
        spec, repaired_args, actions, primary_target_path=primary_target_path
    )

    initial_issues = validate_tool_args(spec.schema, repaired_args) + _catalog_shape_issues(spec, repaired_args)
    if not initial_issues and not actions:
        return ToolCallRepairResult(
            valid_initially=True,
            valid_after_repair=True,
            repaired=False,
            args=args,
            issues=[],
        )

    for issue in validate_tool_args(spec.schema, repaired_args):
        if issue.kind == "type" and _path_schema_type(spec.schema, issue.path) == "array":
            _repair_array_field(spec.name, spec.schema, repaired_args, issue.path, actions)
        elif issue.kind == "type" and _path_schema_type(spec.schema, issue.path) == "string":
            _repair_markdown_path(spec.schema, repaired_args, issue.path, actions)

    _repair_optional_nulls(spec.schema, repaired_args, actions)
    _repair_optional_none_sentinels(spec.schema, repaired_args, actions)
    _repair_markdown_paths_recursive(spec.schema, repaired_args, (), actions)
    _repair_paired_range(getattr(spec, "name", ""), repaired_args, actions)

    after_shape_issues = validate_tool_args(spec.schema, repaired_args) + _catalog_shape_issues(spec, repaired_args)
    stripped: list[str] = []
    if _only_extra_field_issues(after_shape_issues):
        for issue in after_shape_issues:
            if issue.path and _delete_path(repaired_args, issue.path):
                stripped.append(".".join(str(part) for part in issue.path))
                actions.append(
                    ToolCallRepairAction(
                        kind="extra_fields_strip_or_warn",
                        path=issue.path,
                        before_preview="<extra field>",
                        after_preview="<omitted>",
                        message=f"unknown field {issue.path[-1]} was ignored",
                    )
                )

    final_issues = validate_tool_args(spec.schema, repaired_args) + _catalog_shape_issues(spec, repaired_args)
    hint = _build_hint(actions)
    return ToolCallRepairResult(
        valid_initially=False,
        valid_after_repair=not final_issues,
        repaired=bool(actions) and not final_issues,
        args=repaired_args,
        issues=initial_issues,
        actions=actions if not final_issues else [],
        stripped_extra_fields=stripped if not final_issues else [],
        hint=hint if not final_issues else "",
    )


def repair_pending_tool_call_args(harness: Any, pending: Any) -> ToolCallRepairResult | None:
    registry = getattr(harness, "registry", None)
    get_spec = getattr(registry, "get", None)
    if not callable(get_spec):
        return None
    spec = get_spec(str(getattr(pending, "tool_name", "") or ""))
    if spec is None:
        return None
    args = getattr(pending, "args", None)
    if not isinstance(args, dict):
        return None
    target_path = primary_task_target_path(harness)
    return repair_tool_call_args(spec, args, primary_target_path=target_path)


def _validate_schema(schema: dict[str, Any], value: Any, path: tuple[PathPart, ...]) -> list[ToolCallValidationIssue]:
    issues: list[ToolCallValidationIssue] = []
    expected_types = _schema_types(schema)
    if expected_types and not _matches_any_type(value, expected_types):
        issues.append(
            ToolCallValidationIssue(
                path=path,
                kind="type",
                expected="|".join(expected_types),
                actual=_type_name(value),
                message=f"expected {'|'.join(expected_types)} but got {_type_name(value)}",
            )
        )
        return issues

    if "enum" in schema and value not in schema.get("enum", []):
        issues.append(ToolCallValidationIssue(path=path, kind="enum", expected=str(schema.get("enum")), actual=repr(value)))

    if isinstance(value, dict) and "object" in (expected_types or ["object"]):
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        for key in required:
            if key not in value or value[key] is None:
                issues.append(ToolCallValidationIssue(path=path + (key,), kind="required", message=f"missing required field {key}"))
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    issues.append(ToolCallValidationIssue(path=path + (key,), kind="additional_property", message=f"unknown field {key}"))
        for key, child_schema in properties.items():
            if key in value:
                issues.extend(_validate_schema(child_schema, value[key], path + (key,)))

    if isinstance(value, list) and "array" in (expected_types or ["array"]):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                issues.extend(_validate_schema(item_schema, item, path + (index,)))
    return issues


def _catalog_shape_issues(spec: ToolSpec, args: dict[str, Any]) -> list[ToolCallValidationIssue]:
    issues: list[ToolCallValidationIssue] = []
    _collect_optional_null_issues(spec.schema, args, (), issues)
    _collect_optional_none_sentinel_issues(spec.schema, args, (), issues)
    _collect_markdown_path_issues(spec.schema, args, (), issues)
    pair = _PAIRED_RANGE_FIELDS.get(getattr(spec, "name", ""))
    if pair:
        start_path, end_path = pair
        start = _get_path(args, start_path)
        end = _get_path(args, end_path)
        if start is None and isinstance(end, int):
            issues.append(ToolCallValidationIssue(path=start_path, kind="missing_paired_range", message="start_line missing while end_line is present"))
        elif isinstance(start, int) and end is None:
            issues.append(ToolCallValidationIssue(path=end_path, kind="missing_paired_range", message="end_line missing while start_line is present"))
    return issues


def _collect_optional_null_issues(schema: dict[str, Any], args: Any, path: tuple[PathPart, ...], issues: list[ToolCallValidationIssue]) -> None:
    if not isinstance(args, dict):
        return
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    for key, child_schema in properties.items():
        if key not in args:
            continue
        child_path = path + (key,)
        if args[key] is None and key not in required and "null" not in _schema_types(child_schema):
            issues.append(ToolCallValidationIssue(path=child_path, kind="optional_null", expected="omit", actual="null"))
        elif isinstance(args[key], dict):
            _collect_optional_null_issues(child_schema, args[key], child_path, issues)


def _collect_optional_none_sentinel_issues(
    schema: dict[str, Any], args: Any, path: tuple[PathPart, ...], issues: list[ToolCallValidationIssue]
) -> None:
    if not isinstance(args, dict):
        return
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    for key, child_schema in properties.items():
        if key not in args:
            continue
        child_path = path + (key,)
        value = args[key]
        if (
            isinstance(value, str)
            and str(value).strip().lower() in _OPTIONAL_NONE_SENTINELS
            and key not in required
            and "null" not in _schema_types(child_schema)
        ):
            issues.append(
                ToolCallValidationIssue(
                    path=child_path,
                    kind="optional_none_sentinel",
                    expected="omit",
                    actual=repr(value),
                    message=f"optional field {key} had sentinel value {value!r}",
                )
            )
        elif isinstance(value, dict):
            _collect_optional_none_sentinel_issues(child_schema, value, child_path, issues)


def _collect_markdown_path_issues(schema: dict[str, Any], args: Any, path: tuple[PathPart, ...], issues: list[ToolCallValidationIssue]) -> None:
    if not isinstance(args, dict):
        return
    for key, child_schema in (schema.get("properties") or {}).items():
        if key not in args:
            continue
        child_path = path + (key,)
        value = args[key]
        if isinstance(value, str) and _is_path_like_field(child_path) and _MARKDOWN_LINK_RE.match(value):
            issues.append(ToolCallValidationIssue(path=child_path, kind="markdown_link_path", expected="plain path", actual="markdown link"))
        elif isinstance(value, dict):
            _collect_markdown_path_issues(child_schema, value, child_path, issues)


def _schema_types(schema: dict[str, Any]) -> list[str]:
    type_value = schema.get("type")
    if isinstance(type_value, str):
        return [type_value]
    if isinstance(type_value, list):
        return [item for item in type_value if isinstance(item, str)]
    return []


def _matches_any_type(value: Any, expected_types: list[str]) -> bool:
    return any(_matches_type(value, expected_type) for expected_type in expected_types)


def _matches_type(value: Any, expected_type: str) -> bool:
    if expected_type == "null":
        return value is None
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (isinstance(value, int | float) and not isinstance(value, bool))
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    return True


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _repair_wrong_object_wrapper(spec: ToolSpec, args: dict[str, Any]) -> tuple[dict[str, Any], ToolCallRepairAction] | None:
    if len(args) != 1:
        return None
    key, value = next(iter(args.items()))
    if key not in (_WRAPPER_KEYS | {getattr(spec, "name", "")}) or not isinstance(value, dict):
        return None
    result = repair_tool_call_args(spec, value)
    if result.valid_after_repair:
        return result.args, ToolCallRepairAction(
            kind="wrong_object_wrapper_unwrap",
            path=(key,),
            before_preview=f"<{key} wrapper>",
            after_preview="<top-level arguments>",
            message=f"arguments were nested under {key}, so the wrapper was removed",
        )
    return None


def _repair_array_field(tool_name: str, schema: dict[str, Any], args: dict[str, Any], path: tuple[PathPart, ...], actions: list[ToolCallRepairAction]) -> None:
    value = _get_path(args, path)
    if not isinstance(value, str):
        return
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        _set_path(args, path, parsed)
        actions.append(ToolCallRepairAction("json_string_to_array", path, value, parsed, f"field {path[-1]} was parsed as an array"))
        return
    item_schema = _path_schema(schema, path).get("items", {})
    if _allow_single_item_array(tool_name, path, item_schema) and value:
        wrapped = [value]
        _set_path(args, path, wrapped)
        actions.append(ToolCallRepairAction("string_to_single_item_array", path, value, wrapped, f"field {path[-1]} was wrapped as a single-item array"))


def _allow_single_item_array(tool_name: str, path: tuple[PathPart, ...], item_schema: dict[str, Any]) -> bool:
    return (tool_name, path) in _SINGLE_ITEM_ARRAY_ALLOWLIST and _schema_types(item_schema) == ["string"]


def _repair_optional_nulls(schema: dict[str, Any], args: dict[str, Any], actions: list[ToolCallRepairAction], path: tuple[PathPart, ...] = ()) -> None:
    if not isinstance(args, dict):
        return
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    for key, child_schema in properties.items():
        if key not in args:
            continue
        child_path = path + (key,)
        if args[key] is None and key not in required and "null" not in _schema_types(child_schema):
            del args[key]
            actions.append(ToolCallRepairAction("null_optional_to_omit", child_path, None, "<omitted>", f"optional field {key} was null, so it was omitted"))
        elif isinstance(args.get(key), dict):
            _repair_optional_nulls(child_schema, args[key], actions, child_path)


def _repair_optional_none_sentinels(
    schema: dict[str, Any], args: dict[str, Any], actions: list[ToolCallRepairAction], path: tuple[PathPart, ...] = ()
) -> None:
    if not isinstance(args, dict):
        return
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    for key, child_schema in properties.items():
        if key not in args:
            continue
        child_path = path + (key,)
        value = args[key]
        if (
            isinstance(value, str)
            and str(value).strip().lower() in _OPTIONAL_NONE_SENTINELS
            and key not in required
            and "null" not in _schema_types(child_schema)
        ):
            del args[key]
            actions.append(
                ToolCallRepairAction(
                    kind="optional_none_sentinel_to_omit",
                    path=child_path,
                    before_preview=value,
                    after_preview="<omitted>",
                    message=f"optional field {key} had sentinel value {value!r}, so it was omitted",
                )
            )
        elif isinstance(value, dict):
            _repair_optional_none_sentinels(child_schema, value, actions, child_path)


def _repair_patch_argument_aliases(spec: ToolSpec, args: dict[str, Any], actions: list[ToolCallRepairAction]) -> None:
    if getattr(spec, "name", "") not in _PATCH_ALIAS_TOOLS or not isinstance(args, dict):
        return
    for alias, canonical in _PATCH_ARGUMENT_ALIASES.items():
        if alias in args and canonical not in args:
            args[canonical] = args.pop(alias)
            actions.append(
                ToolCallRepairAction(
                    kind="patch_argument_alias",
                    path=(canonical,),
                    before_preview=f"{alias}=...",
                    after_preview=f"{canonical}=...",
                    message=f"patch alias {alias!r} was renamed to {canonical!r}",
                )
            )


def _repair_ssh_exec_malformed_args(spec: ToolSpec, args: dict[str, Any], actions: list[ToolCallRepairAction]) -> None:
    if getattr(spec, "name", "") != "ssh_exec" or not isinstance(args, dict):
        return
    nested = args.get("arguments")
    if isinstance(nested, dict):
        nested_cmd = nested.get("arg") or nested.get("command")
        if nested_cmd:
            if not str(args.get("command") or "").strip():
                args["command"] = str(nested_cmd).strip()
                after_preview = args["command"]
            else:
                after_preview = "<removed nested arguments>"
            args.pop("arguments", None)
            actions.append(
                ToolCallRepairAction(
                    kind="ssh_exec_nested_command_unwrap",
                    path=("arguments",),
                    before_preview="<nested arguments>",
                    after_preview=after_preview,
                    message="ssh_exec command was nested under arguments, so it was unwrapped",
                )
            )
    inner_name = args.get("name")
    if isinstance(inner_name, str) and inner_name.strip() and inner_name.strip() not in _SSH_EXEC_TOOL_NAMES:
        args.pop("name", None)
        actions.append(
            ToolCallRepairAction(
                kind="ssh_exec_hallucinated_name_strip",
                path=("name",),
                before_preview=inner_name,
                after_preview="<omitted>",
                message=f"unexpected field name ({inner_name!r}) was removed from ssh_exec arguments",
            )
        )


def _repair_markdown_paths_recursive(schema: dict[str, Any], args: Any, path: tuple[PathPart, ...], actions: list[ToolCallRepairAction]) -> None:
    if not isinstance(args, dict):
        return
    for key, child_schema in (schema.get("properties") or {}).items():
        if key not in args:
            continue
        child_path = path + (key,)
        if isinstance(args[key], str):
            _repair_markdown_path(schema, args, child_path, actions)
        elif isinstance(args[key], dict):
            _repair_markdown_paths_recursive(child_schema, args[key], child_path, actions)


def _repair_markdown_path(schema: dict[str, Any], args: dict[str, Any], path: tuple[PathPart, ...], actions: list[ToolCallRepairAction]) -> None:
    value = _get_path(args, path)
    if not isinstance(value, str) or not _is_path_like_field(path):
        return
    match = _MARKDOWN_LINK_RE.match(value)
    if not match:
        return
    label, target = match.groups()
    if _is_url(label) or _is_url(target) or not _path_equivalent(label, target):
        return
    _set_path(args, path, target)
    actions.append(ToolCallRepairAction("markdown_link_to_path", path, value, target, f"field {path[-1]} was emitted as a Markdown link, so it was unwrapped"))


def _repair_paired_range(tool_name: str, args: dict[str, Any], actions: list[ToolCallRepairAction]) -> None:
    pair = _PAIRED_RANGE_FIELDS.get(tool_name)
    if not pair:
        return
    start_path, end_path = pair
    start = _get_path(args, start_path)
    end = _get_path(args, end_path)
    if start is None and isinstance(end, int):
        _set_path(args, start_path, 1)
        actions.append(ToolCallRepairAction("missing_paired_range_default", start_path, None, 1, "start_line=1 was assumed"))
    elif isinstance(start, int) and end is None:
        new_end = start + _DEFAULT_RANGE_WINDOW - 1
        _set_path(args, end_path, new_end)
        actions.append(ToolCallRepairAction("missing_paired_range_default", end_path, None, new_end, f"end_line={new_end} was assumed"))


def _repair_missing_path_from_task_targets(
    spec: ToolSpec,
    args: dict[str, Any],
    actions: list[ToolCallRepairAction],
    *,
    primary_target_path: str | None,
) -> None:
    """Fill a missing required path field from the task's primary target path.

    Small local models (especially via llama.cpp/LM Studio) sometimes emit a
    `file_write` call with the full file content but omit the `path` field.
    Without this repair the harness rejects the call, and the model often
    responds by dumping the same content into chat.  Filling in the obvious
    target path keeps the write on the tool path.
    """
    tool_name = str(getattr(spec, "name", "") or "")
    if tool_name not in _PATH_BASED_FILE_TOOLS:
        return
    if not primary_target_path:
        return

    required = set(spec.schema.get("required") or [])
    properties = spec.schema.get("properties") or {}
    path_field: str | None = None
    for candidate in ("path", "file_path", "dir_path"):
        if candidate in required and candidate in properties:
            path_field = candidate
            break
    if path_field is None:
        return
    if args.get(path_field) is not None and str(args[path_field]).strip():
        return

    args[path_field] = primary_target_path
    actions.append(
        ToolCallRepairAction(
            kind="missing_path_from_task_targets",
            path=(path_field,),
            before_preview="<missing>",
            after_preview=primary_target_path,
            message=f"{path_field} was missing, so the task target path was inserted",
        )
    )


def _path_schema(schema: dict[str, Any], path: tuple[PathPart, ...]) -> dict[str, Any]:
    current = schema
    for part in path:
        if isinstance(part, int):
            current = current.get("items", {})
        else:
            current = (current.get("properties") or {}).get(part, {})
    return current


def _path_schema_type(schema: dict[str, Any], path: tuple[PathPart, ...]) -> str | None:
    types = _schema_types(_path_schema(schema, path))
    return types[0] if len(types) == 1 else None


def _get_path(obj: Any, path: tuple[PathPart, ...]) -> Any:
    current = obj
    for part in path:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and isinstance(part, int) and 0 <= part < len(current):
            current = current[part]
        else:
            return None
    return current


def _set_path(obj: Any, path: tuple[PathPart, ...], value: Any) -> None:
    parent = _get_path(obj, path[:-1]) if path[:-1] else obj
    if isinstance(parent, dict):
        parent[path[-1]] = value
    elif isinstance(parent, list) and isinstance(path[-1], int):
        parent[path[-1]] = value


def _delete_path(obj: Any, path: tuple[PathPart, ...]) -> bool:
    parent = _get_path(obj, path[:-1]) if path[:-1] else obj
    if isinstance(parent, dict) and path[-1] in parent:
        del parent[path[-1]]
        return True
    return False


def _only_extra_field_issues(issues: list[ToolCallValidationIssue]) -> bool:
    return bool(issues) and all(issue.kind == "additional_property" for issue in issues)


def _is_path_like_field(path: tuple[PathPart, ...]) -> bool:
    name = str(path[-1]) if path else ""
    if name in _CONTENT_FIELD_NAMES:
        return False
    return name in {"path", "file_path", "dir_path", "directory", "artifact_path"} or name.endswith("_path")


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _path_equivalent(left: str, right: str) -> bool:
    return left.strip().rstrip("/") == right.strip().rstrip("/") and bool(left.strip())


def _build_hint(actions: list[ToolCallRepairAction]) -> str:
    if not actions:
        return ""
    if len(actions) == 1:
        return f"Your tool call was repaired: {actions[0].message}. Next time send arguments in the schema's expected shape."
    messages = [f"{i + 1}. {action.message}" for i, action in enumerate(actions)]
    return "Your tool call was repaired: " + " ".join(messages) + " Next time send arguments in the schema's expected shape."
