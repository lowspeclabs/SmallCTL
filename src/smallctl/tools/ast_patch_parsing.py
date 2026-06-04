from __future__ import annotations

import textwrap

try:
    import libcst as cst
except Exception:  # pragma: no cover - import fallback is only for degraded envs
    class _FallbackCSTModule:
        class CSTVisitor:  # pragma: no cover - degraded env import shim
            pass

        class CSTTransformer:  # pragma: no cover - degraded env import shim
            pass

    cst = _FallbackCSTModule()

from .ast_cst_transformers import (
    AstPatchError,
    _CallCollector,
    _CandidateCollector,
    _FunctionCandidate,
)


def parse_python_module(source_text: str) -> cst.Module:
    try:
        return cst.parse_module(source_text or "")
    except cst.ParserSyntaxError as exc:
        raise AstPatchError(
            "parse_failed",
            f"Unable to parse Python source: {exc.message}.",
            extra={
                "line": int(getattr(exc, "raw_line", 0) or 0),
                "column": int(getattr(exc, "raw_column", 0) or 0),
            },
        ) from exc


def parse_function_replacement(source: str) -> cst.FunctionDef:
    candidate = textwrap.dedent(source).strip() + "\n"
    try:
        replacement_module = cst.parse_module(candidate)
    except cst.ParserSyntaxError as exc:
        raise AstPatchError(
            "replacement_parse_failed",
            f"Unable to parse replacement function: {exc.message}.",
            extra={
                "line": int(getattr(exc, "raw_line", 0) or 0),
                "column": int(getattr(exc, "raw_column", 0) or 0),
            },
        ) from exc
    if len(replacement_module.body) != 1 or not isinstance(replacement_module.body[0], cst.FunctionDef):
        raise AstPatchError(
            "ast_operation_invalid",
            "replace_function requires `payload.source` to contain exactly one function definition.",
        )
    return replacement_module.body[0]


def parse_statement_block(source: str) -> list[cst.BaseStatement]:
    candidate = textwrap.dedent(source).strip()
    if not candidate:
        raise AstPatchError("ast_operation_invalid", "The statement block cannot be empty.")
    try:
        module = cst.parse_module(candidate + "\n")
    except cst.ParserSyntaxError as exc:
        raise AstPatchError(
            "replacement_parse_failed",
            f"Unable to parse inserted statements: {exc.message}.",
            extra={
                "line": int(getattr(exc, "raw_line", 0) or 0),
                "column": int(getattr(exc, "raw_column", 0) or 0),
            },
        ) from exc
    if not module.body:
        raise AstPatchError("ast_operation_invalid", "The statement block cannot be empty.")
    return list(module.body)


def parse_expression(source: str, *, error_kind: str) -> cst.BaseExpression:
    try:
        return cst.parse_expression(source)
    except cst.ParserSyntaxError as exc:
        raise AstPatchError(
            error_kind,
            f"Unable to parse expression: {exc.message}.",
            extra={
                "line": int(getattr(exc, "raw_line", 0) or 0),
                "column": int(getattr(exc, "raw_column", 0) or 0),
                "value": source,
            },
        ) from exc


def parse_class_field_statement(
    *,
    field_name: str,
    annotation: str,
    default_value: str,
) -> cst.BaseStatement:
    field_line = f"{field_name}: {annotation}"
    if default_value:
        field_line += f" = {default_value}"
    try:
        module = cst.parse_module(f"class _Temp:\n    {field_line}\n")
    except cst.ParserSyntaxError as exc:
        raise AstPatchError(
            "replacement_parse_failed",
            f"Unable to parse field definition for `{field_name}`: {exc.message}.",
            extra={
                "line": int(getattr(exc, "raw_line", 0) or 0),
                "column": int(getattr(exc, "raw_column", 0) or 0),
            },
        ) from exc
    class_node = module.body[0]
    assert isinstance(class_node, cst.ClassDef)
    return class_node.body.body[0]


def find_function_candidates(
    module: cst.Module,
    *,
    function_name: str,
    class_name: str | None,
) -> list[_FunctionCandidate]:
    collector = _CandidateCollector()
    module.visit(collector)
    if class_name:
        class_matches = [name for name in collector.class_names if name == class_name]
        if not class_matches:
            raise AstPatchError(
                "ast_target_not_found",
                f"Class `{class_name}` was not found.",
                extra={
                    "candidate_node_names": collector.class_names[:12],
                    "next_action_hint": "Read the containing class or confirm the class name before retrying.",
                },
            )
        if len(class_matches) > 1:
            raise AstPatchError(
                "ast_target_ambiguous",
                f"Multiple classes named `{class_name}` were found.",
                extra={
                    "candidate_node_names": collector.class_names[:12],
                    "next_action_hint": "Add a more specific locator or read the file to disambiguate the class target.",
                },
            )
    matches = [
        candidate
        for candidate in collector.function_candidates
        if candidate.function_name == function_name and (class_name is None or candidate.class_name == class_name)
    ]
    if not matches:
        raise AstPatchError(
            "ast_target_not_found",
            (
                f"Function `{function_name}` was not found."
                if not class_name
                else f"Method `{class_name}.{function_name}` was not found."
            ),
            extra={
                "candidate_node_names": [candidate.qualified_name for candidate in collector.function_candidates[:12]],
                "next_action_hint": "Read the containing function or add `target.class` if this is a method.",
            },
        )
    if len(matches) > 1 and class_name is None:
        raise AstPatchError(
            "ast_target_ambiguous",
            f"Multiple functions named `{function_name}` were found. Add `target.class` to disambiguate.",
            extra={
                "candidate_node_names": [candidate.qualified_name for candidate in matches[:12]],
                "next_action_hint": "Add `target.class` or read the file and choose a narrower structural locator.",
            },
        )
    return matches


def find_class_candidates(module: cst.Module, class_name: str) -> list[str]:
    collector = _CandidateCollector()
    module.visit(collector)
    matches = [name for name in collector.class_names if name == class_name]
    if not matches:
        raise AstPatchError(
            "ast_target_not_found",
            f"Class `{class_name}` was not found.",
            extra={
                "candidate_node_names": collector.class_names[:12],
                "next_action_hint": "Read the containing class or confirm the class name before retrying.",
            },
        )
    if len(matches) > 1:
        raise AstPatchError(
            "ast_target_ambiguous",
            f"Multiple classes named `{class_name}` were found.",
            extra={
                "candidate_node_names": collector.class_names[:12],
                "next_action_hint": "Add a more specific locator or read the file to disambiguate the class target.",
            },
        )
    return matches


def collect_matching_calls_in_scope(
    module: cst.Module,
    *,
    function_name: str,
    class_name: str | None,
    callee: str,
) -> list[str]:
    collector = _CallCollector(function_name=function_name, class_name=class_name, callee=callee)
    module.visit(collector)
    return collector.matches


def call_candidate_names_for_scope(
    module: cst.Module,
    *,
    function_name: str,
    class_name: str | None,
) -> list[str]:
    collector = _CallCollector(function_name=function_name, class_name=class_name, callee=None)
    module.visit(collector)
    return collector.candidates[:12]
