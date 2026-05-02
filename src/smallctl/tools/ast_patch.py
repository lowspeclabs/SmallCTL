from __future__ import annotations

import difflib
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import libcst as cst
    from libcst.codemod import CodemodContext
    from libcst.codemod.visitors import AddImportsVisitor
except Exception:  # pragma: no cover - import fallback is only for degraded envs
    class _FallbackCSTModule:
        class CSTVisitor:  # pragma: no cover - degraded env import shim
            pass

        class CSTTransformer:  # pragma: no cover - degraded env import shim
            pass

    cst = _FallbackCSTModule()
    CodemodContext = None
    AddImportsVisitor = None

from ..state import LoopState
from .common import fail, ok
from .fs_sessions import (
    _clone_section_ranges,
    _record_file_change,
)
from .fs_write_session_policy import _guard_write_session_staging_mutation
from .fs_write_sessions import (
    _content_hash,
    _read_text_file,
    _resolve_patch_source,
    _write_text_file,
    format_write_session_status_block,
    write_session_status_snapshot,
)


class AstPatchError(Exception):
    def __init__(
        self,
        error_kind: str,
        message: str,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_kind = error_kind
        self.message = message
        self.extra = dict(extra or {})


@dataclass(slots=True)
class PythonAstPatchOutcome:
    updated_text: str
    changed: bool
    matched_node_count: int
    touched_symbols: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _FunctionCandidate:
    function_name: str
    class_name: str | None

    @property
    def qualified_name(self) -> str:
        if self.class_name:
            return f"{self.class_name}.{self.function_name}"
        return self.function_name


async def handle_ast_patch(
    *,
    path: str,
    language: str | None = "python",
    operation: str | None = None,
    target: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    cwd: str | None = None,
    encoding: str = "utf-8",
    state: LoopState | None = None,
    session_id: str | None = None,
    write_session_id: str | None = None,
    expected_followup_verifier: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    from .fs import (
        _guard_suspicious_temp_root_path,
        _mark_repeat_patch,
        _repair_cycle_allows_patch,
        _repair_cycle_reads,
        _resolve,
    )

    suspicious_path = _guard_suspicious_temp_root_path(path)
    if suspicious_path is not None:
        return suspicious_path

    normalized_language = str(language or "python").strip().lower() or "python"
    normalized_operation = str(operation or "").strip()
    normalized_target = dict(target or {})
    normalized_payload = dict(payload or {})
    normalized_dry_run = bool(dry_run)
    target_path = _resolve(path, cwd)

    if not write_session_id and session_id:
        write_session_id = session_id

    if not normalized_operation:
        return fail(
            "AST patch operation is required.",
            metadata={
                "path": str(target_path),
                "requested_path": path,
                "error_kind": "ast_operation_invalid",
                "language": normalized_language,
            },
        )

    if normalized_language != "python":
        return _unsupported_language_failure(
            path=target_path,
            requested_path=path,
            language=normalized_language,
            operation=normalized_operation,
        )
    if cst is None:
        return fail(
            "`ast_patch` requires `libcst` for Python structural edits, but the dependency is not available in this environment.",
            metadata={
                "path": str(target_path),
                "requested_path": path,
                "error_kind": "ast_operation_invalid",
                "language": normalized_language,
                "operation": normalized_operation,
                "next_action_hint": "Install `libcst` in the active environment, then retry the structural patch.",
            },
        )

    staging_guard = _guard_write_session_staging_mutation(
        tool_name="ast_patch",
        path=path,
        state=state,
        cwd=cwd,
        write_session_id=write_session_id,
        encoding=encoding,
    )
    if staging_guard is not None:
        return staging_guard

    if write_session_id:
        try:
            source_path, _, session, staged_only = _resolve_patch_source(
                state,
                path,
                cwd=cwd,
                encoding=encoding,
                write_session_id=write_session_id,
            )
        except (LookupError, ValueError) as exc:
            return fail(
                str(exc),
                metadata={
                    "path": str(target_path),
                    "requested_path": path,
                    "error_kind": "session_id_mismatch",
                    "write_session_id": write_session_id,
                    "language": normalized_language,
                    "operation": normalized_operation,
                },
            )
    else:
        session = getattr(state, "write_session", None) if state is not None else None
        source_path, _, session, staged_only = _resolve_patch_source(
            state,
            path,
            cwd=cwd,
            encoding=encoding,
        )

    if not staged_only and not normalized_dry_run and not _repair_cycle_allows_patch(state, target_path):
        _mark_repeat_patch(state)
        return fail(
            "Repair cycle requires reading the target file before patching it again.",
            metadata={
                "path": str(target_path),
                "requested_path": path,
                "system_repair_cycle_id": getattr(state, "repair_cycle_id", ""),
                "required_read_paths": _repair_cycle_reads(state),
                "error_kind": "repair_cycle_read_required",
                "language": normalized_language,
                "operation": normalized_operation,
            },
        )

    if staged_only and not source_path.exists():
        return fail(
            f"Active staged copy `{source_path}` is missing for target `{path}`.",
            metadata={
                "path": str(target_path),
                "requested_path": path,
                "source_path": str(source_path),
                "staged_only": True,
                "error_kind": "ast_target_not_found",
                "language": normalized_language,
                "operation": normalized_operation,
                "write_session_id": str(getattr(session, "write_session_id", "") or "").strip(),
            },
        )

    if not staged_only and not normalized_dry_run:
        from ..risk_policy import evaluate_risk_policy

        risk_decision = evaluate_risk_policy(
            state if state is not None else LoopState(cwd=str(Path.cwd())),
            tool_name="ast_patch",
            tool_risk="high",
            phase=str((state.current_phase if state is not None else "") or ""),
            action=f"Structurally patch file {path}",
            expected_effect="Apply a targeted structural edit to the target file.",
            rollback="Restore the previous file contents from the snapshot or staging file.",
            verification="Read back the file and run the smallest useful verifier.",
        )
        if not risk_decision.allowed:
            return fail(
                risk_decision.reason,
                metadata={
                    "path": path,
                    "reason": "missing_supported_claim",
                    "proof_bundle": risk_decision.proof_bundle,
                },
            )

    try:
        source_text = _read_text_file(source_path, encoding=encoding)
    except Exception as exc:
        return fail(
            f"Unable to read file for AST patching: {exc}",
            metadata={
                "path": str(target_path),
                "requested_path": path,
                "source_path": str(source_path),
                "error_kind": "parse_failed",
                "language": normalized_language,
                "operation": normalized_operation,
            },
        )

    try:
        outcome = _apply_python_ast_patch(
            source_text=source_text,
            operation=normalized_operation,
            target=normalized_target,
            payload=normalized_payload,
        )
    except AstPatchError as exc:
        return fail(
            exc.message,
            metadata={
                "path": str(target_path),
                "requested_path": path,
                "source_path": str(source_path),
                "staged_only": staged_only,
                "error_kind": exc.error_kind,
                "language": normalized_language,
                "operation": normalized_operation,
                "target": normalized_target,
                **exc.extra,
                **(
                    {
                        "write_session_id": str(getattr(session, "write_session_id", "") or "").strip(),
                    }
                    if session is not None
                    else {}
                ),
            },
        )

    status_block = None
    if outcome.changed and not normalized_dry_run:
        try:
            _write_text_file(source_path, outcome.updated_text, encoding=encoding)
        except Exception as exc:
            return fail(
                f"Unable to patch file: {exc}",
                metadata={
                    "path": str(target_path),
                    "requested_path": path,
                    "source_path": str(source_path),
                    "staged_only": staged_only,
                    "error_kind": "ast_operation_invalid",
                    "language": normalized_language,
                    "operation": normalized_operation,
                },
            )

        if session is not None:
            session.write_last_staged_hash = _content_hash(outcome.updated_text)
            session.write_last_attempt_sections = list(getattr(session, "write_sections_completed", []) or [])
            session.write_last_attempt_ranges = _clone_section_ranges(getattr(session, "write_section_ranges", {}) or {})
            status_snapshot = write_session_status_snapshot(
                session,
                cwd=cwd,
                finalized=False,
                encoding=encoding,
            )
            status_block = format_write_session_status_block(status_snapshot)

        _record_file_change(state, target_path)

    metadata = _build_ast_patch_metadata(
        path=target_path,
        requested_path=path,
        source_path=source_path,
        session=session,
        staged_only=staged_only,
        language=normalized_language,
        operation=normalized_operation,
        target=normalized_target,
        payload=normalized_payload,
        changed=outcome.changed,
        updated_text=outcome.updated_text,
        original_text=source_text,
        matched_node_count=outcome.matched_node_count,
        touched_symbols=outcome.touched_symbols,
        dry_run=normalized_dry_run,
        expected_followup_verifier=expected_followup_verifier,
        staging_path=source_path if staged_only else None,
        status_block=status_block,
    )

    if normalized_dry_run:
        if outcome.changed:
            return ok(
                f"Dry run prepared structural patch for `{path}`.",
                metadata=metadata,
            )
        return ok(
            f"Dry run found no structural change needed for `{path}`.",
            metadata=metadata,
        )

    if not outcome.changed:
        return ok(
            f"No structural change was needed for `{path}`.",
            metadata=metadata,
        )

    message = f"Structurally patched `{path}` with `{normalized_operation}`."
    if staged_only:
        message += f" Staged copy: `{source_path}`."
        if status_block:
            message += f"\n{status_block}"
    return ok(message, metadata=metadata)


def _apply_python_ast_patch(
    *,
    source_text: str,
    operation: str,
    target: dict[str, Any],
    payload: dict[str, Any],
) -> PythonAstPatchOutcome:
    module = _parse_python_module(source_text)
    normalized_operation = str(operation or "").strip().lower()
    if normalized_operation == "add_import":
        return _python_add_import(source_text, module, target)
    if normalized_operation == "replace_function":
        return _python_replace_function(source_text, module, target, payload)
    if normalized_operation == "insert_in_function":
        return _python_insert_in_function(source_text, module, target, payload)
    if normalized_operation == "update_call_keyword":
        return _python_update_call_keyword(source_text, module, target, payload)
    if normalized_operation == "add_dataclass_field":
        return _python_add_dataclass_field(source_text, module, target, payload)
    raise AstPatchError(
        "ast_operation_invalid",
        f"Unsupported AST patch operation `{operation}`.",
        extra={"supported_operations": _supported_ast_patch_operations()},
    )


def _parse_python_module(source_text: str) -> cst.Module:
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


def _python_add_import(
    source_text: str,
    module: cst.Module,
    target: dict[str, Any],
) -> PythonAstPatchOutcome:
    module_name = str(target.get("module") or "").strip()
    import_name = str(target.get("name") or "").strip()
    style = str(target.get("style") or ("from" if import_name else "import")).strip().lower()
    if not module_name:
        raise AstPatchError("ast_operation_invalid", "add_import requires `target.module`.")
    if style == "from" and not import_name:
        raise AstPatchError("ast_operation_invalid", "add_import with `style='from'` requires `target.name`.")
    if style not in {"from", "import"}:
        raise AstPatchError(
            "ast_operation_invalid",
            "add_import only supports `style` values `from` and `import`.",
        )

    context = CodemodContext()
    if style == "from":
        AddImportsVisitor.add_needed_import(context, module_name, import_name)
    else:
        AddImportsVisitor.add_needed_import(context, module_name)
    updated_module = AddImportsVisitor(context).transform_module(module)
    updated_text = updated_module.code
    touched = [module_name]
    if import_name:
        touched.append(import_name)
    return PythonAstPatchOutcome(
        updated_text=updated_text,
        changed=(updated_text != source_text),
        matched_node_count=1,
        touched_symbols=_dedupe_symbols(touched),
    )


def _python_replace_function(
    source_text: str,
    module: cst.Module,
    target: dict[str, Any],
    payload: dict[str, Any],
) -> PythonAstPatchOutcome:
    function_name = str(target.get("function") or "").strip()
    class_name = str(target.get("class") or "").strip()
    replace_mode = str(payload.get("replace") or "entire_node").strip().lower()
    replacement_source = str(payload.get("source") or "").strip()
    if replace_mode != "entire_node":
        raise AstPatchError(
            "ast_operation_invalid",
            "replace_function currently supports only `payload.replace='entire_node'`.",
        )
    if not function_name or not replacement_source:
        raise AstPatchError(
            "ast_operation_invalid",
            "replace_function requires `target.function` and `payload.source`.",
        )

    matches = _find_function_candidates(module, function_name=function_name, class_name=class_name or None)
    replacement_node = _parse_function_replacement(replacement_source)
    transformer = _ReplaceFunctionTransformer(
        function_name=function_name,
        class_name=class_name or None,
        replacement_node=replacement_node,
    )
    updated_module = module.visit(transformer)
    touched = [function_name]
    if class_name:
        touched.append(class_name)
    updated_text = updated_module.code
    return PythonAstPatchOutcome(
        updated_text=updated_text,
        changed=(updated_text != source_text),
        matched_node_count=len(matches),
        touched_symbols=_dedupe_symbols(touched),
    )


def _python_insert_in_function(
    source_text: str,
    module: cst.Module,
    target: dict[str, Any],
    payload: dict[str, Any],
) -> PythonAstPatchOutcome:
    function_name = str(target.get("function") or "").strip()
    class_name = str(target.get("class") or "").strip()
    position = str(target.get("position") or "start").strip().lower()
    statements_source = str(payload.get("statements") or "").strip()
    if not function_name or not statements_source:
        raise AstPatchError(
            "ast_operation_invalid",
            "insert_in_function requires `target.function` and `payload.statements`.",
        )
    if position not in {"start", "end", "before_return"}:
        raise AstPatchError(
            "ast_operation_invalid",
            "insert_in_function supports `position` values `start`, `end`, and `before_return`.",
        )

    matches = _find_function_candidates(module, function_name=function_name, class_name=class_name or None)
    statements = _parse_statement_block(statements_source)
    transformer = _InsertInFunctionTransformer(
        function_name=function_name,
        class_name=class_name or None,
        position=position,
        statements=statements,
    )
    updated_module = module.visit(transformer)
    touched = [function_name]
    if class_name:
        touched.append(class_name)
    return PythonAstPatchOutcome(
        updated_text=updated_module.code,
        changed=(updated_module.code != source_text),
        matched_node_count=len(matches),
        touched_symbols=_dedupe_symbols(touched),
    )


def _python_update_call_keyword(
    source_text: str,
    module: cst.Module,
    target: dict[str, Any],
    payload: dict[str, Any],
) -> PythonAstPatchOutcome:
    scope_function = str(target.get("scope_function") or target.get("function") or "").strip()
    class_name = str(target.get("class") or "").strip()
    callee = str(target.get("callee") or "").strip()
    keyword = str(target.get("keyword") or "").strip()
    occurrence_raw = target.get("occurrence")
    if not scope_function or not callee or not keyword:
        raise AstPatchError(
            "ast_operation_invalid",
            "update_call_keyword requires `target.scope_function`, `target.callee`, and `target.keyword`.",
        )
    mode = str(payload.get("mode") or "set").strip().lower()
    if mode not in {"set", "remove"}:
        raise AstPatchError(
            "ast_operation_invalid",
            "update_call_keyword supports `payload.mode` values `set` and `remove`.",
        )
    occurrence = int(occurrence_raw) if occurrence_raw is not None else 1
    if occurrence < 1:
        raise AstPatchError("ast_operation_invalid", "`target.occurrence` must be at least 1.")

    _find_function_candidates(module, function_name=scope_function, class_name=class_name or None)
    if mode == "set":
        value_text = str(payload.get("value") or "").strip()
        if not value_text:
            raise AstPatchError(
                "ast_operation_invalid",
                "update_call_keyword with `mode='set'` requires `payload.value`.",
            )
        value_expr = _parse_expression(value_text, error_kind="replacement_parse_failed")
    else:
        value_text = ""
        value_expr = None

    call_candidates = _collect_matching_calls_in_scope(
        module,
        function_name=scope_function,
        class_name=class_name or None,
        callee=callee,
    )
    if not call_candidates:
        raise AstPatchError(
            "ast_target_not_found",
            f"No call to `{callee}` was found inside function `{scope_function}`.",
            extra={
                "candidate_node_names": _call_candidate_names_for_scope(
                    module,
                    function_name=scope_function,
                    class_name=class_name or None,
                ),
                "next_action_hint": "Read the containing function and add a narrower locator if multiple calls are nearby.",
            },
        )
    if occurrence > len(call_candidates):
        raise AstPatchError(
            "ast_target_not_found",
            f"Only {len(call_candidates)} matching call(s) to `{callee}` were found inside function `{scope_function}`.",
            extra={
                "candidate_node_names": [callee for _ in call_candidates[:8]],
                "next_action_hint": "Read the containing function to confirm the call shape before retrying.",
            },
        )
    if len(call_candidates) > 1 and occurrence_raw is None:
        raise AstPatchError(
            "ast_target_ambiguous",
            f"Multiple calls to `{callee}` were found inside function `{scope_function}`. Add `target.occurrence` to disambiguate.",
            extra={
                "candidate_node_names": [f"{callee}@{index}" for index in range(1, min(len(call_candidates), 8) + 1)],
                "next_action_hint": "Add `target.occurrence` or read the function and choose a narrower structural locator.",
            },
        )

    transformer = _UpdateCallKeywordTransformer(
        function_name=scope_function,
        class_name=class_name or None,
        callee=callee,
        keyword=keyword,
        mode=mode,
        occurrence=occurrence,
        value_expr=value_expr,
    )
    updated_module = module.visit(transformer)
    touched = [scope_function, keyword]
    if class_name:
        touched.append(class_name)
    return PythonAstPatchOutcome(
        updated_text=updated_module.code,
        changed=(updated_module.code != source_text),
        matched_node_count=1,
        touched_symbols=_dedupe_symbols(touched),
    )


def _python_add_dataclass_field(
    source_text: str,
    module: cst.Module,
    target: dict[str, Any],
    payload: dict[str, Any],
) -> PythonAstPatchOutcome:
    class_name = str(target.get("class") or "").strip()
    field_name = str(target.get("field") or "").strip()
    annotation = str(payload.get("annotation") or "").strip()
    default_value = str(payload.get("default") or "").strip()
    if_exists = str(payload.get("if_exists") or "").strip().lower()
    if not class_name or not field_name or not annotation:
        raise AstPatchError(
            "ast_operation_invalid",
            "add_dataclass_field requires `target.class`, `target.field`, and `payload.annotation`.",
        )

    _find_class_candidates(module, class_name)
    if default_value:
        _parse_expression(default_value, error_kind="replacement_parse_failed")
    field_statement = _parse_class_field_statement(
        field_name=field_name,
        annotation=annotation,
        default_value=default_value,
    )
    transformer = _AddDataclassFieldTransformer(
        class_name=class_name,
        field_name=field_name,
        field_statement=field_statement,
        if_exists=if_exists,
    )
    updated_module = module.visit(transformer)
    return PythonAstPatchOutcome(
        updated_text=updated_module.code,
        changed=(updated_module.code != source_text),
        matched_node_count=1,
        touched_symbols=_dedupe_symbols([class_name, field_name]),
    )


def _parse_function_replacement(source: str) -> cst.FunctionDef:
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


def _parse_statement_block(source: str) -> list[cst.BaseStatement]:
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


def _parse_expression(source: str, *, error_kind: str) -> cst.BaseExpression:
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


def _parse_class_field_statement(
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


def _find_function_candidates(
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


def _find_class_candidates(module: cst.Module, class_name: str) -> list[str]:
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


def _collect_matching_calls_in_scope(
    module: cst.Module,
    *,
    function_name: str,
    class_name: str | None,
    callee: str,
) -> list[str]:
    collector = _CallCollector(function_name=function_name, class_name=class_name, callee=callee)
    module.visit(collector)
    return collector.matches


def _call_candidate_names_for_scope(
    module: cst.Module,
    *,
    function_name: str,
    class_name: str | None,
) -> list[str]:
    collector = _CallCollector(function_name=function_name, class_name=class_name, callee=None)
    module.visit(collector)
    return collector.candidates[:12]


class _CandidateCollector(cst.CSTVisitor):
    def __init__(self) -> None:
        self.class_names: list[str] = []
        self.function_candidates: list[_FunctionCandidate] = []
        self._class_stack: list[str] = []

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self.class_names.append(node.name.value)
        self._class_stack.append(node.name.value)

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        del original_node
        self._class_stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.function_candidates.append(
            _FunctionCandidate(
                function_name=node.name.value,
                class_name=self._class_stack[-1] if self._class_stack else None,
            )
        )


class _ReplaceFunctionTransformer(cst.CSTTransformer):
    def __init__(
        self,
        *,
        function_name: str,
        class_name: str | None,
        replacement_node: cst.FunctionDef,
    ) -> None:
        self.function_name = function_name
        self.class_name = class_name
        self.replacement_node = replacement_node
        self._class_stack: list[str] = []

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self._class_stack.append(node.name.value)

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        del original_node
        self._class_stack.pop()
        return updated_node

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.BaseStatement:
        del original_node
        if _matches_function_target(
            updated_node.name.value,
            current_class=self._class_stack[-1] if self._class_stack else None,
            target_function=self.function_name,
            target_class=self.class_name,
        ):
            return self.replacement_node
        return updated_node


class _InsertInFunctionTransformer(cst.CSTTransformer):
    def __init__(
        self,
        *,
        function_name: str,
        class_name: str | None,
        position: str,
        statements: list[cst.BaseStatement],
    ) -> None:
        self.function_name = function_name
        self.class_name = class_name
        self.position = position
        self.statements = tuple(statements)
        self._class_stack: list[str] = []

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self._class_stack.append(node.name.value)

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        del original_node
        self._class_stack.pop()
        return updated_node

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> cst.FunctionDef:
        del original_node
        if not _matches_function_target(
            updated_node.name.value,
            current_class=self._class_stack[-1] if self._class_stack else None,
            target_function=self.function_name,
            target_class=self.class_name,
        ):
            return updated_node
        return _insert_statements_into_function(updated_node, position=self.position, statements=self.statements)

class _UpdateCallKeywordTransformer(cst.CSTTransformer):
    def __init__(
        self,
        *,
        function_name: str,
        class_name: str | None,
        callee: str,
        keyword: str,
        mode: str,
        occurrence: int,
        value_expr: cst.BaseExpression | None,
    ) -> None:
        self.function_name = function_name
        self.class_name = class_name
        self.callee = callee
        self.keyword = keyword
        self.mode = mode
        self.occurrence = occurrence
        self.value_expr = value_expr
        self._class_stack: list[str] = []
        self._active_scope_depth = 0
        self._nested_class_depth = 0
        self._match_index = 0

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        if self._active_scope_depth > 0:
            self._nested_class_depth += 1
        self._class_stack.append(node.name.value)

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        del original_node
        self._class_stack.pop()
        if self._active_scope_depth > 0 and self._nested_class_depth > 0:
            self._nested_class_depth -= 1
        return updated_node

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        is_target = _matches_function_target(
            node.name.value,
            current_class=self._class_stack[-1] if self._class_stack else None,
            target_function=self.function_name,
            target_class=self.class_name,
        )
        if is_target and self._active_scope_depth == 0:
            self._active_scope_depth = 1
        elif self._active_scope_depth > 0:
            self._active_scope_depth += 1

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        del original_node
        if self._active_scope_depth > 0:
            self._active_scope_depth -= 1
        return updated_node

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        del original_node
        if self._active_scope_depth != 1 or self._nested_class_depth > 0:
            return updated_node
        if _expr_to_dotted_name(updated_node.func) != self.callee:
            return updated_node
        self._match_index += 1
        if self._match_index != self.occurrence:
            return updated_node
        return _rewrite_call_keyword_args(
            updated_node,
            keyword=self.keyword,
            mode=self.mode,
            value_expr=self.value_expr,
        )


class _AddDataclassFieldTransformer(cst.CSTTransformer):
    def __init__(
        self,
        *,
        class_name: str,
        field_name: str,
        field_statement: cst.BaseStatement,
        if_exists: str,
    ) -> None:
        self.class_name = class_name
        self.field_name = field_name
        self.field_statement = field_statement
        self.if_exists = if_exists

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        del original_node
        if updated_node.name.value != self.class_name:
            return updated_node
        body = list(updated_node.body.body)
        existing_names = [_class_field_name(stmt) for stmt in body]
        if self.field_name in existing_names:
            if self.if_exists == "ignore":
                return updated_node
            raise AstPatchError(
                "ast_noop",
                f"Field `{self.field_name}` already exists on class `{self.class_name}`.",
                extra={
                    "candidate_node_names": [name for name in existing_names if name][:12],
                    "next_action_hint": "Set `payload.if_exists='ignore'` to treat an existing field as a safe no-op.",
                },
            )
        docstring_index = 1 if body and _is_docstring_statement(body[0]) else 0
        body_after_docstring = body[docstring_index:]
        if len(body_after_docstring) == 1 and _is_pass_statement(body_after_docstring[0]):
            new_body = body[:docstring_index] + [self.field_statement]
        else:
            insert_index = len(body)
            for index, stmt in enumerate(body[docstring_index:], start=docstring_index):
                if _starts_type_or_function_block(stmt):
                    insert_index = index
                    break
            new_body = body[:insert_index] + [self.field_statement] + body[insert_index:]
        return updated_node.with_changes(body=updated_node.body.with_changes(body=new_body))


class _CallCollector(cst.CSTVisitor):
    def __init__(
        self,
        *,
        function_name: str,
        class_name: str | None,
        callee: str | None,
    ) -> None:
        self.function_name = function_name
        self.class_name = class_name
        self.callee = callee
        self.matches: list[str] = []
        self.candidates: list[str] = []
        self._class_stack: list[str] = []
        self._active_scope_depth = 0
        self._nested_class_depth = 0

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        if self._active_scope_depth > 0:
            self._nested_class_depth += 1
        self._class_stack.append(node.name.value)

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        del original_node
        self._class_stack.pop()
        if self._active_scope_depth > 0 and self._nested_class_depth > 0:
            self._nested_class_depth -= 1

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        is_target = _matches_function_target(
            node.name.value,
            current_class=self._class_stack[-1] if self._class_stack else None,
            target_function=self.function_name,
            target_class=self.class_name,
        )
        if is_target and self._active_scope_depth == 0:
            self._active_scope_depth = 1
        elif self._active_scope_depth > 0:
            self._active_scope_depth += 1

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        del original_node
        if self._active_scope_depth > 0:
            self._active_scope_depth -= 1

    def visit_Call(self, node: cst.Call) -> None:
        if self._active_scope_depth != 1 or self._nested_class_depth > 0:
            return
        name = _expr_to_dotted_name(node.func)
        if not name:
            return
        if name not in self.candidates:
            self.candidates.append(name)
        if self.callee is None or name == self.callee:
            self.matches.append(name)


def _matches_function_target(
    function_name: str,
    *,
    current_class: str | None,
    target_function: str,
    target_class: str | None,
) -> bool:
    if function_name != target_function:
        return False
    if target_class is None:
        return True
    return current_class == target_class


def _insert_statements_into_function(
    node: cst.FunctionDef,
    *,
    position: str,
    statements: tuple[cst.BaseStatement, ...],
) -> cst.FunctionDef:
    body = list(node.body.body)
    if position == "start":
        insert_index = 1 if body and _is_docstring_statement(body[0]) else 0
        new_body = body[:insert_index] + list(statements) + body[insert_index:]
    elif position == "end":
        new_body = body + list(statements)
    else:
        return_indices = [index for index, stmt in enumerate(body) if _is_return_statement(stmt)]
        if not return_indices:
            raise AstPatchError(
                "ast_target_not_found",
                f"No return statement was found inside function `{node.name.value}`.",
                extra={
                    "candidate_node_names": [node.name.value],
                    "next_action_hint": "Read the containing function and confirm there is a top-level return to anchor the insertion.",
                },
            )
        if len(return_indices) > 1:
            raise AstPatchError(
                "ast_target_ambiguous",
                f"Multiple return statements matched inside function `{node.name.value}`. Add a narrower locator before using `before_return`.",
                extra={
                    "candidate_node_names": [f"return@{index + 1}" for index in return_indices[:8]],
                    "next_action_hint": "Read the containing function and choose a narrower insertion point.",
                },
            )
        insert_index = return_indices[0]
        new_body = body[:insert_index] + list(statements) + body[insert_index:]
    return node.with_changes(body=node.body.with_changes(body=new_body))


def _rewrite_call_keyword_args(
    node: cst.Call,
    *,
    keyword: str,
    mode: str,
    value_expr: cst.BaseExpression | None,
) -> cst.Call:
    args = list(node.args)
    keyword_indices = [
        index
        for index, arg in enumerate(args)
        if arg.keyword is not None and arg.keyword.value == keyword
    ]
    if mode == "remove":
        if not keyword_indices:
            return node
        return node.with_changes(args=[arg for index, arg in enumerate(args) if index not in keyword_indices])

    tight_equal = cst.AssignEqual(
        whitespace_before=cst.SimpleWhitespace(""),
        whitespace_after=cst.SimpleWhitespace(""),
    )
    new_arg = cst.Arg(value=value_expr, keyword=cst.Name(keyword), equal=tight_equal)
    if keyword_indices:
        first_index = keyword_indices[0]
        args[first_index] = args[first_index].with_changes(value=value_expr, equal=tight_equal)
    else:
        args.append(new_arg)
    return node.with_changes(args=args)


def _expr_to_dotted_name(node: cst.BaseExpression) -> str:
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        base = _expr_to_dotted_name(node.value)
        return f"{base}.{node.attr.value}" if base else node.attr.value
    return ""


def _is_docstring_statement(statement: cst.BaseStatement) -> bool:
    if not isinstance(statement, cst.SimpleStatementLine) or len(statement.body) != 1:
        return False
    expression = statement.body[0]
    return isinstance(expression, cst.Expr) and isinstance(expression.value, cst.SimpleString)


def _is_return_statement(statement: cst.BaseStatement) -> bool:
    if not isinstance(statement, cst.SimpleStatementLine):
        return False
    return any(isinstance(item, cst.Return) for item in statement.body)


def _is_pass_statement(statement: cst.BaseStatement) -> bool:
    if not isinstance(statement, cst.SimpleStatementLine):
        return False
    return any(isinstance(item, cst.Pass) for item in statement.body)


def _class_field_name(statement: cst.BaseStatement) -> str:
    if not isinstance(statement, cst.SimpleStatementLine) or len(statement.body) != 1:
        return ""
    inner = statement.body[0]
    if isinstance(inner, cst.AnnAssign) and isinstance(inner.target, cst.Name):
        return inner.target.value
    if isinstance(inner, cst.Assign) and len(inner.targets) == 1 and isinstance(inner.targets[0].target, cst.Name):
        return inner.targets[0].target.value
    return ""


def _starts_type_or_function_block(statement: cst.BaseStatement) -> bool:
    if isinstance(statement, (cst.FunctionDef, cst.ClassDef)):
        return True
    return False


def _supported_ast_patch_operations() -> list[str]:
    return [
        "add_import",
        "replace_function",
        "insert_in_function",
        "update_call_keyword",
        "add_dataclass_field",
    ]


def _unsupported_language_failure(
    *,
    path: Path,
    requested_path: str,
    language: str,
    operation: str,
) -> dict[str, Any]:
    return fail(
        f"Language `{language}` is not supported by `ast_patch` yet. Use `language='python'` for v1.",
        metadata={
            "path": str(path),
            "requested_path": requested_path,
            "language": language,
            "operation": operation,
            "error_kind": "unsupported_language",
            "supported_languages": ["python"],
        },
    )


def _build_ast_patch_metadata(
    *,
    path: Path,
    requested_path: str,
    source_path: Path,
    session: Any | None,
    staged_only: bool,
    language: str,
    operation: str,
    target: dict[str, Any],
    payload: dict[str, Any],
    changed: bool,
    updated_text: str,
    original_text: str,
    matched_node_count: int,
    touched_symbols: list[str],
    dry_run: bool,
    expected_followup_verifier: str | None,
    staging_path: Path | None,
    status_block: str | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "path": str(path),
        "requested_path": requested_path,
        "source_path": str(source_path),
        "staged_only": staged_only,
        "language": language,
        "operation": operation,
        "target": target,
        "payload": payload,
        "changed": changed,
        "matched_node_count": matched_node_count,
        "touched_symbols": list(touched_symbols),
        "diff_preview": _build_diff_preview(original_text, updated_text) if changed else "",
        "bytes": len((updated_text if changed else original_text).encode("utf-8")),
        "dry_run": dry_run,
        "expected_followup_verifier": str(expected_followup_verifier or ""),
    }
    if staging_path is not None:
        metadata["staging_path"] = str(staging_path)
    if session is not None:
        metadata["write_session_id"] = str(getattr(session, "write_session_id", "") or "").strip()
        metadata["write_session_status_block"] = status_block or ""
    return metadata


def _dedupe_symbols(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        candidate = str(value or "").strip()
        if not candidate or candidate in deduped:
            continue
        deduped.append(candidate)
    return deduped


def _build_diff_preview(before: str, after: str, *, limit: int = 1400) -> str:
    if before == after:
        return ""
    preview = "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile="before",
            tofile="after",
            lineterm="",
            n=1,
        )
    ).strip()
    if len(preview) <= limit:
        return preview
    return preview[: limit - 14].rstrip() + "\n...[truncated]"
