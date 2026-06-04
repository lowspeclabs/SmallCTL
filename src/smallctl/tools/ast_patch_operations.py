from __future__ import annotations

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

from .ast_cst_transformers import (
    AstPatchError,
    PythonAstPatchOutcome,
    _AddDataclassFieldTransformer,
    _InsertInFunctionTransformer,
    _ReplaceFunctionTransformer,
    _UpdateCallKeywordTransformer,
)
from .ast_patch_parsing import (
    call_candidate_names_for_scope,
    collect_matching_calls_in_scope,
    find_class_candidates,
    find_function_candidates,
    parse_class_field_statement,
    parse_expression,
    parse_function_replacement,
    parse_python_module,
    parse_statement_block,
)
from .ast_patch_results import dedupe_symbols, supported_ast_patch_operations


def apply_python_ast_patch(
    *,
    source_text: str,
    operation: str,
    target: dict[str, Any],
    payload: dict[str, Any],
) -> PythonAstPatchOutcome:
    module = parse_python_module(source_text)
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
        extra={"supported_operations": supported_ast_patch_operations()},
    )


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
        touched_symbols=dedupe_symbols(touched),
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

    matches = find_function_candidates(module, function_name=function_name, class_name=class_name or None)
    replacement_node = parse_function_replacement(replacement_source)
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
        touched_symbols=dedupe_symbols(touched),
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

    matches = find_function_candidates(module, function_name=function_name, class_name=class_name or None)
    statements = parse_statement_block(statements_source)
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
        touched_symbols=dedupe_symbols(touched),
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

    find_function_candidates(module, function_name=scope_function, class_name=class_name or None)
    if mode == "set":
        value_text = str(payload.get("value") or "").strip()
        if not value_text:
            raise AstPatchError(
                "ast_operation_invalid",
                "update_call_keyword with `mode='set'` requires `payload.value`.",
            )
        value_expr = parse_expression(value_text, error_kind="replacement_parse_failed")
    else:
        value_expr = None

    call_candidates = collect_matching_calls_in_scope(
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
                "candidate_node_names": call_candidate_names_for_scope(
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
                "next_action_hint": "Add `target.occurrence` or read the file and choose a narrower structural locator.",
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
        touched_symbols=dedupe_symbols(touched),
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

    find_class_candidates(module, class_name)
    if default_value:
        parse_expression(default_value, error_kind="replacement_parse_failed")
    field_statement = parse_class_field_statement(
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
        touched_symbols=dedupe_symbols([class_name, field_name]),
    )
