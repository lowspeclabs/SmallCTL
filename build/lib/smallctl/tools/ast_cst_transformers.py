from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import libcst as cst
except Exception:  # pragma: no cover - import fallback is only for degraded envs
    class _FallbackCSTModule:
        class CSTVisitor:  # pragma: no cover - degraded env import shim
            pass

        class CSTTransformer:  # pragma: no cover - degraded env import shim
            pass

    cst = _FallbackCSTModule()


@dataclass(slots=True)
class PythonAstPatchOutcome:
    updated_text: str
    changed: bool
    matched_node_count: int
    touched_symbols: list[str] = field(default_factory=list)


class _CandidateCollector(cst.CSTVisitor):
    def __init__(self) -> None:
        self.class_names: list[str] = []
        self.function_candidates: list[Any] = []
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
class _FunctionCandidate:
    function_name: str
    class_name: str | None

    @property
    def qualified_name(self) -> str:
        if self.class_name:
            return f"{self.class_name}.{self.function_name}"
        return self.function_name


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
