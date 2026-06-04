from __future__ import annotations

import ast
from pathlib import Path


def _file_has_symbol(path: Path, symbol: str) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False
    parts = [part for part in str(symbol or "").split(".") if part]
    if not parts:
        return False
    if len(parts) == 1:
        return any(_node_defines_name(node, parts[0]) for node in tree.body)
    class_name, member = parts[0], parts[1]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return any(_node_defines_name(child, member) for child in node.body)
    return False


def _node_defines_name(node: ast.AST, name: str) -> bool:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name == name
    if isinstance(node, ast.Assign):
        return any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    if isinstance(node, ast.AnnAssign):
        return isinstance(node.target, ast.Name) and node.target.id == name
    return False
