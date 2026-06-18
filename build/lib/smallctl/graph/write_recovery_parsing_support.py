from __future__ import annotations

import re
from typing import Any


def _python_content_looks_complete(content: str) -> bool:
    line_count = len([line for line in str(content or "").splitlines() if line.strip()])
    if line_count >= 120:
        return True

    has_imports = bool(re.search(r"(?m)^\s*(?:from\s+\S+\s+import|import\s+\S+)", content))
    has_defs_or_classes = bool(re.search(r"(?m)^\s*(?:async\s+def|def|class)\s+\w+", content))
    has_tests = bool(
        re.search(r"(?m)^\s*(?:class\s+Test\w*|def\s+test_\w+)", content)
        or "unittest.main" in content
        or "pytest" in content
    )
    has_entrypoint = bool(re.search(r'(?m)^\s*if\s+__name__\s*==\s*["\']__main__["\']\s*:', content))
    if has_imports and has_defs_or_classes and (has_tests or has_entrypoint):
        return True
    if line_count >= 60 and has_defs_or_classes and has_entrypoint:
        return True
    return False


def _javascript_content_looks_complete(content: str) -> bool:
    line_count = len([line for line in str(content or "").splitlines() if line.strip()])
    if line_count >= 140:
        return True

    has_imports = bool(re.search(r"(?m)^\s*(?:import\s+.+from\s+|const\s+\w+\s*=\s*require\(", content))
    has_exports = bool(re.search(r"(?m)^\s*export\s+", content) or "module.exports" in content)
    has_tests = bool(re.search(r"\b(?:describe|it|test)\s*\(", content))
    return has_imports and has_exports and has_tests


def _go_content_looks_complete(content: str) -> bool:
    line_count = len([line for line in str(content or "").splitlines() if line.strip()])
    if line_count >= 140:
        return True

    has_package = bool(re.search(r"(?m)^\s*package\s+\w+", content))
    has_imports = bool(re.search(r"(?m)^\s*import\s+(?:\(|\w)", content))
    has_functions = bool(re.search(r"(?m)^\s*func\s+\w+", content))
    has_tests = bool(re.search(r"(?m)^\s*func\s+Test\w+\s*\(", content))
    return has_package and has_imports and has_functions and has_tests


def recovered_content_looks_like_complete_file(intent: Any) -> bool:
    target_path = str(getattr(intent, "path", "") or "").strip().lower()
    content = str(getattr(intent, "content", "") or "")
    if not target_path or not content.strip():
        return False
    if target_path.endswith(".py"):
        return _python_content_looks_complete(content)
    if target_path.endswith((".js", ".ts", ".tsx")):
        return _javascript_content_looks_complete(content)
    if target_path.endswith(".go"):
        return _go_content_looks_complete(content)
    return False


def _force_finalize_if_complete_file(intent: Any) -> bool:
    if not recovered_content_looks_like_complete_file(intent):
        return False
    intent.next_section_name = ""
    intent.next_section_name_origin = ""
    intent.evidence.append("complete_file_content")
    return True


def maybe_finalize_recovered_assistant_write(intent: Any) -> bool:
    source = str(getattr(intent, "source", "") or "").strip()
    if source not in {"assistant_text", "tool_args_content", "partial_tool_arguments"}:
        return False
    origin = str(getattr(intent, "next_section_name_origin", "") or "").strip()
    if origin not in {"session_default", "tool_args", "partial_tool_calls", "assistant_payload"}:
        return False

    if _force_finalize_if_complete_file(intent):
        intent.evidence.append("cleared_session_default_next_section_name")
        return True
    return False
