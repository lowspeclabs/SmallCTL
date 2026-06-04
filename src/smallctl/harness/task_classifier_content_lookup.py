from __future__ import annotations

import re


def needs_loop_for_content_lookup(task: str) -> bool:
    text = task.strip().lower()
    if not text:
        return False

    file_markers = (
        "file",
        "log",
        "logs",
        ".log",
        ".jsonl",
        ".txt",
        ".md",
        ".py",
        "/",
        "\\",
        "code",
        "source",
        "src",
    )
    content_queries = (
        "what is",
        "what's",
        "show",
        "read",
        "tell me",
        "summarize",
        "contents",
        "content",
        "line ",
        "lines ",
        "bug",
        "error",
        "inconsistent",
        "inconsistency",
        "issue",
        "debug",
        "still",
    )
    asks_for_specific_line = bool(re.search(r"\bline(?:s)?\s+\d+\b", text))
    asks_for_range = bool(re.search(r"\b\d+\s*-\s*\d+\b", text))
    asks_for_log_or_file_content = any(marker in text for marker in file_markers) and any(
        query in text for query in content_queries
    )
    asks_for_command_execution = (
        bool(re.search(r"\b(run|execute|exec)\b", text))
        and bool(
            re.search(
                r"\b(ls|dir|pwd|cd|cat|type|findstr|grep|git\s+status|get-childitem|pytest|python|powershell|pwsh|cmd)\b",
                text,
            )
        )
    )
    asks_for_directory_contents = (
        any(
            phrase in text
            for phrase in (
                "what files",
                "which files",
                "list files",
                "show files",
                "what folders",
                "which folders",
                "list folders",
                "show folders",
                "list directory",
                "show directory",
                "directory contents",
                "folder contents",
                "current directory",
                "this directory",
                "current folder",
                "this folder",
                "what is in this directory",
                "what is in the current directory",
                "what is in this folder",
                "what is in the current folder",
            )
        )
        or bool(
            re.search(
                r"\b(list|show|what|which)\b.*\b(files?|folders?|directories?|contents?)\b",
                text,
            )
        )
    )
    asks_where_specific_line_is = asks_for_specific_line and any(
        phrase in text for phrase in ("what is", "what's", "show", "read")
    )
    return (
        asks_for_specific_line
        or asks_for_range
        or asks_for_log_or_file_content
        or asks_where_specific_line_is
        or asks_for_directory_contents
        or asks_for_command_execution
    )
