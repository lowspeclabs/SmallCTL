from __future__ import annotations

import re
from pathlib import Path
from typing import Any


# English stop-words and Python keywords that commonly appear in task descriptions
# but are not implementation symbols we should expect to find in source code.
_STOP_WORDS: set[str] = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "don",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
"where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "won",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    # python keywords / builtins that often appear in prose
    "true",
    "false",
    "none",
    "import",
    "from",
    "class",
    "def",
    "return",
    "yield",
    "pass",
    "raise",
    "try",
    "except",
    "finally",
    "for",
    "while",
    "if",
    "else",
    "elif",
    "break",
    "continue",
    "lambda",
    "with",
    "as",
    "assert",
    "del",
    "global",
    "nonlocal",
    "print",
    "len",
    "range",
    "list",
    "dict",
    "set",
    "str",
    "int",
    "float",
    "bool",
    "tuple",
    "type",
    "object",
    "name",
    "main",
    "init",
    "self",
    "cls",
    # common prose verbs
    "build",
    "create",
    "write",
    "make",
    "implement",
    "include",
    "contains",
    "uses",
    "using",
    "use",
    "add",
    "ensure",
    "provide",
    "need",
    "needs",
    "must",
    "should",
    "shall",
    "will",
    "would",
    "may",
    "might",
    "let",
    "like",
    "want",
    "wanted",
    "get",
    "got",
    "given",
    "give",
    "put",
    "set",
    "run",
    "called",
    "call",
    "used",
    "made",
    "done",
    "does",
    "did",
    "doing",
    "having",
    "being",
    "seen",
    "see",
    "know",
    "known",
    "take",
    "taken",
    "come",
    "came",
    "go",
    "went",
    "going",
    "say",
    "said",
    "says",
    "saying",
    "look",
    "looked",
    "looking",
    "find",
    "found",
    "finding",
    "founding",
    "think",
    "thought",
    "thinking",
    "thoughts",
    "thoughtful",
    "thoughtless",
    "feel",
    "felt",
    "feeling",
    "seem",
    "seemed",
    "seeming",
    "seems",
    "turn",
    "turned",
    "turning",
    "turns",
    "show",
    "showed",
    "showing",
    "shows",
    "shown",
    "leave",
    "left",
    "leaving",
    "leaves",
    "hand",
    "hands",
    "part",
    "parts",
    "place",
    "places",
    "placed",
    "placing",
    "point",
    "points",
    "pointed",
    "pointing",
    "case",
    "cases",
    "group",
    "groups",
    "grouped",
    "grouping",
    "number",
    "numbers",
    "numbered",
    "numbering",
    "fact",
    "facts",
    "way",
    "ways",
    "day",
    "days",
    "thing",
    "things",
    "man",
    "men",
    "woman",
    "women",
    "child",
    "children",
    "time",
    "times",
    "year",
    "years",
    "work",
    "works",
    "worked",
    "working",
    "life",
    "lives",
    "world",
    "worlds",
    "people",
    " Peoples",
    "person",
    "persons",
    "right",
    "rights",
    "left",
    "public",
    "publics",
    "old",
    "new",
    "good",
    "bad",
    "big",
    "small",
    "high",
    "low",
    "long",
    "short",
    "great",
    "little",
    "own",
    "last",
    "first",
    "next",
    "early",
    "late",
    "young",
    "old",
    "same",
    "different",
    "able",
    "unable",
    "back",
    "forward",
    "backward",
    "up",
    "down",
    "off",
    "on",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
}


def _strip_path_like_text(text: str) -> str:
    def replace_quoted_path(match: re.Match[str]) -> str:
        body = match.group(2)
        if "/" in body or "\\" in body:
            return " "
        if re.search(r"\.[A-Za-z0-9]{1,8}\b", body):
            return " "
        return match.group(0)

    stripped = re.sub(r"(['\"`])([^'\"`]*)(\1)", replace_quoted_path, text)
    stripped = re.sub(r"\bfile://\S+", " ", stripped)
    stripped = re.sub(r"(?<!\w)(?:~?/|\.{1,2}/|/)?(?:[\w.-]+/)+[\w.-]+", " ", stripped)
    stripped = re.sub(
        r"\b[\w.-]+\.(?:py|pyi|js|jsx|ts|tsx|go|rs|java|c|cc|cpp|h|hpp|md|txt|json|ya?ml|toml|ini|cfg)\b",
        " ",
        stripped,
    )
    return stripped


def _quoted_identifier_symbols(text: str) -> set[str]:
    symbols: set[str] = set()
    for match in re.finditer(r"(['\"`])([^'\"`]+)(\1)", text):
        body = match.group(2).strip()
        if "/" in body or "\\" in body or "." in body:
            continue
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", body):
            normalized = body.strip("_")
            if len(normalized) > 2 and normalized.lower() not in _STOP_WORDS:
                symbols.add(body)
    return symbols


def extract_symbols_from_task(task_description: str) -> set[str]:
    """
    Conservatively extract likely class and function names mentioned in the task.
    """
    text = str(task_description or "")
    text_without_paths = _strip_path_like_text(text)
    quoted_symbols = _quoted_identifier_symbols(text_without_paths)
    classes = set(re.findall(r"\b([A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9_]*)+)\b", text_without_paths))
    functions = set(re.findall(r"\b([a-z_][A-Za-z0-9_]*_[A-Za-z0-9_]*)\b", text_without_paths))
    call_symbols = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text_without_paths))
    symbols = classes | functions | call_symbols | quoted_symbols
    return {symbol for symbol in symbols if len(symbol.strip("_")) > 2 and symbol.lower() not in _STOP_WORDS}


def extract_defined_symbols(content: str) -> set[str]:
    """
    Extract class and function names defined in the source content.
    """
    class_pattern = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
    func_pattern = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
    classes = set(class_pattern.findall(content))
    functions = set(func_pattern.findall(content))
    return classes | functions


def has_class_or_function_bodies(content: str) -> bool:
    """
    Returns True if any class or function definition has a non-trivial body
    (more than just pass/.../docstring/return None).
    """
    block_pattern = re.compile(
        r"^(?:\s*(?:class|def)\s+[A-Za-z_][A-Za-z0-9_]*[^{:]*:?\s*\n)"
        r"((?:[ \t]+[^\n]*\n)+)",
        re.MULTILINE,
    )
    for match in block_pattern.finditer(content):
        body = match.group(1)
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        if not lines:
            continue
        if all(
            line in {"pass", "...", "return None", "return"}
            or line.startswith('"""')
            or line.startswith("'''")
            or line.startswith("#")
            for line in lines
        ):
            continue
        return True
    return False


def is_staged_artifact_recoverable(artifact_path: str, task_description: str) -> bool:
    """
    Returns True if the staged artifact contains enough structural
    evidence that the model can patch it rather than rewrite it.
    """
    try:
        content = Path(artifact_path).read_text(encoding="utf-8")
    except Exception:
        return False

    required_symbols = extract_symbols_from_task(task_description)
    defined_symbols = extract_defined_symbols(content)
    coverage = (
        len(defined_symbols & required_symbols) / max(len(required_symbols), 1)
        if required_symbols
        else 1.0
    )

    has_implementation = has_class_or_function_bodies(content)

    return coverage >= 0.5 and has_implementation


def _abandon_staged_artifact(harness: Any, stage_path: str, reason: str) -> None:
    """
    Rename a staged artifact so the model starts fresh, and log the event.
    """
    path = Path(stage_path)
    if not path.exists():
        return
    if not (path.parent.name == "write_sessions" and path.parent.parent.name == ".smallctl"):
        runlog = getattr(harness, "_runlog", None)
        if callable(runlog):
            runlog(
                "staged_artifact_abandon_skipped",
                "Refused to abandon a path outside .smallctl/write_sessions.",
                stage_path=str(stage_path),
                reason=reason,
            )
        return
    abandoned = path.with_suffix(path.suffix + ".abandoned")
    counter = 1
    original_abandoned = abandoned
    while abandoned.exists():
        abandoned = Path(f"{original_abandoned}.{counter}")
        counter += 1
    try:
        path.rename(abandoned)
    except Exception:
        try:
            path.unlink()
        except Exception:
            pass

    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "staged_artifact_abandoned",
            reason,
            stage_path=str(stage_path),
            abandoned_path=str(abandoned),
        )
