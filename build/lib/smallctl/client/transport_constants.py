import re

_LLAMACPP_CONTEXT_OVERFLOW_RE = re.compile(
    r"request\s*\((?P<request_tokens>\d+)\s+tokens?\)\s+exceeds\s+the\s+available\s+context\s+size\s*\((?P<context_tokens>\d+)\s+tokens?\)",
    re.IGNORECASE,
)
_LOCAL_WRITE_INTENT_RE = re.compile(
    r"\b(build|create|implement|write|generate|add|make)\b.*\b(file|script|module|\.py|\.js|\.ts|\.md|\.txt)\b"
    r"|\b(file|script|module)\b.*\b(build|create|implement|write|generate|add|make)\b",
    re.IGNORECASE | re.DOTALL,
)
_LOCAL_PATCH_INTENT_RE = re.compile(
    r"\b(fix|patch|update|modify|edit|change|repair|refactor)\b",
    re.IGNORECASE,
)
_UNSET = object()
_DEFAULT_MAX_COMPLETION_TOKENS = 2048
