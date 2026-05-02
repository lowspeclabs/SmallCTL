from __future__ import annotations

import re

from .base import ToolProfile

CORE_PROFILE: ToolProfile = "core"
DATA_PROFILE: ToolProfile = "data"
NETWORK_PROFILE: ToolProfile = "network"
NETWORK_READ_PROFILE: ToolProfile = "network_read"
NETWORK_RAW_PROFILE: ToolProfile = "network_raw"
SUPPORT_PROFILE: ToolProfile = "support"
MUTATE_PROFILE: ToolProfile = "mutate"
INDEXER_PROFILE: ToolProfile = "indexer"

DEFAULT_PROFILES: set[ToolProfile] = {CORE_PROFILE}
PUBLIC_PROFILES: tuple[ToolProfile, ...] = (
    CORE_PROFILE,
    DATA_PROFILE,
    NETWORK_PROFILE,
    NETWORK_READ_PROFILE,
    NETWORK_RAW_PROFILE,
    MUTATE_PROFILE,
    INDEXER_PROFILE,
)

_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_USER_AT_HOST_RE = re.compile(r"\b[A-Za-z0-9._-]+@[A-Za-z0-9._-]+\b")


def classify_tool_profiles(
    task: str,
) -> set[ToolProfile]:
    text = task.strip().lower()
    profiles: set[ToolProfile] = set(DEFAULT_PROFILES)
    if not text:
        return profiles

    if _matches_any(
        text,
        (
            "yaml",
            ".yml",
            ".yaml",
            "json",
            ".json",
            "jmespath",
            "structured data",
            "compare",
            "diff",
            "patch",
        ),
    ):
        profiles.add(DATA_PROFILE)

    if _matches_any(
        text,
        (
            "http",
            "https://",
            "http://",
            "url",
            "api",
            "endpoint",
            "webhook",
            "request",
            "download",
            "upload",
            "post ",
            "get ",
            "fetch",
            "curl",
        ),
    ):
        profiles.add(NETWORK_PROFILE)
        profiles.add(NETWORK_RAW_PROFILE)

    if _matches_any(
        text,
        (
            "ssh",
            "scp",
            "sftp",
            "sshd",
            "remote host",
            "remote server",
            "remote command",
            "username",
            "password",
        ),
    ) or _looks_like_remote_access_request(text):
        profiles.add(NETWORK_PROFILE)

    if _matches_any(
        text,
        (
            "latest",
            "current",
            "recent",
            "today",
            "web",
            "website",
            "internet",
            "online",
            "search web",
            "search the web",
            "web search",
            "browse",
            "docs",
            "documentation",
            "release",
            "releases",
            "pricing",
            "announcement",
            "announcements",
            "news",
            "look up",
            "lookup",
            "research",
            "reserach",
            "reseach",
            "reseearch",
            "investigate",
            "options",
            "choose one",
            "find out",
            "search",
        ),
    ):
        profiles.add(NETWORK_READ_PROFILE)

    if _matches_any(
        text,
        (
            "directory tree",
            "dir tree",
            "dir_list",
            "dir_tree",
            "tree view",
            "list directory",
            "list files",
            "working directory",
            "current directory",
            "this directory",
            "current folder",
            "this folder",
            "folder contents",
            "directory contents",
            "change directory",
            "cwd",
            "background process",
            "background job",
            "kill process",
            "process_kill",
            "scratchpad",
        ),
    ):
        profiles.add(SUPPORT_PROFILE)

    if _matches_any(
        text,
        (
            "delete file",
            "remove file",
            "delete ",
            "remove ",
            "cleanup files",
            "clean up files",
        ),
    ):
        profiles.add(MUTATE_PROFILE)

    if _matches_any(
        text,
        (
            "index",
            "indexer",
            "manifest",
            "symbol",
            "reference",
            "definition",
        ),
    ):
        profiles.add(INDEXER_PROFILE)

    return profiles


def parse_public_profiles(value: str | list[str] | tuple[str, ...] | None) -> list[ToolProfile] | None:
    if value in (None, "", []):
        return None
    if isinstance(value, str):
        raw_items = [item.strip().lower() for item in value.split(",")]
    else:
        raw_items = [str(item).strip().lower() for item in value]

    parsed: list[ToolProfile] = []
    for item in raw_items:
        if not item:
            continue
        if item not in PUBLIC_PROFILES:
            raise ValueError(
                f"Unknown tool profile '{item}'. Expected one of: {', '.join(PUBLIC_PROFILES)}"
            )
        if item not in parsed:
            parsed.append(item)  # type: ignore[arg-type]
    return parsed or None


def _matches_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _looks_like_remote_access_request(text: str) -> bool:
    if _USER_AT_HOST_RE.search(text):
        return True
    has_ip = bool(_IPV4_RE.search(text))
    if not has_ip:
        return False
    return _matches_any(
        text,
        (
            "username",
            "password",
            "remote",
            "host",
            "server",
            "ssh",
            "docker",
            "install",
            "deploy",
            "configure",
            "spin up",
        ),
    )
