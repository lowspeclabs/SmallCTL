from __future__ import annotations

from .base import ToolProfile

CORE_PROFILE: ToolProfile = "core"
DATA_PROFILE: ToolProfile = "data"
NETWORK_PROFILE: ToolProfile = "network"
OPS_PROFILE: ToolProfile = "ops"
SUPPORT_PROFILE: ToolProfile = "support"
MUTATE_PROFILE: ToolProfile = "mutate"
INDEXER_PROFILE: ToolProfile = "indexer"

DEFAULT_PROFILES: set[ToolProfile] = {CORE_PROFILE}
PUBLIC_PROFILES: tuple[ToolProfile, ...] = (
    CORE_PROFILE,
    DATA_PROFILE,
    NETWORK_PROFILE,
    OPS_PROFILE,
    MUTATE_PROFILE,
    INDEXER_PROFILE,
)


def classify_tool_profiles(
    task: str,
    *,
    use_ansible: bool,
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
        ),
    ):
        profiles.add(NETWORK_PROFILE)

    if use_ansible and _matches_any(
        text,
        (
            "ansible",
            "playbook",
            "inventory",
            "hosts",
            "host group",
            "check mode",
            "deploy",
            "configure",
            "configuration management",
            "provision",
            "multi-host",
            "multihost",
        ),
    ):
        profiles.add(OPS_PROFILE)

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
            "cwd_get",
            "cwd_set",
            "environment variable",
            "env var",
            "env_get",
            "env_set",
            "set env",
            "background process",
            "background job",
            "shell_background",
            "kill process",
            "process_kill",
            "checkpoint",
            "scratch_set",
            "scratch_get",
            "scratch_list",
            "scratch_delete",
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
