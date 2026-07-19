from __future__ import annotations

import os
import re
import time
import difflib
from pathlib import Path
from typing import Any

from ..redaction import REDACTED
from ..state import LoopState
from .common import fail, ok
from .fs_loop_guard import clear_loop_guard_verification_requirement
from .fs_sessions import _record_repair_cycle_read


_SENSITIVE_FILE_NAMES = {
    ".netrc",
}
_SENSITIVE_FILE_SUFFIXES = (
    ".pem",
    ".key",
    ".p12",
    ".pfx",
)
_SENSITIVE_ENV_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    ".env.testing",
    ".env.test",
    ".envrc",
}
_ENV_FILE_TEMPLATE_NAMES = {
    ".env.example",
    ".env.sample",
    ".env.template",
    ".env.defaults",
    ".env.dist",
}

# Out-of-workspace locations that require explicit approval before mutation.
_SENSITIVE_LOCATION_PATTERNS = (
    "~/.ssh",
    "~/.gnupg",
    "~/.aws",
    "~/.docker",
    "/etc",
    "/root",
    "/usr/local/etc",
    "/var",
)

# Tool operations for which _resolve enforces workspace containment by default.
_MUTATING_OPERATIONS = frozenset({
    "file_write",
    "file_append",
    "file_delete",
    "file_patch",
    "ast_patch",
})


class WorkspaceContainmentError(Exception):
    """Raised when a mutating operation targets a path outside the workspace."""

    def __init__(self, message: str, metadata: dict[str, Any]) -> None:
        super().__init__(message)
        self.metadata = metadata


def _is_sensitive_location(path: Path) -> bool:
    """Return True if a resolved path points to a sensitive system location."""
    normalized = path.as_posix().lower()
    try:
        home = str(Path.home().resolve()).lower()
    except Exception:
        home = ""
    for pattern in _SENSITIVE_LOCATION_PATTERNS:
        try:
            expanded = str(Path(os.path.expanduser(pattern)).resolve()).lower()
        except Exception:
            continue
        if normalized.startswith(expanded + "/") or normalized == expanded:
            return True
        raw = pattern.lower().rstrip("/")
        if raw.startswith("~") and home:
            raw = raw.replace("~", home, 1)
        if normalized.startswith(raw + "/") or normalized == raw:
            return True
    name = path.name.lower()
    if name in {
        "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
        "authorized_keys", "known_hosts", "known_hosts2",
        "passwd", "shadow", "sudoers", "htpasswd",
    }:
        return True
    return False


def _resolve(
    path: str,
    cwd: str | None = None,
    *,
    operation: str | None = None,
    approved_out_of_workspace: bool = False,
) -> Path:
    base = Path(cwd) if cwd else Path.cwd()
    candidate = Path(os.path.expanduser(path))
    if not candidate.is_absolute():
        # Guard against paths that already encode the CWD as relative segments
        base_str = str(base).rstrip("/")
        path_str = str(path).lstrip("./")
        if base_str and path_str.startswith(base_str.lstrip("/") + "/"):
            candidate = Path("/" + path_str)
        else:
            candidate = base / candidate
    resolved = candidate.resolve()

    if operation in _MUTATING_OPERATIONS and not approved_out_of_workspace:
        workspace = _workspace_root(cwd)
        if not _is_within_workspace(resolved, cwd):
            reason = (
                "sensitive_location_unapproved"
                if _is_sensitive_location(resolved)
                else "workspace_path_traversal"
            )
            raise WorkspaceContainmentError(
                f"{operation} blocked: path must stay within the active workspace.",
                metadata={
                    "path": str(resolved),
                    "workspace": str(workspace),
                    "requested_path": path,
                    "error_kind": reason,
                    "operation": operation,
                },
            )

    return resolved


def _workspace_root(cwd: str | None = None) -> Path:
    return Path(cwd).resolve() if cwd else Path.cwd().resolve()


def _is_within_workspace(path: Path, cwd: str | None = None) -> bool:
    try:
        path.relative_to(_workspace_root(cwd))
        return True
    except ValueError:
        return False


def _guard_workspace_containment(
    path: str,
    cwd: str | None = None,
    *,
    operation: str = "file operation",
    approved_out_of_workspace: bool = False,
) -> dict[str, Any] | None:
    """Return a failure envelope if the resolved path escapes the workspace.

    The actual containment check is delegated to :func:`_resolve` so that no
    caller can bypass enforcement by calling :func:`_resolve` directly.
    """
    try:
        _resolve(
            path,
            cwd,
            operation=operation,
            approved_out_of_workspace=approved_out_of_workspace,
        )
    except WorkspaceContainmentError as exc:
        workspace = _workspace_root(cwd)
        relative = _workspace_relative_hint(path, cwd)
        hint = (
            f" Try `{relative!r}` if you meant a workspace-relative path."
            if relative
            else " Use a workspace-relative path or ask the user for explicit approval."
        )
        return fail(
            f"{operation} blocked: path must stay within the active workspace.{hint}",
            metadata={
                **exc.metadata,
                "requested_path": path,
                "workspace": str(workspace),
            },
        )
    except Exception as exc:
        return fail(f"Unable to resolve path: {exc}")
    return None


def _looks_like_sensitive_read_path(path: str | Path) -> bool:
    target = Path(str(path or ""))
    name = target.name.lower()
    if name in _SENSITIVE_FILE_NAMES:
        return True
    if name in {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"}:
        return True
    return any(name.endswith(suffix) for suffix in _SENSITIVE_FILE_SUFFIXES)


def _looks_like_sensitive_env_path(path: str | Path) -> bool:
    """Return True for real .env files that should not be blindly overwritten.

    Template files such as `.env.example` and `.env.sample` are intentionally
    excluded so agents can read/write them as documentation.
    """
    target = Path(str(path or ""))
    name = target.name.lower()
    if name in _ENV_FILE_TEMPLATE_NAMES:
        return False
    if name in _SENSITIVE_ENV_FILE_NAMES:
        return True
    if name.startswith(".env.") and not any(name.endswith(suffix) for suffix in (".example", ".sample", ".template", ".defaults")):
        return True
    return False


_DOTENV_ASSIGNMENT_RE = re.compile(r"^(\s*(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*\s*=\s*)(.*)$")


def _redact_dotenv_text(text: str) -> str:
    """Redact assignment values in dotenv content, preserving keys and comments."""
    redacted_lines: list[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            redacted_lines.append(line)
            continue
        match = _DOTENV_ASSIGNMENT_RE.match(line)
        if match is None:
            redacted_lines.append(line)
            continue
        value = match.group(2).strip()
        if not value:
            redacted_lines.append(line)
            continue
        redacted_lines.append(f"{match.group(1)}{REDACTED}")
    return "\n".join(redacted_lines)


def _dotenv_permission_warning(path: Path) -> str:
    """Warn when a dotenv file is group/other-readable without exposing values."""
    try:
        mode = path.stat().st_mode
    except OSError:
        return ""
    if mode & 0o044:
        return (
            f"Dotenv file `{path.name}` is readable by group/other users "
            f"(mode {oct(mode & 0o777)}); restrict it with `chmod 600`."
        )
    return ""


def _active_session_staging_path(
    state: LoopState | None,
    path: str,
    cwd: str | None = None,
) -> Path | None:
    session = getattr(state, "write_session", None) if state is not None else None
    if session is None or str(getattr(session, "status", "")).strip().lower() == "complete":
        return None
    staging_path = str(getattr(session, "write_staging_path", "") or "").strip()
    try:
        target = _resolve(path, cwd)
        session_target = _resolve(session.write_target_path, cwd)
    except Exception:
        return None
    if target != session_target:
        return None
    staging = Path(staging_path)
    target_exists = target.exists()
    try:
        first_chunk_at = float(getattr(session, "write_first_chunk_at", 0.0) or 0.0)
    except (TypeError, ValueError):
        first_chunk_at = 0.0
    has_staged_progress = bool(
        getattr(session, "write_sections_completed", None)
        or getattr(session, "write_last_staged_hash", None)
        or getattr(session, "write_section_ranges", None)
        or first_chunk_at > 0.0
    )
    if staging_path and staging.exists():
        if staging.stat().st_size == 0 and not has_staged_progress:
            return None
        return staging
    session_id = str(getattr(session, "write_session_id", "") or "").strip()
    if not session_id:
        return None
    try:
        from .fs_write_sessions import _session_stage_path

        expected_staging = _session_stage_path(session_id, target, cwd)
    except Exception:
        expected_staging = None
    if expected_staging is not None and expected_staging.exists():
        if expected_staging.stat().st_size == 0 and not has_staged_progress:
            return None
        try:
            session.write_staging_path = str(expected_staging)
        except Exception:
            pass
        return expected_staging
    if target_exists and not has_staged_progress:
        return None
    try:
        from .fs import _ensure_write_session_files

        restored = _ensure_write_session_files(session, target, cwd=cwd)
    except Exception:
        return None
    if restored.exists():
        if restored.stat().st_size == 0 and not has_staged_progress:
            return None
        if not target_exists or restored.stat().st_size > 0:
            return restored
        try:
            if target.stat().st_size == 0:
                return restored
        except OSError:
            pass
    return None


def active_write_session_source_path(
    state: LoopState | None,
    path: str,
    cwd: str | None = None,
) -> str | None:
    staging = _active_session_staging_path(state, path, cwd)
    return str(staging) if staging is not None else None


def _workspace_relative_hint(path: str, cwd: str | None = None) -> str | None:
    raw = str(path or "").strip()
    if not raw:
        return None

    candidate = Path(raw)
    base = Path(cwd) if cwd else Path.cwd()

    if candidate.is_absolute():
        try:
            relative = candidate.resolve().relative_to(base.resolve())
        except Exception:
            trimmed = raw.lstrip("\\/")
            if not trimmed:
                return None
            workspace_candidate = (base / Path(trimmed)).resolve()
            if not (workspace_candidate.exists() or workspace_candidate.parent.exists()):
                return None
            try:
                relative = workspace_candidate.relative_to(base.resolve())
            except Exception:
                return None
        return "." if str(relative) == "." else f"./{relative}"

    if raw[0] not in {"\\", "/"}:
        return None

    trimmed = raw.lstrip("\\/")
    if not trimmed:
        return None
    suggested = (base / Path(trimmed)).resolve()
    try:
        relative = suggested.relative_to(base.resolve())
    except ValueError:
        return None
    return str(relative)


def _missing_path_error(*, requested_path: str, resolved_path: Path, cwd: str | None = None) -> str:
    near_match = _nearby_path_suggestion(requested_path=requested_path, resolved_path=resolved_path)
    if near_match:
        return f"{requested_path} not found; did you mean {near_match}?"
    message = f"File does not exist: {resolved_path}"
    suggestion = _workspace_relative_hint(requested_path, cwd)
    if suggestion:
        message = (
            f"{message}. The requested path {requested_path!r} was treated as absolute. "
            f"If you meant a workspace-relative path, retry with {suggestion!r}."
        )
    # If the parent directory is also missing, suggest creating the file via file_write
    try:
        parent = resolved_path.parent
        if parent and not parent.exists():
            message += (
                f" The directory `{parent}` does not exist either. "
                f"If you need to create this file, use `file_write(path='{requested_path}', content='...')` "
                f"which will auto-create missing parent directories."
            )
    except Exception:
        pass
    # If the path contains a literal tilde, warn about expansion
    if "~" in requested_path:
        message += (
            f" Note: the path contains `~`, which was expanded to `{resolved_path}`. "
            f"If you meant a literal `~` directory, use an absolute path or `shell_exec` to expand it."
        )
    return message


def _nearby_path_suggestion(*, requested_path: str, resolved_path: Path) -> str | None:
    parent = resolved_path.parent
    try:
        if not parent.exists() or not parent.is_dir():
            return None
        names = [child.name for child in parent.iterdir()]
    except Exception:
        return None
    matches = difflib.get_close_matches(resolved_path.name, names, n=1, cutoff=0.72)
    if not matches:
        return None
    raw = str(requested_path or "").strip()
    suggested = resolved_path.with_name(matches[0])
    if raw and not Path(raw).is_absolute():
        try:
            return str(Path(raw).with_name(matches[0]))
        except Exception:
            return matches[0]
    return str(suggested)


def _nearby_path_suggestion_confidence(*, requested_path: str, resolved_path: Path, suggested_path: str) -> str:
    requested_name = resolved_path.name
    suggested_name = Path(suggested_path).name
    similarity = difflib.SequenceMatcher(None, requested_name, suggested_name).ratio()
    same_parent = False
    try:
        suggested = Path(suggested_path)
        if not suggested.is_absolute():
            suggested = resolved_path.parent / suggested.name
        same_parent = suggested.resolve().parent == resolved_path.parent.resolve()
    except Exception:
        same_parent = False
    return "high" if same_parent and similarity >= 0.84 else "low"


def _missing_dir_error(*, requested_path: str, resolved_path: Path, cwd: str | None = None) -> str:
    message = f"Directory does not exist: {resolved_path}"
    suggestion = _workspace_relative_hint(requested_path, cwd)
    if suggestion:
        message = (
            f"{message}. The requested path {requested_path!r} was treated as absolute. "
            f"If you meant a workspace-relative path, retry with {suggestion!r}."
        )
    return message


def _build_dir_tree(
    path: Path,
    *,
    depth: int,
    max_depth: int,
    max_children: int,
    remaining_nodes: list[int],
) -> dict[str, Any]:
    node = {
        "name": path.name or str(path),
        "path": str(path),
        "type": "dir" if path.is_dir() else "file",
        "size": path.stat().st_size if path.is_file() else None,
    }
    if not path.is_dir() or depth >= max_depth or remaining_nodes[0] <= 0:
        return node

    try:
        children = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    except Exception:
        return node

    preview_children: list[dict[str, Any]] = []
    child_limit = min(max_children, len(children))
    for child in children[:child_limit]:
        if remaining_nodes[0] <= 0:
            break
        remaining_nodes[0] -= 1
        preview_children.append(
            _build_dir_tree(
                child,
                depth=depth + 1,
                max_depth=max_depth,
                max_children=max_children,
                remaining_nodes=remaining_nodes,
            )
        )

    if preview_children:
        node["children"] = preview_children
        node["children_count"] = len(children)
        if len(children) > len(preview_children):
            node["children_truncated"] = True
    return node


async def file_read(
    path: str,
    cwd: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    max_bytes: int = 100_000,
    state: LoopState | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    target = _resolve(path, cwd)
    source = _active_session_staging_path(state, path, cwd) or target
    session = getattr(state, "write_session", None) if state is not None else None
    sensitive_dotenv_read = _looks_like_sensitive_env_path(target)
    if _looks_like_sensitive_read_path(target):
        return fail(
            f"Refusing to read likely secret-bearing file `{path}`. "
            "Ask the user for the specific non-secret value needed, or use a safer command that prints only non-sensitive keys.",
            metadata={
                "path": str(target),
                "requested_path": path,
                "reason": "sensitive_file_read_blocked",
                "failure_class": "sensitive_file_read_blocked",
                "sensitive_path": True,
            },
        )
    if not source.exists():
        _record_repair_cycle_read(state, target)
        suggested = _nearby_path_suggestion(requested_path=path, resolved_path=target)
        return fail(
            _missing_path_error(requested_path=path, resolved_path=target, cwd=cwd),
            metadata={
                "path": str(target),
                "requested_path": path,
                "read_result": "missing",
                "suggested_path": suggested or "",
                "suggestion_confidence": (
                    _nearby_path_suggestion_confidence(
                        requested_path=path,
                        resolved_path=target,
                        suggested_path=suggested,
                    )
                    if suggested
                    else ""
                ),
            },
        )
    try:
        source_size = source.stat().st_size
        raw = source.read_bytes()
        raw = raw[:max_bytes]
        text = raw.decode("utf-8", errors="replace")
    except Exception as exc:
        return fail(f"Unable to read file: {exc}")

    if sensitive_dotenv_read:
        text = _redact_dotenv_text(text)

    lines = text.splitlines()
    total_lines = len(lines)
    requested_start = start_line
    requested_end = end_line
    if start_line is not None and end_line is not None and end_line < start_line:
        # Normalize reversed ranges by swapping the bounds rather than failing.
        start_line, end_line = end_line, start_line
    s = 0 if start_line is None else max(start_line - 1, 0)
    e = len(lines) if end_line is None else min(end_line, len(lines))
    sliced = "\n".join(lines[s:e])
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    complete_file = (
        total_lines == 0
        or (
            (requested_start is None or requested_start <= 1)
            and (requested_end is None or requested_end >= total_lines)
            and source_size <= max_bytes
        )
    )
    truncated = source_size > max_bytes
    _record_repair_cycle_read(state, target)
    if source != target:
        _record_repair_cycle_read(state, source)
    if complete_file:
        clear_loop_guard_verification_requirement(
            state,
            path=str(target),
            cwd=cwd,
        )
    return ok(
        sliced,
        metadata={
            "path": str(target),
            "source_path": str(source),
            "bytes": len(raw),
            "elapsed_ms": elapsed_ms,
            "requested_start_line": requested_start,
            "requested_end_line": requested_end,
            "max_bytes": max_bytes,
            "line_start": s + 1 if lines else 0,
            "line_end": e,
            "total_lines": total_lines,
            "complete_file": complete_file,
            "truncated": truncated,
            "read_from_staging": source != target,
            "staged_only": source != target,
            "dotenv_read_redacted": sensitive_dotenv_read,
            "dotenv_permissions_warning": (
                _dotenv_permission_warning(source) if sensitive_dotenv_read else ""
            ),
            "write_session_id": (
                str(getattr(session, "write_session_id", "") or "") if source != target else ""
            ),
            "write_session_status": (
                str(getattr(session, "status", "") or "") if source != target else ""
            ),
            "system_repair_cycle_id": str(getattr(state, "repair_cycle_id", "") or ""),
        },
    )


async def dir_list(path: str = ".", cwd: str | None = None) -> dict[str, Any]:
    target = _resolve(path, cwd)
    if not target.exists():
        return fail(
            _missing_dir_error(requested_path=path, resolved_path=target, cwd=cwd),
            metadata={"path": str(target), "requested_path": path},
        )
    if not target.is_dir():
        return fail(f"Path is not a directory: {target}")
    items: list[dict[str, Any]] = []
    root_limit = 120
    children = sorted(target.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    for p in children[:root_limit]:
        items.append(
            _build_dir_tree(
                p,
                depth=0,
                max_depth=2,
                max_children=8,
                remaining_nodes=[root_limit],
            )
        )
    return ok(
        items,
        metadata={
            "path": str(target),
            "count": len(items),
            "total_items": len(children),
            "truncated": len(children) > len(items),
            "tree_depth": 2,
            "tree_children_limit": 8,
        },
    )


async def dir_tree(
    path: str = ".",
    cwd: str | None = None,
    max_depth: int = 3,
    max_entries: int = 500,
) -> dict[str, Any]:
    root = _resolve(path, cwd)
    if not root.exists() or not root.is_dir():
        return fail(f"Invalid directory: {root}")

    entries: list[dict[str, Any]] = []
    root_depth = len(root.parts)
    for p in root.rglob("*"):
        depth = len(p.parts) - root_depth
        if depth > max_depth:
            continue
        entries.append(
            {
                "path": str(p),
                "relative": str(p.relative_to(root)),
                "depth": depth,
                "type": "dir" if p.is_dir() else "file",
            }
        )
        if len(entries) >= max_entries:
            break
    return ok(entries, metadata={"path": str(root), "count": len(entries), "truncated": len(entries) >= max_entries})
