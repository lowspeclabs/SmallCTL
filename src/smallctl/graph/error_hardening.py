from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..logging_utils import log_kv
from ..models.conversation import ConversationMessage
from ..state import json_safe_value
from .state import GraphRunState, PendingToolCall, ToolExecutionRecord

logger = logging.getLogger("smallctl.graph.error_hardening")

# ---------------------------------------------------------------------------
# B) Harness hardening: detect nginx sites-enabled misconfiguration
# ---------------------------------------------------------------------------

_NGINX_CONFIG_TEST_RE = re.compile(
    r"nginx:\s*configuration\s*file.*test\s*failed",
    re.IGNORECASE,
)
_NGINX_SITES_ENABLED_PATH_RE = re.compile(
    r"/etc/nginx/sites-enabled/([^\s:]+)",
)
_NGINX_UNEXPECTED_EOF_RE = re.compile(
    r"unexpected\s+end\s+of\s*file",
    re.IGNORECASE,
)


def _record_output_text(record: ToolExecutionRecord) -> str:
    result = record.result
    metadata = result.metadata if isinstance(getattr(result, "metadata", None), dict) else {}
    output = result.output if isinstance(getattr(result, "output", None), dict) else {}
    if not output:
        metadata_output = metadata.get("output")
        if isinstance(metadata_output, dict):
            output = metadata_output
    parts = [
        str(getattr(result, "error", "") or "").strip(),
        str(output.get("stdout") or "").strip() if isinstance(output, dict) else "",
        str(output.get("stderr") or "").strip() if isinstance(output, dict) else "",
    ]
    return "\n".join(part for part in parts if part)


def _record_has_failure_evidence(record: ToolExecutionRecord) -> bool:
    if not record.result.success:
        return True
    text = _record_output_text(record)
    return bool(_NGINX_CONFIG_TEST_RE.search(text) or _NGINX_UNEXPECTED_EOF_RE.search(text))


def _maybe_emit_nginx_sites_enabled_nudge(
    harness: Any,
    record: ToolExecutionRecord,
) -> bool:
    """Detect when an ssh_exec/shell_exec nginx config test fails because
    sites-enabled contains an invalid text file (e.g. 'enabled') instead of
    a symlink to sites-available. Inject a hardening nudge with the correct
    pattern.
    """
    if record.tool_name not in {"ssh_exec", "shell_exec"}:
        return False
    if not _record_has_failure_evidence(record):
        return False

    error = _record_output_text(record)
    if not error:
        return False

    # Must be an nginx config test failure
    if not _NGINX_CONFIG_TEST_RE.search(error):
        return False

    # Must reference sites-enabled
    site_match = _NGINX_SITES_ENABLED_PATH_RE.search(error)
    if not site_match:
        return False

    site_name = site_match.group(1)

    # Deduplicate per site per session
    nudge_key = f"nginx_sites_enabled_hardening:{site_name}"
    scratchpad = getattr(harness.state, "scratchpad", {})
    if not isinstance(scratchpad, dict):
        scratchpad = {}
    if scratchpad.get("_nginx_sites_enabled_nudged") == nudge_key:
        return False
    scratchpad["_nginx_sites_enabled_nudged"] = nudge_key

    message = (
        f"NGINX ENABLE ERROR: The nginx config test failed because "
        f"`/etc/nginx/sites-enabled/{site_name}` contains invalid nginx syntax. "
        f"On Debian/Ubuntu systems, `sites-enabled/` should contain symlinks to "
        f"configs in `sites-available/`, not regular text files. "
        f"To enable a site, use: "
        f"`ln -s /etc/nginx/sites-available/{site_name}.conf /etc/nginx/sites-enabled/{site_name}`. "
        f"Do NOT use `echo 'enabled' > /etc/nginx/sites-enabled/{site_name}`."
    )

    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=message,
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "nginx_sites_enabled_hardening",
                "tool_name": record.tool_name,
                "site_name": site_name,
                "error_excerpt": error[:300],
            },
        )
    )
    harness._runlog(
        "nginx_sites_enabled_hardening",
        "injected nginx sites-enabled hardening nudge",
        tool_name=record.tool_name,
        site_name=site_name,
        error_excerpt=error[:300],
    )
    return True


# ---------------------------------------------------------------------------
# A2) Auto-Ground-Truth Diffusion
# ---------------------------------------------------------------------------

_ERROR_PATH_RE = re.compile(
    r"(/(?:etc|var|usr|home|opt|srv|tmp|lib|run)/[a-zA-Z0-9_./\-]+)"
)


def _maybe_emit_ground_truth_diffusion(
    harness: Any,
    record: ToolExecutionRecord,
) -> bool:
    """When a tool fails and references a file path, if we already have an
    artifact showing the current content of that file, inject a ground-truth
    observation so the model cannot misattribute the error to a different file.
    """
    if not _record_has_failure_evidence(record):
        return False

    error = _record_output_text(record)
    if not error:
        return False

    paths = _ERROR_PATH_RE.findall(error)
    if not paths:
        return False

    scratchpad = getattr(harness.state, "scratchpad", {})
    if not isinstance(scratchpad, dict):
        scratchpad = {}

    emitted = False
    for path in paths:
        # Look for an artifact whose source matches this path
        for artifact_id, artifact in (getattr(harness.state, "artifacts", {}) or {}).items():
            source = str(getattr(artifact, "source", "") or "").strip()
            if not source:
                continue
            if source != path and not source.endswith(path):
                continue

            # Found ground-truth artifact
            content = ""
            if hasattr(artifact, "text"):
                content = str(getattr(artifact, "text", "") or "").strip()
            elif hasattr(artifact, "summary"):
                content = str(getattr(artifact, "summary", "") or "").strip()
            if not content:
                continue

            nudge_key = f"ground_truth:{path}:{artifact_id}"
            if scratchpad.get("_ground_truth_nudged") == nudge_key:
                continue
            scratchpad["_ground_truth_nudged"] = nudge_key

            # Truncate very long content for the nudge
            display = content[:400]
            if len(content) > 400:
                display += " …"

            message = (
                f"Ground truth: The error references `{path}`. "
                f"The current content of this file is:\n{display}\n"
                f"Ensure your diagnosis and fix target this exact file, not a different path."
            )

            harness.state.append_message(
                ConversationMessage(
                    role="system",
                    content=message,
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "ground_truth_diffusion",
                        "path": path,
                        "artifact_id": artifact_id,
                    },
                )
            )
            harness._runlog(
                "ground_truth_diffusion",
                "injected ground-truth observation for error-referenced file",
                path=path,
                artifact_id=artifact_id,
            )
            emitted = True
            break

    return emitted


# ---------------------------------------------------------------------------
# A3) Dynamic Web Search on repeated errors
# ---------------------------------------------------------------------------

_MAX_ERROR_SIGNATURE_LEN = 200


def _maybe_schedule_web_search_for_repeated_error(
    graph_state: GraphRunState,
    harness: Any,
    record: ToolExecutionRecord,
) -> bool:
    """If the same error text appears twice, auto-schedule a web_search with
    the exact error string (when the tool is available).
    """
    if not _record_has_failure_evidence(record):
        return False

    error = _record_output_text(record)
    if not error or len(error) < 20:
        return False

    # Normalize signature: first 200 chars, collapsed whitespace
    signature = " ".join(error[:_MAX_ERROR_SIGNATURE_LEN].split())

    scratchpad = getattr(harness.state, "scratchpad", {})
    if not isinstance(scratchpad, dict):
        scratchpad = {}

    seen_errors: dict[str, int] = scratchpad.get("_repeated_error_signatures", {})
    if not isinstance(seen_errors, dict):
        seen_errors = {}

    count = seen_errors.get(signature, 0) + 1
    seen_errors[signature] = count
    scratchpad["_repeated_error_signatures"] = seen_errors

    # Only act on the second occurrence
    if count != 2:
        return False

    # Check whether web_search is registered
    registry = getattr(harness, "registry", None)
    names_fn = getattr(registry, "names", None) if registry is not None else None
    try:
        has_web_search = callable(names_fn) and "web_search" in names_fn()
    except Exception:
        has_web_search = False

    query = f"nginx error: {error[:150]}"

    if has_web_search:
        # Schedule the search as the very next tool call
        graph_state.pending_tool_calls = [
            PendingToolCall(
                tool_name="web_search",
                args={"query": query, "limit": 5},
                raw_arguments=json.dumps(
                    {"query": query, "limit": 5}, ensure_ascii=True, sort_keys=True
                ),
                source="system",
            )
        ]
        harness.state.append_message(
            ConversationMessage(
                role="system",
                content=(
                    f"This error has occurred twice. Auto-searching the web for: `{query}`. "
                    "Use the search results to diagnose the root cause before making another fix."
                ),
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "repeated_error_web_search",
                    "query": query,
                },
            )
        )
        harness._runlog(
            "repeated_error_web_search_scheduled",
            "scheduled automatic web_search after repeated identical error",
            query=query,
            error_signature=signature[:200],
        )
        return True
    else:
        # Tool not available; inject a nudge instead
        harness.state.append_message(
            ConversationMessage(
                role="system",
                content=(
                    f"This error has occurred twice. Consider searching the web for: `{query}`. "
                    "If `web_search` is unavailable, explain the blocker and ask for guidance."
                ),
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "repeated_error_web_search_unavailable",
                    "query": query,
                },
            )
        )
        harness._runlog(
            "repeated_error_web_search_nudge",
            "nudged model to search web after repeated identical error (tool unavailable)",
            query=query,
            error_signature=signature[:200],
        )
        return True
