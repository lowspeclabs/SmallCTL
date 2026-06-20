from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..models.conversation import ConversationMessage
from ..state import json_safe_value
from ..tools.ansi_utils import strip_ansi
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
_LOCAL_REMOTE_BLOCKER_RE = re.compile(
    r'The account\s+"[^"]+"\s+already exists'
    r"|Please remove the account"
    r"|set a new service username"
    r"|\buserdel\s+\S+"
    r"|\bfailed to create symbolic link\b.*?\bFile exists\b"
    r"|\b(?:bash|sh):\s+line\s+\d+:\s+\S+:\s+command not found\b"
    r"|\bpermission denied\b"
    r"|\bSorry,\s+answer not recognized\b"
    r"|\bAre you sure you wish to continue\b",
    re.IGNORECASE | re.DOTALL,
)
_HARNESS_POLICY_BLOCK_RE = re.compile(
    r"\b(?:raw_ssh_shell_blocked|nested_raw_ssh_in_ssh_exec|missing_supported_claim|"
    r"patch_over_rewrite_guard|ssh_host_key_recovery_required|spec_not_approved)\b"
    r"|Raw `ssh`/`scp`/`sftp` shell commands are (?:not allowed|blocked)"
    r"|Shell execution is blocked until the spec contract is approved"
    r"|SSH execution is blocked until the spec contract is approved",
    re.IGNORECASE | re.DOTALL,
)
_TERMINAL_UNKNOWN_RE = re.compile(r"\berror\s+opening\s+terminal:\s*unknown\b", re.IGNORECASE)
_ANSI_ART_NOISE_RE = re.compile(r"^[\s.,;:'`\-_/\\|()\[\]{}<>~=+*!?A-Za-z0-9]*$")


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


def _clean_error_for_recovery_query(error: str) -> str:
    text = strip_ansi(str(error or ""))
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.strip().split())
        if not line:
            continue
        if _TERMINAL_UNKNOWN_RE.search(line):
            cleaned_lines.append("Error opening terminal: unknown")
            continue
        if _ANSI_ART_NOISE_RE.fullmatch(line) and not re.search(r"[a-zA-Z]{3,}", line):
            continue
        cleaned_lines.append(line)
    return " ".join(cleaned_lines)


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
_LOCAL_SOURCE_TRACEBACK_RE = re.compile(
    r"(?:"
    r"Traceback \(most recent call last\)"
    r"|\bFile \"[^\"]+\.py\", line \d+"
    r"|\bFAILED \((?:errors|failures)=\d+\)"
    r")",
    re.IGNORECASE,
)


def _artifact_kind(artifact: Any) -> str:
    return str(getattr(artifact, "kind", "") or "").strip().lower()


def _artifact_text(artifact: Any) -> str:
    return str(getattr(artifact, "text", "") or "").strip()


def _artifact_is_authoritative_file_content(artifact: Any) -> bool:
    kind = _artifact_kind(artifact)
    if kind and kind not in {"file_read", "file", "ssh_file_read"}:
        return False
    return bool(_artifact_text(artifact))


def _artifact_matches_path(artifact: Any, path: str) -> bool:
    source = str(getattr(artifact, "source", "") or "").strip()
    if not source:
        return False
    return source == path or source.endswith(path)


def _ground_truth_nudged_keys(scratchpad: dict[str, Any]) -> set[str]:
    raw = scratchpad.get("_ground_truth_nudged_keys")
    if isinstance(raw, list):
        return {str(item) for item in raw if str(item)}
    legacy = str(scratchpad.get("_ground_truth_nudged") or "").strip()
    return {legacy} if legacy else set()


def _remember_ground_truth_nudge(scratchpad: dict[str, Any], key: str) -> None:
    keys = _ground_truth_nudged_keys(scratchpad)
    keys.add(key)
    scratchpad["_ground_truth_nudged_keys"] = sorted(keys)[-64:]
    scratchpad["_ground_truth_nudged"] = key


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
    for path in dict.fromkeys(paths):
        nudge_key = f"ground_truth:{path}"
        if nudge_key in _ground_truth_nudged_keys(scratchpad):
            continue

        matching_artifact: tuple[str, Any] | None = None
        for artifact_id, artifact in reversed(list((getattr(harness.state, "artifacts", {}) or {}).items())):
            if not _artifact_matches_path(artifact, path):
                continue
            if not _artifact_is_authoritative_file_content(artifact):
                continue
            matching_artifact = (artifact_id, artifact)
            break

        if matching_artifact is None:
            continue

        artifact_id, artifact = matching_artifact
        content = _artifact_text(artifact)
        _remember_ground_truth_nudge(scratchpad, nudge_key)

        # Truncate very long content for the nudge
        display = content[:400]
        if len(content) > 400:
            display += " …"

        message = (
            f"Ground truth: The error references `{path}`. "
            f"The latest authoritative file-read content available for this path is:\n{display}\n"
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

    return emitted


# ---------------------------------------------------------------------------
# A3) Dynamic Web Search on repeated errors
# ---------------------------------------------------------------------------

_MAX_ERROR_SIGNATURE_LEN = 200
_URL_HOST_RE = re.compile(r"https?://([^/\s'\"]+)", re.IGNORECASE)
_HTML_FETCH_RE = re.compile(r"<(?:!doctype\s+html|html|head|body)\b", re.IGNORECASE)
_HTML_SHELL_ERROR_RE = re.compile(r"syntax error near unexpected token\s+`?<", re.IGNORECASE)
_FETCH_404_RE = re.compile(r"\b404\b|\bnot found\b", re.IGNORECASE)
_PUBLIC_DNS_CMD_RE = re.compile(r"\b(?:dig|nslookup|host)\b", re.IGNORECASE)
_PUBLIC_RESOLVER_RE = re.compile(r"(?:@(?:8\.8\.8\.8|1\.1\.1\.1)|\b8\.8\.8\.8\b|\b1\.1\.1\.1\b)")
_NXDOMAIN_RE = re.compile(r"\bNXDOMAIN\b", re.IGNORECASE)
_RESOLVE_FAIL_RE = re.compile(r"Could not resolve '?([A-Za-z0-9.-]+)'?", re.IGNORECASE)
_INSTALL_MARKER_RE = re.compile(r"\b(?:install|setup|deploy|configure|bootstrap|apt-get|apt)\b", re.IGNORECASE)
_DOCKER_IMAGE_NOT_FOUND_RE = re.compile(
    r"Unable to find image\s+'([^']+)'\s+locally"
    r"|manifest for\s+(\S+)\s+not found"
    r"|pull access denied for\s+(\S+)",
    re.IGNORECASE,
)
_DOCKER_IMAGE_EXTRACT_RE = re.compile(r"^([^/:]+/[^/:]+)", re.IGNORECASE)


def _record_command(record: ToolExecutionRecord) -> str:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    return str(args.get("command") or "").strip()


def _extract_command_host(command: str) -> str:
    match = _URL_HOST_RE.search(str(command or ""))
    return str(match.group(1) if match else "").strip()


def _install_source_diagnosis(harness: Any) -> dict[str, Any]:
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return {}
    diagnosis = scratchpad.get("_install_source_diagnosis")
    if not isinstance(diagnosis, dict):
        diagnosis = {}
        scratchpad["_install_source_diagnosis"] = diagnosis
    return diagnosis


def _observe_install_source_diagnosis(harness: Any, record: ToolExecutionRecord) -> None:
    if record.tool_name not in {"ssh_exec", "shell_exec"}:
        return
    diagnosis = _install_source_diagnosis(harness)
    if not diagnosis and getattr(harness, "state", None) is None:
        return
    command = _record_command(record)
    output = _record_output_text(record)
    command_lower = command.lower()

    if command and _INSTALL_MARKER_RE.search(command):
        diagnosis["install_context_seen"] = True

    if any(token in command_lower for token in ("curl ", "wget ")) and (
        _HTML_FETCH_RE.search(output) or _HTML_SHELL_ERROR_RE.search(output) or _FETCH_404_RE.search(output)
    ):
        diagnosis["invalid_fetch_count"] = int(diagnosis.get("invalid_fetch_count", 0) or 0) + 1
        host = _extract_command_host(command)
        if host:
            diagnosis["source_host"] = host

    resolve_match = _RESOLVE_FAIL_RE.search(output)
    if resolve_match:
        diagnosis["resolve_fail_count"] = int(diagnosis.get("resolve_fail_count", 0) or 0) + 1
        diagnosis["source_host"] = str(resolve_match.group(1) or "").strip()
        if _INSTALL_MARKER_RE.search(command):
            diagnosis["install_context_resolve_failed"] = True

    if command and _PUBLIC_DNS_CMD_RE.search(command) and _PUBLIC_RESOLVER_RE.search(command) and _NXDOMAIN_RE.search(output):
        diagnosis["public_dns_nxdomain"] = True
        command_parts = command.split()
        if len(command_parts) > 1 and not command_parts[1].startswith("@"):
            diagnosis["nxdomain_host"] = command_parts[1]
            diagnosis.setdefault("source_host", command_parts[1])

    if record.result.success and (
        ("ping" in command_lower and "1.1.1.1" in command_lower)
        or ("curl" in command_lower and "example.com" in command_lower)
        or ("curl" in command_lower and "example.org" in command_lower)
    ):
        diagnosis["network_ok"] = True


def _install_source_invalid_query(harness: Any) -> str:
    state = getattr(harness, "state", None)
    diagnosis = _install_source_diagnosis(harness)
    host = str(diagnosis.get("source_host") or diagnosis.get("nxdomain_host") or "installer host").strip()
    task_text = " ".join(
        str(part or "")
        for part in (
            getattr(getattr(state, "run_brief", None), "original_task", ""),
            getattr(getattr(state, "working_memory", None), "current_goal", ""),
        )
    ).lower()
    if "fog" in task_text or "fogproject" in host:
        return f"{host} NXDOMAIN fog install official instructions"
    if "debian" in task_text or "ubuntu" in task_text:
        return f"current official install instructions Debian {host} NXDOMAIN"
    return f"{host} NXDOMAIN official install instructions"


def _web_search_query_for_repeated_error(error: str, *, record: ToolExecutionRecord | None = None, harness: Any = None) -> str:
    if harness is not None:
        diagnosis = _install_source_diagnosis(harness)
        if bool(diagnosis.get("public_dns_nxdomain")) and bool(diagnosis.get("network_ok")):
            return _install_source_invalid_query(harness)
    if record is not None:
        command = _record_command(record).lower()
        # Docker image pull/manifest failures: extract the image name and search for
        # current tags/instructions instead of pasting the whole noisy error string.
        docker_match = _DOCKER_IMAGE_NOT_FOUND_RE.search(error)
        if docker_match:
            image = (
                docker_match.group(1)
                or docker_match.group(2)
                or docker_match.group(3)
                or ""
            ).strip()
            image = re.sub(r":\w+$", "", image)
            if "/" in image:
                return f"{image} docker hub current tags manifest not found"
            return f"docker {image} image current tag official"
        if _TERMINAL_UNKNOWN_RE.search(error):
            installer_context = " installer" if _INSTALL_MARKER_RE.search(command) else ""
            return f"Error opening terminal unknown ssh_exec{installer_context} TERM noninteractive"
        if _INSTALL_MARKER_RE.search(command) and (
            _HTML_FETCH_RE.search(error) or _HTML_SHELL_ERROR_RE.search(error) or _FETCH_404_RE.search(error) or _RESOLVE_FAIL_RE.search(error)
        ):
            host = _extract_command_host(command)
            if host:
                return f"{host} official install instructions"
    prefix = "nginx error" if _NGINX_CONFIG_TEST_RE.search(error) else "error"
    cleaned_error = _clean_error_for_recovery_query(error)
    return f"{prefix}: {cleaned_error[:150] or str(error or '')[:150]}"


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
    if record.tool_name in {"ssh_exec", "shell_exec"} and _LOCAL_REMOTE_BLOCKER_RE.search(error):
        return False
    if _HARNESS_POLICY_BLOCK_RE.search(error):
        return False
    if record.tool_name in {"ssh_exec", "shell_exec"} and _LOCAL_SOURCE_TRACEBACK_RE.search(error):
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

    # Act on 2nd occurrence: web search. Escalate on 4th+: stagnation nudge.
    if count == 2:
        # Check whether web_search is registered
        registry = getattr(harness, "registry", None)
        names_fn = getattr(registry, "names", None) if registry is not None else None
        try:
            has_web_search = callable(names_fn) and "web_search" in names_fn()
        except Exception:
            has_web_search = False

        query = _web_search_query_for_repeated_error(error, record=record, harness=harness)

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

    # Escalate on repeated occurrences beyond count 2
    if count >= 4:
        harness.state.append_message(
            ConversationMessage(
                role="system",
                content=(
                    f"The same error has occurred {count} times: `{signature[:120]}`. "
                    "Stop retrying the same approach. This strategy is not working. "
                    "Reassess the problem from scratch. Use a completely different tool or approach."
                ),
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "repeated_error_stagnation",
                    "error_signature": signature[:200],
                    "count": count,
                },
            )
        )
        harness._runlog(
            "repeated_error_stagnation_nudge",
            "escalated nudge after repeated identical error stagnation",
            error_signature=signature[:200],
            count=count,
        )
        counters = getattr(harness.state, "stagnation_counters", None)
        if isinstance(counters, dict):
            counters["repeat_command"] = int(counters.get("repeat_command", 0)) + 1
        return True

    return False


def _maybe_pivot_upstream_install_source_invalid(
    graph_state: GraphRunState,
    harness: Any,
) -> bool:
    diagnosis = _install_source_diagnosis(harness)
    if not diagnosis:
        return False
    if not bool(diagnosis.get("public_dns_nxdomain")) or not bool(diagnosis.get("network_ok")):
        return False
    if int(diagnosis.get("invalid_fetch_count", 0) or 0) <= 0 and int(diagnosis.get("resolve_fail_count", 0) or 0) <= 0:
        return False

    scratchpad = getattr(harness.state, "scratchpad", None)
    if not isinstance(scratchpad, dict) or scratchpad.get("_install_source_invalid_pivoted"):
        return False
    scratchpad["_install_source_invalid_pivoted"] = True
    blocker = {
        "reason": "upstream_install_source_invalid",
        "host": str(diagnosis.get("source_host") or diagnosis.get("nxdomain_host") or "").strip(),
        "invalid_fetch_count": int(diagnosis.get("invalid_fetch_count", 0) or 0),
        "resolve_fail_count": int(diagnosis.get("resolve_fail_count", 0) or 0),
        "network_ok": True,
    }
    scratchpad["_install_source_invalid_blocker"] = blocker
    scratchpad["_ui_recovery_banner"] = "Blocked: installer source invalid or unavailable"
    _suppress_unrelated_subtasks_on_pivot(harness)

    registry = getattr(harness, "registry", None)
    names_fn = getattr(registry, "names", None) if registry is not None else None
    try:
        has_web_search = callable(names_fn) and "web_search" in names_fn()
    except Exception:
        has_web_search = False

    query = _install_source_invalid_query(harness)
    host = str(blocker.get("host") or "the installer/package host").strip()
    question = (
        "I found a blocker while trying to continue the install. "
        f"The upstream install source `{host}` appears invalid or unavailable: public DNS returned NXDOMAIN, "
        "general internet connectivity still works, and earlier installer fetches were invalid. "
        "Should I research the current official install path, use an alternate/manual source you provide, or stop and report this as blocked?"
    )
    payload = {
        "kind": "ask_human",
        "question": question,
        "reason": "upstream_install_source_invalid",
        "blocker": blocker,
        "suggested_actions": [
            "research_current_official_install_path",
            "use_user_provided_alternate_source",
            "stop_blocked_with_summary",
        ],
    }
    scratchpad["_ask_human"] = True
    scratchpad["_ask_human_question"] = question
    harness.state.pending_interrupt = payload
    graph_state.interrupt_payload = payload
    graph_state.pending_tool_calls = []
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                "Blocked: installer source invalid or unavailable. Public DNS returned NXDOMAIN for the package host, "
                "general connectivity still works, and earlier installer fetches were invalid. Stop local DNS repair and retry churn. "
                "Ask the user whether to research the current official install path, use an alternate/manual source, or stop blocked. "
                f"If researching is approved, use a task-aware query such as `{query}`."
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "upstream_install_source_invalid",
                "query": query,
                "blocker": blocker,
                "interrupt": payload,
            },
        )
    )
    harness._runlog(
        "upstream_install_source_invalid_pivot",
        "pivoted from local installer retries to source validation",
        blocker=json_safe_value(blocker),
        query=query,
        web_search_available=has_web_search,
        interrupt_kind="ask_human",
    )
    return True


def _suppress_unrelated_subtasks_on_pivot(harness: Any) -> None:
    """Mark completed subtasks/artifacts from unrelated side quests as suppressed
    once the run pivots to upstream-source-invalid handling.
    """
    state = getattr(harness, "state", None)
    if state is None:
        return
    ledger = getattr(state, "subtask_ledger", None)
    if ledger is None:
        return
    suppressed_ids: list[str] = []
    install_markers = ("install", "setup", "deploy", "configure")
    for task in ledger.subtasks:
        if task.status != "done":
            continue
        text = " ".join((task.title or "", task.goal or "")).lower()
        # Escalation detours are always unrelated to the install objective
        is_escalation = "escalat" in text and ("bigger" in text or "stronger" in text or "larger" in text or "model" in text)
        if is_escalation or not any(m in text for m in install_markers):
            suppressed_ids.append(task.subtask_id)
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        state.scratchpad = scratchpad
    if suppressed_ids:
        existing = set(scratchpad.get("_pivot_suppressed_subtask_ids", []))
        existing.update(suppressed_ids)
        scratchpad["_pivot_suppressed_subtask_ids"] = sorted(existing)


# ---------------------------------------------------------------------------
# A4) Dead-URL 404 spiral detection for remote SSH download attempts
# ---------------------------------------------------------------------------

_404_URL_RE = re.compile(
    r"(?:curl|wget)\s*.*?(?:404|not found|Not Found|Failed to download)",
    re.IGNORECASE,
)


def _maybe_nudge_repeated_404_remote_url(
    graph_state: GraphRunState,
    harness: Any,
    record: ToolExecutionRecord,
) -> bool:
    """Detect repeated 404 errors from curl/wget on remote hosts and nudge
    the model to stop retrying the same URL family.
    """
    if record.tool_name not in {"ssh_exec", "shell_exec"}:
        return False
    if not _record_has_failure_evidence(record):
        return False

    error = _record_output_text(record)
    if not error or len(error) < 20:
        return False
    if not _404_URL_RE.search(error):
        return False

    # Extract the failing URL from the command
    command = str(getattr(record.args, "get", lambda k: None)("command") or "").strip()
    if not command:
        return False

    # Extract host
    host = str(getattr(record.args, "get", lambda k: None)("host") or "").strip()

    # Derive a URL family from the command (e.g. firecow/fog)
    family = ""
    for marker in ("github.com/", "raw.githubusercontent.com/"):
        idx = command.find(marker)
        if idx >= 0:
            rest = command[idx + len(marker):]
            parts = rest.replace("/", " ").split()
            # Take org/repo (first two path components)
            family = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]
            break

    if not family:
        return False

    scratchpad = getattr(harness.state, "scratchpad", {})
    if not isinstance(scratchpad, dict):
        scratchpad = {}

    # Track per-family 404 counts across the session
    url_counts: dict[str, int] = scratchpad.get("_repeated_404_url_counts", {})
    if not isinstance(url_counts, dict):
        url_counts = {}
    count = url_counts.get(family, 0) + 1
    url_counts[family] = count
    scratchpad["_repeated_404_url_counts"] = url_counts

    # Nudge at count 2 (second 404 from same URL family)
    if count < 2:
        return False

    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                f"The URL family `{family}` has returned 404 {count} times "
                f"on remote host `{host}`. "
                "These URLs do not exist. Stop retrying this URL family. "
                "Use `web_search` to find the correct source or call `ask_human` for the right download location."
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "repeated_404_url_spiral",
                "url_family": family,
                "host": host,
                "count": count,
            },
        )
    )
    harness._runlog(
        "repeated_404_url_spiral_nudge",
        "nudged model to stop retrying dead URLs after repeated 404",
        url_family=family,
        host=host,
        count=count,
    )
    return True


_SSH_AUTH_FAILURE_RE = re.compile(
    r"(?:permission denied|publickey|authentication failed|"
    r"password required|password:)",
    re.IGNORECASE,
)


_SSH_FAILURE_COUNT_KEY = "_ssh_auth_failure_count"


def _maybe_nudge_ssh_auth_fallback(
    graph_state: GraphRunState,
    harness: Any,
    record: ToolExecutionRecord,
) -> bool:
    """After repeated SSH auth failures, nudge the model to try a different
    host, use local shell_exec, or ask the user for corrected credentials.
    """
    if record.tool_name != "ssh_exec":
        return False
    if not _record_has_failure_evidence(record):
        return False
    error = _record_output_text(record)
    if not error or not _SSH_AUTH_FAILURE_RE.search(error):
        return False

    scratchpad = getattr(harness.state, "scratchpad", {})
    if not isinstance(scratchpad, dict):
        scratchpad = {}
    count = int(scratchpad.get(_SSH_FAILURE_COUNT_KEY, 0) or 0) + 1
    scratchpad[_SSH_FAILURE_COUNT_KEY] = count

    if count < 2:
        return False

    host = str(getattr(record.args, "get", lambda k: None)("host") or "").strip()
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                f"SSH authentication to `{host}` failed {count} times. "
                "The credential/access issue will not resolve by retrying. "
                "Try a different host, use local shell_exec to run commands "
                "locally, or ask the user for corrected credentials."
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "ssh_auth_fallback",
                "host": host,
                "count": count,
            },
        )
    )
    harness._runlog(
        "ssh_auth_fallback_nudge",
        "nudged model to stop retrying SSH after auth failures",
        host=host,
        count=count,
    )
    return True


_APT_KEY_DEPRECATION_RE = re.compile(
    r"\bapt-key:\s+command not found\b",
    re.IGNORECASE,
)
_APT_KEY_DEBIAN_VER_RE = re.compile(
    r"(?:Debian|Raspbian|Ubuntu)\s+\d+",
    re.IGNORECASE,
)
_APT_KEY_NUDGE_KEY = "_apt_key_deprecation_nudged"


def _maybe_emit_apt_key_deprecation_nudge(
    graph_state: GraphRunState,
    harness: Any,
    record: ToolExecutionRecord,
) -> bool:
    """Detect when `apt-key: command not found` appears on a remote host
    (Debian 13+/Ubuntu 22.04+ where apt-key is removed) and inject a
    recovery hint explaining the gpg --dearmor / signed-by pattern.

    https://manpages.debian.org/testing/apt/apt-key.8.en.html
    """
    if record.tool_name not in {"ssh_exec", "shell_exec"}:
        return False
    if not _record_has_failure_evidence(record):
        return False

    error = _record_output_text(record)
    if not error or not _APT_KEY_DEPRECATION_RE.search(error):
        return False

    scratchpad = getattr(harness.state, "scratchpad", {})
    if not isinstance(scratchpad, dict):
        scratchpad = {}
    if scratchpad.get(_APT_KEY_NUDGE_KEY):
        return False
    scratchpad[_APT_KEY_NUDGE_KEY] = True

    command = str(getattr(record.args, "get", lambda k: None)("command") or "").strip()
    host = str(getattr(record.args, "get", lambda k: None)("host") or "").strip()

    message = (
        "APT-KEY DEPRECATION: `apt-key` has been removed in Debian 13 (Trixie) and "
        "Ubuntu 22.04+. It was deprecated upstream for years. "
        "The modern approach is:\n"
        "1. Download the key:  "
        "`wget -qO- <key-url> | gpg --dearmor -o /usr/share/keyrings/<name>.gpg`\n"
        "2. Add the repo:  "
        "`echo \"deb [signed-by=/usr/share/keyrings/<name>.gpg] <repo-url> <dist> <component>\" "
        "> /etc/apt/sources.list.d/<name>.list`\n"
        "3. Update:  `apt update`\n"
        "4. Install:  `apt install <package> -y`\n"
        "Do NOT retry `apt-key`. Use the gpg --dearmor / signed-by pattern shown above."
    )
    if host:
        message += f"\nApplies to remote host: {host}"

    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=message,
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "apt_key_deprecation",
                "tool_name": record.tool_name,
                "host": host,
                "command": command[:200],
            },
        )
    )
    harness._runlog(
        "apt_key_deprecation_nudge",
        "injected apt-key deprecation recovery nudge",
        tool_name=record.tool_name,
        host=host,
        command=command[:200],
    )
    return True
