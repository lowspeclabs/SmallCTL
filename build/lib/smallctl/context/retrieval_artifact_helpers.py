from __future__ import annotations

import re
from pathlib import Path

from ..state import ArtifactRecord, LoopState, normalize_intent_label

MUTATION_RESULT_TOOLS = {
    "ssh_file_write",
    "ssh_file_patch",
    "ssh_file_replace_between",
    "file_write",
    "file_append",
    "file_patch",
    "ast_patch",
}
INTERACTIVE_PROMPT_MARKERS = (
    "(y/n)",
    "[y/n]",
    "[y/n",
    "choice: [",
    "hit enter",
    "hit [enter]",
    "are you sure you wish to continue",
    "answer not recognized",
    "sorry, answer not recognized",
)
FILE_LIKE_PATH_RE = re.compile(r"(?:^|\s)(?:\.{0,2}/|/)?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*(?:\.[A-Za-z0-9_.-]+)")


def file_like_paths(text: str) -> set[str]:
    paths: set[str] = set()
    for match in FILE_LIKE_PATH_RE.finditer(str(text or "")):
        value = match.group(0).strip()
        if not value or "." not in Path(value).name:
            continue
        paths.add(Path(value).as_posix().lower())
    return paths


def artifact_text(artifact: ArtifactRecord) -> str:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    verifier_verdict = str(metadata.get("verifier_verdict") or "").strip()
    if verifier_verdict:
        target = str(metadata.get("verifier_target") or artifact.source or artifact.tool_name or "").strip()
        exit_code = metadata.get("verifier_exit_code")
        stdout = str(metadata.get("verifier_stdout") or "").strip()
        stderr = str(metadata.get("verifier_stderr") or "").strip()
        transcript = stdout or stderr
        if len(transcript) > 320:
            transcript = f"{transcript[:320].rstrip()}..."
        details = [f"Verifier {verifier_verdict}: {artifact.summary or target or artifact.tool_name}"]
        if target:
            details.append(f"Target: {target}")
        if exit_code not in ("", None):
            details.append(f"Exit code: {exit_code}")
        if transcript:
            details.append(f"Key output: {transcript}")
        return "\n".join(details)[:900]

    if artifact_category(artifact) == "mutation_result":
        path = str(metadata.get("path") or artifact.source or "").strip()
        host = str(metadata.get("host") or "").strip()
        changed = metadata.get("changed")
        bits = [f"Mutation result: {artifact.summary or artifact.tool_name}"]
        if host or path:
            target = f"{host}:{path}" if host and path else host or path
            bits.append(f"Target: {target}")
        if isinstance(changed, bool):
            bits.append(f"Changed: {'yes' if changed else 'no'}")
        for key in ("bytes_written", "actual_occurrences", "expected_occurrences", "new_sha256"):
            value = metadata.get(key)
            if value not in (None, ""):
                bits.append(f"{key}: {value}")
        readback_sha = str(metadata.get("readback_sha256") or "").strip()
        new_sha = str(metadata.get("new_sha256") or "").strip()
        verification = metadata.get("verification") if isinstance(metadata.get("verification"), dict) else {}
        if readback_sha or verification:
            verified = bool(verification.get("readback_sha256_matches")) or bool(new_sha and readback_sha == new_sha)
            bits.append(f"Readback verified: {'yes' if verified else 'no'}")
        return "\n".join(bits)[:900]

    base = f"{artifact.source or artifact.tool_name} | {artifact.summary}"
    preview = artifact.preview_text or artifact.inline_content or artifact_body_excerpt(artifact)
    if metadata.get("complete_file") and preview:
        preview = f"Full file already captured; excerpt below is preview only.\n{preview[:500].rstrip()}"
    combined = f"{base}\n{preview}".strip()
    return combined[:900]


def artifact_body_excerpt(artifact: ArtifactRecord, *, limit: int = 500) -> str:
    content_path = str(getattr(artifact, "content_path", "") or "").strip()
    if not content_path:
        return ""
    try:
        text = Path(content_path).read_text(encoding="utf-8")
    except OSError:
        return ""
    excerpt = text[:limit].strip()
    if not excerpt:
        return ""
    if len(text) > limit:
        return f"{excerpt}..."
    return excerpt


def artifact_category(artifact: ArtifactRecord) -> str:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    if str(metadata.get("verifier_verdict") or "").strip():
        return "verifier"
    if str(artifact.tool_name or artifact.kind or "").strip() in MUTATION_RESULT_TOOLS:
        return "mutation_result"
    if artifact.kind == "file_read" and bool(metadata.get("complete_file")):
        return "primary_file"
    return "other"


def artifact_dedupe_key(artifact: ArtifactRecord) -> str:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    category = artifact_category(artifact)
    if category == "verifier":
        family = str(metadata.get("attempt_family") or "").strip()
        if family:
            return f"verifier:{family}"
        target = str(
            metadata.get("verifier_target")
            or metadata.get("command")
            or artifact.source
            or artifact.summary
            or ""
        ).strip()
        return f"verifier:{target.lower()}"
    if category == "mutation_result":
        path = str(metadata.get("path") or artifact.source or "").strip()
        host = str(metadata.get("host") or "").strip().lower()
        if path:
            return f"mutation_result:{host}:{Path(path).as_posix().lower()}"
        return f"mutation_result:{artifact.artifact_id}"
    path = str(metadata.get("path") or artifact.source or "").strip()
    if path:
        normalized = Path(path).as_posix().lower()
        return f"{category}:{normalized}"
    return f"{category}:{artifact.artifact_id}"


def artifact_success(artifact: ArtifactRecord) -> bool:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    if "success" in metadata:
        return bool(metadata.get("success"))
    return True


def artifact_host(artifact: ArtifactRecord) -> str:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    host = str(metadata.get("host") or "").strip().lower()
    if host:
        return host
    arguments = metadata.get("arguments")
    if isinstance(arguments, dict):
        return str(arguments.get("host") or "").strip().lower()
    return ""


def artifact_path(artifact: ArtifactRecord) -> str:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    path = str(metadata.get("path") or "").strip()
    if not path:
        arguments = metadata.get("arguments")
        if isinstance(arguments, dict):
            path = str(arguments.get("path") or "").strip()
    if not path:
        source = str(artifact.source or "").strip()
        if source.startswith("/"):
            path = source
    return Path(path).as_posix().lower() if path else ""


def artifact_has_resolved_successor(
    *,
    state: LoopState,
    artifact: ArtifactRecord,
    max_gap: int = 6,
) -> bool:
    if artifact_success(artifact):
        return False
    artifact_items = list(state.artifacts.items())
    artifact_id = str(getattr(artifact, "artifact_id", "") or "").strip()
    current_index = next(
        (index for index, (candidate_id, _) in enumerate(artifact_items) if candidate_id == artifact_id),
        -1,
    )
    if current_index < 0:
        return False

    failure_path = artifact_path(artifact)
    failure_host = artifact_host(artifact)
    if not failure_path:
        return False

    failure_tool = str(getattr(artifact, "tool_name", "") or getattr(artifact, "kind", "") or "").strip().lower()
    for _, successor in artifact_items[current_index + 1 : current_index + 1 + max_gap]:
        if not artifact_success(successor):
            continue
        successor_path = artifact_path(successor)
        if successor_path != failure_path:
            continue
        successor_host = artifact_host(successor)
        if failure_host and successor_host and successor_host != failure_host:
            continue
        successor_tool = str(
            getattr(successor, "tool_name", "") or getattr(successor, "kind", "") or ""
        ).strip().lower()
        if failure_tool and successor_tool and successor_tool != failure_tool:
            continue
        return True
    return False


def query_requests_specific_detail(query: str) -> bool:
    lowered = str(query or "").lower()
    if not lowered:
        return False
    detail_markers = (
        "specific line",
        "specific lines",
        "line-level",
        "line level",
        "line numbers",
        "line number",
        "start_line",
        "end_line",
        "artifact_read",
        "quote the line",
        "show the line",
        "show lines",
        "inspect lines",
        "page forward",
        "narrow excerpt",
        "exact excerpt",
        "specific excerpt",
        "prior evidence",
        "artifact summary",
        "artifact summaries",
        "artifact snippet",
        "artifact snippets",
        "show evidence",
        "reuse evidence",
    )
    return any(marker in lowered for marker in detail_markers)


def handoff_recent_research_artifact_ids(state: LoopState) -> set[str]:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return set()
    handoff = scratchpad.get("_last_task_handoff")
    if not isinstance(handoff, dict):
        return set()
    artifact_ids = handoff.get("recent_research_artifact_ids")
    if not isinstance(artifact_ids, list):
        return set()
    return {
        str(artifact_id).strip()
        for artifact_id in artifact_ids
        if str(artifact_id).strip()
    }


def should_pin_recent_research_artifacts(state: LoopState) -> bool:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return False
    if scratchpad.get("_task_boundary_previous_task"):
        return True
    if isinstance(scratchpad.get("_resolved_followup"), dict):
        return True
    if isinstance(scratchpad.get("_resolved_remote_followup"), dict):
        return True
    return False


def artifact_tool_name(artifact: ArtifactRecord) -> str:
    return str(getattr(artifact, "tool_name", "") or getattr(artifact, "kind", "") or "").strip().lower()


def artifact_failure_text(artifact: ArtifactRecord) -> str:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    output = metadata.get("output")
    output_bits: list[str] = []
    if isinstance(output, dict):
        output_bits.extend([
            str(output.get("stdout") or ""),
            str(output.get("stderr") or ""),
        ])
    return "\n".join(
        bit
        for bit in [
            str(artifact.summary or ""),
            str(artifact.preview_text or ""),
            str(artifact.inline_content or ""),
            str(metadata.get("error") or ""),
            str(metadata.get("failure_mode") or ""),
            str(metadata.get("failure_kind") or ""),
            *output_bits,
        ]
        if bit
    )


def artifact_contains_interactive_prompt(artifact: ArtifactRecord) -> bool:
    text = artifact_failure_text(artifact).lower()
    return bool(text and any(marker in text for marker in INTERACTIVE_PROMPT_MARKERS))


def is_remote_repair_state(state: LoopState) -> bool:
    phase = str(getattr(state, "current_phase", "") or "").strip().lower()
    task_mode = str(getattr(state, "task_mode", "") or "").strip().lower()
    active_intent = normalize_intent_label(getattr(state, "active_intent", "") or "")
    intent_tags = {
        str(tag or "").strip().lower()
        for tag in getattr(state, "intent_tags", []) or []
        if str(tag or "").strip()
    }
    return (
        phase == "repair"
        and (
            task_mode == "remote_execute"
            or active_intent == "requested_ssh_exec"
            or "ssh_exec" in intent_tags
        )
    )


def is_causal_remote_failure_artifact(artifact: ArtifactRecord) -> bool:
    if artifact_tool_name(artifact) != "ssh_exec":
        return False
    if artifact_success(artifact):
        return False
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    failure_kind = str(metadata.get("failure_kind") or "").strip().lower()
    ssh_transport_succeeded = bool(metadata.get("ssh_transport_succeeded"))
    output_received = bool(metadata.get("output_received"))
    return (
        failure_kind == "remote_command"
        or ssh_transport_succeeded
        or output_received
        or artifact_contains_interactive_prompt(artifact)
    )


def latest_causal_remote_failure_artifact_id(state: LoopState) -> str:
    latest = ""
    for artifact_id, artifact in getattr(state, "artifacts", {}).items():
        if is_causal_remote_failure_artifact(artifact):
            latest = str(artifact_id or "")
    return latest
