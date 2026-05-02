from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..state import EvidenceRecord, LoopState

_ADAPTER_KIND_MAP = {
    "artifact_replay": "artifact_replay",
    "file_read_fact": "file_fact",
    "file_state": "file_fact",
    "verifier_verdict": "verifier_verdict",
    "shell_observation_list": "observation_list",
    "search_observation_list": "observation_list",
    "web_fetch_observation": "web_observation",
    "artifact_observation_list": "observation_list",
    "shell_observation": "shell_observation",
}


@dataclass(slots=True)
class ObservationPacket:
    observation_id: str
    kind: str
    summary: str
    phase: str = ""
    tool_name: str = ""
    confidence: float = 0.0
    artifact_id: str = ""
    operation_id: str = ""
    path: str = ""
    command: str = ""
    failure_mode: str = ""
    adapter: str = ""
    verdict: str = ""
    query: str = ""
    replayed: bool = False
    stale: bool = False


def build_observation_packets(
    state: LoopState,
    *,
    limit: int = 8,
) -> list[ObservationPacket]:
    if limit <= 0:
        return []
    records = list(getattr(state.reasoning_graph, "evidence_records", []) or [])
    if not records:
        return []

    packets: list[ObservationPacket] = []
    seen_keys: set[str] = set()
    for record in reversed(records):
        packet = _packet_from_evidence(record)
        if packet is None:
            continue
        packet.stale = _is_stale_packet(state, packet)
        key = _dedupe_key(packet)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        packets.append(packet)
        if len(packets) >= limit:
            break

    packets.reverse()
    return packets


def _packet_from_evidence(record: EvidenceRecord) -> ObservationPacket | None:
    metadata = record.metadata if isinstance(record.metadata, dict) else {}
    adapter = _normalize_adapter(metadata.get("observation_adapter"))
    kind = _classify_observation_kind(record, adapter=adapter)
    path = str(metadata.get("path") or "").strip()
    command = str(metadata.get("command") or "").strip()
    failure_mode = str(metadata.get("failure_mode") or "").strip()
    verdict = str(metadata.get("verdict") or "").strip().lower()
    query = str(metadata.get("query") or metadata.get("pattern") or "").strip()
    observation_items = _coerce_string_list(metadata.get("observation_items"))
    statement = _clean_statement(record)
    summary = _normalize_observation_summary(
        statement=statement,
        kind=kind,
        path=path,
        command=command,
        failure_mode=failure_mode,
        verdict=verdict,
        query=query,
        observation_items=observation_items,
        negative=bool(record.negative),
    )
    if not summary:
        return None

    return ObservationPacket(
        observation_id=str(record.evidence_id or "").strip(),
        kind=kind,
        summary=summary,
        phase=str(record.phase or "").strip(),
        tool_name=str(record.tool_name or "").strip(),
        confidence=float(record.confidence or 0.0),
        artifact_id=str(record.artifact_id or "").strip(),
        operation_id=str(record.operation_id or "").strip(),
        path=path,
        command=command,
        failure_mode=failure_mode,
        adapter=adapter,
        verdict=verdict,
        query=query,
        replayed=bool(record.replayed),
    )


def _classify_observation_kind(record: EvidenceRecord, *, adapter: str) -> str:
    metadata = record.metadata if isinstance(record.metadata, dict) else {}
    if adapter in _ADAPTER_KIND_MAP:
        return _ADAPTER_KIND_MAP[adapter]
    observation_kind = str(metadata.get("observation_kind") or "").strip().lower()
    if observation_kind:
        return observation_kind
    tool = str(record.tool_name or "").strip().lower()
    if record.replayed or str(record.evidence_type or "") == "replayed_or_cached":
        return "artifact_replay"
    if tool in {"file_read", "file_patch", "ast_patch", "file_write", "file_append", "file_delete", "dir_list"}:
        return "file_fact"
    if tool in {"shell_exec", "ssh_exec"}:
        if "verdict" in metadata or "exit_code" in metadata:
            return "verifier_verdict"
        return "shell_observation"
    if tool in {"search", "artifact_read"}:
        return "observation_list"
    if record.negative:
        return "negative_observation"
    return "observation"


def _normalize_observation_summary(
    *,
    statement: str,
    kind: str,
    path: str,
    command: str,
    failure_mode: str,
    verdict: str,
    query: str,
    observation_items: list[str],
    negative: bool,
) -> str:
    if kind == "file_fact":
        fact = statement or (observation_items[0] if observation_items else "")
        if not fact and path:
            fact = "file state updated"
        if not fact:
            return ""
        if path:
            return f"File fact ({path}): {fact}"[:340]
        return f"File fact: {fact}"[:320]
    if kind == "verifier_verdict":
        verdict_label = verdict or ("fail" if (failure_mode or negative) else "pass")
        prefix = f"Verifier verdict ({verdict_label})"
        if command:
            prefix += f" [{command}]"
        body = statement or (observation_items[0] if observation_items else "no output captured")
        return f"{prefix}: {body}"[:360]
    if kind == "artifact_replay":
        if statement:
            return f"Replayed evidence: {statement}"[:320]
        if observation_items:
            return f"Replayed evidence: {observation_items[0]}"[:320]
        return "Replayed evidence"[:320]
    if kind == "observation_list":
        if observation_items:
            head = "Observation list"
            if query:
                head += f" [{query}]"
            elif command:
                head += f" [{command}]"
            lead = observation_items[0]
            suffix = f" (+{len(observation_items) - 1} more)" if len(observation_items) > 1 else ""
            return f"{head}: {lead}{suffix}"[:360]
        if statement:
            return statement[:320]
        return ""
    if kind == "web_observation":
        body = observation_items[0] if observation_items else statement
        if not body:
            return ""
        return f"Web finding: {body}"[:360]
    if kind == "shell_observation":
        if command and statement:
            return f"Shell observation [{command}]: {statement}"[:340]
        if command:
            return f"Shell observation [{command}]"[:280]
        if statement:
            return f"Shell observation: {statement}"[:320]
        return ""
    if kind == "negative_observation":
        if statement:
            return f"Negative observation: {statement}"[:320]
        if observation_items:
            return f"Negative observation: {observation_items[0]}"[:320]
        return ""
    if statement:
        return statement[:320]
    if observation_items:
        return observation_items[0][:320]
    return ""


def _clean_statement(record: EvidenceRecord) -> str:
    statement = str(record.statement or "").strip()
    if not statement:
        return ""
    tool = str(record.tool_name or "").strip()
    if tool and statement.lower().startswith(f"{tool.lower()}:"):
        statement = statement[len(tool) + 1 :].strip()
    return statement


def _normalize_adapter(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return text


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    cleaned: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        if text not in cleaned:
            cleaned.append(text)
    return cleaned


def _dedupe_key(packet: ObservationPacket) -> str:
    if packet.observation_id:
        return f"id:{packet.observation_id}"
    return "|".join(
        [
            packet.kind,
            packet.adapter.lower(),
            packet.path.lower(),
            packet.command.lower(),
            packet.summary.lower(),
        ]
    )


def _is_stale_packet(state: LoopState, packet: ObservationPacket) -> bool:
    invalidations = state.scratchpad.get("_context_invalidations")
    if not isinstance(invalidations, list):
        return False
    for item in invalidations[-24:]:
        if not isinstance(item, dict):
            continue
        reason = str(item.get("reason") or "").strip().lower()
        paths = item.get("paths")
        path_list = [str(path).strip() for path in paths] if isinstance(paths, list) else []
        if reason == "file_changed" and packet.path:
            if any(_path_match(packet.path, changed) for changed in path_list):
                return True
        if reason == "verifier_failed" and packet.kind == "verifier_verdict":
            return True
    return False


def _path_match(left: str, right: str) -> bool:
    lhs = str(left or "").strip().lower()
    rhs = str(right or "").strip().lower()
    if not lhs or not rhs:
        return False
    return lhs == rhs or lhs.endswith(rhs) or rhs.endswith(lhs)


__all__ = ["ObservationPacket", "build_observation_packets"]
