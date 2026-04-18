from __future__ import annotations

from dataclasses import dataclass

from ..state import EvidenceRecord, LoopState


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
    if not str(record.statement or "").strip():
        return None

    metadata = record.metadata if isinstance(record.metadata, dict) else {}
    kind = _classify_observation_kind(record)
    path = str(metadata.get("path") or "").strip()
    command = str(metadata.get("command") or "").strip()
    failure_mode = str(metadata.get("failure_mode") or "").strip()
    summary = _normalize_observation_summary(record, kind=kind, path=path, command=command)
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
        replayed=bool(record.replayed),
    )


def _classify_observation_kind(record: EvidenceRecord) -> str:
    metadata = record.metadata if isinstance(record.metadata, dict) else {}
    tool = str(record.tool_name or "").strip().lower()
    if record.replayed or str(record.evidence_type or "") == "replayed_or_cached":
        return "artifact_replay"
    if tool in {"file_read", "file_patch", "file_write", "file_append", "file_delete", "dir_list"}:
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
    record: EvidenceRecord,
    *,
    kind: str,
    path: str,
    command: str,
) -> str:
    statement = str(record.statement or "").strip()
    if not statement:
        return ""

    if kind == "file_fact":
        if path:
            return f"File fact ({path}): {statement}"[:320]
        return f"File fact: {statement}"[:320]
    if kind == "verifier_verdict":
        verdict = "pass"
        if record.negative:
            verdict = "fail"
        prefix = f"Verifier verdict ({verdict})"
        if command:
            prefix += f" [{command}]"
        return f"{prefix}: {statement}"[:340]
    if kind == "artifact_replay":
        return f"Replayed evidence: {statement}"[:300]
    if kind == "shell_observation":
        if command:
            return f"Shell observation [{command}]: {statement}"[:340]
        return f"Shell observation: {statement}"[:320]
    if kind == "negative_observation":
        return f"Negative observation: {statement}"[:320]
    return statement[:320]


def _dedupe_key(packet: ObservationPacket) -> str:
    if packet.observation_id:
        return f"id:{packet.observation_id}"
    return "|".join(
        [
            packet.kind,
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
