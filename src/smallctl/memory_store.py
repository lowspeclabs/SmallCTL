from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .experience_tags import normalize_experience_tags
from .memory_namespace import infer_memory_namespace, normalize_memory_namespace
from .redaction import redact_sensitive_text
from .state import ExperienceMemory, _coerce_experience_memory, json_safe_value

log = logging.getLogger("smallctl.memory")


class ExperienceStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def get(self, memory_id: str) -> ExperienceMemory | None:
        memories = self.list()
        for m in memories:
            if m.memory_id == memory_id:
                return m
        return None

    def list(self) -> list[ExperienceMemory]:
        if not self.path.exists():
            return []
        records = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = _coerce_experience_memory(json.loads(line))
                        records.append(record)
                    except (json.JSONDecodeError, TypeError):
                        continue
        except Exception as exc:
            log.warning("Failed to load experience memory from %s: %s", self.path, exc)
        return records

    def upsert(self, memory: ExperienceMemory) -> ExperienceMemory:
        records = self.list()
        found = False
        for i, existing in enumerate(records):
            if existing.memory_id == memory.memory_id:
                records[i] = memory
                found = True
                break
        if not found:
            records.append(memory)
        self.write_all(records)
        return memory

    def delete(self, memory_id: str) -> bool:
        records = self.list()
        initial_len = len(records)
        records = [r for r in records if r.memory_id != memory_id]
        if len(records) < initial_len:
            self.write_all(records)
            return True
        return False

    def write_all(self, records: list[ExperienceMemory]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                for record in records:
                    record.namespace = normalize_memory_namespace(record.namespace) or infer_memory_namespace(
                        tool_name=record.tool_name,
                        intent=record.intent,
                        intent_tags=record.intent_tags,
                        environment_tags=record.environment_tags,
                        entity_tags=record.entity_tags,
                        notes=record.notes,
                    )
                    record.intent_tags = normalize_experience_tags(record.intent_tags)
                    record.environment_tags = normalize_experience_tags(record.environment_tags)
                    record.entity_tags = normalize_experience_tags(record.entity_tags)
                    f.write(json.dumps(json_safe_value(record)) + "\n")
        except Exception as exc:
            log.error("Failed to write experience memory to %s: %s", self.path, exc)

    def scrub_sensitive_notes(self, *, write: bool = False) -> dict[str, int]:
        records = self.list()
        changed = 0
        for record in records:
            original = str(record.notes or "")
            redacted = redact_sensitive_text(original)
            if redacted != original:
                record.notes = redacted
                changed += 1
        if write and changed:
            self.write_all(records)
        return {
            "records": len(records),
            "changed": changed,
            "written": int(bool(write and changed)),
        }


def search_memories(
    memories: list[ExperienceMemory],
    *,
    query: str = "",
    intent: str = "",
    tool_name: str = "",
    outcome: str = "",
    failure_mode: str = "",
    namespace: str = "",
) -> list[ExperienceMemory]:
    matches = []
    q = query.lower()
    i = intent.lower()
    t = tool_name.lower()
    o = outcome.lower()
    f = failure_mode.lower()
    n = normalize_memory_namespace(namespace)
    
    for m in memories:
        if q and q not in m.notes.lower() and q not in m.intent.lower():
            continue
        if i and i not in m.intent.lower():
            continue
        if t and t != m.tool_name.lower():
            continue
        if o and o != m.outcome.lower():
            continue
        if f and f not in m.failure_mode.lower():
            continue
        if n and normalize_memory_namespace(getattr(m, "namespace", "")) != n:
            continue
        matches.append(m)
    return matches
