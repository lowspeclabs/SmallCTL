from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

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
                    f.write(json.dumps(json_safe_value(record)) + "\n")
        except Exception as exc:
            log.error("Failed to write experience memory to %s: %s", self.path, exc)


def search_memories(
    memories: list[ExperienceMemory],
    *,
    query: str = "",
    intent: str = "",
    tool_name: str = "",
    outcome: str = "",
    failure_mode: str = "",
) -> list[ExperienceMemory]:
    matches = []
    q = query.lower()
    i = intent.lower()
    t = tool_name.lower()
    o = outcome.lower()
    f = failure_mode.lower()
    
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
        matches.append(m)
    return matches
