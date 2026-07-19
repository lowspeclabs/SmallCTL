from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
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
        self._cache_lock = threading.Lock()
        self._cache: tuple[int, int, list[Any]] | None = None

    def get(self, memory_id: str) -> ExperienceMemory | None:
        memories = self.list()
        for m in memories:
            if m.memory_id == memory_id:
                return m
        return None

    def list(self) -> list[ExperienceMemory]:
        records = []
        for payload in self._read_payloads():
            try:
                records.append(_coerce_experience_memory(payload))
            except (TypeError, ValueError):
                continue
        return records

    def upsert(self, memory: ExperienceMemory) -> ExperienceMemory | None:
        with self._write_lock():
            records = self._read_records_locked()
            found = False
            for i, existing in enumerate(records):
                if existing.memory_id == memory.memory_id:
                    records[i] = memory
                    found = True
                    break
            if not found:
                records.append(memory)
            if not self._write_all_locked(records):
                return None
        return memory

    def delete(self, memory_id: str) -> bool:
        with self._write_lock():
            records = self._read_records_locked()
            remaining = [r for r in records if r.memory_id != memory_id]
            if len(remaining) == len(records):
                return False
            return self._write_all_locked(remaining)

    def write_all(self, records: list[ExperienceMemory]) -> bool:
        with self._write_lock():
            return self._write_all_locked(records)

    def scrub_sensitive_notes(self, *, write: bool = False) -> dict[str, int]:
        if not write:
            records = [_coerce_experience_memory(json_safe_value(record)) for record in self.list()]
            changed = sum(1 for record in records if self._redact_record_notes(record))
            return {"records": len(records), "changed": changed, "written": 0}
        # The read-modify-write must hold the sidecar write lock for the entire
        # transaction; reading first and locking only around the write loses any
        # concurrent upsert that lands in between.
        with self._write_lock():
            records = self._read_records_locked()
            changed = sum(1 for record in records if self._redact_record_notes(record))
            written = False
            if changed:
                written = self._write_all_locked(records)
            return {"records": len(records), "changed": changed, "written": int(bool(written))}

    @staticmethod
    def _redact_record_notes(record: ExperienceMemory) -> bool:
        original = str(record.notes or "")
        redacted = redact_sensitive_text(original)
        if redacted == original:
            return False
        record.notes = redacted
        return True

    def _read_records_locked(self) -> list[ExperienceMemory]:
        records = []
        for payload in self._read_payloads(force=True):
            try:
                records.append(_coerce_experience_memory(payload))
            except (TypeError, ValueError):
                continue
        return records

    def _read_payloads(self, *, force: bool = False) -> list[Any]:
        try:
            stat = self.path.stat()
        except OSError:
            return []
        with self._cache_lock:
            if not force and self._cache is not None and self._cache[:2] == (stat.st_mtime_ns, stat.st_size):
                return list(self._cache[2])
        payloads: list[Any] = []
        skipped = 0
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        payloads.append(json.loads(line))
                    except json.JSONDecodeError:
                        skipped += 1
        except Exception as exc:
            log.warning("Failed to load experience memory from %s: %s", self.path, exc)
            return payloads
        if skipped:
            log.warning("Skipped %d unparseable experience memory lines in %s", skipped, self.path)
        with self._cache_lock:
            self._cache = (stat.st_mtime_ns, stat.st_size, list(payloads))
        return payloads

    def _write_all_locked(self, records: list[ExperienceMemory]) -> bool:
        temp_path: Path | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
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
                # Defense-in-depth at the persistence boundary: secrets pasted
                # into free-text notes must never reach disk in plaintext, even
                # if the caller skipped the redaction layer. Runs after
                # namespace inference so classification still sees the raw text.
                record.notes = redact_sensitive_text(str(record.notes or ""))
                lines.append(json.dumps(json_safe_value(record)))
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                os.chmod(temp_path, 0o600)
                for line in lines:
                    handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        except Exception as exc:
            log.error("Failed to write experience memory to %s: %s", self.path, exc)
            return False
        finally:
            if temp_path is not None:
                with contextlib.suppress(FileNotFoundError):
                    temp_path.unlink()
        try:
            stat = self.path.stat()
        except OSError:
            return True
        with self._cache_lock:
            self._cache = (stat.st_mtime_ns, stat.st_size, [json.loads(line) for line in lines])
        return True

    @contextlib.contextmanager
    def _write_lock(self):
        """Serialize read-modify-write transactions across threads and processes."""
        lock_path = self.path.with_name(f"{self.path.name}.lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as handle:
            try:
                import fcntl
            except ImportError:
                yield
                return
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except (ImportError, OSError):
                    pass


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
