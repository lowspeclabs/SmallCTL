from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import random
import tempfile
import threading
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.memory import InMemorySaver


logger = logging.getLogger("smallctl.graph.checkpoint")

DEFAULT_MAX_CHECKPOINTS_PER_THREAD = 32


def _checkpoint_backup_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.bak")


def _parse_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _is_valid_typed_payload(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if not isinstance(value.get("type"), str) or not isinstance(value.get("data"), str):
        return False
    try:
        base64.b64decode(value["data"].encode("ascii"), validate=True)
    except Exception:
        return False
    return True


def _validate_checkpoint_record(section: str, item: Any) -> str | None:
    if not isinstance(item, dict):
        return "record is not an object"
    for field_name in ("thread_id", "checkpoint_ns"):
        if not isinstance(item.get(field_name), str):
            return f"{field_name} missing or not a string"
    if section == "storage":
        if not isinstance(item.get("checkpoint_id"), str):
            return "checkpoint_id missing or not a string"
        for field_name in ("checkpoint", "metadata"):
            if not _is_valid_typed_payload(item.get(field_name)):
                return f"{field_name} is not a valid typed payload"
        parent = item.get("parent_checkpoint_id")
        if parent is not None and not isinstance(parent, str):
            return "parent_checkpoint_id is not a string or null"
        return None
    if section == "writes":
        for field_name in ("checkpoint_id", "task_id", "channel"):
            if not isinstance(item.get(field_name), str):
                return f"{field_name} missing or not a string"
        idx = item.get("idx")
        if isinstance(idx, bool) or not isinstance(idx, int):
            return "idx missing or not an int"
        if not _is_valid_typed_payload(item.get("value")):
            return "value is not a valid typed payload"
        if not isinstance(item.get("task_path", ""), str):
            return "task_path is not a string"
        return None
    if not isinstance(item.get("channel"), str):
        return "channel missing or not a string"
    version = item.get("version")
    if (
        not isinstance(version, dict)
        or version.get("kind") not in ("str", "int", "float")
        or "value" not in version
    ):
        return "version is not a valid encoded key"
    if not _is_valid_typed_payload(item.get("value")):
        return "value is not a valid typed payload"
    return None


def _validate_checkpoint_payload(payload: Any) -> list[str]:
    """Structural validation of the checkpoint database payload.

    The schema mirrors exactly what ``FileCheckpointSaver._flush_to_disk``
    serializes: three list sections (``storage``, ``writes``, ``blobs``) of dict
    records with typed ``{"type", "data"}`` blobs. A legitimately empty database
    is ``{"storage": [], "writes": [], "blobs": []}`` — that is what a fresh
    saver flushes, and it must load cleanly. Anything missing sections or
    containing malformed records (including a bare ``{}``) is reported so the
    caller treats the file as corrupt and falls back to the ``.bak`` backup
    instead of silently loading partial state and later flushing that
    incomplete state over both files.
    """
    if not isinstance(payload, dict):
        return ["payload is not a JSON object"]
    problems: list[str] = []
    for section in ("storage", "writes", "blobs"):
        records = payload.get(section)
        if not isinstance(records, list):
            problems.append(f"{section}: missing or not a list")
            continue
        for index, item in enumerate(records):
            problem = _validate_checkpoint_record(section, item)
            if problem is not None:
                problems.append(f"{section}[{index}]: {problem}")
    return problems


def _parse_validated_payload(path: Path) -> dict[str, Any] | None:
    payload = _parse_payload(path)
    if payload is None:
        return None
    problems = _validate_checkpoint_payload(payload)
    if not problems:
        return payload
    logger.warning(
        "checkpoint_validation_failed %s",
        {
            "event": "checkpoint_validation_failed",
            "path": str(path),
            "problem_count": len(problems),
            "problems": problems[:5],
        },
    )
    return None


def _atomic_write_text(path: Path, serialized: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            os.chmod(temp_path, 0o600)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp_path is not None:
            with contextlib.suppress(FileNotFoundError):
                temp_path.unlink()


class FileCheckpointSaver(BaseCheckpointSaver[str]):
    def __init__(
        self,
        path: str | Path,
        *,
        max_checkpoints_per_thread: int = DEFAULT_MAX_CHECKPOINTS_PER_THREAD,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._mtime_ns: int | None = None
        self._degraded = False
        self._max_checkpoints_per_thread = max(1, int(max_checkpoints_per_thread))
        # Monotonic write order per (thread_id, checkpoint_ns, checkpoint_id);
        # retention pruning ranks by this instead of lexical checkpoint ids so
        # a non-lexically-ordered id never gets the newest checkpoint pruned.
        self._write_sequence: int = 0
        self._checkpoint_recency: dict[tuple[str, str, str], int] = {}
        self.storage: dict[
            str,
            dict[str, dict[str, tuple[tuple[str, bytes], tuple[str, bytes], str | None]]],
        ] = {}
        self.writes: dict[
            tuple[str, str, str],
            dict[tuple[str, int], tuple[str, str, tuple[str, bytes], str]],
        ] = {}
        self.blobs: dict[tuple[str, str, str, str | int | float], tuple[str, bytes]] = {}
        self._load_from_disk()

    @property
    def degraded(self) -> bool:
        return self._degraded

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        with self._lock:
            self._load_if_changed()
            thread_id = str(config["configurable"]["thread_id"])
            checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
            thread_storage = self.storage.get(thread_id, {})
            checkpoints = thread_storage.get(checkpoint_ns, {})
            checkpoint_id = get_checkpoint_id(config)
            if checkpoint_id:
                saved = checkpoints.get(checkpoint_id)
                if saved is None:
                    return None
                resolved_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                }
                return self._try_build_checkpoint_tuple(
                    config=resolved_config,
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    saved=saved,
                )
            if not checkpoints:
                return None
            for checkpoint_id, saved in sorted(checkpoints.items(), reverse=True):
                item = self._try_build_checkpoint_tuple(
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                        }
                    },
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    saved=saved,
                )
                if item is not None:
                    return item
            return None

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ):
        with self._lock:
            self._load_if_changed()
            thread_ids = (
                [str(config["configurable"]["thread_id"])] if config else sorted(self.storage.keys())
            )
            requested_ns = str(config["configurable"].get("checkpoint_ns", "")) if config else None
            requested_id = get_checkpoint_id(config) if config else None
            before_id = get_checkpoint_id(before) if before else None
            remaining = limit
            for thread_id in thread_ids:
                thread_storage = self.storage.get(thread_id, {})
                namespaces = [requested_ns] if requested_ns is not None else sorted(thread_storage.keys())
                for checkpoint_ns in namespaces:
                    checkpoints = thread_storage.get(checkpoint_ns, {})
                    for checkpoint_id, saved in sorted(checkpoints.items(), reverse=True):
                        if requested_id and checkpoint_id != requested_id:
                            continue
                        if before_id and checkpoint_id >= before_id:
                            continue
                        item = self._try_build_checkpoint_tuple(
                            config={
                                "configurable": {
                                    "thread_id": thread_id,
                                    "checkpoint_ns": checkpoint_ns,
                                    "checkpoint_id": checkpoint_id,
                                }
                            },
                            thread_id=thread_id,
                            checkpoint_ns=checkpoint_ns,
                            checkpoint_id=checkpoint_id,
                            saved=saved,
                        )
                        if item is None:
                            continue
                        if filter and not all(
                            item.metadata.get(key) == value for key, value in filter.items()
                        ):
                            continue
                        yield item
                        if remaining is not None:
                            remaining -= 1
                            if remaining <= 0:
                                return

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        with self._lock:
            with self._write_lock():
                self._load_from_disk()
                c = checkpoint.copy()
                thread_id = str(config["configurable"]["thread_id"])
                checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
                raw_values = c.pop("channel_values", {})
                values = raw_values if isinstance(raw_values, dict) else {}
                version_map = new_versions if isinstance(new_versions, dict) else {}
                for channel, version in version_map.items():
                    self.blobs[(thread_id, checkpoint_ns, channel, version)] = (
                        self.serde.dumps_typed(values[channel]) if channel in values else ("empty", b"")
                    )
                self.storage.setdefault(thread_id, {}).setdefault(checkpoint_ns, {})[checkpoint["id"]] = (
                    self.serde.dumps_typed(c),
                    self.serde.dumps_typed(get_checkpoint_metadata(config, metadata)),
                    config["configurable"].get("checkpoint_id"),
                )
                self._write_sequence += 1
                self._checkpoint_recency[(thread_id, checkpoint_ns, str(checkpoint["id"]))] = (
                    self._write_sequence
                )
                self._flush_to_disk()
                return {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint["id"],
                    }
                }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]] | tuple[tuple[str, Any], ...],
        task_id: str,
        task_path: str = "",
    ) -> None:
        with self._lock:
            with self._write_lock():
                self._load_from_disk()
                thread_id = str(config["configurable"]["thread_id"])
                checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
                checkpoint_id = str(config["configurable"]["checkpoint_id"])
                outer_key = (thread_id, checkpoint_ns, checkpoint_id)
                existing = self.writes.get(outer_key)
                for idx, (channel, value) in enumerate(writes):
                    inner_key = (task_id, WRITES_IDX_MAP.get(channel, idx))
                    if inner_key[1] >= 0 and existing and inner_key in existing:
                        continue
                    self.writes.setdefault(outer_key, {})[inner_key] = (
                        task_id,
                        channel,
                        self.serde.dumps_typed(value),
                        task_path,
                    )
                self._flush_to_disk()

    def checkpoint_history_summary(
        self,
        thread_id: str | None = None,
        checkpoint_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return a chronological, metadata-only summary of checkpoint writes.

        The summary contains only the identifiers that describe *which* task and
        channel wrote to *which* checkpoint. It deliberately does not include
        decoded checkpoint values or pending-write payloads.
        """
        with self._lock:
            self._load_if_changed()
            if thread_id is None:
                thread_id = self.latest_thread_id()
                if thread_id is None:
                    return []
            summary: list[dict[str, Any]] = []
            for (tid, cns, cid), entries in self.writes.items():
                if tid != thread_id:
                    continue
                if checkpoint_id is not None and cid != checkpoint_id:
                    continue
                timestamp = self._checkpoint_timestamp(tid, cns, cid)
                for (task_id, _idx), record in entries.items():
                    if (
                        not isinstance(record, tuple)
                        or len(record) != 4
                        or not isinstance(record[0], str)
                        or not isinstance(record[1], str)
                        or not isinstance(record[2], tuple)
                        or len(record[2]) != 2
                    ):
                        continue
                    _, channel, _value, task_path = record
                    summary.append(
                        {
                            "checkpoint_id": str(cid),
                            "checkpoint_ns": str(cns),
                            "task_id": record[0],
                            "channel": str(channel),
                            "task_path": str(task_path),
                            "timestamp": timestamp,
                        }
                    )
            return sorted(
                summary,
                key=lambda entry: (
                    entry["timestamp"] is None,
                    entry["timestamp"] or "",
                    entry["checkpoint_ns"],
                    entry["checkpoint_id"],
                    entry["task_id"],
                    entry["channel"],
                ),
            )

    def _checkpoint_timestamp(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> str | None:
        thread_storage = self.storage.get(thread_id, {})
        checkpoints = thread_storage.get(checkpoint_ns, {})
        saved = checkpoints.get(checkpoint_id)
        if saved is None:
            return None
        try:
            metadata = self.serde.loads_typed(saved[1])
        except Exception:
            return None
        if not isinstance(metadata, dict):
            return None
        return str(metadata["timestamp"]) if metadata.get("timestamp") else None

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            with self._write_lock():
                self._load_from_disk()
                self.storage.pop(thread_id, None)
                for key in list(self.writes.keys()):
                    if key[0] == thread_id:
                        del self.writes[key]
                for key in list(self.blobs.keys()):
                    if key[0] == thread_id:
                        del self.blobs[key]
                for key in list(self._checkpoint_recency):
                    if key[0] == thread_id:
                        del self._checkpoint_recency[key]
                self._flush_to_disk()

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ):
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]] | tuple[tuple[str, Any], ...],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        self.delete_thread(thread_id)

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            try:
                current_v = int(str(current).split(".")[0])
            except (TypeError, ValueError):
                current_v = 0
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    def latest_thread_id(self) -> str | None:
        with self._lock:
            self._load_if_changed()
            latest: tuple[str, str] | None = None
            for thread_id, namespaces in self.storage.items():
                for checkpoint_ns, checkpoints in namespaces.items():
                    for checkpoint_id, saved in sorted(checkpoints.items(), reverse=True):
                        item = self._try_build_checkpoint_tuple(
                            config={
                                "configurable": {
                                    "thread_id": thread_id,
                                    "checkpoint_ns": checkpoint_ns,
                                    "checkpoint_id": checkpoint_id,
                                }
                            },
                            thread_id=thread_id,
                            checkpoint_ns=checkpoint_ns,
                            checkpoint_id=checkpoint_id,
                            saved=saved,
                        )
                        if item is None:
                            continue
                        if latest is None or checkpoint_id > latest[1]:
                            latest = (thread_id, checkpoint_id)
                        break
            return latest[0] if latest else None

    def _try_build_checkpoint_tuple(
        self,
        *,
        config: RunnableConfig,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        saved: tuple[tuple[str, bytes], tuple[str, bytes], str | None],
    ) -> CheckpointTuple | None:
        try:
            return self._build_checkpoint_tuple(
                config=config,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint_id,
                saved=saved,
            )
        except Exception:
            return None

    def _build_checkpoint_tuple(
        self,
        *,
        config: RunnableConfig,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        saved: tuple[tuple[str, bytes], tuple[str, bytes], str | None],
    ) -> CheckpointTuple:
        checkpoint_blob, metadata_blob, parent_checkpoint_id = saved
        checkpoint = self.serde.loads_typed(checkpoint_blob)
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint payload must decode to a dict")
        channel_versions = checkpoint.get("channel_versions")
        if not isinstance(channel_versions, dict):
            raise ValueError("checkpoint payload missing channel_versions")
        metadata = self.serde.loads_typed(metadata_blob)
        if not isinstance(metadata, dict):
            raise ValueError("checkpoint metadata must decode to a dict")
        writes = self.writes.get((thread_id, checkpoint_ns, checkpoint_id), {})
        return CheckpointTuple(
            config=config,
            checkpoint={
                **checkpoint,
                "channel_values": self._load_blobs(
                    thread_id,
                    checkpoint_ns,
                    channel_versions,
                ),
            },
            metadata=metadata,
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }
                if parent_checkpoint_id
                else None
            ),
            pending_writes=[
                (task_id, channel, loaded_value)
                for task_id, channel, loaded_value in self._load_pending_writes(writes)
            ],
        )

    def _load_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        versions: ChannelVersions,
    ) -> dict[str, Any]:
        values: dict[str, Any] = {}
        if not isinstance(versions, dict):
            return values
        for channel, version in versions.items():
            payload = self.blobs.get((thread_id, checkpoint_ns, channel, version))
            if payload and payload[0] != "empty":
                try:
                    values[channel] = self.serde.loads_typed(payload)
                except Exception:
                    continue
        return values

    def _load_pending_writes(
        self,
        writes: dict[tuple[str, int], tuple[str, str, tuple[str, bytes], str]],
    ) -> list[tuple[str, str, Any]]:
        loaded: list[tuple[str, str, Any]] = []
        for task_id, channel, value, _ in writes.values():
            try:
                loaded.append((task_id, channel, self.serde.loads_typed(value)))
            except Exception:
                continue
        return loaded

    def _load_if_changed(self) -> None:
        if not self.path.exists():
            if self._mtime_ns is not None or self.storage or self.writes or self.blobs:
                self._load_from_disk()
            return
        mtime_ns = self.path.stat().st_mtime_ns
        if self._mtime_ns != mtime_ns:
            self._load_from_disk()

    def _read_payload_with_backup(self) -> dict[str, Any] | None:
        backup_path = _checkpoint_backup_path(self.path)
        if self.path.exists():
            payload = _parse_validated_payload(self.path)
            if payload is not None:
                return payload
            logger.warning(
                "checkpoint_load_failed %s",
                {"event": "checkpoint_load_failed", "path": str(self.path)},
            )
            backup_payload = _parse_validated_payload(backup_path)
            if backup_payload is not None:
                logger.warning(
                    "checkpoint_backup_recovery %s",
                    {
                        "event": "checkpoint_backup_recovery",
                        "path": str(self.path),
                        "backup_path": str(backup_path),
                    },
                )
                return backup_payload
            logger.error(
                "checkpoint_unrecoverable %s",
                {"event": "checkpoint_unrecoverable", "path": str(self.path)},
            )
            return None
        backup_payload = _parse_validated_payload(backup_path)
        if backup_payload is not None:
            logger.warning(
                "checkpoint_backup_recovery %s",
                {
                    "event": "checkpoint_backup_recovery",
                    "path": str(self.path),
                    "backup_path": str(backup_path),
                },
            )
            return backup_payload
        if backup_path.exists():
            logger.error(
                "checkpoint_unrecoverable %s",
                {"event": "checkpoint_unrecoverable", "path": str(self.path)},
            )
        return None

    def _load_from_disk(self) -> None:
        self.storage = {}
        self.writes = {}
        self.blobs = {}
        self._write_sequence = 0
        self._checkpoint_recency = {}
        payload = self._read_payload_with_backup()
        if payload is None:
            self._degraded = self.path.exists() or _checkpoint_backup_path(self.path).exists()
            self._mtime_ns = self.path.stat().st_mtime_ns if self.path.exists() else None
            return
        for item in payload.get("storage", []):
            if not isinstance(item, dict):
                continue
            try:
                self.storage.setdefault(str(item["thread_id"]), {}).setdefault(
                    str(item["checkpoint_ns"]),
                    {},
                )[str(item["checkpoint_id"])] = (
                    self._decode_typed(item["checkpoint"]),
                    self._decode_typed(item["metadata"]),
                    item.get("parent_checkpoint_id"),
                )
                # The persisted storage list preserves flush (insertion) order,
                # so enumerating it rebuilds the relative write recency.
                self._write_sequence += 1
                self._checkpoint_recency[
                    (str(item["thread_id"]), str(item["checkpoint_ns"]), str(item["checkpoint_id"]))
                ] = self._write_sequence
            except Exception:
                continue
        for item in payload.get("writes", []):
            if not isinstance(item, dict):
                continue
            try:
                outer_key = (
                    str(item["thread_id"]),
                    str(item["checkpoint_ns"]),
                    str(item["checkpoint_id"]),
                )
                inner_key = (str(item["task_id"]), int(item["idx"]))
                self.writes.setdefault(outer_key, {})[inner_key] = (
                    str(item["task_id"]),
                    str(item["channel"]),
                    self._decode_typed(item["value"]),
                    str(item.get("task_path", "")),
                )
            except Exception:
                continue
        for item in payload.get("blobs", []):
            if not isinstance(item, dict):
                continue
            try:
                self.blobs[
                    (
                        str(item["thread_id"]),
                        str(item["checkpoint_ns"]),
                        str(item["channel"]),
                        self._decode_key(item["version"]),
                    )
                ] = self._decode_typed(item["value"])
            except Exception:
                continue
        self._degraded = False
        self._mtime_ns = self.path.stat().st_mtime_ns if self.path.exists() else None

    def _prune_for_retention(self) -> None:
        # The retention bound applies per thread across all checkpoint
        # namespaces, matching the max_checkpoints_per_thread contract.
        #
        # "Oldest" is determined by write recency: a monotonic sequence
        # assigned at put time and rebuilt from persisted storage order on
        # load. Lexical checkpoint id ordering is only a tie-breaker for
        # records with no recency information, so a non-lexically-ordered id
        # can never cause the newest checkpoint to be pruned.
        limit = self._max_checkpoints_per_thread
        for thread_id, namespaces in list(self.storage.items()):
            total = sum(len(checkpoints) for checkpoints in namespaces.values())
            if total <= limit:
                continue
            recency_ranked = sorted(
                (
                    (
                        self._checkpoint_recency.get((thread_id, checkpoint_ns, checkpoint_id), 0),
                        checkpoint_id,
                        checkpoint_ns,
                    )
                    for checkpoint_ns, checkpoints in namespaces.items()
                    for checkpoint_id in checkpoints
                ),
                reverse=True,
            )
            keep_keys = {
                (checkpoint_ns, checkpoint_id)
                for _sequence, checkpoint_id, checkpoint_ns in recency_ranked[:limit]
            }
            referenced_versions: set[tuple[str, str, str, str | int | float]] = set()
            for checkpoint_ns, checkpoints in list(namespaces.items()):
                for checkpoint_id in list(checkpoints):
                    if (checkpoint_ns, checkpoint_id) in keep_keys:
                        continue
                    del checkpoints[checkpoint_id]
                    self._checkpoint_recency.pop((thread_id, checkpoint_ns, checkpoint_id), None)
            for checkpoint_ns, checkpoints in namespaces.items():
                for checkpoint_id, saved in checkpoints.items():
                    try:
                        decoded = self.serde.loads_typed(saved[0])
                    except Exception:
                        continue
                    versions = decoded.get("channel_versions") if isinstance(decoded, dict) else None
                    if isinstance(versions, dict):
                        for channel, version in versions.items():
                            referenced_versions.add((thread_id, checkpoint_ns, channel, version))
            for key in list(self.writes):
                if key[0] == thread_id and (key[1], key[2]) not in keep_keys:
                    del self.writes[key]
            for key in list(self.blobs):
                if key[0] == thread_id and key not in referenced_versions:
                    del self.blobs[key]

    def _flush_to_disk(self) -> None:
        if self._degraded:
            raise RuntimeError(
                f"checkpoint store at {self.path} is degraded; "
                "refusing to overwrite an unreadable database"
            )
        self._prune_for_retention()
        payload = {
            "storage": [
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint": self._encode_typed(checkpoint),
                    "metadata": self._encode_typed(metadata),
                    "parent_checkpoint_id": parent_checkpoint_id,
                }
                for thread_id, namespaces in self.storage.items()
                for checkpoint_ns, checkpoints in namespaces.items()
                for checkpoint_id, (checkpoint, metadata, parent_checkpoint_id) in checkpoints.items()
            ],
            "writes": [
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "task_id": task_id,
                    "idx": idx,
                    "channel": channel,
                    "value": self._encode_typed(value),
                    "task_path": task_path,
                }
                for (thread_id, checkpoint_ns, checkpoint_id), entries in self.writes.items()
                for (task_id, idx), (_, channel, value, task_path) in entries.items()
            ],
            "blobs": [
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "channel": channel,
                    "version": self._encode_key(version),
                    "value": self._encode_typed(value),
                }
                for (thread_id, checkpoint_ns, channel, version), value in self.blobs.items()
            ],
        }
        serialized = json.dumps(payload, indent=2, sort_keys=True)
        _atomic_write_text(self.path, serialized)
        backup_path = _checkpoint_backup_path(self.path)
        try:
            _atomic_write_text(backup_path, serialized)
        except OSError as exc:
            logger.warning(
                "checkpoint_backup_write_failed %s",
                {
                    "event": "checkpoint_backup_write_failed",
                    "backup_path": str(backup_path),
                    "error": str(exc),
                },
            )
        self._mtime_ns = self.path.stat().st_mtime_ns

    @contextlib.contextmanager
    def _write_lock(self):
        """Serialize writers from separate saver instances and processes."""
        lock_path = self.path.with_name(f"{self.path.name}.lock")
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

    @staticmethod
    def _encode_typed(value: tuple[str, bytes]) -> dict[str, str]:
        type_name, payload = value
        return {
            "type": type_name,
            "data": base64.b64encode(payload).decode("ascii"),
        }

    @staticmethod
    def _decode_typed(value: dict[str, str]) -> tuple[str, bytes]:
        return (
            str(value["type"]),
            base64.b64decode(value["data"].encode("ascii")),
        )

    @staticmethod
    def _encode_key(value: str | int | float) -> dict[str, Any]:
        if isinstance(value, bool):
            return {"kind": "str", "value": str(value)}
        if isinstance(value, int):
            return {"kind": "int", "value": value}
        if isinstance(value, float):
            return {"kind": "float", "value": value}
        return {"kind": "str", "value": str(value)}

    @staticmethod
    def _decode_key(value: dict[str, Any]) -> str | int | float:
        kind = value.get("kind")
        raw = value.get("value")
        if kind == "int":
            return int(raw)
        if kind == "float":
            return float(raw)
        return str(raw)


def create_graph_checkpointer(
    *,
    backend: str,
    path: str | Path | None,
    max_checkpoints_per_thread: int | None = None,
) -> BaseCheckpointSaver[str]:
    if backend == "file":
        checkpoint_path = Path(path) if path else Path.cwd() / ".smallctl-langgraph-checkpoints.json"
        if max_checkpoints_per_thread is None:
            return FileCheckpointSaver(checkpoint_path)
        return FileCheckpointSaver(
            checkpoint_path,
            max_checkpoints_per_thread=max_checkpoints_per_thread,
        )
    return InMemorySaver()
