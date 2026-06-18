from __future__ import annotations

import base64
import json
import random
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
    PendingWrite,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.memory import InMemorySaver


class FileCheckpointSaver(BaseCheckpointSaver[str]):
    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._mtime_ns: int | None = None
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
            self._load_if_changed()
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
            self._load_if_changed()
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

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            self._load_if_changed()
            self.storage.pop(thread_id, None)
            for key in list(self.writes.keys()):
                if key[0] == thread_id:
                    del self.writes[key]
            for key in list(self.blobs.keys()):
                if key[0] == thread_id:
                    del self.blobs[key]
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

    def _load_from_disk(self) -> None:
        self.storage = {}
        self.writes = {}
        self.blobs = {}
        if not self.path.exists():
            self._mtime_ns = None
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self._mtime_ns = self.path.stat().st_mtime_ns
            return
        if not isinstance(payload, dict):
            self._mtime_ns = self.path.stat().st_mtime_ns
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
        self._mtime_ns = self.path.stat().st_mtime_ns

    def _flush_to_disk(self) -> None:
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
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self._mtime_ns = self.path.stat().st_mtime_ns

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
) -> BaseCheckpointSaver[str]:
    if backend == "file":
        checkpoint_path = Path(path) if path else Path.cwd() / ".smallctl-langgraph-checkpoints.json"
        return FileCheckpointSaver(checkpoint_path)
    return InMemorySaver()
