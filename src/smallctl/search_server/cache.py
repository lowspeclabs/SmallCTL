from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CacheEntry:
    key: str
    kind: str
    payload: dict[str, Any]
    created_at: float
    expires_at: float
    negative: bool = False


class SearchCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    kind TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    negative INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(kind, cache_key)
                )
                """
            )
            conn.commit()

    def get(self, *, kind: str, key: str) -> CacheEntry | None:
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM cache_entries WHERE kind = ? AND cache_key = ?",
                (kind, key),
            ).fetchone()
            if row is None:
                return None
            if float(row["expires_at"]) <= now:
                conn.execute(
                    "DELETE FROM cache_entries WHERE kind = ? AND cache_key = ?",
                    (kind, key),
                )
                conn.commit()
                return None
            return CacheEntry(
                key=key,
                kind=kind,
                payload=json.loads(str(row["payload_json"])),
                created_at=float(row["created_at"]),
                expires_at=float(row["expires_at"]),
                negative=bool(row["negative"]),
            )

    def set(
        self,
        *,
        kind: str,
        key: str,
        payload: dict[str, Any],
        ttl_seconds: int,
        negative: bool = False,
    ) -> None:
        created_at = time.time()
        expires_at = created_at + max(1, int(ttl_seconds))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache_entries (kind, cache_key, created_at, expires_at, negative, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (kind, key, created_at, expires_at, int(negative), json.dumps(payload, ensure_ascii=True, default=str)),
            )
            conn.commit()

    def get_payload(self, *, kind: str, key: str) -> dict[str, Any] | None:
        entry = self.get(kind=kind, key=key)
        return None if entry is None else dict(entry.payload)
