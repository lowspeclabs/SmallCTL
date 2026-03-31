from __future__ import annotations

import sqlite3
import json
from pathlib import Path
from typing import Any
from .common import ok, fail

def _get_db_path(cwd: str | None = None) -> Path:
    base = Path(cwd) if cwd else Path.cwd()
    db_dir = base / ".smallctl"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "index.db"

def _init_db(db_path: Path):
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                file TEXT NOT NULL,
                start_line INTEGER,
                end_line INTEGER,
                signature TEXT,
                docstring_short TEXT,
                parent TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS symbol_references (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_symbol TEXT,
                to_symbol TEXT NOT NULL,
                call_site_file TEXT NOT NULL,
                call_site_line INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS imports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file TEXT NOT NULL,
                imported_from TEXT NOT NULL,
                symbols TEXT, -- JSON list of symbols
                is_external BOOLEAN
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_refs_to ON symbol_references(to_symbol)")

async def index_write_symbol(
    name: str,
    kind: str,
    file: str,
    start_line: int | None = None,
    end_line: int | None = None,
    signature: str | None = None,
    docstring_short: str | None = None,
    parent: str | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    db_path = _get_db_path(cwd)
    _init_db(db_path)
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO symbols (name, kind, file, start_line, end_line, signature, docstring_short, parent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (name, kind, file, start_line, end_line, signature, docstring_short, parent)
            )
        return ok({"status": "indexed", "symbol": name, "file": file})
    except Exception as exc:
        return fail(f"Failed to write symbol: {exc}")

async def index_write_reference(
    to_symbol: str,
    call_site_file: str,
    from_symbol: str | None = None,
    call_site_line: int | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    db_path = _get_db_path(cwd)
    _init_db(db_path)
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO symbol_references (from_symbol, to_symbol, call_site_file, call_site_line)
                VALUES (?, ?, ?, ?)
                """,
                (from_symbol, to_symbol, call_site_file, call_site_line)
            )
        return ok({"status": "indexed", "to_symbol": to_symbol, "at": f"{call_site_file}:{call_site_line}"})
    except Exception as exc:
        return fail(f"Failed to write reference: {exc}")

async def index_write_import(
    file: str,
    imported_from: str,
    symbols: list[str] | None = None,
    is_external: bool = False,
    cwd: str | None = None,
) -> dict[str, Any]:
    db_path = _get_db_path(cwd)
    _init_db(db_path)
    try:
        symbols_json = json.dumps(symbols) if symbols else None
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO imports (file, imported_from, symbols, is_external)
                VALUES (?, ?, ?, ?)
                """,
                (file, imported_from, symbols_json, is_external)
            )
        return ok({"status": "indexed", "import": imported_from, "file": file})
    except Exception as exc:
        return fail(f"Failed to write import: {exc}")

async def index_batch_write(
    symbols: list[dict[str, Any]] | None = None,
    references: list[dict[str, Any]] | None = None,
    imports: list[dict[str, Any]] | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Write multiple indexing records in a single transaction."""
    db_path = _get_db_path(cwd)
    _init_db(db_path)
    
    counts = {"symbols": 0, "references": 0, "imports": 0}
    try:
        with sqlite3.connect(db_path) as conn:
            if symbols:
                for s in symbols:
                    conn.execute(
                        """
                        INSERT INTO symbols (name, kind, file, start_line, end_line, signature, docstring_short, parent)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            s.get("name"), s.get("kind"), s.get("file"),
                            s.get("start_line"), s.get("end_line"),
                            s.get("signature"), s.get("docstring_short"), s.get("parent")
                        )
                    )
                    counts["symbols"] += 1
            
            if references:
                for r in references:
                    conn.execute(
                        """
                        INSERT INTO symbol_references (from_symbol, to_symbol, call_site_file, call_site_line)
                        VALUES (?, ?, ?, ?)
                        """,
                        (r.get("from_symbol"), r.get("to_symbol"), r.get("call_site_file"), r.get("call_site_line"))
                    )
                    counts["references"] += 1
            
            if imports:
                for i in imports:
                    symbols_json = json.dumps(i.get("symbols")) if i.get("symbols") else None
                    conn.execute(
                        """
                        INSERT INTO imports (file, imported_from, symbols, is_external)
                        VALUES (?, ?, ?, ?)
                        """,
                        (i.get("file"), i.get("imported_from"), symbols_json, i.get("is_external", False))
                    )
                    counts["imports"] += 1
                    
        return ok({"status": "batch_indexed", "counts": counts})
    except Exception as exc:
        return fail(f"Batch indexing failed: {exc}")

async def index_finalize(cwd: str | None = None) -> dict[str, Any]:
    db_path = _get_db_path(cwd)
    if not db_path.exists():
        return fail("No index found to finalize.")
    
    try:
        with sqlite3.connect(db_path) as conn:
            symbols_count = conn.execute("SELECT count(*) FROM symbols").fetchone()[0]
            refs_count = conn.execute("SELECT count(*) FROM symbol_references").fetchone()[0]
            files_count = conn.execute("SELECT count(DISTINCT file) FROM symbols").fetchone()[0]
            
            # Generate manifest
            top_files = conn.execute(
                "SELECT file, count(*) as c FROM symbols GROUP BY file ORDER BY c DESC LIMIT 5"
            ).fetchall()
            
            # Simple heuristic for entry points
            entry_points = conn.execute(
                "SELECT file || ':' || name FROM symbols WHERE name IN ('main', 'app', 'run', 'start') LIMIT 5"
            ).fetchall()
            
            manifest = {
                "stats": {
                    "files": files_count,
                    "symbols": symbols_count,
                    "refs": refs_count
                },
                "entry_points": [e[0] for e in entry_points],
                "top_files_by_symbol_count": [f[0] for f in top_files],
            }
            
            manifest_path = db_path.parent / "index_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2))
            
        return ok(manifest)
    except Exception as exc:
        return fail(f"Failed to finalize index: {exc}")
