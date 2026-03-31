from __future__ import annotations

import sqlite3
import json
from pathlib import Path
from typing import Any
from .common import ok, fail

def _get_db_path(cwd: str | None = None) -> Path:
    base = Path(cwd) if cwd else Path.cwd()
    return base / ".smallctl" / "index.db"

def _format_result(data: Any, hint: str | None = None) -> dict[str, Any]:
    text = json.dumps(data, indent=2)
    # Very rough token estimate (chars / 4)
    token_estimate = len(text) // 4
    return {
        "result": data,
        "token_estimate": token_estimate,
        "hint": hint
    }

async def index_query_symbol(
    query: str,
    cwd: str | None = None,
) -> dict[str, Any]:
    db_path = _get_db_path(cwd)
    if not db_path.exists():
        return fail("Code index not found. Please run indexer first.")
    
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT name, kind, file, start_line FROM symbols WHERE name LIKE ? LIMIT 20",
                (f"%{query}%",)
            ).fetchall()
            
            results = [dict(r) for r in rows]
            hint = f"Use index_get_definition('{results[0]['name']}') to see source." if results else "Try a different search pattern."
            return ok(_format_result(results, hint))
    except Exception as exc:
        return fail(f"Search failed: {exc}")

async def index_get_references(
    symbol_name: str,
    cwd: str | None = None,
) -> dict[str, Any]:
    db_path = _get_db_path(cwd)
    if not db_path.exists():
        return fail("Code index not found.")
    
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT from_symbol, call_site_file, call_site_line FROM symbol_references WHERE to_symbol = ? LIMIT 20",
                (symbol_name,)
            ).fetchall()
            
            results = [dict(r) for r in rows]
            return ok(_format_result(results, "These are known call sites for the symbol."))
    except Exception as exc:
        return fail(f"Query failed: {exc}")

async def index_get_definition(
    symbol_name: str,
    cwd: str | None = None,
) -> dict[str, Any]:
    db_path = _get_db_path(cwd)
    if not db_path.exists():
        return fail("Code index not found.")
    
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM symbols WHERE name = ? LIMIT 1",
                (symbol_name,)
            ).fetchone()
            
            if not row:
                return fail(f"Symbol '{symbol_name}' not found in index.")
            
            result = dict(row)
            hint = f"Use file_read('{result['file']}', start_line={result['start_line']}, end_line={result['end_line']}) to read the code."
            return ok(_format_result(result, hint))
    except Exception as exc:
        return fail(f"Query failed: {exc}")
