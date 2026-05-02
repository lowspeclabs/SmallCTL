from __future__ import annotations

import argparse
import json
import uuid
import datetime
from typing import Any
from pathlib import Path

from .memory_namespace import infer_memory_namespace
from .state import ExperienceMemory, _coerce_experience_memory, json_safe_value
from .memory_store import ExperienceStore, search_memories


def _memory_store_paths() -> tuple[Path, Path]:
    base = Path(".smallctl") / "memory"
    return base / "warm-experiences.jsonl", base / "cold-experiences.jsonl"


def build_memory_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("memory", help="Manage experience memory tiers")
    memory_sub = parser.add_subparsers(dest="memory_command", required=True)

    # ADD
    add_p = memory_sub.add_parser("add", help="Add a manual experience record")
    add_p.add_argument("--tier", choices=["warm", "cold"], default="cold")
    add_p.add_argument("--intent", required=True)
    add_p.add_argument("--tool", help="Tool name")
    add_p.add_argument("--outcome", choices=["success", "failure", "partial"], default="success")
    add_p.add_argument("--failure-mode", help="Specific failure taxonomy mode")
    add_p.add_argument("--note", help="The operational guidance note", required=False, default="")
    add_p.add_argument("--tag", action="append", help="Intent tags")
    add_p.add_argument("--env-tag", action="append", help="Environment tags")
    add_p.add_argument("--entity-tag", action="append", help="Entity tags")
    add_p.add_argument("--namespace", help="Primary memory namespace")
    add_p.add_argument("--pinned", action="store_true", help="Pin the memory")
    add_p.add_argument("--confidence", type=float, default=1.0)
    add_p.add_argument("--from-json", help="Import memories from JSON file")

    # LIST
    list_p = memory_sub.add_parser("list", help="List stored memories")
    list_p.add_argument("--tier", choices=["warm", "cold", "all"], default="all")
    list_p.add_argument("--intent", help="Filter by intent")
    list_p.add_argument("--tool", help="Filter by tool")
    list_p.add_argument("--namespace", help="Filter by namespace")

    # SEARCH
    search_p = memory_sub.add_parser("search", help="Search memories by query")
    search_p.add_argument("query", nargs="?", default="")
    search_p.add_argument("--query", dest="query_text", help="Query string")
    search_p.add_argument("--query-text", help="Query string")
    search_p.add_argument("--tier", choices=["warm", "cold", "all"], default="all")
    search_p.add_argument("--namespace", help="Filter by namespace")

    # FORGET
    forget_p = memory_sub.add_parser("forget", help="Delete a memory by ID")
    forget_p.add_argument("id", help="Memory ID")
    forget_p.add_argument("--tier", choices=["warm", "cold"], required=True)

    # PROMOTE
    promote_p = memory_sub.add_parser("promote", help="Promote warm memory to cold")
    promote_p.add_argument("id", help="Memory ID")

    # SCRUB
    scrub_p = memory_sub.add_parser("scrub", help="Redact sensitive text from stored memory notes")
    scrub_p.add_argument("--tier", choices=["warm", "cold", "all"], default="all")
    scrub_p.add_argument(
        "--write",
        action="store_true",
        help="Persist the redacted notes back to disk. Default is dry-run.",
    )

def handle_memory_command(args: argparse.Namespace) -> int:
    cmd = args.memory_command
    warm_path, cold_path = _memory_store_paths()
    warm_store = ExperienceStore(warm_path)
    cold_store = ExperienceStore(cold_path)

    if cmd == "add":
        return _handle_add(args, warm_store, cold_store)
    elif cmd == "list":
        return _handle_list(args, warm_store, cold_store)
    elif cmd == "search":
        return _handle_search(args, warm_store, cold_store)
    elif cmd == "forget":
        return _handle_forget(args, warm_store, cold_store)
    elif cmd == "promote":
        return _handle_promote(args, warm_store, cold_store)
    elif cmd == "scrub":
        return _handle_scrub(args, warm_store, cold_store)
    
    return 0

def _handle_add(args: argparse.Namespace, warm_store: ExperienceStore, cold_store: ExperienceStore) -> int:
    all_memories = []
    if args.from_json:
        path = Path(args.from_json)
        if not path.exists():
            print(json.dumps({"status": "failed", "reason": f"File not found: {path}"}))
            return 1
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                all_memories.extend([_coerce_experience_memory(m) for m in data])
            else:
                all_memories.append(_coerce_experience_memory(data))
    else:
        mem = ExperienceMemory(
            memory_id=f"mem-{uuid.uuid4().hex[:10]}",
            tier=args.tier,
            source="manual",
            created_at=datetime.datetime.now().isoformat(),
            intent=args.intent,
            namespace=args.namespace
            or infer_memory_namespace(
                tool_name=args.tool or "",
                intent=args.intent,
                intent_tags=args.tag or [],
                environment_tags=args.env_tag or [],
                entity_tags=args.entity_tag or [],
                notes=args.note or "",
            ),
            intent_tags=args.tag or [],
            environment_tags=args.env_tag or [],
            entity_tags=args.entity_tag or [],
            tool_name=args.tool or "",
            outcome=args.outcome,
            failure_mode=args.failure_mode or "",
            notes=args.note or "",
            confidence=args.confidence,
            pinned=args.pinned,
        )
        all_memories.append(mem)

    ids = [m.memory_id for m in all_memories]
    for m in all_memories:
        target = cold_store if m.tier == "cold" else warm_store
        target.upsert(m)
    
    print(json.dumps({"status": "added", "memory_ids": ids}))
    return 0

def _handle_list(args: argparse.Namespace, warm_store: ExperienceStore, cold_store: ExperienceStore) -> int:
    combined = []
    if args.tier in ["warm", "all"]:
        combined.extend(warm_store.list())
    if args.tier in ["cold", "all"]:
        combined.extend(cold_store.list())
    
    if args.intent:
        combined = [m for m in combined if m.intent == args.intent]
    if args.tool:
        combined = [m for m in combined if m.tool_name == args.tool]
    if args.namespace:
        combined = search_memories(combined, namespace=args.namespace)
    
    print(json.dumps({
        "count": len(combined),
        "records": [json_safe_value(m) for m in combined]
    }, indent=2))
    return 0

def _handle_search(args: argparse.Namespace, warm_store: ExperienceStore, cold_store: ExperienceStore) -> int:
    combined = []
    if args.tier in ["warm", "all"]:
        combined.extend(warm_store.list())
    if args.tier in ["cold", "all"]:
        combined.extend(cold_store.list())
    
    query = getattr(args, "query", "") or getattr(args, "query_text", "") or ""
    matches = search_memories(combined, query=query, namespace=getattr(args, "namespace", "") or "")
    print(json.dumps({
        "count": len(matches),
        "records": [json_safe_value(m) for m in matches]
    }, indent=2))
    return 0

def _handle_forget(args: argparse.Namespace, warm_store: ExperienceStore, cold_store: ExperienceStore) -> int:
    target = cold_store if args.tier == "cold" else warm_store
    memory_id = getattr(args, "memory_id", None) or getattr(args, "id", None)
    target.delete(memory_id)
    print(json.dumps({"status": "deleted", "memory_id": memory_id, "tier": args.tier}))
    return 0

def _handle_promote(args: argparse.Namespace, warm_store: ExperienceStore, cold_store: ExperienceStore) -> int:
    memory_id = getattr(args, "memory_id", None) or getattr(args, "id", None)
    m = warm_store.get(memory_id)
    if not m:
        print(json.dumps({"status": "failed", "reason": f"Memory not found in warm tier: {memory_id}"}))
        return 1
    m.tier = "cold"
    cold_store.upsert(m)
    warm_store.delete(memory_id)
    print(json.dumps({"status": "promoted", "memory_id": memory_id, "tier": "cold"}))
    return 0


def _handle_scrub(args: argparse.Namespace, warm_store: ExperienceStore, cold_store: ExperienceStore) -> int:
    stores: list[tuple[str, ExperienceStore]] = []
    if args.tier in ["warm", "all"]:
        stores.append(("warm", warm_store))
    if args.tier in ["cold", "all"]:
        stores.append(("cold", cold_store))

    tiers: dict[str, dict[str, int]] = {}
    total_records = 0
    total_changed = 0
    total_written = 0
    for tier_name, store in stores:
        summary = store.scrub_sensitive_notes(write=bool(args.write))
        tiers[tier_name] = summary
        total_records += int(summary.get("records", 0))
        total_changed += int(summary.get("changed", 0))
        total_written += int(summary.get("written", 0))

    print(json.dumps({
        "status": "scrubbed" if args.write else "dry_run",
        "tier": args.tier,
        "tiers": tiers,
        "records": total_records,
        "changed": total_changed,
        "written": total_written,
    }))
    return 0


def memory_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    # We don't want the 'memory' prefix in the arguments passed to memory_cli in tests
    memory_sub = parser.add_subparsers(dest="memory_command", required=True)
    
    # We can't reuse build_memory_parser directly because it adds 'memory' subparsers
    # So we inline the subcommand definitions or refactor
    _build_subcommands(memory_sub)
    
    args = parser.parse_args(argv)
    return handle_memory_command(args)

def _build_subcommands(memory_sub: Any) -> None:
    # ADD
    add_p = memory_sub.add_parser("add", help="Add a manual experience record")
    add_p.add_argument("--tier", choices=["warm", "cold"], default="cold")
    add_p.add_argument("--intent", required=True)
    add_p.add_argument("--tool", help="Tool name")
    add_p.add_argument("--outcome", choices=["success", "failure", "partial"], default="success")
    add_p.add_argument("--failure-mode", help="Specific failure taxonomy mode")
    add_p.add_argument("--note", help="The operational guidance note", required=False, default="")
    add_p.add_argument("--tag", action="append", help="Intent tags")
    add_p.add_argument("--env-tag", action="append", help="Environment tags")
    add_p.add_argument("--entity-tag", action="append", help="Entity tags")
    add_p.add_argument("--namespace", help="Primary memory namespace")
    add_p.add_argument("--pinned", action="store_true", help="Pin the memory")
    add_p.add_argument("--confidence", type=float, default=1.0)
    add_p.add_argument("--from-json", help="Import memories from JSON file")

    # LIST
    list_p = memory_sub.add_parser("list", help="List stored memories")
    list_p.add_argument("--tier", choices=["warm", "cold", "all"], default="all")
    list_p.add_argument("--intent", help="Filter by intent")
    list_p.add_argument("--tool", help="Filter by tool")
    list_p.add_argument("--namespace", help="Filter by namespace")

    # SEARCH
    search_p = memory_sub.add_parser("search", help="Search memories by query")
    search_p.add_argument("query", nargs="?", default="")
    search_p.add_argument("--query", dest="query_text", help="Query string")
    search_p.add_argument("--query-text", help="Query string")
    search_p.add_argument("--tier", choices=["warm", "cold", "all"], default="all")
    search_p.add_argument("--namespace", help="Filter by namespace")

    # FORGET
    forget_p = memory_sub.add_parser("forget", help="Delete a memory by ID")
    forget_p.add_argument("--memory-id", help="Memory ID")
    forget_p.add_argument("--tier", choices=["warm", "cold"], required=True)

    # PROMOTE
    promote_p = memory_sub.add_parser("promote", help="Promote warm memory to cold")
    promote_p.add_argument("--memory-id", help="Memory ID")

    # SCRUB
    scrub_p = memory_sub.add_parser("scrub", help="Redact sensitive text from stored memory notes")
    scrub_p.add_argument("--tier", choices=["warm", "cold", "all"], default="all")
    scrub_p.add_argument("--write", action="store_true", help="Persist the redacted notes back to disk")
