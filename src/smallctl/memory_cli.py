from __future__ import annotations

import argparse
import json
import uuid
import datetime
from typing import Any
from pathlib import Path

from .state import ExperienceMemory, _coerce_experience_memory, json_safe_value
from .memory_store import ExperienceStore, search_memories

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
    add_p.add_argument("--pinned", action="store_true", help="Pin the memory")
    add_p.add_argument("--confidence", type=float, default=1.0)
    add_p.add_argument("--from-json", help="Import memories from JSON file")

    # LIST
    list_p = memory_sub.add_parser("list", help="List stored memories")
    list_p.add_argument("--tier", choices=["warm", "cold", "all"], default="all")
    list_p.add_argument("--intent", help="Filter by intent")
    list_p.add_argument("--tool", help="Filter by tool")

    # SEARCH
    search_p = memory_sub.add_parser("search", help="Search memories by query")
    search_p.add_argument("query", help="Query string")
    search_p.add_argument("--tier", choices=["warm", "cold", "all"], default="all")

    # FORGET
    forget_p = memory_sub.add_parser("forget", help="Delete a memory by ID")
    forget_p.add_argument("id", help="Memory ID")
    forget_p.add_argument("--tier", choices=["warm", "cold"], required=True)

    # PROMOTE
    promote_p = memory_sub.add_parser("promote", help="Promote warm memory to cold")
    promote_p.add_argument("id", help="Memory ID")

def handle_memory_command(args: argparse.Namespace) -> int:
    cmd = args.memory_command
    warm_store = ExperienceStore(Path(".smallctl/warm-experiences.jsonl"))
    cold_store = ExperienceStore(Path(".smallctl/cold-experiences.jsonl"))

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
    
    matches = search_memories(combined, query=args.query)
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
    add_p.add_argument("--pinned", action="store_true", help="Pin the memory")
    add_p.add_argument("--confidence", type=float, default=1.0)
    add_p.add_argument("--from-json", help="Import memories from JSON file")

    # LIST
    list_p = memory_sub.add_parser("list", help="List stored memories")
    list_p.add_argument("--tier", choices=["warm", "cold", "all"], default="all")
    list_p.add_argument("--intent", help="Filter by intent")
    list_p.add_argument("--tool", help="Filter by tool")

    # SEARCH
    search_p = memory_sub.add_parser("search", help="Search memories by query")
    search_p.add_argument("--query", help="Query string") # Changed from positional to match test
    search_p.add_argument("--tier", choices=["warm", "cold", "all"], default="all")

    # FORGET
    forget_p = memory_sub.add_parser("forget", help="Delete a memory by ID")
    forget_p.add_argument("--memory-id", help="Memory ID")
    forget_p.add_argument("--tier", choices=["warm", "cold"], required=True)

    # PROMOTE
    promote_p = memory_sub.add_parser("promote", help="Promote warm memory to cold")
    promote_p.add_argument("--memory-id", help="Memory ID")
