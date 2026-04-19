from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize harness log records.")
    parser.add_argument("run_dir", nargs="?", help="Run directory containing harness.log")
    parser.add_argument("--log-file", dest="log_file", help="Explicit harness.log file path")
    return parser.parse_args()


def _resolve_log_path(args: argparse.Namespace) -> Path:
    if args.log_file:
        return Path(str(args.log_file)).expanduser().resolve()
    if args.run_dir:
        return (Path(str(args.run_dir)).expanduser().resolve() / "harness.log")
    return Path("logs") / "harness.log"


def _extract_json_payload(line: str) -> Any:
    start = line.find("{")
    if start < 0:
        return None
    payload = line[start:].strip()
    try:
        return json.loads(payload)
    except Exception:
        return None


def _payload_has_error(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).strip().lower()
            if key_text == "status" and str(item).strip().lower() in {"failed", "error"}:
                return True
            if key_text == "error":
                return True
            if _payload_has_error(item):
                return True
        return False
    if isinstance(value, list):
        return any(_payload_has_error(item) for item in value)
    return False


def _line_has_error(line: str) -> bool:
    parts = line.split(maxsplit=3)
    event = parts[1] if len(parts) >= 2 else ""
    if "error" in event.lower():
        return True
    payload = _extract_json_payload(line)
    if payload is None:
        return False
    return _payload_has_error(payload)


def main() -> int:
    args = _parse_args()
    log_path = _resolve_log_path(args)
    if not log_path.exists() or not log_path.is_file():
        print(f"Log file not found: {log_path}")
        return 1

    parsed_records = 0
    errors = 0
    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parsed_records += 1
        if _line_has_error(line):
            errors += 1

    print(f"Parsed records: {parsed_records}")
    print(f"Errors: {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
