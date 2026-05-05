from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
from typing import Any


class Colors:
    """ANSI color codes for colored terminal output."""

    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"


def colorize(text: str, color: str) -> str:
    """Return text with ANSI color code applied."""

    return f"{color}{text}{Colors.RESET}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize harness log records.")
    parser.add_argument("run_dir", nargs="?", help="Run directory containing harness.log")
    parser.add_argument("--log-file", dest="log_file", help="Explicit harness.log file path")
    parser.add_argument("--log-dir", dest="log_dir", help="Custom log directory (default: logs)")
    return parser.parse_args()


def _resolve_log_path(args: argparse.Namespace) -> Path:
    if args.log_file:
        log_path = Path(str(args.log_file)).expanduser().resolve()
    elif args.run_dir:
        log_path = Path(str(args.run_dir)).expanduser().resolve() / "harness.log"
    else:
        log_dir = Path(str(args.log_dir)).expanduser().resolve() if args.log_dir else Path("logs").resolve()
        log_path = log_dir / "harness.log"

    if not log_path.exists():
        logging.error("Log file does not exist: %s", log_path)
        raise FileNotFoundError(f"Log file not found: {log_path}")

    return log_path


def _extract_json_payload(line: str) -> Any:
    start = line.find("{")
    if start < 0:
        return None
    payload = line[start:].strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        logging.error("JSON decode error in log line: %s", exc)
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


def _event_name(line: str) -> str:
    parts = line.split(maxsplit=2)
    return parts[1] if len(parts) >= 2 else ""


def _load_task_summary(log_path: Path) -> dict[str, Any]:
    summary_path = log_path.with_name("task_summary.json")
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("logwatch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path.with_suffix(".logwatch.log"), mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def main() -> int:
    args = _parse_args()
    log_path = _resolve_log_path(args)
    logger = setup_logging(log_path)
    logger.info("Starting log analysis for %s", log_path)

    if not log_path.exists() or not log_path.is_file():
        logger.error("Log file not found: %s", log_path)
        return 1

    parsed_records = 0
    errors = 0
    events: Counter[str] = Counter()
    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parsed_records += 1
        event = _event_name(line)
        if event:
            events[event] += 1
        if _line_has_error(line):
            logger.warning("Error detected in line %s: %s", parsed_records, line[:100])
            errors += 1

    logger.info("Parsed records: %s", parsed_records)
    logger.info("Errors: %s", errors)
    summary = _load_task_summary(log_path)
    if summary:
        logger.info("Canonical task_summary total_tool_calls: %s", summary.get("total_tool_calls", "unknown"))
        logger.info("Canonical task_summary final_status: %s", summary.get("final_status", summary.get("status", "unknown")))
    logger.info("RCA checklist: cross-reference task_summary.json for canonical metrics before citing tool-call counts.")
    logger.info("RCA checklist: verify diagnosis claims against raw tool stdout, not model paraphrases.")
    logger.info("RCA checklist: action_stall events: %s", events.get("action_stall", 0))
    logger.info("RCA checklist: no_tool_recovery events: %s", events.get("no_tool_recovery", 0))
    logger.info(
        "RCA checklist: inline thinking tool recoveries: %s sanitized: %s",
        events.get("inline_tool_call_recovered_from_thinking", 0),
        events.get("thinking_tool_protocol_sanitized", 0),
    )
    logger.info(
        "RCA checklist: harness auto-scheduled remote verifiers: %s",
        events.get("task_complete_remote_mutation_verifier_autocontinue", 0),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
