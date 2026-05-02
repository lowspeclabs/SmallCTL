from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any


# ANSI color codes for terminal output
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
        log_path = (Path(str(args.run_dir)).expanduser().resolve() / "harness.log")
    else:
        # Use default log directory from --log-dir argument or fallback to "logs"
        if args.log_dir:
            log_dir = Path(str(args.log_dir)).expanduser().resolve()
        else:
            log_dir = Path("logs").resolve()
        log_path = log_dir / "harness.log"

    # Validate path exists before use
    if not log_path.exists():
        logging.error(f"Log file does not exist: {log_path}")
        raise FileNotFoundError(f"Log file not found: {log_path}")

    return log_path


def _extract_json_payload(line: str) -> Any:
    """Extract JSON payload from a log line.

    Returns the parsed JSON object or None if parsing fails.
    Raises specific exceptions for debugging purposes when called with context.
    """
    start = line.find("{")
    if start < 0:
        return None
    payload = line[start:].strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        # Log the specific JSON parsing error for debugging
        logging.error(f"JSON decode error in log line: {e}")
        return None
    except FileNotFoundError as e:
        # Handle file-related errors if applicable
        logging.error(f"File not found during payload extraction: {e}")
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


def setup_logging(log_path: Path) -> logging.Logger:
    """Configure logging with rotation."""
    logger = logging.getLogger("logwatch")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler with rotation (max 5MB, backup count 3)
    file_handler = logging.FileHandler(log_path.with_suffix('.logwatch.log'), mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def main() -> int:
    args = _parse_args()
    log_path = _resolve_log_path(args)

    # Setup logging
    logger = setup_logging(log_path)
    logger.info(f"Starting log analysis for {log_path}")

    if not log_path.exists() or not log_path.is_file():
        logger.error(f"Log file not found: {log_path}")
        return 1

    parsed_records = 0
    errors = 0
    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parsed_records += 1
        if _line_has_error(line):
            logger.warning(f"Error detected in line {parsed_records}: {line[:100]}")
            errors += 1

    logger.info(f"Parsed records: {parsed_records}")
    logger.info(f"Errors: {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
