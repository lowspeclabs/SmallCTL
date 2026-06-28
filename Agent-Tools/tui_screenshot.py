#!/usr/bin/env python3
"""tui_screenshot — run the SmallCTL TUI headlessly and capture a PNG screenshot.

This tool launches `SmallctlApp` via Textual's test/pilot harness, drives a task
to completion (or a fixed wait), saves the terminal screenshot as an SVG, and
converts it to PNG for review.

The intended workflow is:
  1. Start the TUI with a prompt/task.
  2. Wait for harness output to render.
  3. Force a Textual screenshot (SVG).
  4. Convert the SVG to PNG.
  5. Return the PNG (and SVG) paths so a model/agent can view them.

Examples:
  python Agent-Tools/tui_screenshot.py --task "list files in src/smallctl"
  python Agent-Tools/tui_screenshot.py --task "what is 2+2" --timeout 60 --name math_test
  python Agent-Tools/tui_screenshot.py --task "read temp/vikunja-9b.py" --width 120 --height 40
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a SmallCTL TUI screenshot as PNG.")
    parser.add_argument("--task", "-t", required=True, help="Task/prompt to send to the TUI")
    parser.add_argument("--output", "-o", help="Output directory (default: temp/tui_screenshots)")
    parser.add_argument("--name", "-n", help="Base filename (default: tui_<timestamp>)")
    parser.add_argument("--timeout", type=float, default=300.0, help="Max seconds to wait for task completion")
    parser.add_argument("--wait-seconds", type=float, default=0.0, help="Extra seconds to wait after task completes")
    parser.add_argument("--width", type=int, default=120, help="Terminal width")
    parser.add_argument("--height", type=int, default=40, help="Terminal height")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--endpoint", help="OpenAI-compatible API base URL")
    parser.add_argument("--provider-profile", help="Provider profile (auto, generic, openai, ollama, vllm, lmstudio, openrouter, llamacpp)")
    parser.add_argument("--run-mode", help="Runtime mode (auto, chat, loop, planning, indexer, tool_plan)")
    parser.add_argument("--phase", help="Initial phase")
    parser.add_argument("--tool-profiles", help="Comma-separated tool profiles")
    parser.add_argument("--fresh-run", action="store_true", help="Start without prior memory/state")
    parser.add_argument("--config-path", help="Path to custom .smallctl.yaml config")
    parser.add_argument("--keep-svg", action="store_true", help="Keep the intermediate SVG file")
    parser.add_argument("--json", action="store_true", help="Output JSON with paths")
    parser.add_argument("--verbose", action="store_true", help="Print progress to stderr")
    return parser.parse_args()


def _log(args: argparse.Namespace, message: str) -> None:
    if args.verbose:
        print(message, file=sys.stderr)


def _convert_svg_to_png(svg_path: Path, png_path: Path) -> None:
    """Convert an SVG file to PNG, preferring cairosvg, then CLI fallbacks."""
    errors: list[str] = []

    # Try cairosvg first (best quality for Textual SVGs)
    try:
        import cairosvg

        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), scale=2.0)
        return
    except Exception as exc:
        errors.append(f"cairosvg: {exc}")

    # Try rsvg-convert
    rsvg = shutil.which("rsvg-convert")
    if rsvg:
        try:
            subprocess.run(
                [rsvg, "-o", str(png_path), str(svg_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            return
        except Exception as exc:
            errors.append(f"rsvg-convert: {exc}")

    # Try inkscape
    inkscape = shutil.which("inkscape")
    if inkscape:
        try:
            subprocess.run(
                [inkscape, "--export-filename", str(png_path), str(svg_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            return
        except Exception as exc:
            errors.append(f"inkscape: {exc}")

    # Try ImageMagick convert (often handles SVG poorly, so last)
    convert = shutil.which("convert")
    if convert:
        try:
            subprocess.run(
                [convert, str(svg_path), str(png_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            return
        except Exception as exc:
            errors.append(f"convert: {exc}")

    raise RuntimeError(
        f"Could not convert {svg_path} to PNG. Tried:\n" + "\n".join(errors)
    )


def _build_cli_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build the CLI config dict that resolve_config expects."""
    cfg: dict[str, Any] = {"task": args.task}
    if args.model is not None:
        cfg["model"] = args.model
    if args.endpoint is not None:
        cfg["endpoint"] = args.endpoint
    if args.provider_profile is not None:
        cfg["provider_profile"] = args.provider_profile
    if args.run_mode is not None:
        cfg["run_mode"] = args.run_mode
    if args.phase is not None:
        cfg["phase"] = args.phase
    if args.tool_profiles is not None:
        cfg["tool_profiles"] = args.tool_profiles
    if args.fresh_run:
        cfg["fresh_run"] = True
    if args.config_path is not None:
        cfg["config_path"] = args.config_path
    return cfg


class _TaskWakeupRecursionFilter(logging.Filter):
    """Drop the asyncio RecursionError logged during Textual shutdown.

    Textual's test runner cancels a deep task tree on exit; on some runs this
    walks enough nested tasks to hit Python's recursion limit. The resulting
    ``Task.task_wakeup`` RecursionError is logged by asyncio but does not
    affect the screenshot or run log, so we suppress it to keep the tool's
    output usable.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if "Task.task_wakeup" in record.getMessage():
            exc_info = record.exc_info
            if exc_info and exc_info[0] is RecursionError:
                return False
        return True


def _setup_logging() -> None:
    # Suppress noisy smallctl logging during screenshot capture
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("asyncio").addFilter(_TaskWakeupRecursionFilter())


async def _shutdown_app(app: Any) -> None:
    """Cancel the active harness task and tear the harness down before the
    Textual test runner tries to cancel everything at once.

    Textual's run_test context manager cancels all pending tasks on exit.
    If the harness is still retrying model calls in the background, that
    cancellation can walk a deep task tree and hit asyncio's recursion limit.
    Stopping the harness explicitly here avoids the cascade.
    """
    active_task = getattr(app, "active_task", None)
    if active_task is not None and not active_task.done():
        active_task.cancel()
        try:
            await active_task
        except asyncio.CancelledError:
            pass

    harness = getattr(app, "harness", None)
    if harness is not None:
        try:
            await harness.teardown()
        except Exception:
            pass

    # Let any stray callbacks drain before the context manager cleans up.
    await asyncio.sleep(0)


async def _run_tui_and_capture(args: argparse.Namespace, output_dir: Path, base_name: str) -> dict[str, str]:
    sys.path.insert(0, str(SRC_DIR))

    from smallctl.config import resolve_config
    from smallctl.logging_utils import create_run_logger, setup_logging as smallctl_setup_logging
    from smallctl.main import build_harness_config_kwargs
    from smallctl.ui import SmallctlApp

    _setup_logging()

    cli_cfg = _build_cli_config(args)
    config = resolve_config(cli_cfg)

    # Create a run logger so the harness behaves like a normal CLI run
    smallctl_setup_logging(config.debug, stream_to_terminal=False)
    run_logger = create_run_logger(base_dir=str(REPO_ROOT / "logs"))

    harness_kwargs = build_harness_config_kwargs(config, run_logger=run_logger, task=args.task)
    harness_kwargs["show_system_messages"] = True

    app = SmallctlApp(harness_kwargs=harness_kwargs)

    svg_path = output_dir / f"{base_name}.svg"
    png_path = output_dir / f"{base_name}.png"

    _log(args, f"Launching TUI (size {args.width}x{args.height}) with task: {args.task!r}")

    async with app.run_test(headless=True, size=(args.width, args.height)) as pilot:
        # Let the app mount and start the initial task
        await pilot.pause()
        _log(args, "TUI mounted; waiting for harness output...")

        start = time.monotonic()
        # Wait until the initial task finishes or timeout elapses
        while True:
            elapsed = time.monotonic() - start
            if elapsed >= args.timeout:
                _log(args, f"Timeout reached after {elapsed:.1f}s; capturing screenshot now")
                break
            active_task = getattr(app, "active_task", None)
            if active_task is None or active_task.done():
                _log(args, f"Harness task finished after {elapsed:.1f}s")
                break
            await pilot.pause(0.5)

        if args.wait_seconds > 0:
            _log(args, f"Waiting additional {args.wait_seconds}s as requested")
            await asyncio.sleep(args.wait_seconds)
            await pilot.pause()

        # Force the screenshot. Textual returns the filename it actually wrote.
        saved_filename = app.save_screenshot(filename=svg_path.name, path=str(svg_path.parent))
        actual_svg = svg_path.parent / saved_filename
        _log(args, f"Saved SVG screenshot: {actual_svg}")

        # Convert SVG to PNG
        await asyncio.to_thread(_convert_svg_to_png, actual_svg, png_path)
        _log(args, f"Converted to PNG: {png_path}")

        # Clean up SVG unless requested
        if not args.keep_svg and actual_svg.exists():
            actual_svg.unlink()
            _log(args, "Removed intermediate SVG")

        # Stop the harness cleanly before run_test tears tasks down.
        await _shutdown_app(app)

    return {
        "png": str(png_path),
        "svg": str(actual_svg) if args.keep_svg else None,
        "run_log_dir": str(run_logger.run_dir) if run_logger else None,
    }


def main() -> int:
    args = _parse_args()

    output_dir = Path(args.output) if args.output else REPO_ROOT / "temp" / "tui_screenshots"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = args.name or f"tui_{time.strftime('%Y%m%d_%H%M%S')}"

    # Textual's run_test shutdown can walk a deep task tree when cancelling
    # background model-call retries. Raise the recursion limit so the
    # cancellation cascade does not crash the screenshot tool.
    old_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_recursion_limit, 5000))
    try:
        result = asyncio.run(_run_tui_and_capture(args, output_dir, base_name))
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 1
    finally:
        sys.setrecursionlimit(old_recursion_limit)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"PNG screenshot: {result['png']}")
        if result.get("run_log_dir"):
            print(f"Run log dir:   {result['run_log_dir']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
