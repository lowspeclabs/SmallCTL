from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from typing import Any

import httpx

from . import __version__

DEFAULT_REPO = "lowspeclabs/SmallCTL"
GITHUB_API_BASE = "https://api.github.com/repos"


def _normalize_version(version: str) -> str:
    version = str(version or "").strip().lstrip("v")
    if not version:
        return "0.0.0"
    return version


def _version_tuple(version: str) -> tuple[int, ...]:
    version = _normalize_version(version)
    parts = re.split(r"[.-]", version)
    nums: list[int] = []
    for part in parts:
        match = re.match(r"^(\d+)", part)
        nums.append(int(match.group(1)) if match else 0)
    return tuple(nums)


def _is_newer(latest: str, current: str) -> bool:
    return _version_tuple(latest) > _version_tuple(current)


def _github_release_url(repo: str, tag: str) -> str:
    return f"git+https://github.com/{repo}.git@{tag}"


def _fetch_latest_release(repo: str, *, timeout: float = 20.0) -> dict[str, Any]:
    url = f"{GITHUB_API_BASE}/{repo}/releases/latest"
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Unexpected response from GitHub API")
    return data


def _is_git_install() -> bool:
    spec = sys.modules.get("smallctl") or __import__("smallctl")
    file = getattr(spec, "__file__", None)
    if not file:
        return False
    from pathlib import Path

    path = Path(file).resolve()
    git_dir = path.parent / ".git"
    return git_dir.is_dir()


def _detect_install_context() -> dict[str, Any]:
    import importlib.util
    import site
    from pathlib import Path

    spec = importlib.util.find_spec("smallctl")
    location = spec.origin if spec and spec.origin else None
    editable = False
    if location:
        path = Path(location).resolve()
        editable = (path.parent.parent / "pyproject.toml").exists() or (path.parent / ".git").is_dir()
    return {
        "location": location,
        "in_virtualenv": hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        ),
        "prefix": sys.prefix,
        "site_packages": site.getsitepackages(),
        "editable": editable,
        "git_install": _is_git_install(),
    }


def _pip_command(context: dict[str, Any]) -> list[str]:
    return [sys.executable, "-m", "pip"]


def _run_pip_install(target: str, *, dry_run: bool = False) -> dict[str, Any]:
    context = _detect_install_context()
    cmd = _pip_command(context) + ["install", "--upgrade"]
    if dry_run:
        cmd.append("--dry-run")
    if not context["in_virtualenv"]:
        cmd.append("--break-system-packages")
    cmd.append(target)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def check_for_update(
    repo: str = DEFAULT_REPO,
    *,
    include_prerelease: bool = False,
    timeout: float = 20.0,
) -> dict[str, Any]:
    current = _normalize_version(__version__)
    try:
        release = _fetch_latest_release(repo, timeout=timeout)
    except httpx.HTTPStatusError as exc:
        return {
            "status": "failed",
            "reason": f"GitHub API error: {exc.response.status_code}",
            "current_version": current,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "reason": f"Could not check for updates: {exc}",
            "current_version": current,
        }

    tag = str(release.get("tag_name") or "").strip()
    if not tag:
        return {
            "status": "failed",
            "reason": "Latest release has no tag",
            "current_version": current,
        }

    latest = _normalize_version(tag)
    prerelease = bool(release.get("prerelease", False))
    if prerelease and not include_prerelease:
        return {
            "status": "up_to_date",
            "current_version": current,
            "latest_version": latest,
            "reason": "Latest release is a pre-release; use --prerelease to include it",
        }

    if not _is_newer(latest, current):
        return {
            "status": "up_to_date",
            "current_version": current,
            "latest_version": latest,
        }

    return {
        "status": "update_available",
        "current_version": current,
        "latest_version": latest,
        "tag": tag,
        "url": release.get("html_url"),
        "install_target": _github_release_url(repo, tag),
    }


def _confirm(message: str) -> bool:
    try:
        response = input(f"{message} [y/N]: ")
    except (EOFError, KeyboardInterrupt):
        return False
    return response.strip().lower() in {"y", "yes"}


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def handle_update_command(args: argparse.Namespace) -> int:
    repo = str(args.repo or DEFAULT_REPO).strip()
    include_prerelease = bool(args.prerelease)
    dry_run = bool(args.dry_run)
    force = bool(args.yes)

    check = check_for_update(repo, include_prerelease=include_prerelease)
    if check["status"] == "failed":
        _print_json(check)
        return 1

    if check["status"] == "up_to_date":
        _print_json(check)
        return 0

    context = _detect_install_context()
    latest = check["latest_version"]
    current = check["current_version"]
    target = check["install_target"]

    if not context["in_virtualenv"] and not force:
        print(
            json.dumps(
                {
                    "status": "warning",
                    "message": (
                        "You are not inside a virtual environment. "
                        "Installing into the system Python can overwrite files used by other tools. "
                        "Run with --yes to proceed anyway."
                    ),
                    "in_virtualenv": context["in_virtualenv"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    print(
        json.dumps(
            {
                "status": "update_available",
                "current_version": current,
                "latest_version": latest,
                "install_target": target,
                "dry_run": dry_run,
            },
            indent=2,
            sort_keys=True,
        )
    )

    if dry_run:
        dry_result = _run_pip_install(target, dry_run=True)
        _print_json({"status": "dry_run", "result": dry_result})
        return 0

    if not force and not _confirm(f"Install smallctl {latest} over {current}?"):
        _print_json({"status": "cancelled"})
        return 0

    install_result = _run_pip_install(target)
    success = install_result["returncode"] == 0
    _print_json(
        {
            "status": "installed" if success else "failed",
            "current_version": current,
            "latest_version": latest,
            "install_target": target,
            "result": install_result,
        }
    )
    return 0 if success else 1


def build_update_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("update", help="Check for and install updates from GitHub")
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"GitHub owner/repo to check (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--prerelease",
        action="store_true",
        help="Include pre-releases when checking for updates",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what pip would install without modifying anything",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompts",
    )
