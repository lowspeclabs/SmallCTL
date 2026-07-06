from __future__ import annotations

import json
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from smallctl import __version__
from smallctl.update_cli import (
    _github_release_url,
    _is_newer,
    _normalize_version,
    _version_tuple,
    check_for_update,
    handle_update_command,
)


def _last_json_object(text: str) -> dict:
    """Return the last JSON object printed to stdout."""
    lines = text.split("\n")
    start = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "{":
            start = i
            break
    if start is None:
        raise ValueError("No JSON object found in output")
    return json.loads("\n".join(lines[start:]))


def test_normalize_version_strips_v():
    assert _normalize_version("v0.2.0") == "0.2.0"
    assert _normalize_version("0.2.0") == "0.2.0"
    assert _normalize_version("") == "0.0.0"


def test_version_tuple_parses_prerelease():
    assert _version_tuple("0.1.2") == (0, 1, 2)
    assert _version_tuple("0.1.2-alpha") == (0, 1, 2, 0)
    assert _version_tuple("v1.10.0") == (1, 10, 0)


def test_is_newer():
    assert _is_newer("0.2.0", "0.1.1")
    assert not _is_newer("0.1.1", "0.2.0")
    assert not _is_newer("0.1.1", "0.1.1")


def test_github_release_url():
    assert _github_release_url("lowspeclabs/SmallCTL", "v0.2.0") == (
        "git+https://github.com/lowspeclabs/SmallCTL.git@v0.2.0"
    )


def test_check_for_update_when_up_to_date(monkeypatch):
    mock_release = {"tag_name": __version__, "prerelease": False}

    def _fake_fetch(repo, *, timeout):
        return mock_release

    monkeypatch.setattr("smallctl.update_cli._fetch_latest_release", _fake_fetch)
    result = check_for_update("lowspeclabs/SmallCTL")
    assert result["status"] == "up_to_date"
    assert result["current_version"] == _normalize_version(__version__)


def test_check_for_update_when_update_available(monkeypatch):
    mock_release = {"tag_name": "v0.9.9", "prerelease": False}

    def _fake_fetch(repo, *, timeout):
        return mock_release

    monkeypatch.setattr("smallctl.update_cli._fetch_latest_release", _fake_fetch)
    result = check_for_update("lowspeclabs/SmallCTL")
    assert result["status"] == "update_available"
    assert result["latest_version"] == "0.9.9"
    assert result["install_target"] == _github_release_url("lowspeclabs/SmallCTL", "v0.9.9")


def test_check_for_update_prerelease_skipped_by_default(monkeypatch):
    mock_release = {"tag_name": "v0.9.9-rc1", "prerelease": True}

    def _fake_fetch(repo, *, timeout):
        return mock_release

    monkeypatch.setattr("smallctl.update_cli._fetch_latest_release", _fake_fetch)
    result = check_for_update("lowspeclabs/SmallCTL")
    assert result["status"] == "up_to_date"


def test_check_for_update_prerelease_allowed(monkeypatch):
    mock_release = {"tag_name": "v0.9.9-rc1", "prerelease": True}

    def _fake_fetch(repo, *, timeout):
        return mock_release

    monkeypatch.setattr("smallctl.update_cli._fetch_latest_release", _fake_fetch)
    result = check_for_update("lowspeclabs/SmallCTL", include_prerelease=True)
    assert result["status"] == "update_available"


def test_check_for_update_api_failure(monkeypatch):
    import httpx

    def _fake_fetch(repo, *, timeout):
        response = MagicMock()
        response.status_code = 404
        raise httpx.HTTPStatusError(
            "Not found", request=MagicMock(), response=response
        )

    monkeypatch.setattr("smallctl.update_cli._fetch_latest_release", _fake_fetch)
    result = check_for_update("lowspeclabs/SmallCTL")
    assert result["status"] == "failed"


def test_update_command_dry_run(monkeypatch, capsys):
    monkeypatch.setattr(
        "smallctl.update_cli._detect_install_context",
        lambda: {"in_virtualenv": True, "editable": True, "git_install": True},
    )
    monkeypatch.setattr(
        "smallctl.update_cli.check_for_update",
        lambda repo, include_prerelease: {
            "status": "update_available",
            "current_version": "0.1.1",
            "latest_version": "0.2.0",
            "tag": "v0.2.0",
            "install_target": _github_release_url("lowspeclabs/SmallCTL", "v0.2.0"),
        },
    )
    monkeypatch.setattr(
        "smallctl.update_cli._run_pip_install",
        lambda target, *, dry_run: {
            "command": f"pip install --upgrade --dry-run {target}",
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        },
    )

    args = Namespace(repo="lowspeclabs/SmallCTL", prerelease=False, dry_run=True, yes=True)
    exit_code = handle_update_command(args)
    assert exit_code == 0
    output = _last_json_object(capsys.readouterr().out)
    assert output["status"] == "dry_run"


def test_update_command_cancels_without_confirmation(monkeypatch, capsys):
    monkeypatch.setattr(
        "smallctl.update_cli._detect_install_context",
        lambda: {"in_virtualenv": True, "editable": True, "git_install": True},
    )
    monkeypatch.setattr(
        "smallctl.update_cli.check_for_update",
        lambda repo, include_prerelease: {
            "status": "update_available",
            "current_version": "0.1.1",
            "latest_version": "0.2.0",
            "tag": "v0.2.0",
            "install_target": _github_release_url("lowspeclabs/SmallCTL", "v0.2.0"),
        },
    )
    monkeypatch.setattr("smallctl.update_cli._confirm", lambda message: False)

    args = Namespace(repo="lowspeclabs/SmallCTL", prerelease=False, dry_run=False, yes=False)
    exit_code = handle_update_command(args)
    assert exit_code == 0
    output = _last_json_object(capsys.readouterr().out)
    assert output["status"] == "cancelled"


def test_update_command_warns_outside_venv(monkeypatch, capsys):
    monkeypatch.setattr(
        "smallctl.update_cli._detect_install_context",
        lambda: {"in_virtualenv": False, "editable": True, "git_install": True},
    )
    monkeypatch.setattr(
        "smallctl.update_cli.check_for_update",
        lambda repo, include_prerelease: {
            "status": "update_available",
            "current_version": "0.1.1",
            "latest_version": "0.2.0",
            "tag": "v0.2.0",
            "install_target": _github_release_url("lowspeclabs/SmallCTL", "v0.2.0"),
        },
    )

    args = Namespace(repo="lowspeclabs/SmallCTL", prerelease=False, dry_run=False, yes=False)
    exit_code = handle_update_command(args)
    assert exit_code == 1
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "warning"
    assert "virtual environment" in output["message"]


def test_update_command_up_to_date(monkeypatch, capsys):
    monkeypatch.setattr(
        "smallctl.update_cli.check_for_update",
        lambda repo, include_prerelease: {
            "status": "up_to_date",
            "current_version": "0.1.1",
            "latest_version": "0.1.1",
        },
    )

    args = Namespace(repo="lowspeclabs/SmallCTL", prerelease=False, dry_run=False, yes=False)
    exit_code = handle_update_command(args)
    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "up_to_date"
