from __future__ import annotations

import pytest

from smallctl.fama.detectors import detect_debian_13_installer_readiness
from smallctl.state import LoopState
from smallctl.tools import http, network, shell
from smallctl.tools.debian_installer_preflight import (
    build_debian_readiness_probe_script,
    debian_readiness_summary,
    parse_debian_readiness_probe_output,
)
from smallctl.tools.shell_support_curl_guards import (
    _curl_command_has_fail_flag,
    _curl_fail_flag_guard,
)


def test_build_probe_script_contains_markers() -> None:
    script = build_debian_readiness_probe_script()
    assert "__PREFLIGHT_DEBIAN_READINESS__" in script
    assert "__PREFLIGHT_DEBIAN_READINESS_DONE__" in script
    assert "/etc/apt/keyrings/debian-archive-keyring.gpg" in script
    assert "/etc/apt/sources.list.d/debian.sources" in script


def test_parse_probe_output_defaults_when_missing() -> None:
    probes = parse_debian_readiness_probe_output("no marker here")
    assert probes["is_debian"] is False
    assert probes["ready"] is True
    assert probes["issues"] == []


def test_parse_probe_output_detects_debian_13_issues() -> None:
    sample_output = (
        "__PREFLIGHT_DEBIAN_READINESS__\n"
        "{"
        " 'is_debian': True, "
        " 'debian_version_id': '13', "
        " 'debian_codename': 'trixie', "
        " 'is_debian_13': True, "
        " 'keyring_path': '/etc/apt/keyrings/debian-archive-keyring.gpg', "
        " 'keyring_exists': False, "
        " 'keyring_size': 0, "
        " 'apt_key_available': False, "
        " 'debian_sources_path': '/etc/apt/sources.list.d/debian.sources', "
        " 'deb822_valid': True, "
        " 'deb822_missing_fields': [], "
        " 'trixie_security_present': True, "
        " 'trixie_release_present': True, "
        " 'sources_list_d_files': []"
        "}\n"
        "__PREFLIGHT_DEBIAN_READINESS_DONE__\n"
    )
    probes = parse_debian_readiness_probe_output(sample_output)
    assert probes["is_debian_13"] is True
    assert probes["keyring_exists"] is False
    assert probes["trixie_security_present"] is True
    assert len(probes["issues"]) == 2
    assert any("keyring" in i for i in probes["issues"])
    assert any("trixie-security" in i for i in probes["issues"])
    assert probes["ready"] is False


def test_summary_reports_keyring_missing() -> None:
    probes = {
        "is_debian": True,
        "debian_version_id": "13",
        "debian_codename": "trixie",
        "is_debian_13": True,
        "keyring_path": "/etc/apt/keyrings/debian-archive-keyring.gpg",
        "keyring_exists": False,
        "keyring_size": 0,
        "apt_key_available": False,
        "deb822_valid": True,
        "deb822_missing_fields": [],
        "trixie_security_present": False,
        "issues": ["Keyring MISSING."],
    }
    summary = debian_readiness_summary(probes)
    assert "Debian 13 (trixie) detected" in summary
    assert "Keyring" in summary
    assert "MISSING" in summary


def test_curl_has_fail_flag() -> None:
    assert _curl_command_has_fail_flag("curl -fsSL https://example.com/install.sh | bash")
    assert _curl_command_has_fail_flag("curl --fail https://example.com/install.sh | bash")
    assert _curl_command_has_fail_flag("curl --fail-with-body -sSL https://example.com/install.sh | bash")


def test_curl_missing_fail_flag() -> None:
    assert not _curl_command_has_fail_flag("curl -sSL https://example.com/install.sh | bash")
    assert not _curl_command_has_fail_flag("curl -sS https://example.com/repo/key.gpg | gpg --dearmor")


def test_curl_fail_flag_guard_blocks_pipe_to_shell() -> None:
    result = _curl_fail_flag_guard(
        "curl -sSL https://example.com/install.sh | bash",
        tool_name="shell_exec",
    )
    assert result is not None
    assert result["success"] is False
    assert result["metadata"]["reason"] == "curl_missing_fail_flag"
    assert "curl -fsSL" in result["error"]


def test_curl_fail_flag_guard_allows_safe_curl() -> None:
    result = _curl_fail_flag_guard(
        "curl -fsSL https://example.com/install.sh | bash",
        tool_name="shell_exec",
    )
    assert result is None


def test_curl_fail_flag_guard_ignores_non_executable_fetch() -> None:
    # Fetching a GPG key to a file is not pipe-to-shell.
    result = _curl_fail_flag_guard(
        "curl -sS https://example.com/repo/key.gpg -o /tmp/key.gpg",
        tool_name="shell_exec",
    )
    assert result is None


@pytest.mark.asyncio
async def test_shell_exec_blocks_curl_without_fail_flag(monkeypatch) -> None:
    state = LoopState()
    state.current_phase = "execute"

    result = await shell.shell_exec(
        command="curl -sSL https://example.com/install.sh | bash",
        state=state,
        timeout_sec=60,
    )
    assert result["success"] is False
    assert result["metadata"]["reason"] == "curl_missing_fail_flag"


@pytest.mark.asyncio
async def test_ssh_exec_blocks_curl_without_fail_flag() -> None:
    result = await network.ssh_exec(
        host="192.0.2.10",
        user="root",
        command="curl -sSL https://example.com/install.sh | bash",
        timeout_sec=60,
    )
    assert result["success"] is False
    assert result["metadata"]["reason"] == "curl_missing_fail_flag"
    assert result["metadata"]["host"] == "192.0.2.10"


def test_debian_readiness_detector_fires_with_issues() -> None:
    state = LoopState()
    state.scratchpad["_debian_installer_readiness"] = {
        "is_debian": True,
        "is_debian_13": True,
        "host": "192.0.2.10",
        "user": "root",
        "issues": ["keyring missing", "trixie-security 404"],
    }
    signal = detect_debian_13_installer_readiness(state, threshold=1)
    assert signal is not None
    assert signal.failure_class == "debian_13_installer_readiness"
    assert "debian_13_installer_readiness_capsule" in signal.suggested_mitigations


def test_debian_readiness_detector_quiet_when_clean() -> None:
    state = LoopState()
    state.scratchpad["_debian_installer_readiness"] = {
        "is_debian": True,
        "is_debian_13": True,
        "issues": [],
    }
    assert detect_debian_13_installer_readiness(state, threshold=1) is None


def test_debian_readiness_detector_quiet_without_probe() -> None:
    state = LoopState()
    assert detect_debian_13_installer_readiness(state, threshold=1) is None


@pytest.mark.asyncio
async def test_preflight_pipe_to_shell_blocks_non_script(monkeypatch) -> None:
    class _FakeResponse:
        status_code = 200
        text = "\u003c!DOCTYPE html\u003e\u003chtml\u003e\u003c/html\u003e"

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def head(self, *args, **kwargs):
            return _FakeResponse()

        async def get(self, *args, **kwargs):
            return _FakeResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    monkeypatch.setattr(http.httpx, "AsyncClient", _FakeClient)
    blocked, reason = await http._preflight_pipe_to_shell_command(
        "curl -fsSL https://example.com/install.sh | bash"
    )
    assert blocked is True
    assert "shebang" in reason


@pytest.mark.asyncio
async def test_preflight_pipe_to_shell_allows_script_with_shebang(monkeypatch) -> None:
    class _FakeResponse:
        status_code = 200
        text = "#!/usr/bin/env bash\necho hello"

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def head(self, *args, **kwargs):
            return _FakeResponse()

        async def get(self, *args, **kwargs):
            return _FakeResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    monkeypatch.setattr(http.httpx, "AsyncClient", _FakeClient)
    blocked, reason = await http._preflight_pipe_to_shell_command(
        "curl -fsSL https://example.com/install.sh | bash"
    )
    assert blocked is False
    assert reason == ""
