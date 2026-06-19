from __future__ import annotations

from smallctl.tools.shell_foreground import _classify_shell_failure


def test_shell_failure_classifies_connection_refused_as_environment_unavailable() -> None:
    metadata = _classify_shell_failure(
        "python3 ./temp/vikunja-9b.py projects list",
        "Network error: [Errno 111] Connection refused",
        {"stdout": "", "stderr": "", "exit_code": 1},
    )

    assert metadata["failure_class"] == "environment_unavailable"
    assert metadata["reason"] == "connection_refused"


def test_shell_failure_classifies_empty_port_probe_as_service_not_listening() -> None:
    metadata = _classify_shell_failure(
        "ss -tuln | grep 3456",
        "Command exited with code 1",
        {"stdout": "", "stderr": "", "exit_code": 1},
    )

    assert metadata["failure_class"] == "environment_unavailable"
    assert metadata["reason"] == "service_not_listening"
