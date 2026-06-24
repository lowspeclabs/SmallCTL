from __future__ import annotations

from typing import Any
from smallctl.tools.network_ssh_helpers import is_purely_diagnostic, ssh_diagnostic_not_found

def test_is_purely_diagnostic() -> None:
    # Diagnostic cases
    assert is_purely_diagnostic("docker ps -a | grep -i dagu") is True
    assert is_purely_diagnostic("ls -la /var/lib/dagu") is True
    assert is_purely_diagnostic("cat /var/lib/dagu/dagu.yaml") is True
    assert is_purely_diagnostic("grep -q 'dagu' /etc/passwd") is True
    assert is_purely_diagnostic("ls -la") is True
    assert is_purely_diagnostic("df -h") is True
    assert is_purely_diagnostic("docker inspect dagu") is True
    assert is_purely_diagnostic("apt list --upgradable 2>/dev/null") is True
    assert is_purely_diagnostic("git status") is True
    
    # Mutating / Chained cases (Should be False)
    assert is_purely_diagnostic("chown -R root:root /var/lib/dagu && docker restart dagu") is False
    assert is_purely_diagnostic("chown -R root:root /var/lib/dagu") is False
    assert is_purely_diagnostic("docker restart dagu") is False
    assert is_purely_diagnostic("docker run -d --name dagu nginx") is False
    assert is_purely_diagnostic("rm -rf /tmp/foo") is False
    assert is_purely_diagnostic("mkdir -p /tmp/bar") is False
    assert is_purely_diagnostic("echo 'hello' > /tmp/test.txt") is False
    assert is_purely_diagnostic("systemctl restart nginx") is False
    assert is_purely_diagnostic("git pull origin main") is False
    assert is_purely_diagnostic("apt install curl") is False

def test_ssh_diagnostic_not_found_with_grep_no_match() -> None:
    # Grep with exit code 1 and empty output is considered an informational success
    cmd = "docker ps -a | grep -i dagu"
    output: dict[str, Any] = {
        "stdout": "",
        "stderr": "",
        "exit_code": 1
    }
    assert ssh_diagnostic_not_found(cmd, output) is True

def test_ssh_diagnostic_not_found_with_mutating_error() -> None:
    # Mutating command failing with "No such file" error must NOT be matched as informational success
    cmd = "chown -R root:root /var/lib/dagu && docker restart dagu"
    output: dict[str, Any] = {
        "stdout": "",
        "stderr": "chown: cannot access '/var/lib/dagu': No such file or directory\n",
        "exit_code": 1
    }
    assert ssh_diagnostic_not_found(cmd, output) is False

def test_ssh_diagnostic_not_found_standard_diagnostic_errors() -> None:
    # ls command with exit code 2 and "No such file" is a diagnostic not found success
    cmd = "ls -la /nonexistent"
    output: dict[str, Any] = {
        "stdout": "",
        "stderr": "ls: cannot access '/nonexistent': No such file or directory\n",
        "exit_code": 2
    }
    assert ssh_diagnostic_not_found(cmd, output) is True


def test_ssh_diagnostic_not_found_command_missing_exit_127() -> None:
    # A purely diagnostic chain probing for an optional tool (e.g. getenforce)
    # should be informational when that tool is not installed.
    cmd = "ls -ld /opt/app/data; echo '---'; getenforce"
    output: dict[str, Any] = {
        "stdout": "drwxr-xr-x 2 root root 2 Jun 23 21:14 /opt/app/data\n---\n",
        "stderr": "bash: line 1: getenforce: command not found\n",
        "exit_code": 127,
    }
    assert ssh_diagnostic_not_found(cmd, output) is True


def test_ssh_mutating_command_missing_exit_127_is_failure() -> None:
    # A mutating command that cannot run because the binary is missing is a
    # real failure, not an informational probe result.
    cmd = "chmown root:root /opt/app/data"
    output: dict[str, Any] = {
        "stdout": "",
        "stderr": "bash: line 1: chmown: command not found\n",
        "exit_code": 127,
    }
    assert ssh_diagnostic_not_found(cmd, output) is False

