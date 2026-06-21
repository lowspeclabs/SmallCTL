from __future__ import annotations

import asyncio
import os
import sys

import pytest

from smallctl.tools.shell import _tokenize_command, create_process
from smallctl.tools.shell_support import _command_requires_shell


def test_tokenize_simple_command() -> None:
    assert _tokenize_command("echo hello", posix=True) == ["echo", "hello"]


def test_tokenize_preserves_quoted_whitespace() -> None:
    assert _tokenize_command("echo 'hello world'", posix=True) == ["echo", "hello world"]
    assert _tokenize_command('echo "hello world"', posix=True) == ["echo", "hello world"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX backslash rules")
def test_tokenize_posix_backslash() -> None:
    assert _tokenize_command(r"echo hello\ world", posix=True) == ["echo", "hello world"]


def test_tokenize_empty_command_raises() -> None:
    with pytest.raises(ValueError, match="Empty command"):
        _tokenize_command("", posix=True)


def test_tokenize_unbalanced_quotes_raises() -> None:
    with pytest.raises(ValueError, match="Invalid command quoting"):
        _tokenize_command("echo '", posix=True)


def test_command_requires_shell_detects_pipe() -> None:
    assert _command_requires_shell("ls | grep foo") is True


def test_command_requires_shell_detects_redirect() -> None:
    assert _command_requires_shell("echo x > file") is True
    assert _command_requires_shell("cat < file") is True


def test_command_requires_shell_detects_env_expansion() -> None:
    assert _command_requires_shell("echo $HOME") is True
    assert _command_requires_shell('echo "$HOME"') is True


def test_command_requires_shell_ignores_single_quoted_env() -> None:
    assert _command_requires_shell("echo '$HOME'") is False


def test_command_requires_shell_detects_command_substitution() -> None:
    assert _command_requires_shell("echo $(date)") is True
    assert _command_requires_shell("echo `date`") is True


def test_command_requires_shell_detects_boolean_operators() -> None:
    assert _command_requires_shell("cmd1 && cmd2") is True
    assert _command_requires_shell("cmd1 || cmd2") is True
    assert _command_requires_shell("cmd1; cmd2") is True


def test_command_requires_shell_detects_tilde() -> None:
    assert _command_requires_shell("cat ~/.bashrc") is True


def test_command_requires_shell_simple_command_false() -> None:
    assert _command_requires_shell("echo hello") is False
    assert _command_requires_shell("python3 script.py --flag value") is False


def test_command_requires_shell_quoted_metachar_false() -> None:
    assert _command_requires_shell("echo 'a;b'") is False


@pytest.mark.asyncio
async def test_create_process_runs_simple_command_without_shell() -> None:
    proc = await create_process(command="echo hello", cwd=os.getcwd())
    stdout, _ = await proc.communicate()
    assert stdout.strip() == b"hello"


@pytest.mark.asyncio
async def test_create_process_preserves_quoted_whitespace() -> None:
    proc = await create_process(command="echo 'hello world'", cwd=os.getcwd())
    stdout, _ = await proc.communicate()
    assert stdout.strip() == b"hello world"


@pytest.mark.asyncio
async def test_create_process_without_shell_does_not_expand_env() -> None:
    proc = await create_process(command="echo $HOME", cwd=os.getcwd())
    stdout, _ = await proc.communicate()
    assert stdout.strip() == b"$HOME"


@pytest.mark.asyncio
async def test_create_process_without_shell_treats_pipe_as_literal() -> None:
    proc = await create_process(command="echo '|'", cwd=os.getcwd())
    stdout, _ = await proc.communicate()
    assert stdout.strip() == b"|"


@pytest.mark.asyncio
async def test_create_process_shell_true_preserves_pipe() -> None:
    proc = await create_process(command="echo hello | cat", cwd=os.getcwd(), shell=True)
    stdout, _ = await proc.communicate()
    assert stdout.strip() == b"hello"


@pytest.mark.asyncio
async def test_create_process_shell_true_expands_env(monkeypatch) -> None:
    monkeypatch.setenv("SMALLCTL_TEST_VAR", "testvalue")
    proc = await create_process(
        command="echo $SMALLCTL_TEST_VAR", cwd=os.getcwd(), shell=True
    )
    stdout, _ = await proc.communicate()
    assert stdout.strip() == b"testvalue"


@pytest.mark.asyncio
async def test_create_process_shell_true_redirects_stdout(tmp_path) -> None:
    out_file = tmp_path / "out.txt"
    proc = await create_process(
        command=f"echo redirected > {out_file}", cwd=os.getcwd(), shell=True
    )
    await proc.communicate()
    assert out_file.read_text().strip() == "redirected"


@pytest.mark.skipif(os.name == "nt", reason="Uses POSIX executable lookup")
@pytest.mark.asyncio
async def test_create_process_without_shell_runs_executable_directly() -> None:
    proc = await create_process(command="/bin/echo direct", cwd=os.getcwd())
    stdout, _ = await proc.communicate()
    assert stdout.strip() == b"direct"
