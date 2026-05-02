from __future__ import annotations

import json
from pathlib import Path

from smallctl.memory_cli import memory_cli


def test_memory_scrub_dry_run_does_not_modify_store(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    memory_dir = tmp_path / ".smallctl" / "memory"
    memory_dir.mkdir(parents=True)
    cold_path = memory_dir / "cold-experiences.jsonl"
    cold_path.write_text(
        json.dumps(
            {
                "memory_id": "mem-1",
                "tier": "cold",
                "intent": "requested_ssh_exec",
                "tool_name": "task_complete",
                "outcome": "success",
                "notes": 'SSH to root@192.168.1.63 with password "@S02v1735" succeeded.',
            }
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = memory_cli(["scrub", "--tier", "cold"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "dry_run"
    assert payload["changed"] == 1
    assert "@S02v1735" in cold_path.read_text(encoding="utf-8")


def test_memory_scrub_write_redacts_store(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    memory_dir = tmp_path / ".smallctl" / "memory"
    memory_dir.mkdir(parents=True)
    cold_path = memory_dir / "cold-experiences.jsonl"
    cold_path.write_text(
        json.dumps(
            {
                "memory_id": "mem-1",
                "tier": "cold",
                "intent": "requested_ssh_exec",
                "tool_name": "task_complete",
                "outcome": "success",
                "notes": 'SSH to root@192.168.1.63 with password "@S02v1735" succeeded.',
            }
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = memory_cli(["scrub", "--tier", "cold", "--write"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "scrubbed"
    assert payload["changed"] == 1
    rewritten = cold_path.read_text(encoding="utf-8")
    assert "[REDACTED]" in rewritten
    assert "@S02v1735" not in rewritten
