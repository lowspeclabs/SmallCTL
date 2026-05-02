from __future__ import annotations

import json

from smallctl.memory_cli import memory_cli


def test_memory_cli_add_and_list_filter_by_namespace(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = memory_cli(
        [
            "add",
            "--tier",
            "cold",
            "--intent",
            "requested_ssh_exec",
            "--tool",
            "ssh_exec",
            "--note",
            "Run whoami on the remote host.",
            "--namespace",
            "ssh_remote",
        ]
    )

    assert exit_code == 0
    capsys.readouterr()

    exit_code = memory_cli(["list", "--tier", "cold", "--namespace", "ssh_remote"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 1
    assert payload["records"][0]["namespace"] == "ssh_remote"


def test_memory_cli_search_filters_namespace_and_preserves_json_import_namespace(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    payload_path = tmp_path / "import.json"
    payload_path.write_text(
        json.dumps(
            [
                {
                    "memory_id": "mem-plan",
                    "tier": "cold",
                    "intent": "plan_execution",
                    "namespace": "planning",
                    "tool_name": "task_complete",
                    "outcome": "success",
                    "notes": "Plan the rollout before execution.",
                },
                {
                    "memory_id": "mem-ssh",
                    "tier": "cold",
                    "intent": "requested_ssh_exec",
                    "namespace": "ssh_remote",
                    "tool_name": "ssh_exec",
                    "outcome": "success",
                    "notes": "Run the remote command.",
                },
            ]
        ),
        encoding="utf-8",
    )

    exit_code = memory_cli(["add", "--tier", "cold", "--intent", "ignored", "--from-json", str(payload_path)])

    assert exit_code == 0
    capsys.readouterr()

    exit_code = memory_cli(["search", "--query", "plan", "--tier", "cold", "--namespace", "planning"])

    assert exit_code == 0
    results = json.loads(capsys.readouterr().out)
    assert results["count"] == 1
    assert results["records"][0]["memory_id"] == "mem-plan"
    assert results["records"][0]["namespace"] == "planning"
