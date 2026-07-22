from __future__ import annotations

from smallctl.tools.shell_support_apt_and_outcome import classify_shell_outcome
from smallctl.tools.shell_support_foreground_guards import _masked_compose_lifecycle_guard


def test_classifies_proxmox_pipeline_error_even_when_exit_code_is_masked() -> None:
    result = classify_shell_outcome(
        "python scripts/Proxmox-cli.py get-hosts 2>&1 | tail -30; echo \"---EXIT:$?\"",
        0,
        "usage: proxmox-cli ...\nproxmox-cli: error: argument resource: invalid choice: 'get-hosts'\n---EXIT:0\n",
        "",
    )

    assert result["status"] == "failure"
    assert result["kind"] == "masked_pipeline_failure"
    assert result["failure_mode"] == "masked_pipeline_failure"


def test_does_not_reclassify_normal_pipelined_output() -> None:
    result = classify_shell_outcome("printf 'hello\\n' | tail -1", 0, "hello\n", "")

    assert result == {"status": "success", "kind": "ok"}


def test_blocks_compose_down_and_truncated_up_chain() -> None:
    result = _masked_compose_lifecycle_guard(
        "docker compose down 2>&1; docker compose up -d --build 2>&1 | head -50",
        tool_name="ssh_exec",
    )

    assert result is not None
    assert result["success"] is False
    assert result["metadata"]["reason"] == "masked_compose_lifecycle_command"
    assert result["metadata"]["next_required_action"]["strategy"] == "run_compose_lifecycle_then_verify"


def test_allows_unmasked_compose_lifecycle_and_readonly_diagnostic() -> None:
    assert _masked_compose_lifecycle_guard("docker compose up -d --build", tool_name="ssh_exec") is None
    assert _masked_compose_lifecycle_guard("docker compose ps | head -30", tool_name="ssh_exec") is None
