from __future__ import annotations

from smallctl.tools.shell_support_apt_and_outcome import classify_shell_outcome


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
