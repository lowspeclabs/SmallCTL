from __future__ import annotations

from smallctl.graph.cancel_result import cancellation_message, cancellation_result
from smallctl.graph.error_hardening import _INTERNAL_POLICY_ERROR_RE
from smallctl.state import LoopState
from smallctl.tools.shell_support_apt_and_outcome import classify_shell_outcome


def test_cancellation_result_preserves_failed_verifier_context() -> None:
    state = LoopState()
    state.last_verifier_verdict = {
        "verdict": "fail",
        "failure_mode": "remote_installer_download_error",
        "exit_code": 127,
        "command": "curl -o setup-webmin-repo.sh URL && sh setup-webmin-repo.sh",
    }
    state.recent_errors.append("setup-webmin-repo.sh: 1: 404:: not found")

    result = cancellation_result(state, reason="cancel_requested")

    assert result["status"] == "cancelled_after_verifier_failure"
    assert result["last_verifier_verdict"]["exit_code"] == 127
    assert "failing verifier" in cancellation_message(state)


def test_classify_shell_outcome_detects_executed_http_error_body() -> None:
    outcome = classify_shell_outcome(
        "curl -o setup-webmin-repo.sh https://raw.githubusercontent.com/webmin/webmin/master/setup-webmin-repo.sh && sh setup-webmin-repo.sh",
        127,
        "",
        "setup-webmin-repo.sh: 1: 404:: not found",
    )

    assert outcome["failure_mode"] == "remote_installer_download_error"
    assert "Stop executing the downloaded file" in outcome["next_required_action"]


def test_repeated_error_web_search_suppresses_internal_remote_guards() -> None:
    assert _INTERNAL_POLICY_ERROR_RE.search(
        "Remote package/repository install preflight required before running this package mutation. Verify OS and remote network/DNS readiness first."
    )
    assert _INTERNAL_POLICY_ERROR_RE.search(
        "Destructive apt source mutation blocked. Do not remove or truncate apt source files blindly."
    )

