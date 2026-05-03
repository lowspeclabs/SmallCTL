"""Tests for error_hardening.py (B, A2, A3)."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, PropertyMock

from smallctl.graph.error_hardening import (
    _maybe_emit_ground_truth_diffusion,
    _maybe_emit_nginx_sites_enabled_nudge,
    _maybe_schedule_web_search_for_repeated_error,
)
from smallctl.graph.state import GraphRunState, PendingToolCall, ToolExecutionRecord
from smallctl.harness.tool_results import _store_verifier_verdict
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState
from smallctl.tools import memory
from smallctl.tools.base import ToolSpec, build_tool_schema
from smallctl.tools.dispatcher import ToolDispatcher
from smallctl.tools.registry import ToolRegistry


class FakeArtifact:
    def __init__(self, artifact_id: str, source: str, text: str = "", summary: str = ""):
        self.artifact_id = artifact_id
        self.source = source
        self.text = text
        self.summary = summary


def _make_harness() -> MagicMock:
    harness = MagicMock()
    harness.state.scratchpad = {}
    harness.state.artifacts = {}
    harness.state.append_message = MagicMock()
    harness._runlog = MagicMock()
    harness.registry = MagicMock()
    harness.registry.names = MagicMock(return_value=[])
    return harness


def _make_record(
    tool_name: str,
    success: bool,
    error: str = "",
    command: str = "",
) -> ToolExecutionRecord:
    return ToolExecutionRecord(
        operation_id="op-1",
        tool_name=tool_name,
        args={"command": command} if command else {},
        tool_call_id="tc-1",
        result=ToolEnvelope(
            success=success,
            error=error,
            metadata={"command": command} if command else {},
        ),
    )


# ---------------------------------------------------------------------------
# B) Nginx sites-enabled hardening
# ---------------------------------------------------------------------------

class TestNginxSitesEnabledHardening:
    def test_ignores_success(self):
        harness = _make_harness()
        record = _make_record("ssh_exec", success=True)
        assert _maybe_emit_nginx_sites_enabled_nudge(harness, record) is False

    def test_ignores_non_shell_tool(self):
        harness = _make_harness()
        record = _make_record("file_read", success=False, error="nginx: test failed")
        assert _maybe_emit_nginx_sites_enabled_nudge(harness, record) is False

    def test_ignores_non_nginx_error(self):
        harness = _make_harness()
        record = _make_record("ssh_exec", success=False, error="command not found")
        assert _maybe_emit_nginx_sites_enabled_nudge(harness, record) is False

    def test_fires_on_sites_enabled_syntax_error(self):
        harness = _make_harness()
        record = _make_record(
            "ssh_exec",
            success=False,
            error="2026/05/02 15:19:19 [emerg] 12665#12665: unexpected end of file, expecting \";\" or \"}\" in /etc/nginx/sites-enabled/test-1:2\nnginx: configuration file /etc/nginx/nginx.conf test failed",
            command="echo 'enabled' > /etc/nginx/sites-enabled/test-1 && nginx -t && service nginx reload",
        )
        assert _maybe_emit_nginx_sites_enabled_nudge(harness, record) is True
        harness.state.append_message.assert_called_once()
        msg = harness.state.append_message.call_args[0][0]
        assert "sites-enabled/test-1" in msg.content
        assert "ln -s" in msg.content
        assert msg.metadata["recovery_kind"] == "nginx_sites_enabled_hardening"

    def test_fires_on_masked_pipeline_success_with_nginx_failure_stdout(self):
        harness = _make_harness()
        record = ToolExecutionRecord(
            operation_id="op-1",
            tool_name="ssh_exec",
            args={"command": "cat /etc/nginx/sites-available/test-1.conf; nginx -t 2>&1 | head -50"},
            tool_call_id="tc-1",
            result=ToolEnvelope(
                success=True,
                output={
                    "stdout": (
                        "2026/05/02 15:36:12 [emerg] 13109#13109: unexpected end of file, "
                        "expecting \";\" or \"}\" in /etc/nginx/sites-enabled/test-1:2\n"
                        "nginx: configuration file /etc/nginx/nginx.conf test failed\n"
                    ),
                    "stderr": "",
                    "exit_code": 0,
                },
            ),
        )

        assert _maybe_emit_nginx_sites_enabled_nudge(harness, record) is True
        msg = harness.state.append_message.call_args[0][0]
        assert "sites-enabled/test-1" in msg.content

    def test_deduplicates_per_site(self):
        harness = _make_harness()
        record = _make_record(
            "ssh_exec",
            success=False,
            error="nginx: configuration file /etc/nginx/nginx.conf test failed\nunexpected end of file in /etc/nginx/sites-enabled/test-1:2",
            command="echo 'enabled' > /etc/nginx/sites-enabled/test-1 && nginx -t",
        )
        assert _maybe_emit_nginx_sites_enabled_nudge(harness, record) is True
        harness.state.append_message.reset_mock()
        assert _maybe_emit_nginx_sites_enabled_nudge(harness, record) is False


# ---------------------------------------------------------------------------
# A2) Ground-truth diffusion
# ---------------------------------------------------------------------------

class TestGroundTruthDiffusion:
    def test_ignores_success(self):
        harness = _make_harness()
        record = _make_record("ssh_exec", success=True)
        assert _maybe_emit_ground_truth_diffusion(harness, record) is False

    def test_fires_when_artifact_matches_error_path(self):
        harness = _make_harness()
        harness.state.artifacts = {
            "A0013": FakeArtifact("A0013", "/etc/nginx/sites-enabled/test-1", text="enabled\n"),
        }
        record = _make_record(
            "ssh_exec",
            success=False,
            error="unexpected end of file in /etc/nginx/sites-enabled/test-1:2",
        )
        assert _maybe_emit_ground_truth_diffusion(harness, record) is True
        harness.state.append_message.assert_called_once()
        msg = harness.state.append_message.call_args[0][0]
        assert "Ground truth" in msg.content
        assert "/etc/nginx/sites-enabled/test-1" in msg.content
        assert "enabled" in msg.content
        assert msg.metadata["recovery_kind"] == "ground_truth_diffusion"

    def test_deduplicates(self):
        harness = _make_harness()
        harness.state.artifacts = {
            "A0013": FakeArtifact("A0013", "/etc/nginx/sites-enabled/test-1", text="enabled\n"),
        }
        record = _make_record(
            "ssh_exec",
            success=False,
            error="unexpected end of file in /etc/nginx/sites-enabled/test-1:2",
        )
        assert _maybe_emit_ground_truth_diffusion(harness, record) is True
        harness.state.append_message.reset_mock()
        assert _maybe_emit_ground_truth_diffusion(harness, record) is False

    def test_no_artifact_no_nudge(self):
        harness = _make_harness()
        record = _make_record(
            "ssh_exec",
            success=False,
            error="missing file /etc/nginx/sites-enabled/other:2",
        )
        assert _maybe_emit_ground_truth_diffusion(harness, record) is False


def test_nginx_verifier_stdout_failure_overrides_masked_exit_zero() -> None:
    state = LoopState()
    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=ToolEnvelope(
            success=True,
            output={
                "stdout": (
                    "# config preview\n"
                    "2026/05/02 15:36:12 [emerg] 13109#13109: unexpected end of file, "
                    "expecting \";\" or \"}\" in /etc/nginx/sites-enabled/test-1:2\n"
                    "nginx: configuration file /etc/nginx/nginx.conf test failed\n"
                ),
                "stderr": "",
                "exit_code": 0,
            },
        ),
        arguments={"host": "192.168.1.89", "command": "cat /etc/nginx/sites-available/test-1.conf; nginx -t 2>&1 | head -50"},
    )

    assert verdict is not None
    assert verdict["verdict"] == "fail"
    assert verdict["exit_code"] == 0
    assert "unexpected end of file" in verdict["acceptance_delta"]["notes"][0]


@pytest.mark.asyncio
async def test_memory_update_rejects_verifier_success_fact_when_latest_verifier_fails() -> None:
    state = LoopState()
    state.last_verifier_verdict = {
        "verdict": "fail",
        "command": "nginx -t",
        "key_stdout": "nginx: configuration file /etc/nginx/nginx.conf test failed",
    }

    result = await memory.memory_update(
        state,
        section="known_facts",
        content="nginx config syntax verified with 'nginx -t'",
    )

    assert result["success"] is False
    assert result["metadata"]["reason"] == "contradictory_verifier_success_claim"


@pytest.mark.asyncio
async def test_dispatcher_exposes_shell_failure_output_from_metadata() -> None:
    registry = ToolRegistry()

    async def failing_ssh_exec(command: str, host: str) -> dict:
        del host
        return {
            "success": False,
            "output": None,
            "error": "Remote SSH command exited with code 1",
            "metadata": {
                "output": {
                    "stdout": "bad config\n",
                    "stderr": "nginx: configuration file /etc/nginx/nginx.conf test failed\n",
                    "exit_code": 1,
                }
            },
        }

    registry.register(
            ToolSpec(
                name="ssh_exec",
                description="fake ssh",
                schema=build_tool_schema(
                    required=["host", "command"],
                    properties={"host": {"type": "string"}, "command": {"type": "string"}},
                ),
                handler=failing_ssh_exec,
                category="network",
                risk="high",
            )
        )
    dispatcher = ToolDispatcher(registry, phase="execute")

    envelope = await dispatcher.dispatch("ssh_exec", {"host": "192.168.1.89", "command": "nginx -t"})

    assert envelope.success is False
    assert isinstance(envelope.output, dict)
    assert envelope.output["exit_code"] == 1
    assert "test failed" in envelope.output["stderr"]


# ---------------------------------------------------------------------------
# A3) Web search on repeated errors
# ---------------------------------------------------------------------------

class TestWebSearchOnRepeatedError:
    def test_ignores_success(self):
        gs = GraphRunState(loop_state=MagicMock(), thread_id="t1", run_mode="loop")
        harness = _make_harness()
        record = _make_record("ssh_exec", success=True)
        assert _maybe_schedule_web_search_for_repeated_error(gs, harness, record) is False

    def test_schedules_on_second_identical_error(self):
        gs = GraphRunState(loop_state=MagicMock(), thread_id="t1", run_mode="loop")
        harness = _make_harness()
        harness.registry.names.return_value = ["web_search"]
        error = "nginx: configuration file /etc/nginx/nginx.conf test failed\nunexpected end of file"
        record = _make_record("ssh_exec", success=False, error=error)

        # First occurrence — nothing scheduled
        assert _maybe_schedule_web_search_for_repeated_error(gs, harness, record) is False
        assert gs.pending_tool_calls == []

        # Second occurrence — web_search scheduled
        assert _maybe_schedule_web_search_for_repeated_error(gs, harness, record) is True
        assert len(gs.pending_tool_calls) == 1
        assert gs.pending_tool_calls[0].tool_name == "web_search"
        assert "nginx error:" in gs.pending_tool_calls[0].args["query"]
        harness.state.append_message.assert_called_once()
        msg = harness.state.append_message.call_args[0][0]
        assert "Auto-searching" in msg.content

    def test_nudges_when_web_search_unavailable(self):
        gs = GraphRunState(loop_state=MagicMock(), thread_id="t1", run_mode="loop")
        harness = _make_harness()
        harness.registry.names.return_value = ["shell_exec"]
        error = "nginx: configuration file /etc/nginx/nginx.conf test failed"
        record = _make_record("ssh_exec", success=False, error=error)

        # Prime first occurrence
        _maybe_schedule_web_search_for_repeated_error(gs, harness, record)
        # Second occurrence
        assert _maybe_schedule_web_search_for_repeated_error(gs, harness, record) is True
        assert gs.pending_tool_calls == []
        msg = harness.state.append_message.call_args[0][0]
        assert "Consider searching" in msg.content
