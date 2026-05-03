from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

from smallctl.state import LoopState
from smallctl.models.tool_result import ToolEnvelope
from smallctl.harness import tool_result_artifact_updates
from smallctl.state_schema import ArtifactRecord
from smallctl.tools.control import task_complete
from smallctl.tools import ssh_files
from smallctl.tools.dispatcher import normalize_tool_request
from smallctl.harness.task_intent import infer_requested_tool_name
from smallctl.tools.profiles import NETWORK_PROFILE
from smallctl.tools.register import build_registry


def test_apply_exact_patch_content_fails_on_zero_match() -> None:
    success, updated, metadata = ssh_files.apply_exact_patch_content(
        "alpha beta\n",
        target_text="gamma",
        replacement_text="delta",
        expected_occurrences=1,
    )

    assert success is False
    assert updated == "alpha beta\n"
    assert metadata["error_kind"] == "patch_target_not_found"
    assert metadata["actual_occurrences"] == 0
    assert metadata["expected_occurrences"] == 1
    assert metadata["best_match"]["preview"] == "alpha beta"
    assert metadata["best_match"]["start_line"] == 1


def test_apply_exact_patch_content_fails_on_occurrence_mismatch() -> None:
    success, _updated, metadata = ssh_files.apply_exact_patch_content(
        "alpha alpha\n",
        target_text="alpha",
        replacement_text="beta",
        expected_occurrences=1,
    )

    assert success is False
    assert metadata["error_kind"] == "patch_occurrence_mismatch"
    assert metadata["actual_occurrences"] == 2


def test_replace_between_handles_multiline_style_block() -> None:
    content = "<html><style>\nbody { color: red; }\n</style><main></main></html>"

    success, updated, metadata = ssh_files.apply_replace_between_content(
        content,
        start_text="<style>",
        end_text="</style>",
        replacement_text='<link rel="stylesheet" href="/llm-explainer-theme.css">',
        include_bounds=True,
        expected_occurrences=1,
    )

    assert success is True
    assert "<style>" not in updated
    assert "llm-explainer-theme.css" in updated
    assert metadata["actual_occurrences"] == 1


def test_replace_between_fails_when_end_bound_missing() -> None:
    success, updated, metadata = ssh_files.apply_replace_between_content(
        "<style>\nbody { color: red; }\n",
        start_text="<style>",
        end_text="</style>",
        replacement_text="<link>",
    )

    assert success is False
    assert updated.startswith("<style>")
    assert metadata["error_kind"] == "bounded_region_not_found"


def test_apply_exact_patch_content_whitespace_normalized_matches_tab_differences() -> None:
    success, updated, metadata = ssh_files.apply_exact_patch_content(
        "server {\n\troot /var/www/demo-site;\n}\n",
        target_text="root /var/www/demo-site;",
        replacement_text="root /srv/www/demo-site;",
        whitespace_normalized=True,
    )

    assert success is True
    assert "/srv/www/demo-site;" in updated
    assert metadata["match_mode"] == "whitespace_normalized"
    assert metadata["matched_region_previews"]


def test_network_profile_registers_typed_ssh_file_tools(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    harness = SimpleNamespace(
        state=state,
        log=SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    registry = build_registry(harness, registry_profiles={NETWORK_PROFILE})

    assert {"ssh_exec", "ssh_file_read", "ssh_file_write", "ssh_file_patch", "ssh_file_replace_between"} <= set(
        registry.names()
    )


def test_remote_file_guard_suggests_typed_read_tool() -> None:
    state = LoopState(cwd="/workspace")
    state.task_mode = "remote_execute"
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }

    class _Registry:
        @staticmethod
        def get(tool_name: str):
            if tool_name == "ssh_exec":
                return SimpleNamespace(
                    phase_allowed=lambda phase: True,
                    profile_allowed=lambda profiles: True,
                )
            return SimpleNamespace() if tool_name == "file_read" else None

    _tool_name, _args, envelope, _metadata = normalize_tool_request(
        _Registry(),
        "file_read",
        {"path": "/var/www/html/index.html"},
        phase="execute",
        state=state,
    )

    assert envelope is not None
    assert envelope.success is False
    assert envelope.metadata["suggested_tool"] == "ssh_file_read"
    assert "Use `ssh_file_read`" in envelope.error


def test_remote_file_guard_suggests_typed_patch_tool() -> None:
    state = LoopState(cwd="/workspace")
    state.task_mode = "remote_execute"
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }

    class _Registry:
        @staticmethod
        def get(tool_name: str):
            if tool_name == "ssh_exec":
                return SimpleNamespace(
                    phase_allowed=lambda phase: True,
                    profile_allowed=lambda profiles: True,
                )
            return SimpleNamespace() if tool_name == "file_patch" else None

    _tool_name, _args, envelope, _metadata = normalize_tool_request(
        _Registry(),
        "file_patch",
        {"path": "/var/www/html/index.html", "target_text": "old", "replacement_text": "new"},
        phase="execute",
        state=state,
    )

    assert envelope is not None
    assert envelope.success is False
    assert envelope.metadata["suggested_tool"] == "ssh_file_patch"
    assert "Use `ssh_file_patch`" in envelope.error


def test_remote_style_task_infers_replace_between() -> None:
    harness = SimpleNamespace(state=LoopState(cwd="."))

    tool_name = infer_requested_tool_name(
        harness,
        "ssh into root@192.168.1.63 and replace the style block in the remote page",
    )

    assert tool_name == "ssh_file_replace_between"


def test_remote_service_task_still_infers_ssh_exec() -> None:
    harness = SimpleNamespace(state=LoopState(cwd="."))

    tool_name = infer_requested_tool_name(
        harness,
        "ssh into root@192.168.1.63 and restart nginx",
    )

    assert tool_name == "ssh_exec"


def test_ssh_file_write_readback_mismatch_fails_and_keeps_requirement(monkeypatch) -> None:
    state = LoopState(cwd=".")
    state.scratchpad[ssh_files.REMOTE_MUTATION_VERIFICATION_KEY] = {
        "host": "192.168.1.63",
        "guessed_paths": ["/var/www/html/index.html"],
    }

    async def _fake_run_ssh_command(**kwargs):
        payload = {
            "ok": True,
            "path": "/var/www/html/index.html",
            "bytes_written": 3,
            "old_sha256": None,
            "new_sha256": "wrong",
            "readback_sha256": "wrong",
            "changed": True,
        }
        return {
            "success": True,
            "output": {"stdout": json.dumps(payload), "stderr": "", "exit_code": 0},
            "error": None,
            "metadata": {},
        }

    monkeypatch.setattr(ssh_files.network, "run_ssh_command", _fake_run_ssh_command)

    result = asyncio.run(
        ssh_files.ssh_file_write(
            target="root@192.168.1.63",
            path="/var/www/html/index.html",
            content="abc",
            state=state,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "readback_mismatch"
    assert ssh_files.REMOTE_MUTATION_VERIFICATION_KEY in state.scratchpad


def test_ssh_file_patch_success_clears_matching_remote_requirement(monkeypatch) -> None:
    state = LoopState(cwd=".")
    state.scratchpad[ssh_files.REMOTE_MUTATION_VERIFICATION_KEY] = {
        "host": "192.168.1.63",
        "guessed_paths": ["/var/www/html/index.html"],
    }

    async def _fake_run_ssh_command(**kwargs):
        payload = {
            "ok": True,
            "path": "/var/www/html/index.html",
            "actual_occurrences": 1,
            "expected_occurrences": 1,
            "old_sha256": "old",
            "new_sha256": "new",
            "readback_sha256": "new",
            "changed": True,
            "verification": {"readback_sha256_matches": True},
        }
        return {
            "success": True,
            "output": {"stdout": json.dumps(payload), "stderr": "", "exit_code": 0},
            "error": None,
            "metadata": {},
        }

    monkeypatch.setattr(ssh_files.network, "run_ssh_command", _fake_run_ssh_command)

    result = asyncio.run(
        ssh_files.ssh_file_patch(
            target="root@192.168.1.63",
            path="/var/www/html/index.html",
            target_text="old",
            replacement_text="new",
            state=state,
        )
    )

    assert result["success"] is True
    assert ssh_files.REMOTE_MUTATION_VERIFICATION_KEY not in state.scratchpad


def test_ssh_file_patch_dry_run_resolves_source_artifact_precondition(monkeypatch) -> None:
    state = LoopState(cwd=".")
    state.scratchpad[ssh_files.REMOTE_MUTATION_VERIFICATION_KEY] = {
        "host": "192.168.1.63",
        "user": "root",
        "guessed_paths": ["/var/www/html/index.html"],
    }
    state.artifacts["A-ssh-read"] = ArtifactRecord(
        artifact_id="A-ssh-read",
        kind="artifact",
        source="/var/www/html/index.html",
        created_at="2026-05-01T00:00:00Z",
        size_bytes=0,
        summary="remote readback",
        metadata={
            "path": "/var/www/html/index.html",
            "sha256": "abc123",
        },
    )

    async def _fake_run_ssh_command(**kwargs):
        payload = {
            "ok": True,
            "path": "/var/www/html/index.html",
            "actual_occurrences": 1,
            "expected_occurrences": 1,
            "old_sha256": "abc123",
            "planned_new_sha256": "def456",
            "changed": True,
            "dry_run": True,
            "match_mode": "whitespace_normalized",
            "matched_region_previews": [{"preview": "root /var/www/demo-site;", "chars": 24, "truncated": False, "start": 10, "end": 34}],
            "verification": {"replacement_occurrences": 1, "target_occurrences_after": 0},
        }
        return {
            "success": True,
            "output": {"stdout": json.dumps(payload), "stderr": "", "exit_code": 0},
            "error": None,
            "metadata": {},
        }

    monkeypatch.setattr(ssh_files.network, "run_ssh_command", _fake_run_ssh_command)

    result = asyncio.run(
        ssh_files.ssh_file_patch(
            target="root@192.168.1.63",
            path="/var/www/html/index.html",
            target_text="\troot /var/www/demo-site;",
            replacement_text="\troot /srv/www/demo-site;",
            source_artifact_id="A-ssh-read",
            whitespace_normalized=True,
            dry_run=True,
            state=state,
        )
    )

    assert result["success"] is True
    assert result["metadata"]["dry_run"] is True
    assert result["metadata"]["resolved_expected_sha256"] == "abc123"
    assert result["metadata"]["source_artifact_id"] == "A-ssh-read"
    assert ssh_files.REMOTE_MUTATION_VERIFICATION_KEY in state.scratchpad


def test_task_complete_remote_mutation_block_surfaces_exact_ssh_file_read_hint() -> None:
    state = LoopState(cwd=".")
    state.scratchpad[ssh_files.REMOTE_MUTATION_VERIFICATION_KEY] = {
        "host": "192.168.1.63",
        "user": "root",
        "guessed_paths": ["/var/www/html/index.html"],
    }

    blocked = asyncio.run(task_complete("done", state=state, harness=None))

    assert blocked["success"] is False
    assert "ssh_file_read(host='192.168.1.63', user='root', path='/var/www/html/index.html')" in blocked["error"]
    assert blocked["metadata"]["next_required_action"]["required_arguments"] == {
        "host": "192.168.1.63",
        "user": "root",
        "path": "/var/www/html/index.html",
    }


def test_ssh_file_read_can_fail_on_max_size_when_truncation_disabled(monkeypatch) -> None:
    async def _fake_run_ssh_command(**kwargs):
        payload = {
            "ok": False,
            "error_kind": "max_size_exceeded",
            "path": "/var/www/html/index.html",
            "bytes": 5000,
            "max_bytes": 100,
            "message": "Remote file exceeded max_bytes and truncation was disabled.",
        }
        return {
            "success": True,
            "output": {"stdout": json.dumps(payload), "stderr": "", "exit_code": 0},
            "error": None,
            "metadata": {},
        }

    monkeypatch.setattr(ssh_files.network, "run_ssh_command", _fake_run_ssh_command)

    result = asyncio.run(
        ssh_files.ssh_file_read(
            target="root@192.168.1.63",
            path="/var/www/html/index.html",
            max_bytes=100,
            truncate=False,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "max_size_exceeded"


def test_raw_ssh_sed_mutation_blocks_task_complete() -> None:
    state = LoopState(cwd=".")
    state.acceptance_waived = True
    harness = SimpleNamespace(state=state)
    service = SimpleNamespace(harness=harness)

    tool_result_artifact_updates._record_remote_mutation_requirement(
        service,
        result=ToolEnvelope(success=True, output={"stdout": "ok", "exit_code": 0}),
        arguments={
            "host": "192.168.1.63",
            "command": (
                "sed -i 's/<style>[^<]*<\\/style>/<link rel=\"stylesheet\" "
                "href=\"\\/llm-explainer-theme.css\">/' /var/www/html/llm-explainer-page-001.html"
            ),
        },
    )

    result = asyncio.run(task_complete("done", state=state, harness=harness))

    assert result["success"] is False
    assert result["metadata"]["reason"] == "remote_mutation_requires_verification"
    assert "ssh_file_replace_between" in result["error"]


def test_795c2db6_regression_bad_grep_verifier_stays_blocked() -> None:
    state = LoopState(cwd=".")
    state.acceptance_waived = True
    harness = SimpleNamespace(state=state, _runlog=lambda *args, **kwargs: None)
    service = SimpleNamespace(harness=harness)

    tool_result_artifact_updates._record_remote_mutation_requirement(
        service,
        result=ToolEnvelope(success=True, output={"stdout": "ok", "exit_code": 0}),
        arguments={
            "host": "192.168.1.63",
            "command": (
                "sed -i 's/<style>[^<]*<\\/style>/<link rel=\"stylesheet\" "
                "href=\"\\/llm-explainer-theme.css\">/' /var/www/html/llm-explainer-page-001.html"
            ),
        },
    )
    tool_result_artifact_updates._handle_remote_mutation_verifier_result(
        service,
        result=ToolEnvelope(
            success=True,
            output={
                "stdout": "/var/www/html/llm-explainer-page-001.html\n",
                "stderr": "",
                "exit_code": 0,
            },
        ),
        arguments={
            "host": "192.168.1.63",
            "command": "grep -l '<link.*theme' /var/www/html/llm-explainer-page-001.html",
        },
    )

    result = asyncio.run(task_complete("done", state=state, harness=harness))

    assert result["success"] is False
    assert result["metadata"]["reason"] == "remote_mutation_requires_verification"
    assert "ssh_file_replace_between" in result["error"]


def test_ssh_file_replace_between_success_allows_completion() -> None:
    state = LoopState(cwd=".")
    state.acceptance_waived = True
    harness = SimpleNamespace(state=state, _runlog=lambda *args, **kwargs: None)
    service = SimpleNamespace(harness=harness)
    state.scratchpad[ssh_files.REMOTE_MUTATION_VERIFICATION_KEY] = {
        "host": "192.168.1.63",
        "guessed_paths": ["/var/www/html/llm-explainer-page-001.html"],
        "verification_profile": "html_stylesheet_swap",
        "verification_patterns": {
            "old_absent": ["<style>"],
            "new_present": ["llm-explainer-theme.css"],
        },
    }

    tool_result_artifact_updates._clear_remote_mutation_requirement_from_tool(
        service,
        tool_name="ssh_file_replace_between",
        result=ToolEnvelope(
            success=True,
            output={
                "path": "/var/www/html/llm-explainer-page-001.html",
                "verification": {
                    "readback_sha256_matches": True,
                    "replacement_occurrences": 1,
                },
            },
            metadata={
                "path": "/var/www/html/llm-explainer-page-001.html",
            },
        ),
        arguments={"path": "/var/www/html/llm-explainer-page-001.html"},
    )

    result = asyncio.run(task_complete("done", state=state, harness=harness))

    assert result["success"] is True


def test_weak_positive_only_ssh_verifier_does_not_clear_requirement() -> None:
    state = LoopState(cwd=".")
    harness = SimpleNamespace(state=state, _runlog=lambda *args, **kwargs: None)
    service = SimpleNamespace(harness=harness)
    state.scratchpad[ssh_files.REMOTE_MUTATION_VERIFICATION_KEY] = {
        "host": "192.168.1.63",
        "guessed_paths": ["/var/www/html/llm-explainer-page-001.html"],
        "verification_profile": "html_stylesheet_swap",
        "verification_patterns": {
            "old_absent": ["<style>"],
            "new_present": ["llm-explainer-theme.css"],
        },
    }

    tool_result_artifact_updates._handle_remote_mutation_verifier_result(
        service,
        result=ToolEnvelope(
            success=True,
            output={
                "stdout": "/var/www/html/llm-explainer-page-001.html\n",
                "stderr": "",
                "exit_code": 0,
            },
        ),
        arguments={
            "host": "192.168.1.63",
            "command": "grep -l '<link.*theme' /var/www/html/llm-explainer-page-001.html",
        },
    )

    assert ssh_files.REMOTE_MUTATION_VERIFICATION_KEY in state.scratchpad
    assert state.recent_messages[-1].metadata["recovery_kind"] == "remote_weak_replacement_verifier"


def test_strong_ssh_verifier_clears_requirement() -> None:
    state = LoopState(cwd=".")
    harness = SimpleNamespace(state=state, _runlog=lambda *args, **kwargs: None)
    service = SimpleNamespace(harness=harness)
    state.scratchpad[ssh_files.REMOTE_MUTATION_VERIFICATION_KEY] = {
        "host": "192.168.1.63",
        "guessed_paths": ["/var/www/html/llm-explainer-page-001.html"],
        "verification_profile": "html_stylesheet_swap",
        "verification_patterns": {
            "old_absent": ["<style>"],
            "new_present": ["llm-explainer-theme.css"],
        },
    }

    tool_result_artifact_updates._handle_remote_mutation_verifier_result(
        service,
        result=ToolEnvelope(
            success=True,
            output={
                "stdout": "/var/www/html/llm-explainer-page-001.html NO_STYLE HAS_LINK\n",
                "stderr": "",
                "exit_code": 0,
            },
        ),
        arguments={
            "host": "192.168.1.63",
            "command": (
                "for f in /var/www/html/llm-explainer-page-001.html; do "
                "grep -q '<style>' \"$f\" && printf 'HAS_STYLE ' || printf 'NO_STYLE '; "
                "grep -q 'rel=[\"'\"']stylesheet[\"'\"'].*llm-explainer-theme.css' \"$f\" "
                "&& printf 'HAS_LINK\\n' || printf 'NO_LINK\\n'; done"
            ),
        },
    )

    assert ssh_files.REMOTE_MUTATION_VERIFICATION_KEY not in state.scratchpad


def test_ssh_file_read_only_clears_requirement_when_content_satisfies_patterns() -> None:
    state = LoopState(cwd=".")
    harness = SimpleNamespace(state=state, _runlog=lambda *args, **kwargs: None)
    service = SimpleNamespace(harness=harness)
    state.scratchpad[ssh_files.REMOTE_MUTATION_VERIFICATION_KEY] = {
        "host": "192.168.1.63",
        "guessed_paths": ["/var/www/html/llm-explainer-page-001.html"],
        "verification_profile": "html_stylesheet_swap",
        "verification_patterns": {
            "old_absent": ["<style>"],
            "new_present": ["llm-explainer-theme.css"],
        },
    }

    tool_result_artifact_updates._clear_remote_mutation_requirement_from_tool(
        service,
        tool_name="ssh_file_read",
        result=ToolEnvelope(
            success=True,
            output={
                "path": "/var/www/html/llm-explainer-page-001.html",
                "content": '<link rel="stylesheet" href="/llm-explainer-theme.css">',
            },
            metadata={
                "path": "/var/www/html/llm-explainer-page-001.html",
            },
        ),
        arguments={"path": "/var/www/html/llm-explainer-page-001.html"},
    )

    assert ssh_files.REMOTE_MUTATION_VERIFICATION_KEY not in state.scratchpad


def test_failed_ssh_file_read_clears_remote_deletion_requirement() -> None:
    state = LoopState(cwd=".")
    harness = SimpleNamespace(state=state, _runlog=lambda *args, **kwargs: None)
    service = SimpleNamespace(harness=harness)
    state.scratchpad[ssh_files.REMOTE_MUTATION_VERIFICATION_KEY] = {
        "host": "192.168.1.89",
        "guessed_paths": ["/etc/nginx/sites-enabled/default.bak.20260502022009"],
        "mutation_type": "deletion",
    }

    tool_result_artifact_updates._clear_remote_mutation_requirement_from_tool(
        service,
        tool_name="ssh_file_read",
        result=ToolEnvelope(
            success=False,
            error="",
            metadata={
                "host": "192.168.1.89",
                "path": "/etc/nginx/sites-enabled/default.bak.20260502022009",
                "error_kind": "file_not_found",
                "message": "Remote file not found.",
            },
        ),
        arguments={
            "host": "192.168.1.89",
            "path": "/etc/nginx/sites-enabled/default.bak.20260502022009",
        },
    )

    assert ssh_files.REMOTE_MUTATION_VERIFICATION_KEY not in state.scratchpad


def test_remote_deletion_blocks_task_complete_with_deletion_specific_guidance() -> None:
    state = LoopState(cwd=".")
    state.acceptance_waived = True
    state.scratchpad[ssh_files.REMOTE_MUTATION_VERIFICATION_KEY] = {
        "host": "192.168.1.89",
        "guessed_paths": ["/etc/nginx/sites-enabled/default.bak.20260502022009"],
        "mutation_type": "deletion",
    }
    harness = SimpleNamespace(state=state)

    result = asyncio.run(task_complete("done", state=state, harness=harness))

    assert result["success"] is False
    assert result["metadata"]["reason"] == "remote_mutation_requires_verification"
    assert "Verify the target is gone with `ssh_file_read`" in result["error"]
    assert "`not found` / `no such file`" in result["error"]
    assert result["metadata"]["next_required_action"]["tool_names"] == ["ssh_file_read"]


def test_remote_mutation_requirement_ignores_dev_null_redirection() -> None:
    state = LoopState(cwd=".")
    harness = SimpleNamespace(state=state, _runlog=lambda *args, **kwargs: None)
    service = SimpleNamespace(harness=harness)

    tool_result_artifact_updates._record_remote_mutation_requirement(
        service,
        result=ToolEnvelope(success=True),
        arguments={
            "host": "192.168.1.89",
            "command": "apt list --upgradable 2>/dev/null | grep '/' | wc -l",
        },
    )

    assert ssh_files.REMOTE_MUTATION_VERIFICATION_KEY not in state.scratchpad


def test_ssh_file_read_clears_simple_path_based_remote_requirement() -> None:
    state = LoopState(cwd=".")
    harness = SimpleNamespace(state=state, _runlog=lambda *args, **kwargs: None)
    service = SimpleNamespace(harness=harness)
    state.scratchpad[ssh_files.REMOTE_MUTATION_VERIFICATION_KEY] = {
        "host": "192.168.1.89",
        "guessed_paths": ["/etc/apt/sources.list.d/guacamole.list"],
    }

    tool_result_artifact_updates._clear_remote_mutation_requirement_from_tool(
        service,
        tool_name="ssh_file_read",
        result=ToolEnvelope(
            success=True,
            output={
                "path": "/etc/apt/sources.list.d/guacamole.list",
                "content": "deb https://example.invalid stable main\n",
            },
            metadata={
                "host": "192.168.1.89",
                "path": "/etc/apt/sources.list.d/guacamole.list",
            },
        ),
        arguments={
            "host": "192.168.1.89",
            "path": "/etc/apt/sources.list.d/guacamole.list",
        },
    )

    assert ssh_files.REMOTE_MUTATION_VERIFICATION_KEY not in state.scratchpad


def test_sed_mutation_generates_verification_patterns() -> None:
    state = LoopState(cwd=".")
    harness = SimpleNamespace(state=state, _runlog=lambda *args, **kwargs: None)
    service = SimpleNamespace(harness=harness)

    tool_result_artifact_updates._record_remote_mutation_requirement(
        service,
        result=ToolEnvelope(success=True, output={"stdout": "", "exit_code": 0}),
        arguments={
            "host": "192.168.1.89",
            "command": "sed -i 's|/var/ww/demo-site|/var/www/demo-site|' /etc/nginx/sites-enabled/default",
        },
    )

    requirement = state.scratchpad.get(ssh_files.REMOTE_MUTATION_VERIFICATION_KEY)
    assert isinstance(requirement, dict)
    patterns = requirement.get("verification_patterns")
    assert isinstance(patterns, dict)
    assert "/var/ww/demo-site" in patterns["old_absent"]
    assert "/var/www/demo-site" in patterns["new_present"]


def test_ssh_file_read_clears_sed_mutation_when_patterns_satisfied() -> None:
    state = LoopState(cwd=".")
    harness = SimpleNamespace(state=state, _runlog=lambda *args, **kwargs: None)
    service = SimpleNamespace(harness=harness)
    state.scratchpad[ssh_files.REMOTE_MUTATION_VERIFICATION_KEY] = {
        "host": "192.168.1.89",
        "guessed_paths": ["/etc/nginx/sites-enabled/default"],
        "verification_patterns": {
            "old_absent": ["/var/ww/demo-site"],
            "new_present": ["/var/www/demo-site"],
        },
    }

    tool_result_artifact_updates._clear_remote_mutation_requirement_from_tool(
        service,
        tool_name="ssh_file_read",
        result=ToolEnvelope(
            success=True,
            output={
                "path": "/etc/nginx/sites-enabled/default",
                "content": "root /var/www/demo-site;\n",
            },
            metadata={
                "host": "192.168.1.89",
                "path": "/etc/nginx/sites-enabled/default",
            },
        ),
        arguments={"host": "192.168.1.89", "path": "/etc/nginx/sites-enabled/default"},
    )

    assert ssh_files.REMOTE_MUTATION_VERIFICATION_KEY not in state.scratchpad


def test_bounded_region_trap_nudge_emitted_once() -> None:
    state = LoopState(cwd=".")
    harness = SimpleNamespace(state=state, _runlog=lambda *args, **kwargs: None)
    service = SimpleNamespace(harness=harness)
    state.scratchpad[ssh_files.REMOTE_MUTATION_VERIFICATION_KEY] = {
        "host": "192.168.1.89",
        "guessed_paths": ["/etc/nginx/sites-enabled/default"],
    }

    tool_result_artifact_updates._maybe_emit_bounded_region_trap_nudge(
        service,
        tool_name="ssh_file_replace_between",
        result=ToolEnvelope(
            success=False,
            error="Remote bounded region was not found.",
            metadata={
                "host": "192.168.1.89",
                "path": "/etc/nginx/sites-enabled/default",
                "error_kind": "bounded_region_not_found",
            },
        ),
        arguments={"host": "192.168.1.89", "path": "/etc/nginx/sites-enabled/default"},
    )

    assert len(state.recent_messages) == 1
    message = state.recent_messages[-1]
    assert message.metadata["recovery_kind"] == "remote_bounded_region_trap"
    assert "ssh_file_read" in message.content
    assert "already applied this change" in message.content

    # Second identical call should be deduped
    tool_result_artifact_updates._maybe_emit_bounded_region_trap_nudge(
        service,
        tool_name="ssh_file_replace_between",
        result=ToolEnvelope(
            success=False,
            error="Remote bounded region was not found.",
            metadata={
                "host": "192.168.1.89",
                "path": "/etc/nginx/sites-enabled/default",
                "error_kind": "bounded_region_not_found",
            },
        ),
        arguments={"host": "192.168.1.89", "path": "/etc/nginx/sites-enabled/default"},
    )

    assert len(state.recent_messages) == 1



def test_build_remote_command_small_payload_uses_argv() -> None:
    payload = {"action": "write", "path": "/tmp/test.txt", "content": "small content"}
    command, stdin_payload = ssh_files._build_remote_command(payload)
    assert not command.endswith(" --stdin")
    assert stdin_payload is None


def test_build_remote_command_large_payload_uses_stdin() -> None:
    payload = {"action": "write", "path": "/tmp/test.txt", "content": "x" * (200 * 1024)}
    command, stdin_payload = ssh_files._build_remote_command(payload)
    assert command.endswith(" --stdin")
    assert stdin_payload is not None
    assert len(stdin_payload) > ssh_files._MAX_ARGV_PAYLOAD_SIZE


def test_build_remote_command_exact_threshold_payload() -> None:
    # Binary search for the largest content size that still fits via argv
    content = "x" * ssh_files._MAX_ARGV_PAYLOAD_SIZE
    low, high = 0, len(content)
    while low < high:
        mid = (low + high + 1) // 2
        payload = {"action": "write", "path": "/tmp/test.txt", "content": content[:mid]}
        _command, stdin_payload = ssh_files._build_remote_command(payload)
        if stdin_payload is None:
            low = mid
        else:
            high = mid - 1

    # 'low' is the largest size that still uses argv
    payload = {"action": "write", "path": "/tmp/test.txt", "content": content[:low]}
    command, stdin_payload = ssh_files._build_remote_command(payload)
    assert not command.endswith(" --stdin")
    assert stdin_payload is None

    # One more character should flip to stdin
    payload["content"] += "x"
    command, stdin_payload = ssh_files._build_remote_command(payload)
    assert command.endswith(" --stdin")
    assert stdin_payload is not None
