"""Snapshot-style safety tests for prompts.py before refactoring.

These tests capture current prompt output behavior so future refactors
can prove parity.
"""
from __future__ import annotations

from smallctl.prompts import build_system_prompt
from smallctl.state import LoopState


def _make_state(model_name: str = "") -> LoopState:
    state = LoopState()
    if model_name:
        state.scratchpad["_model_name"] = model_name
    return state


class TestSystemPromptSnapshots:
    """Verify prompt content for different model configurations."""

    def test_large_model_includes_structured_reasoning(self) -> None:
        state = _make_state("qwen3:32b")
        prompt = build_system_prompt(state, "execute")
        assert "### STRUCTURED REASONING" in prompt
        assert "OBSERVE" in prompt
        assert "ORIENT" in prompt
        assert "DECIDE" in prompt
        assert "ACT" in prompt
        assert "VERIFY" in prompt

    def test_small_model_omits_structured_reasoning(self) -> None:
        state = _make_state("qwen3.5:4b")
        prompt = build_system_prompt(state, "execute")
        assert "### STRUCTURED REASONING" not in prompt

    def test_all_models_include_tool_call_format(self) -> None:
        for model in ["qwen3:32b", "qwen3.5:4b", "gemma-4-e2b-it"]:
            state = _make_state(model)
            prompt = build_system_prompt(state, "execute")
            assert 'TOOL CALL FORMAT: If tools are available' in prompt
            assert 'JSON format' in prompt

    def test_all_models_include_task_goal(self) -> None:
        for model in ["qwen3:32b", "qwen3.5:4b", "gemma-4-e2b-it"]:
            state = _make_state(model)
            prompt = build_system_prompt(state, "execute")
            assert "task goal" in prompt.lower() or "primary obligation" in prompt.lower()

    def test_all_models_include_phase_info(self) -> None:
        state = _make_state("qwen3:32b")
        prompt = build_system_prompt(state, "execute")
        assert "Phase: execute" in prompt
        assert "Active tool profiles: core" in prompt

    def test_small_model_includes_patch_verbatim_rule(self) -> None:
        state = _make_state("qwen3.5:4b")
        prompt = build_system_prompt(state, "execute")
        assert "PATCH VERBATIM RULE" in prompt

    def test_small_model_tool_routing_card_present(self) -> None:
        state = _make_state("qwen3.5:4b")
        prompt = build_system_prompt(state, "execute", available_tool_names=["shell_exec", "ssh_exec"])
        assert "Remote host/IP/user/password mentioned means" in prompt
        assert "`shell_exec` is local-only" in prompt

    def test_gemma_model_response_structure(self) -> None:
        state = _make_state("google_gemma-4-e2b-it")
        prompt = build_system_prompt(state, "execute")
        assert "Start EVERY response with a <think> block" in prompt

    def test_small_gemma_format_notes(self) -> None:
        state = _make_state("gemma-4-e2b-it")
        prompt = build_system_prompt(state, "execute")
        assert "SMALL GEMMA-4 FORMAT" in prompt
        assert "Example:" in prompt
        assert '"name":"ssh_exec"' in prompt

    def test_large_gemma_26b_contract(self) -> None:
        state = _make_state("google_gemma-4-26b-a4b-it")
        prompt = build_system_prompt(state, "execute")
        assert "STRICT: NEVER use text-based tool tags" in prompt
        assert "CONCISENESS: Do not paste long tool output" in prompt
        assert "ANTI-LOOP RULE" in prompt
        assert "Do not restart from the beginning" in prompt

    def test_small_model_includes_redundancy_rules(self) -> None:
        state = _make_state("qwen3.5:4b")
        prompt = build_system_prompt(state, "execute")
        assert "REDUNDANCY: Prefer the compressed summary" in prompt

    def test_large_model_includes_artifact_completeness(self) -> None:
        state = _make_state("qwen3:32b")
        prompt = build_system_prompt(state, "execute")
        assert "ARTIFACT COMPLETENESS" in prompt

    def test_small_model_includes_artifact_paging(self) -> None:
        state = _make_state("qwen3.5:4b")
        prompt = build_system_prompt(state, "execute")
        assert "ARTIFACT PAGING" in prompt

    def test_all_models_include_workspace_paths(self) -> None:
        for model in ["qwen3:32b", "qwen3.5:4b"]:
            state = _make_state(model)
            prompt = build_system_prompt(state, "execute")
            assert "WORKSPACE" in prompt

    def test_small_model_includes_privileges(self) -> None:
        state = _make_state("qwen3.5:4b")
        prompt = build_system_prompt(state, "execute")
        assert "PRIVILEGES: Do not invent or guess a sudo password" in prompt

    def test_all_models_include_secret_handling_rule(self) -> None:
        for model in ["qwen3:32b", "qwen3.5:4b"]:
            state = _make_state(model)
            prompt = build_system_prompt(state, "execute")
            assert ".netrc" in prompt
            assert "SECRET HANDLING" in prompt
            assert "do not echo them" in prompt

    def test_repair_phase_includes_failure_class(self) -> None:
        state = _make_state("qwen3:32b")
        state.last_failure_class = "syntax_error"
        state.scratchpad["_contract_phase"] = "repair"
        prompt = build_system_prompt(state, "execute")
        assert "REPAIR FOCUS" in prompt
        assert "failure class: syntax_error" in prompt

    def test_step_budget_prompt_included_when_set(self) -> None:
        state = _make_state("qwen3:32b")
        state.scratchpad["_graph_steps_remaining"] = 5
        state.scratchpad["_graph_recursion_limit"] = 10
        prompt = build_system_prompt(state, "execute")
        assert "STEP BUDGET" in prompt
        assert "5 graph steps remaining" in prompt

    def test_stderr_circuit_breaker_included_when_set(self) -> None:
        state = _make_state("qwen3:32b")
        state.scratchpad["_stderr_signature_circuit_breaker"] = {"signature": "foo"}
        prompt = build_system_prompt(state, "execute")
        assert "STDERR CIRCUIT BREAKER" in prompt

    def test_local_artifact_task_included_when_set(self) -> None:
        state = _make_state("qwen3:32b")
        state.task_mode = "remote_execute"
        state.run_brief.original_task = "SSH to host and save report to ./temp/report.txt"
        prompt = build_system_prompt(state, "execute")
        assert "LOCAL ARTIFACT TASK" in prompt

    def test_no_hallucinations_in_all_models(self) -> None:
        for model in ["qwen3:32b", "qwen3.5:4b", "gemma-4-e2b-it"]:
            state = _make_state(model)
            prompt = build_system_prompt(state, "execute")
            assert "No hallucinations" in prompt

    def test_task_complete_format_in_all_models(self) -> None:
        for model in ["qwen3:32b", "qwen3.5:4b"]:
            state = _make_state(model)
            prompt = build_system_prompt(state, "execute")
            assert "task_complete" in prompt

    def test_all_models_include_deliverable_verification(self) -> None:
        for model in ["qwen3:32b", "qwen3.5:4b", "gemma-4-e2b-it", "google_gemma-4-26b-a4b-it"]:
            state = _make_state(model)
            prompt = build_system_prompt(state, "execute")
            assert "DELIVERABLE VERIFICATION" in prompt
            assert "verify every file, path, or artifact" in prompt

    def test_all_models_include_docker_inspect_hint(self) -> None:
        for model in ["qwen3:32b", "qwen3.5:4b", "gemma-4-e2b-it", "google_gemma-4-26b-a4b-it"]:
            state = _make_state(model)
            prompt = build_system_prompt(state, "execute")
            assert "DOCKER INSPECT HINT" in prompt, f"missing header for {model}"
            assert "NetworkSettings.Ports" in prompt, f"missing NetworkSettings.Ports for {model}"
            assert "`.PortMappings` is not a valid key" in prompt, f"missing PortMappings note for {model}: {prompt[:500]}..."

    def test_all_models_include_secret_handling(self) -> None:
        for model in ["qwen3:32b", "qwen3.5:4b", "gemma-4-e2b-it", "google_gemma-4-26b-a4b-it"]:
            state = _make_state(model)
            prompt = build_system_prompt(state, "execute")
            assert "SECRET HANDLING" in prompt, f"missing secret handling for {model}"
            assert "`.netrc`" in prompt, f"missing .netrc mention for {model}"
            assert "NEVER read `.env`" not in prompt, f"env read gate still present for {model}"
            assert "do not echo them" in prompt, f"missing echo prohibition for {model}"
            # Should be near the top of the prompt, before most other directives
            system_lines = [line.strip() for line in prompt.split("  ") if line.strip()]
            secret_idx = next(i for i, line in enumerate(system_lines) if line.startswith("SECRET HANDLING"))
            assert secret_idx < 5, f"secret handling too late in prompt for {model}: index {secret_idx}"

    def test_all_models_include_installer_timeout_recovery(self) -> None:
        for model in ["qwen3:32b", "qwen3.5:4b", "gemma-4-e2b-it", "google_gemma-4-26b-a4b-it"]:
            state = _make_state(model)
            prompt = build_system_prompt(state, "execute")
            assert "INSTALLER TIMEOUT RECOVERY" in prompt
            assert "larger `timeout_sec`" in prompt
