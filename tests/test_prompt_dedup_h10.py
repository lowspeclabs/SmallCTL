"""Regression tests for H10: system prompt guidance block must not be duplicated.

Previously the NETWORK/SHELL/ARTIFACTS/WEB RESEARCH/REMOTE FILES guidance block
was emitted twice for large models (an else-branch copy plus an all-models copy),
and the else-branch copy carried an SSH_EXEC example missing its opening brace,
so 9-14B models received two contradictory SSH JSON examples.
"""
from __future__ import annotations

import json
import re

import pytest

from smallctl.prompts import build_system_prompt
from smallctl.state import LoopState

_TOOLS = [
    "shell_exec",
    "ssh_exec",
    "ssh_file_read",
    "ssh_file_write",
    "ssh_file_patch",
    "ssh_file_replace_between",
    "artifact_read",
    "artifact_grep",
    "web_search",
    "web_fetch",
    "file_read",
    "file_write",
    "task_complete",
]

_MODELS = {
    "large": "qwen3:32b",
    "small": "qwen3.5:4b",
    "mid_12b": "gemma-4-12b",
}

_GUIDANCE_BLOCK_PREFIXES = [
    "NETWORK: Use `ssh_exec`",
    "SHELL: For long-running commands",
    "ARTIFACTS: Use `artifact_read(",
    "WEB RESEARCH: Use `web_search`",
    "REMOTE FILES: Prefer typed SSH file tools",
]

_BACKTICK_JSON_RE = re.compile(r"`(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})`")


def _build_prompt(model_name: str) -> str:
    state = LoopState()
    state.scratchpad["_model_name"] = model_name
    return build_system_prompt(state, "execute", available_tool_names=list(_TOOLS))


def _ssh_json_examples(prompt: str) -> list[str]:
    examples: list[str] = []
    for match in _BACKTICK_JSON_RE.finditer(prompt):
        candidate = match.group(1)
        if '"host"' not in candidate and "ssh_exec" not in candidate:
            continue
        context = prompt[max(0, match.start() - 60):match.start()]
        if "INVALID" in context:
            continue
        examples.append(candidate)
    return examples


class TestPromptDedupH10:
    @pytest.mark.parametrize("model_label", sorted(_MODELS))
    def test_guidance_blocks_occur_exactly_once(self, model_label: str) -> None:
        prompt = _build_prompt(_MODELS[model_label])
        for prefix in _GUIDANCE_BLOCK_PREFIXES:
            count = prompt.count(prefix)
            assert count == 1, (
                f"{model_label} model ({_MODELS[model_label]}): guidance block "
                f"{prefix!r} occurs {count} times, expected exactly 1"
            )

    @pytest.mark.parametrize("model_label", sorted(_MODELS))
    def test_ssh_json_examples_are_valid_json(self, model_label: str) -> None:
        prompt = _build_prompt(_MODELS[model_label])
        examples = _ssh_json_examples(prompt)
        for example in examples:
            try:
                json.loads(example)
            except json.JSONDecodeError as exc:
                pytest.fail(
                    f"{model_label} model ({_MODELS[model_label]}): SSH JSON "
                    f"example is malformed: {example!r} ({exc})"
                )

    @pytest.mark.parametrize("model_label", ["small", "mid_12b"])
    def test_small_model_routing_card_has_exactly_one_ssh_example(self, model_label: str) -> None:
        prompt = _build_prompt(_MODELS[model_label])
        assert prompt.count("SMALL MODEL TOOL ROUTING:") == 1
        assert prompt.count("SSH_EXEC EXAMPLE:") == 1
        examples = [
            e for e in _ssh_json_examples(prompt)
            if e.startswith('{"host"')
        ]
        assert len(examples) == 1
        parsed = json.loads(examples[0])
        assert set(parsed) >= {"host", "user", "password", "command"}
