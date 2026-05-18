from __future__ import annotations

from smallctl.graph.solver_refine import (
    SolverRefineResult,
    build_critique_prompt,
    parse_critique_response,
)


def test_build_critique_prompt_contains_checks() -> None:
    prompt = build_critique_prompt(draft="do X", observations_text="found Y")
    assert "TOOL PLAN OBSERVATIONS" in prompt
    assert "SOLVER DRAFT" in prompt
    assert "pass" in prompt
    assert "revise" in prompt
    assert "block" in prompt


def test_build_critique_prompt_includes_subtask_and_signals() -> None:
    prompt = build_critique_prompt(
        draft="d",
        observations_text="o",
        active_subtask="sub-1",
        verifier_signals={"ok": True},
    )
    assert "ACTIVE SUBTASK" in prompt
    assert "sub-1" in prompt
    assert "VERIFIER SIGNALS" in prompt


def test_build_critique_prompt_prefers_rewoo_context_frame() -> None:
    prompt = build_critique_prompt(
        draft="claim",
        observations_text="legacy observations",
        context_frame=(
            "REWOO PLAN STATE\n"
            "Acceptance criteria:\n- tests pass\n\n"
            "REWOO EVIDENCE\n"
            "- TP-E1-E2 [failed] file_read; summary=missing execution record"
        ),
    )
    assert "REWOO CONTEXT FRAME" in prompt
    assert "tests pass" in prompt
    assert "missing execution record" in prompt
    assert "legacy observations" not in prompt


def test_parse_critique_response_pass() -> None:
    text = '{"verdict": "pass", "issues": [], "revised_output": ""}'
    result = parse_critique_response(text)
    assert isinstance(result, SolverRefineResult)
    assert result.verdict == "pass"
    assert result.issues == []


def test_parse_critique_response_revise() -> None:
    text = '{"verdict": "revise", "issues": ["typo"], "revised_output": "fixed text"}'
    result = parse_critique_response(text)
    assert result.verdict == "revise"
    assert result.issues == ["typo"]
    assert result.revised_output == "fixed text"


def test_parse_critique_response_block() -> None:
    text = '{"verdict": "block", "issues": ["unsafe"], "revised_output": ""}'
    result = parse_critique_response(text)
    assert result.verdict == "block"
    assert result.issues == ["unsafe"]


def test_parse_critique_response_fenced_json() -> None:
    text = '```json\n{"verdict": "pass", "issues": [], "revised_output": ""}\n```'
    result = parse_critique_response(text)
    assert result is not None
    assert result.verdict == "pass"


def test_parse_critique_response_invalid_verdict() -> None:
    text = '{"verdict": "maybe", "issues": [], "revised_output": ""}'
    assert parse_critique_response(text) is None


def test_parse_critique_response_malformed() -> None:
    assert parse_critique_response("not json") is None
    assert parse_critique_response("") is None
