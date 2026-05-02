from __future__ import annotations

import asyncio
from pathlib import Path

from smallctl.state import ClaimRecord, LoopState
from smallctl.tools import fs, shell


def test_diagnosis_task_blocks_shell_exec_without_supported_claim(tmp_path: Path) -> None:
    async def _run() -> dict:
        state = LoopState(cwd=str(tmp_path))
        state.current_phase = "execute"
        state.scratchpad["_task_classification"] = "diagnosis_remediation"
        return await shell.shell_exec(command="echo hello", state=state, harness=None)

    result = asyncio.run(_run())

    assert result["success"] is False
    assert result["metadata"]["reason"] == "missing_supported_claim"
    assert "supported claim" in result["error"].lower()
    assert "`memory_update`" in result["error"]
    assert "restating the intended command do not count" in result["error"]


def test_diagnosis_task_blocks_file_write_without_supported_claim(tmp_path: Path) -> None:
    async def _run() -> dict:
        state = LoopState(cwd=str(tmp_path))
        state.current_phase = "repair"
        state.scratchpad["_task_classification"] = "diagnosis_remediation"
        return await fs.file_write(
            path=str(tmp_path / "example.txt"),
            content="hello\n",
            cwd=str(tmp_path),
            state=state,
        )

    result = asyncio.run(_run())

    assert result["success"] is False
    assert result["metadata"]["reason"] == "missing_supported_claim"


def test_diagnosis_task_blocks_file_patch_without_supported_claim(tmp_path: Path) -> None:
    async def _run() -> dict:
        state = LoopState(cwd=str(tmp_path))
        state.current_phase = "repair"
        state.scratchpad["_task_classification"] = "diagnosis_remediation"
        target = tmp_path / "example.txt"
        target.write_text("hello\n", encoding="utf-8")
        return await fs.file_patch(
            path=str(target),
            target_text="hello",
            replacement_text="goodbye",
            cwd=str(tmp_path),
            state=state,
        )

    result = asyncio.run(_run())

    assert result["success"] is False
    assert result["metadata"]["reason"] == "missing_supported_claim"


def test_diagnosis_task_blocks_ast_patch_without_supported_claim(tmp_path: Path) -> None:
    async def _run() -> dict:
        state = LoopState(cwd=str(tmp_path))
        state.current_phase = "repair"
        state.scratchpad["_task_classification"] = "diagnosis_remediation"
        target = tmp_path / "example.py"
        target.write_text("def run():\n    return 1\n", encoding="utf-8")
        return await fs.ast_patch(
            path=str(target),
            language="python",
            operation="insert_in_function",
            target={"function": "run", "position": "start"},
            payload={"statements": "value = 2"},
            cwd=str(tmp_path),
            state=state,
        )

    result = asyncio.run(_run())

    assert result["success"] is False
    assert result["metadata"]["reason"] == "missing_supported_claim"


def test_supported_claim_allows_risky_actions_for_diagnosis_task(tmp_path: Path) -> None:
    async def _run() -> tuple[dict, dict, dict, dict]:
        state = LoopState(cwd=str(tmp_path))
        state.current_phase = "execute"
        state.scratchpad["_task_classification"] = "diagnosis_remediation"
        state.reasoning_graph.claim_records.append(
            ClaimRecord(
                claim_id="C1",
                kind="causal",
                statement="The build fails because the test fixture is missing.",
                supporting_evidence_ids=["E1"],
                status="confirmed",
            )
        )
        shell_result = await shell.shell_exec(command="echo hello", state=state, harness=None)
        file_result = await fs.file_write(
            path=str(tmp_path / "example.txt"),
            content="hello\n",
            cwd=str(tmp_path),
            state=state,
        )
        patch_result = await fs.file_patch(
            path=str(tmp_path / "example.txt"),
            target_text="hello",
            replacement_text="goodbye",
            cwd=str(tmp_path),
            state=state,
        )
        python_target = tmp_path / "example.py"
        python_target.write_text("def run():\n    return 1\n", encoding="utf-8")
        ast_patch_result = await fs.ast_patch(
            path=str(python_target),
            language="python",
            operation="insert_in_function",
            target={"function": "run", "position": "start"},
            payload={"statements": "value = 2"},
            cwd=str(tmp_path),
            state=state,
        )
        return shell_result, file_result, patch_result, ast_patch_result

    (shell_result, file_result, patch_result, ast_patch_result) = asyncio.run(_run())

    assert shell_result["success"] is True
    assert file_result["success"] is True
    assert patch_result["success"] is True
    assert ast_patch_result["success"] is True
