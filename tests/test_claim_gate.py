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


def test_supported_claim_allows_risky_actions_for_diagnosis_task(tmp_path: Path) -> None:
    async def _run() -> tuple[dict, dict]:
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
        return shell_result, file_result, patch_result

    shell_result, file_result, patch_result = asyncio.run(_run())

    assert shell_result["success"] is True
    assert file_result["success"] is True
    assert patch_result["success"] is True
