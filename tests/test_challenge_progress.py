from __future__ import annotations

from pathlib import Path

from aho.run_baseline import generate_report
from smallctl.challenge_progress import (
    challenge_progress_report,
    record_code_change,
    record_verifier_result,
    redundant_verifier_block,
)
from smallctl.state import LoopState


def test_coding_verifier_pass_blocks_identical_reverify() -> None:
    state = LoopState()
    state.run_brief.original_task = (
        "Build a self-contained Python script at `./temp/example.py` "
        "and includes built-in unittest cases."
    )

    record_code_change(state, tool_name="file_write", path="./temp/example.py")
    record_verifier_result(
        state,
        tool_name="shell_exec",
        command="python3 ./temp/example.py -v",
        verifier_kind="run_target",
        verdict="pass",
        exit_code=0,
    )

    blocked = redundant_verifier_block(
        state,
        tool_name="shell_exec",
        arguments={"command": "python3 ./temp/example.py -v"},
    )

    assert blocked is not None
    assert blocked.metadata["reason"] == "redundant_verifier_after_pass"
    assert state.challenge_progress.redundant_verifier_count == 1


def test_code_change_resets_verified_after_last_change() -> None:
    state = LoopState()
    state.run_brief.original_task = "Build a self-contained Python script at `./temp/example.py`."

    record_code_change(state, tool_name="file_write", path="./temp/example.py")
    record_verifier_result(
        state,
        tool_name="shell_exec",
        command="python3 ./temp/example.py",
        verifier_kind="run_target",
        verdict="pass",
        exit_code=0,
    )
    assert state.challenge_progress.verified_after_last_change is True

    record_code_change(state, tool_name="file_patch", path="./temp/example.py")

    assert state.challenge_progress.verified_after_last_change is False
    assert state.challenge_progress.redundant_verifier_count == 0


def test_coding_no_change_loop_blocks_after_twenty_steps() -> None:
    state = LoopState()
    state.run_brief.original_task = "Build a self-contained Python script at `./temp/example.py`."

    record_code_change(state, tool_name="file_write", path="./temp/example.py")

    blocked = None
    for _ in range(21):
        blocked = redundant_verifier_block(state, tool_name="file_read", arguments={"path": "./temp/example.py"})

    assert blocked is not None
    assert blocked.metadata["reason"] == "coding_no_change_after_write_early_stop"
    assert state.challenge_progress.no_change_steps_after_write == 21


def test_challenge_progress_round_trips_through_loop_state() -> None:
    state = LoopState()
    state.run_brief.original_task = "Build a self-contained Python script at `./temp/example.py`."
    record_code_change(state, tool_name="file_write", path="./temp/example.py")

    restored = LoopState.from_dict(state.to_dict())

    assert restored.challenge_progress.task_category == "coding"
    assert restored.challenge_progress.code_change_count == 1
    assert restored.challenge_progress.last_code_change_paths == ["./temp/example.py"]


def test_baseline_report_includes_challenge_state_from_task_summary(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    log_dir.mkdir()
    (log_dir / "task_summary.json").write_text(
        """
{
  "challenge_progress": {
    "task_category": "coding",
    "phase": "finalize",
    "code_change_count": 1,
    "last_verifier_verdict": "pass",
    "last_verifier_command": "python3 ./temp/example.py",
    "verified_after_last_change": true,
    "redundant_verifier_count": 0
  }
}
""".strip(),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.md"

    generate_report(
        [
            {
                "challenge_id": "challenge-01",
                "status": "completed",
                "elapsed_sec": 1.0,
                "returncode": 0,
                "log_dir": str(log_dir),
                "log_review": {},
            }
        ],
        report_path,
    )

    report = report_path.read_text(encoding="utf-8")
    assert "- **Challenge state:**" in report
    assert "phase: `finalize`" in report
    assert "last verifier: `pass` `python3 ./temp/example.py`" in report
