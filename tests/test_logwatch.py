from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGWATCH = ROOT / "logwatch.py"


def test_logwatch_accepts_run_directory_input(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "harness.log").write_text(
        "\n".join(
            [
                "2026-04-11T17:46:21.431+00:00 stream_chunk_error sub-4b write timeout {'ignored': 'not json'}",
                "2026-04-11T17:47:24.297+00:00 stream_text_write_fallback_succeeded converted no-tools write response into synthetic file_write "
                '{"write_session_id": "ws_1", "target_path": "./temp/task_queue.py"}',
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(LOGWATCH), str(run_dir)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    logwatch_output = (run_dir / "harness.logwatch.log").read_text(encoding="utf-8")
    assert "Parsed records: 2" in logwatch_output
    assert "Errors: 1" in logwatch_output


def test_logwatch_accepts_log_file_input(tmp_path: Path) -> None:
    log_path = tmp_path / "harness.log"
    log_path.write_text(
        "2026-04-11T17:59:11.294+00:00 task_finalize task finished "
        '{"result": {"status": "failed", "error": {"message": "500 error"}}}',
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(LOGWATCH), "--log-file", str(log_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    logwatch_output = log_path.with_suffix(".logwatch.log").read_text(encoding="utf-8")
    assert "Parsed records: 1" in logwatch_output
    assert "Errors: 1" in logwatch_output


def test_logwatch_rca_checklist_uses_task_summary_and_counts_stalls(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "task_summary.json").write_text(
        '{"total_tool_calls": 10, "final_status": "Completed"}',
        encoding="utf-8",
    )
    (run_dir / "harness.log").write_text(
        "\n".join(
            [
                "2026-05-05T14:13:58.000+00:00 action_stall improper tool format {}",
                "2026-05-05T14:13:59.000+00:00 no_tool_recovery injected nudge {}",
                "2026-05-05T14:14:00.000+00:00 inline_tool_call_recovered_from_thinking recovered {}",
                "2026-05-05T14:14:01.000+00:00 thinking_tool_protocol_sanitized sanitized {}",
                "2026-05-05T14:14:02.000+00:00 task_complete_remote_mutation_verifier_autocontinue scheduled {}",
            ]
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(LOGWATCH), str(run_dir)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    logwatch_output = (run_dir / "harness.logwatch.log").read_text(encoding="utf-8")
    assert "Canonical task_summary total_tool_calls: 10" in logwatch_output
    assert "RCA checklist: action_stall events: 1" in logwatch_output
    assert "inline thinking tool recoveries: 1 sanitized: 1" in logwatch_output
    assert "harness auto-scheduled remote verifiers: 1" in logwatch_output
