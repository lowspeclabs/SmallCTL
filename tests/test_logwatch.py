from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGWATCH = ROOT / "temp" / "logwatch.py"


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
    assert "Parsed records: 2" in completed.stdout
    assert "Errors: 1" in completed.stdout


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
    assert "Parsed records: 1" in completed.stdout
    assert "Errors: 1" in completed.stdout
