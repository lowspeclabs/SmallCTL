from __future__ import annotations

import json

from smallctl.logging_utils import RunLogger


def test_run_logger_trace_id_is_written_to_all_channels(tmp_path) -> None:
    logger = RunLogger(tmp_path / "run")
    logger.set_trace_id("thread-1:7")

    logger.log("harness", "turn_start", "turn started")
    logger.log("tools", "dispatch_start", "tool started")
    logger.log("chat", "chunk", "chat chunk")
    logger.log("model_output", "model_token", "assistant token", token="ok")

    for channel in ("harness", "tools", "chat", "model_output"):
        row = json.loads((logger.run_dir / f"{channel}.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert row["trace_id"] == "thread-1:7"
