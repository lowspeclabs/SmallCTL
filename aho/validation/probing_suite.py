import argparse
import asyncio
import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Any

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.smallctl.harness import Harness
from src.smallctl.state import LoopState
from src.smallctl.logging_utils import create_run_logger
from src.smallctl.tools.dispatcher import ToolDispatcher

def get_harness(temp_dir: str, fama: bool = True, loop_guard: bool = True) -> Harness:
    run_logger = create_run_logger(f"{temp_dir}/logs")
    harness = Harness(
        endpoint="http://localhost:8080/v1",
        model="mock-model",
        max_prompt_tokens=4000,
        context_limit=4096,
        chat_endpoint="/chat/completions",
        api_key="local-key",
        use_ansible=False,
        run_logger=run_logger,
        summarize_at_ratio=0.8,
        strategy={
            "max_steps": 10,
            "tool_call_format": "strict_xml",
            "required_tool_calls": [],
        },
        strategy_prompt="You are a helpful assistant.",
        provider_profile="lmstudio",
        fama_enabled=fama,
        loop_guard_enabled=loop_guard,
    )
    harness.state.cwd = temp_dir
    return harness

# Probe 1: Path Traversal Probe
async def test_path_traversal():
    print("Running Path Traversal Probe...")
    with tempfile.TemporaryDirectory() as tmpdir:
        harness = get_harness(tmpdir)
        from src.smallctl.tools.fs import file_write
        from src.smallctl.tools.http import file_download
        
        res = await file_write(path="/temp/test.txt", content="secret", state=harness.state)
        assert not res.get("success"), "Path traversal check should have failed for suspicious root temp path"
        assert "suspicious_temp_root_path" in res.get("metadata", {}).get("error_kind", ""), f"Unexpected metadata: {res}"

        res = await file_download(
            url="http://127.0.0.1:9/no-network-call-required",
            output_path="../../outside.txt",
            state=harness.state,
        )
        assert not res.get("success"), "file_download should reject traversal before HTTP fetch"
        assert res.get("metadata", {}).get("error_kind") == "workspace_path_traversal", f"Unexpected metadata: {res}"
        print("  - Path Traversal Probe Passed (workspace escapes correctly blocked)")

# Probe 2: Done-Gate Escaper Test
async def test_done_gate_escaper():
    print("Running Done-Gate Escaper Test...")
    with tempfile.TemporaryDirectory() as tmpdir:
        harness = get_harness(tmpdir)
        harness.state.cwd = tmpdir
        
        # Force a done gate constraint: verifier verdict is "fail"
        harness.state.last_verifier_verdict = {"verdict": "fail", "message": "Tests are failing"}
        harness.state.planning_mode_enabled = False
        
        from src.smallctl.tools.control import task_complete, task_fail
        
        res = await task_complete(message="Done", state=harness.state, harness=harness)
        assert not res.get("success"), f"task_complete should have been blocked by verifier failure. (res={res})"
        print("  - Done-Gate Escaper Probe: task_complete successfully blocked.")

        res_fail = await task_fail(message="Unresolvable failures", state=harness.state)
        assert res_fail.get("success"), f"task_fail should be allowed even if verification fails. (res={res_fail})"
        print("  - Done-Gate Escaper Probe Passed")

# Probe 3: Stale Read Repair Probe
async def test_stale_read_repair():
    print("Running Stale Read Repair Probe...")
    with tempfile.TemporaryDirectory() as tmpdir:
        harness = get_harness(tmpdir)
        harness.state.cwd = tmpdir
        
        # Configure state to be in a repair cycle
        harness.state.repair_cycle_id = "repair-cycle-123"
        harness.state.files_changed_this_cycle = []
        
        from src.smallctl.tools.fs import file_write
        
        target_file = Path(tmpdir) / "test.py"
        target_file.write_text("initial content")
        
        res = await file_write(path="test.py", content="updated content", cwd=tmpdir, state=harness.state)
        assert not res.get("success"), "File write should be blocked in repair cycle before reading it"
        assert "reading the target file before patching" in res.get("error", ""), f"Unexpected error message: {res.get('error')}"
        print("  - Stale Read Repair Probe Passed")

# Probe 4: Compaction Overflow Probe
async def test_compaction_overflow():
    print("Running Compaction Overflow Probe...")
    with tempfile.TemporaryDirectory() as tmpdir:
        harness = get_harness(tmpdir)
        large_content = "X" * 10240
        from src.smallctl.models.conversation import ConversationMessage
        from src.smallctl.state_memory import trim_recent_messages
        
        messages = [
            ConversationMessage(role="system", content="You are a coding assistant."),
            ConversationMessage(role="user", content="Run analysis."),
            ConversationMessage(role="assistant", content="I will run the weather tool.", tool_calls=[{"id": "1", "type": "function", "function": {"name": "weather_lookup", "arguments": "{}"}}]),
            ConversationMessage(role="tool", tool_call_id="1", content=large_content)
        ]
        
        compacted = trim_recent_messages(messages, limit=2)
        assert len(compacted) <= 2, f"Expected message list to be trimmed to 2, got {len(compacted)}"
        print(f"  - Compaction Overflow Probe Passed (original size: {len(messages)}, compacted size: {len(compacted)})")

async def main_async(args: argparse.Namespace):
    print("==================================================")
    print("Running AHO Boundary Probing Suite...")
    if args.mock_llm:
        print("Mode: MOCK-LLM (no live API calls)")
    print("==================================================")
    try:
        await test_path_traversal()
        await test_done_gate_escaper()
        await test_stale_read_repair()
        await test_compaction_overflow()
        print("\n[SUCCESS] All Probing Suite tests completed successfully!")
    except AssertionError as e:
        print(f"\n[FAILURE] Assertion failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="AHO Boundary Probing Suite")
    parser.add_argument("--mock-llm", action="store_true", help="Run in mock-LLM mode without calling live APIs")
    args = parser.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
