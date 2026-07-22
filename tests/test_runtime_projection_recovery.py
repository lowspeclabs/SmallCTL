from types import SimpleNamespace

from smallctl.harness.tool_result_remote_mutation import (
    observe_runtime_projection_read,
    record_remote_mutation_provenance,
)
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState


def test_docker_read_of_old_host_patch_emits_runtime_projection_stale() -> None:
    state = LoopState()
    service = SimpleNamespace(harness=SimpleNamespace(state=state))
    record_remote_mutation_provenance(
        service,
        host="server",
        path="/srv/api/config.py",
        arguments={"target_text": "DEBUG = True", "replacement_text": "DEBUG = False"},
    )

    observe_runtime_projection_read(
        service,
        result=ToolEnvelope(
            success=True,
            output={"exit_code": 0, "stdout": "DEBUG = True\n"},
        ),
        arguments={
            "host": "server",
            "command": "docker exec api cat /app/config.py",
        },
    )

    recovery = state.scratchpad["_runtime_projection_stale"]
    assert recovery["recovery_kind"] == "runtime_projection_stale"
    assert recovery["service"] == "api"
    assert "Recreate only" in recovery["guidance"]
    assert state.recent_messages[-1].metadata["recovery_kind"] == "runtime_projection_stale"
