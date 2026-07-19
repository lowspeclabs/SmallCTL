from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

from smallctl.shell_utils import (
    is_read_only_shell_evidence_action,
    is_read_only_shell_segment,
)
from smallctl.state import LoopState
from smallctl.state_schema import WriteSession
from smallctl.tools.base import ToolSpec, build_tool_schema
from smallctl.tools.dispatcher import ToolDispatcher
from smallctl.tools.register import build_registry
from smallctl.tools.registry import ToolRegistry


# --- M2: unknown args reject mutating tools, warn visibly on read-only tools -


def _make_real_dispatcher(state: LoopState) -> ToolDispatcher:
    provider = SimpleNamespace(state=state, log=logging.getLogger("test.phase2"))
    registry = build_registry(provider)
    return ToolDispatcher(registry, state=state, phase="execute")


def test_m2_file_patch_unknown_dryrun_arg_rejected_without_mutation(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("hello world\n", encoding="utf-8")
    state = LoopState(cwd=str(tmp_path))
    dispatcher = _make_real_dispatcher(state)

    result = asyncio.run(
        dispatcher.dispatch(
            "file_patch",
            {
                "path": str(target),
                "target_text": "hello",
                "replacement_text": "goodbye",
                "dryrun": True,
            },
        )
    )

    assert result.success is False
    assert "dryrun" in str(result.error)
    assert result.metadata["validation_error"] == "schema_validation"
    kinds = [issue["kind"] for issue in result.metadata["validation_issues"]]
    assert "additional_property" in kinds
    assert target.read_text(encoding="utf-8") == "hello world\n"


def test_m2_mutating_tool_multiple_unknown_args_all_named(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("hello world\n", encoding="utf-8")
    state = LoopState(cwd=str(tmp_path))
    dispatcher = _make_real_dispatcher(state)

    result = asyncio.run(
        dispatcher.dispatch(
            "file_patch",
            {
                "path": str(target),
                "target_text": "hello",
                "replacement_text": "goodbye",
                "dryrun": True,
                "backup_file": True,
            },
        )
    )

    assert result.success is False
    assert "dryrun" in str(result.error)
    assert "backup_file" in str(result.error)
    assert target.read_text(encoding="utf-8") == "hello world\n"


def test_m2_file_patch_legit_dry_run_still_previews(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("hello world\n", encoding="utf-8")
    state = LoopState(cwd=str(tmp_path))
    dispatcher = _make_real_dispatcher(state)

    result = asyncio.run(
        dispatcher.dispatch(
            "file_patch",
            {
                "path": str(target),
                "target_text": "hello",
                "replacement_text": "goodbye",
                "dry_run": True,
            },
        )
    )

    assert result.success is True
    assert target.read_text(encoding="utf-8") == "hello world\n"


def test_m2_read_only_tool_unknown_arg_warns_visibly(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("some content\n", encoding="utf-8")
    state = LoopState(cwd=str(tmp_path))
    dispatcher = _make_real_dispatcher(state)

    result = asyncio.run(
        dispatcher.dispatch("file_read", {"path": str(target), "checksum": True})
    )

    assert result.success is True
    assert "checksum" in str(result.output)
    assert "Warning" in str(result.output)
    assert result.metadata["ignored_arguments"] == ["checksum"]


def test_m2_dispatcher_injected_write_session_path_not_flagged() -> None:
    captured: dict[str, object] = {}

    def _handler(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"success": True, "output": dict(kwargs), "error": None, "metadata": {}}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="file_write",
            description="write a file",
            schema=build_tool_schema(
                properties={
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "write_session_id": {"type": "string"},
                },
                required=["path", "content"],
            ),
            handler=_handler,
            risk="high",
        )
    )
    state = LoopState(cwd="/tmp")
    state.write_session = WriteSession(
        write_session_id="ws-1",
        write_target_path="target.py",
        status="open",
    )
    dispatcher = ToolDispatcher(registry, state=state, phase="execute")

    result = asyncio.run(
        dispatcher.dispatch(
            "file_write", {"content": "x = 1", "write_session_id": "ws-1"}
        )
    )

    assert result.success is True
    assert captured.get("path") == "target.py"
    assert "ignored_arguments" not in result.metadata
    assert result.metadata.get("repaired_write_session_path") is True


# --- M17: read-only shell classifier fails closed on docker/sed mutations ----


def test_m17_docker_mutating_subcommands_not_read_only() -> None:
    assert is_read_only_shell_evidence_action("docker system prune -af") is False
    assert is_read_only_shell_evidence_action("docker volume rm data") is False
    assert is_read_only_shell_evidence_action("docker volume prune") is False
    assert is_read_only_shell_evidence_action("docker network rm front") is False
    assert is_read_only_shell_evidence_action("docker rm app") is False
    assert is_read_only_shell_evidence_action("docker rmi img:1") is False
    assert is_read_only_shell_evidence_action("docker kill app") is False
    assert is_read_only_shell_evidence_action("docker stop app") is False
    assert is_read_only_shell_evidence_action("docker run alpine true") is False
    assert is_read_only_shell_evidence_action("docker exec app ls") is False
    assert is_read_only_shell_evidence_action("docker pull alpine") is False
    assert is_read_only_shell_evidence_action("podman system prune -af") is False
    assert is_read_only_shell_evidence_action("docker system df") is False


def test_m17_docker_read_only_subcommands_stay_read_only() -> None:
    assert is_read_only_shell_evidence_action("docker ps") is True
    assert is_read_only_shell_evidence_action("docker ps -a") is True
    assert is_read_only_shell_evidence_action("docker images") is True
    assert is_read_only_shell_evidence_action("docker inspect app") is True
    assert is_read_only_shell_evidence_action("docker logs --tail 50 app") is True
    assert is_read_only_shell_evidence_action("docker stats --no-stream") is True
    assert is_read_only_shell_evidence_action("docker version") is True
    assert is_read_only_shell_evidence_action("docker info") is True
    assert is_read_only_shell_evidence_action("docker volume ls") is True
    assert is_read_only_shell_evidence_action("docker volume inspect data") is True
    assert is_read_only_shell_evidence_action("docker network ls") is True
    assert is_read_only_shell_evidence_action("docker container ls") is True
    assert is_read_only_shell_evidence_action("docker image inspect img:1") is True
    assert is_read_only_shell_evidence_action("docker system info") is True
    assert is_read_only_shell_evidence_action("podman ps") is True


def test_m17_sed_script_injection_not_read_only() -> None:
    assert is_read_only_shell_evidence_action("sed -n '1e id' /etc/hostname") is False
    assert is_read_only_shell_evidence_action("sed -n 'w /tmp/pwn' file") is False
    assert is_read_only_shell_evidence_action("sed -n '1r /etc/passwd' file") is False
    assert is_read_only_shell_evidence_action("sed 's/foo/bar/w /tmp/out' file") is False
    assert is_read_only_shell_evidence_action("sed 's/foo/bar/e' file") is False
    assert is_read_only_shell_evidence_action("sed -i 's/foo/bar/' file") is False
    assert is_read_only_shell_evidence_action("sed -n -f script.sed file") is False
    assert is_read_only_shell_segment("sed -n '1,5p' file > /tmp/out") is False


def test_m17_sed_read_only_invocations_stay_read_only() -> None:
    assert is_read_only_shell_evidence_action("sed -n '1,5p' file") is True
    assert is_read_only_shell_evidence_action("sed -n -e '1,5p' file") is True
    assert is_read_only_shell_evidence_action("sed 's/foo/bar/' file") is True
    assert is_read_only_shell_evidence_action("sed -n '/error/p' app.log") is True
