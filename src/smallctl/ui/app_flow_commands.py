from __future__ import annotations

from typing import Any

from ..models.events import UIEvent, UIEventType


async def handle_slash_command(app: Any, task: str) -> bool:
    harness = app.harness
    command = task.strip().lower()
    if harness is None:
        return False
    if command == "/plan-mode":
        bridge = getattr(app, "_harness_bridge", None)
        if bridge is not None:
            snapshot = await bridge.set_planning_mode(True)
        else:
            setter = getattr(harness, "set_planning_mode", None)
            if callable(setter):
                snapshot = setter(True)
            else:
                harness.state.planning_mode_enabled = True
                harness.state.planner_resume_target_mode = "loop"
                harness.state.touch()
                snapshot = app._capture_status_snapshot_from_harness()
        app._set_activity("planning mode active")
        app._refresh_status(snapshot=snapshot)
        await app._append_system_line("Planning mode enabled.", force=True)
        await app.on_harness_event(
            UIEvent(
                event_type=UIEventType.ALERT,
                content="Planning mode enabled.",
                data={"status_activity": "planning mode active"},
            )
        )
        return True
    if command == "/exit-plan-mode":
        pending = harness.get_pending_interrupt() or {}
        if str(pending.get("kind") or "") == "plan_execute_approval":
            await app._append_system_line(
                "Planning mode is awaiting approval; finish that prompt before exiting.",
                force=True,
            )
            return True
        bridge = getattr(app, "_harness_bridge", None)
        if bridge is not None:
            snapshot = await bridge.set_planning_mode(False)
        else:
            setter = getattr(harness, "set_planning_mode", None)
            if callable(setter):
                snapshot = setter(False)
            else:
                harness.state.planning_mode_enabled = False
                harness.state.planner_requested_output_path = ""
                harness.state.planner_requested_output_format = ""
                harness.state.touch()
                snapshot = app._capture_status_snapshot_from_harness()
        app._set_activity("planning mode off")
        app._refresh_status(snapshot=snapshot)
        await app._append_system_line("Planning mode disabled.", force=True)
        await app.on_harness_event(
            UIEvent(
                event_type=UIEventType.ALERT,
                content="Planning mode disabled.",
                data={"status_activity": "planning mode off"},
            )
        )
        return True
    return False
