from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..logging_utils import log_kv
from ..models.events import UIEvent, UIEventType
from .input import InputPane


async def handle_input_pane_submitted(app: Any, event: InputPane.Submitted) -> None:
    task = event.value.strip()
    input_widget = app.query_one(InputPane)
    input_widget.text = ""
    if not task:
        return
    if task.startswith("/"):
        handled = await app._handle_slash_command(task)
        if handled:
            return
    if app.active_task and not app.active_task.done():
        console = app._get_console()
        if console is not None:
            await app._append_system_line("Task already running.")
        return
    app.task_history.append(task)
    app.history_index = len(app.task_history)
    console = app._get_console()
    if console is not None:
        app._pending_user_echo = task
        await console.append_event(
            UIEvent(event_type=UIEventType.USER, content=task)
        )
        await console.begin_assistant_turn()
    else:
        app._pending_user_echo = None
    app._record_chat_session_prompt(task)
    app._set_activity("[thinking...]")
    app._refresh_status(step_override="running")
    app.active_task = asyncio.create_task(app._run_harness_task(task))
    log_kv(app._app_logger, logging.INFO, "ui_task_started", task=task)
