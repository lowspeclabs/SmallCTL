from __future__ import annotations

import asyncio
from typing import Any

from ..logging_utils import log_kv
from ..models.events import UIEvent, UIEventType
from .approval import PlanApprovalDecision
from .approval import PlanApprovalScreen
from .approval import ApprovePromptScreen
from .approval import ShellApprovalDecision
from .approval import SudoPasswordPromptScreen


async def maybe_handle_plan_approval_result(app, result: dict[str, Any]) -> dict[str, Any]:
    harness = app.harness
    if harness is None:
        return result
    if str(result.get("status") or "") != "needs_human":
        return result
    interrupt = result.get("interrupt")
    if not isinstance(interrupt, dict):
        return result
    if str(interrupt.get("kind") or "") != "plan_execute_approval":
        return result

    decision = await prompt_for_plan_approval(app, interrupt)
    if decision is None:
        return result
    bridge = getattr(app, "_harness_bridge", None)
    if bridge is not None:
        return await bridge.resume(decision.choice)
    return await harness.resume_task_with_events(decision.choice, app.on_harness_event)


async def handle_approval_prompt(app, event: UIEvent) -> None:
    harness = app.harness
    if harness is None:
        return
    approval_id = str(event.data.get("approval_id") or "").strip()
    command = str(event.data.get("command") or "").strip()
    cwd = str(event.data.get("cwd") or "").strip()
    timeout_raw = event.data.get("timeout_sec", 30)
    try:
        timeout_sec = int(timeout_raw)
    except (TypeError, ValueError):
        timeout_sec = 30
    proof_bundle = event.data.get("proof_bundle")
    if not isinstance(proof_bundle, dict):
        proof_bundle = {}

    prompt = ApprovePromptScreen(
        approval_id=approval_id or "pending",
        command=command or "(empty command)",
        cwd=cwd or harness.state.cwd,
        timeout_sec=max(1, timeout_sec),
        proof_bundle=proof_bundle,
    )
    app._active_approval_prompt = prompt
    app._refresh_status()

    approved = False
    remember_session = False
    runlog = getattr(harness, "_runlog", None)
    prompt_timeout_sec = max(1, timeout_sec)
    if callable(runlog):
        runlog(
            "shell_approval_prompt",
            "awaiting shell approval",
            approval_id=approval_id or "pending",
            command=command or "(empty command)",
            cwd=cwd or harness.state.cwd,
            timeout_sec=prompt_timeout_sec,
            proof_bundle=proof_bundle,
        )
    try:
        loop = asyncio.get_running_loop()
        decision_future: asyncio.Future[ShellApprovalDecision | bool] = loop.create_future()

        def _resolve_decision(decision: ShellApprovalDecision | bool | None) -> None:
            if decision_future.done():
                return
            if isinstance(decision, ShellApprovalDecision):
                decision_future.set_result(decision)
                return
            decision_future.set_result(bool(decision))

        await app.push_screen(prompt, callback=_resolve_decision)
        decision = await decision_future
        approved = bool(decision)
        if isinstance(decision, ShellApprovalDecision):
            remember_session = decision.remember_session
        else:
            remember_session = bool(getattr(decision, "remember_session", False))
    except Exception as exc:
        app._app_logger.warning("Approval prompt failed: %s", exc)
        if callable(runlog):
            runlog(
                "shell_approval_error",
                "shell approval prompt failed",
                approval_id=approval_id or "pending",
                error=str(exc),
            )
    finally:
        app._active_approval_prompt = None
        if approved and remember_session:
            app._set_shell_approval_session_default(True)
        if approval_id:
            try:
                bridge = getattr(app, "_harness_bridge", None)
                if bridge is not None:
                    bridge.resolve_shell_approval(approval_id, approved)
                else:
                    harness.resolve_shell_approval(approval_id, approved)
            except Exception as exc:
                app._app_logger.warning(
                    "Failed to resolve shell approval %s: %s",
                    approval_id,
                    exc,
                )
        if callable(runlog):
            runlog(
                "shell_approval_decision",
                "shell approval resolved",
                approval_id=approval_id or "pending",
                approved=approved,
                remember_session=remember_session,
                command=command or "(empty command)",
                cwd=cwd or harness.state.cwd,
                timeout_sec=prompt_timeout_sec,
            )
        if approved and remember_session:
            await app._append_system_line(
                "Shell commands will auto-approve for this session.",
                force=True,
            )
        app._set_activity("running shell..." if approved else "thinking...")
        app._refresh_status()


async def prompt_for_plan_approval(app, interrupt: dict[str, Any]) -> PlanApprovalDecision | None:
    harness = app.harness
    if harness is None:
        return None
    prompt = PlanApprovalScreen(
        question=str(interrupt.get("question") or "Plan ready. Execute it now?").strip(),
        plan_id=str(interrupt.get("plan_id") or "").strip(),
        response_mode=str(interrupt.get("response_mode") or "yes/no/revise").strip(),
    )
    app._active_approval_prompt = prompt
    app._refresh_status()

    try:
        loop = asyncio.get_running_loop()
        decision_future: asyncio.Future[PlanApprovalDecision | str | None] = loop.create_future()

        def _resolve_decision(decision: PlanApprovalDecision | str | None) -> None:
            if decision_future.done():
                return
            if isinstance(decision, PlanApprovalDecision):
                decision_future.set_result(decision)
                return
            if decision is None:
                decision_future.set_result(None)
                return
            decision_future.set_result(PlanApprovalDecision(str(decision)))

        await app.push_screen(prompt, callback=_resolve_decision)
        decision = await decision_future
        if isinstance(decision, PlanApprovalDecision):
            return decision
        if isinstance(decision, str) and decision.strip():
            return PlanApprovalDecision(decision.strip())
    except Exception as exc:
        app._app_logger.warning("Plan approval prompt failed: %s", exc)
    finally:
        app._active_approval_prompt = None
    return None


async def handle_sudo_password_prompt(app, event: UIEvent) -> None:
    harness = app.harness
    if harness is None:
        return
    prompt_id = str(event.data.get("prompt_id") or "").strip()
    command = str(event.data.get("command") or "").strip()
    prompt_text = str(event.data.get("prompt_text") or "").strip()

    prompt = SudoPasswordPromptScreen(
        prompt_id=prompt_id or "pending",
        command=command or "(empty command)",
        prompt_text=prompt_text or "Enter the sudo password to continue this command.",
    )
    app._active_approval_prompt = prompt
    app._refresh_status()

    password: str | None = None
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "sudo_password_prompt",
            "awaiting sudo password",
            prompt_id=prompt_id or "pending",
            command=command or "(empty command)",
        )
    try:
        loop = asyncio.get_running_loop()
        decision_future: asyncio.Future[str | None] = loop.create_future()

        def _resolve_decision(decision: str | None) -> None:
            if decision_future.done():
                return
            decision_future.set_result(decision)

        await app.push_screen(prompt, callback=_resolve_decision)
        password = await decision_future
    except Exception as exc:
        app._app_logger.warning("Sudo password prompt failed: %s", exc)
        if callable(runlog):
            runlog(
                "sudo_password_error",
                "sudo password prompt failed",
                prompt_id=prompt_id or "pending",
                error=str(exc),
            )
    finally:
        app._active_approval_prompt = None
        if prompt_id:
            try:
                bridge = getattr(app, "_harness_bridge", None)
                if bridge is not None:
                    bridge.resolve_sudo_password(prompt_id, password)
                else:
                    harness.resolve_sudo_password(prompt_id, password)
            except Exception as exc:
                app._app_logger.warning(
                    "Failed to resolve sudo password prompt %s: %s",
                    prompt_id,
                    exc,
                )
        if callable(runlog):
            runlog(
                "sudo_password_resolved",
                "sudo password prompt resolved",
                prompt_id=prompt_id or "pending",
                provided=bool(password),
                command=command or "(empty command)",
            )
        app._set_activity("running shell..." if password is not None else "thinking...")
        app._refresh_status()
