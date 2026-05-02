from __future__ import annotations

import asyncio
import re
from typing import Any, Awaitable, Callable

from .common import fail, needs_human

SUDO_PROMPT_PATTERNS = [
    re.compile(r"\[sudo\] password for", re.IGNORECASE),
    re.compile(r"^password:", re.IGNORECASE),
    re.compile(r"password for .*:", re.IGNORECASE),
    re.compile(r"sudo:.*no password was provided", re.IGNORECASE),
    re.compile(r"sudo:.*password is required", re.IGNORECASE),
    re.compile(r"sudo:.*a password is required", re.IGNORECASE),
    re.compile(r"sudo:.*no tty present", re.IGNORECASE),
    re.compile(r"sudo:.*a terminal is required", re.IGNORECASE),
    re.compile(r"sudo:.*must be run from a terminal", re.IGNORECASE),
]

SUDO_INVALID_PASSWORD_PATTERNS = [
    re.compile(r"sorry,\s*try again", re.IGNORECASE),
    re.compile(r"incorrect password", re.IGNORECASE),
    re.compile(r"authentication failure", re.IGNORECASE),
    re.compile(r"\bincorrect password attempts?\b", re.IGNORECASE),
]

SUDO_PERMISSION_DENIED_PATTERNS = [
    re.compile(r"not in the sudoers file", re.IGNORECASE),
    re.compile(r"may not run sudo", re.IGNORECASE),
]


def matches_any_pattern(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


async def run_sudo_validation(
    *,
    create_process: Callable[..., Awaitable[Any]],
    cwd: str,
    harness: Any = None,
    timeout_sec: int = 10,
    password: str | None = None,
) -> dict[str, Any]:
    proc = None
    try:
        command = "sudo -n -v" if password is None else "sudo -S -p '' -v"
        stdin = asyncio.subprocess.DEVNULL if password is None else asyncio.subprocess.PIPE
        proc = await create_process(
            command=command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=stdin,
            harness=harness,
        )
        input_bytes = None if password is None else f"{password}\n".encode("utf-8")
        stdout, stderr = await asyncio.wait_for(proc.communicate(input_bytes), timeout=max(1, timeout_sec))
        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        combined = "\n".join(part for part in (stderr_text.strip(), stdout_text.strip()) if part).strip()
        if proc.returncode in (0, None):
            return {"status": "ok", "stdout": stdout_text, "stderr": stderr_text}
        if matches_any_pattern(combined, SUDO_INVALID_PASSWORD_PATTERNS):
            return {
                "status": "invalid_password",
                "stdout": stdout_text,
                "stderr": stderr_text,
                "error": combined or "Incorrect sudo password.",
            }
        if matches_any_pattern(combined, SUDO_PROMPT_PATTERNS):
            return {
                "status": "password_required",
                "stdout": stdout_text,
                "stderr": stderr_text,
                "error": combined or "Sudo password is required.",
            }
        if matches_any_pattern(combined, SUDO_PERMISSION_DENIED_PATTERNS):
            return {
                "status": "permission_denied",
                "stdout": stdout_text,
                "stderr": stderr_text,
                "error": combined or "Sudo permission denied.",
            }
        return {
            "status": "error",
            "stdout": stdout_text,
            "stderr": stderr_text,
            "error": combined or f"sudo validation exited with code {proc.returncode}",
        }
    except asyncio.TimeoutError:
        if proc and proc.returncode is None:
            try:
                proc.kill()
                await asyncio.wait_for(proc.wait(), timeout=1.0)
            except Exception:
                pass
        return {"status": "error", "error": f"sudo validation timed out after {max(1, timeout_sec)}s"}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    finally:
        if harness and proc and hasattr(harness, "_active_processes"):
            try:
                harness._active_processes.discard(proc)
            except Exception:
                pass


async def ensure_sudo_credentials(
    *,
    command: str,
    cwd: str,
    is_leading_sudo: bool,
    create_process: Callable[..., Awaitable[Any]],
    harness: Any = None,
    timeout_sec: int = 30,
    sudo_human_message: str,
) -> dict[str, Any] | None:
    if not is_leading_sudo:
        return None

    validation_timeout = max(5, min(timeout_sec, 15))
    validation = await run_sudo_validation(
        create_process=create_process,
        cwd=cwd,
        harness=harness,
        timeout_sec=validation_timeout,
    )
    status = str(validation.get("status") or "").strip().lower()
    if status == "ok":
        return None
    if status == "permission_denied":
        return fail(
            str(validation.get("error") or "Sudo permission denied."),
            metadata={"command": command, "reason": "sudo_permission_denied", "sudo_validation": validation},
        )
    if status not in {"password_required", "invalid_password"}:
        return fail(
            str(validation.get("error") or "Unable to validate sudo credentials."),
            metadata={"command": command, "reason": "sudo_validation_failed", "sudo_validation": validation},
        )

    password_fn = getattr(harness, "request_sudo_password", None)
    if not callable(password_fn) or getattr(harness, "event_handler", None) is None:
        return needs_human(
            f"Command requires sudo/password input: '{command}'. {sudo_human_message}",
            metadata={"command": command, "reason": "sudo_password_required", "sudo_validation": validation},
        )

    prompt_text = "Enter the sudo password to continue this command."
    validation_error = str(validation.get("error") or "").strip()
    if validation_error:
        prompt_text = f"{prompt_text}\n\n{validation_error}"

    for attempt in range(1, 4):
        password = await password_fn(command=command, prompt_text=prompt_text)
        if password is None:
            return fail(
                "Sudo password entry cancelled by user.",
                metadata={"command": command, "reason": "sudo_password_cancelled"},
            )
        validation = await run_sudo_validation(
            create_process=create_process,
            cwd=cwd,
            harness=harness,
            timeout_sec=validation_timeout,
            password=password,
        )
        status = str(validation.get("status") or "").strip().lower()
        if status == "ok":
            return None
        if status == "permission_denied":
            return fail(
                str(validation.get("error") or "Sudo permission denied."),
                metadata={"command": command, "reason": "sudo_permission_denied", "sudo_validation": validation},
            )
        if status in {"password_required", "invalid_password"} and attempt < 3:
            prompt_text = "Incorrect sudo password. Try again."
            validation_error = str(validation.get("error") or "").strip()
            if validation_error:
                prompt_text = f"{prompt_text}\n\n{validation_error}"
            continue
        if status in {"password_required", "invalid_password"}:
            return fail(
                "Sudo authentication failed after 3 attempts.",
                metadata={"command": command, "reason": "sudo_password_rejected", "sudo_validation": validation},
            )
        return fail(
            str(validation.get("error") or "Unable to validate sudo credentials."),
            metadata={"command": command, "reason": "sudo_validation_failed", "sudo_validation": validation},
        )
    return None
