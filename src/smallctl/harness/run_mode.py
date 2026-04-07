from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ..client import OpenAICompatClient
from .memory import assess_write_task_complexity
from ..guards import is_four_b_or_under_model_name
from ..models.events import UIEvent, UIEventType
from ..models.conversation import ConversationMessage

if TYPE_CHECKING:
    from ..harness import Harness

logger = logging.getLogger("smallctl.harness.run_mode")

def should_enable_complex_write_chat_draft(
    task: str,
    *,
    model_name: str | None = None,
    cwd: str | None = None,
) -> bool:
    """Return True when complex write tasks should take the chat drafting path."""
    text = str(task or "").strip()
    if not text:
        return False
    if not is_four_b_or_under_model_name(model_name):
        return False

    analysis = assess_write_task_complexity(text, cwd=cwd)
    return bool(analysis.get("force_chunk_mode_targets"))


def decide_run_mode_sync(task: str, *, model_name: str | None = None) -> str | None:
    """Sync heuristics for mode decision. Returns mode name or None to continue to model-based decision."""
    if should_enable_complex_write_chat_draft(task, model_name=model_name):
        return "chat"
    return None

class ModeDecisionService:
    def __init__(self, harness: Harness):
        self.harness = harness

    async def _announce_mode_change(self, *, mode: str, reason: str) -> None:
        if mode != "planning":
            return
        await self.harness._emit(
            self.harness.event_handler,
            UIEvent(
                event_type=UIEventType.ALERT,
                content="Planning mode enabled.",
                data={
                    "status_activity": "planning mode active",
                    "mode": mode,
                    "reason": reason,
                },
            ),
        )

    async def decide(self, task: str) -> str:
        plan_request = self._extract_planning_request(task)
        if plan_request is not None:
            output_path, output_format = plan_request
            self._set_planning_request(output_path=output_path, output_format=output_format)
            self.harness._runlog(
                "mode_decision",
                "selected run mode",
                mode="planning",
                raw="planning_intent",
                output_path=output_path,
                output_format=output_format,
            )
            await self._announce_mode_change(mode="planning", reason="planning_intent")
            return "planning"

        model_name = getattr(self.harness.client, "model", None)
        sync_mode = decide_run_mode_sync(task, model_name=model_name)
        if sync_mode is not None:
            self.harness._runlog(
                "mode_decision",
                "selected run mode",
                mode=sync_mode,
                raw="complex_write_sync_heuristic",
            )
            return sync_mode
            
        if self.harness.state.planning_mode_enabled and not (self.harness.state.active_plan and self.harness.state.active_plan.approved):
            self.harness._runlog(
                "mode_decision",
                "selected run mode",
                mode="planning",
                raw="planning_mode_enabled",
            )
            await self._announce_mode_change(mode="planning", reason="planning_mode_enabled")
            return "planning"
            
        if self._is_smalltalk(task):
            self.harness._runlog("mode_decision", "selected run mode", mode="chat", raw="smalltalk_heuristic")
            return "chat"
            
        if self._needs_memory_persistence(task):
            self.harness._runlog(
                "mode_decision",
                "selected run mode",
                mode="loop",
                raw="memory_persistence_heuristic",
            )
            return "loop"
            
        if self._needs_contextual_loop_escalation(task):
            self.harness._runlog(
                "mode_decision",
                "selected run mode",
                mode="loop",
                raw="contextual_execution_followup",
            )
            return "loop"
            
        if (
            self._needs_loop_for_content_lookup(task)
            or self._looks_like_action_request(task)
            or self._looks_like_shell_request(task)
        ):
            self.harness._runlog(
                "mode_decision",
                "selected run mode",
                mode="loop",
                raw="action_lookup_heuristic",
            )
            return "loop"
            
        prompt = (
            "Decide whether the user request requires tool usage in a coding harness. "
            "Reply with exactly one word: chat or loop."
        )
        messages = [
            ConversationMessage(role="system", content=prompt).to_dict(),
            ConversationMessage(role="user", content=task).to_dict(),
        ]
        chunks: list[dict[str, Any]] = []
        try:
            async for event in self.harness.client.stream_chat(messages=messages, tools=[]):
                chunks.append(event)
        except Exception:
            self.harness._runlog("mode_decision_fallback", "mode decision failed, using loop")
            return "loop"
            
        decision_result = OpenAICompatClient.collect_stream(
            chunks,
            reasoning_mode="off",
            thinking_start_tag=self.harness.thinking_start_tag,
            thinking_end_tag=self.harness.thinking_end_tag,
        )
        decision = decision_result.assistant_text.strip().lower()
        if not decision:
            decision = decision_result.thinking_text.strip().lower()
        mode = "loop" if decision.startswith("loop") else "chat"
        self.harness._runlog("mode_decision", "selected run mode", mode=mode, raw=decision)
        return mode

    def _set_planning_request(self, *, output_path: str | None = None, output_format: str | None = None) -> None:
        self.harness.state.planning_mode_enabled = True
        self.harness.state.planner_requested_output_path = str(output_path or "").strip()
        self.harness.state.planner_requested_output_format = str(output_format or "").strip().lower()
        self.harness.state.planner_resume_target_mode = "loop"
        self.harness.state.touch()
        self.harness.state.sync_plan_mirror()

    def _extract_planning_request(self, task: str) -> tuple[str | None, str | None] | None:
        lowered = (task or "").lower()
        if "plan" not in lowered:
            return None
        output_path: str | None = None
        output_format: str | None = None

        path_match = re.search(r"([^\s]+?\.(?:md|txt|text))\b", task, flags=re.IGNORECASE)
        if path_match:
            output_path = path_match.group(1)
            suffix = Path(output_path).suffix.lower()
            if suffix == ".md":
                output_format = "markdown"
            elif suffix in {".txt", ".text"}:
                output_format = "text"
        if "plan.md" in lowered and output_path is None:
            output_format = "markdown"
        if any(
            phrase in lowered
            for phrase in (
                "make a plan",
                "make a short plan",
                "create a plan",
                "create a short plan",
                "create a brief plan",
                "plan this",
                "plan this out",
                "make a plan first",
                "plan out",
                "before doing anything, create a short plan",
                "before doing anything, create a plan",
                "before doing anything, plan",
            )
        ):
            return output_path, output_format
        if output_path:
            return output_path, output_format
        return None

    def _is_smalltalk(self, task: str) -> bool:
        text = task.strip().lower()
        smalltalk = {
            "hi",
            "hello",
            "hey",
            "yo",
            "good morning",
            "good afternoon",
            "good evening",
            "thanks",
            "thank you",
            "how are you",
            "what's up",
            "whats up",
        }
        return text in smalltalk

    def _needs_loop_for_content_lookup(self, task: str) -> bool:
        text = task.strip().lower()
        if not text:
            return False

        file_markers = (
            "file", "log", "logs", ".log", ".jsonl", ".txt", ".md", ".py", "/", "\\", "code", "source", "src",
        )
        content_queries = (
            "what is", "what's", "show", "read", "tell me", "summarize", "contents", "content", "line ", "lines ", "bug", "error", "issue", "debug",
        )
        asks_for_specific_line = bool(re.search(r"\bline(?:s)?\s+\d+\b", text))
        asks_for_range = bool(re.search(r"\b\d+\s*-\s*\d+\b", text))
        asks_for_log_or_file_content = any(marker in text for marker in file_markers) and any(
            query in text for query in content_queries
        )
        asks_for_command_execution = (
            bool(re.search(r"\b(run|execute|exec)\b", text))
            and bool(
                re.search(
                    r"\b(ls|dir|pwd|cd|cat|type|findstr|grep|git\s+status|get-childitem|pytest|python|powershell|pwsh|cmd)\b",
                    text,
                )
            )
        )
        asks_for_directory_contents = (
            any(
                phrase in text
                for phrase in (
                    "what files", "which files", "list files", "show files", "what folders", "which folders", "list folders", "show folders",
                    "list directory", "show directory", "directory contents", "folder contents", "current directory", "this directory",
                    "current folder", "this folder", "what is in this directory", "what is in the current directory", "what is in this folder",
                    "what is in the current folder",
                )
            )
            or bool(
                re.search(
                    r"\b(list|show|what|which)\b.*\b(files?|folders?|directories?|contents?)\b",
                    text,
                )
            )
        )
        asks_where_specific_line_is = asks_for_specific_line and any(
            phrase in text for phrase in ("what is", "what's", "show", "read")
        )
        return (
            asks_for_specific_line
            or asks_for_range
            or asks_for_log_or_file_content
            or asks_where_specific_line_is
            or asks_for_directory_contents
            or asks_for_command_execution
        )

    def _needs_contextual_loop_escalation(self, task: str) -> bool:
        text = task.strip().lower()
        if not text:
            return False
        if not self._looks_like_execution_followup(text):
            return False
        if self._recent_assistant_proposed_command():
            return True
        return self._recent_assistant_referenced_tool_name("shell_exec")

    def _looks_like_execution_followup(self, text: str) -> bool:
        followup_phrases = (
            "use the command", "use that command", "run it", "run that", "execute it", "execute that", "try again",
            "use the shell command", "run the shell command", "execute the shell command",
        )
        return any(phrase in text for phrase in followup_phrases)

    def _recent_assistant_proposed_command(self) -> bool:
        recent_assistants = [
            message.content or ""
            for message in reversed(self.harness.state.recent_messages)
            if message.role == "assistant" and (message.content or "").strip()
        ][:2]
        if not recent_assistants:
            return False
        command_pattern = re.compile(
            r"```(?:bash|sh|shell|zsh|pwsh|powershell)?\s*\n.+?```",
            re.IGNORECASE | re.DOTALL,
        )
        shell_tokens = re.compile(
            r"\b(top|ps|ls|pwd|cd|cat|grep|find|git|pytest|python|bash|sh|systemctl|journalctl)\b",
            re.IGNORECASE,
        )
        for content in recent_assistants:
            if command_pattern.search(content):
                return True
            if shell_tokens.search(content):
                return True
        return False

    def _recent_assistant_referenced_tool_name(self, tool_name: str) -> bool:
        target = str(tool_name or "").strip().lower()
        if not target:
            return False
        for message in reversed(self.harness.state.recent_messages):
            if message.role != "assistant" or not message.content:
                continue
            if target in message.content.lower():
                return True
        return False

    def _looks_like_action_request(self, task: str) -> bool:
        text = task.strip().lower()
        action_markers = ("run", "exec", "shell", "terminal", "ping", "curl", "wget", "git")
        return any(m in text for m in action_markers)

    def _needs_memory_persistence(self, task: str) -> bool:
        text = task.strip().lower()
        if not text:
            return False
        memory_markers = (
            "save this in memory",
            "save memory",
            "remember this",
            "store this in memory",
            "store this",
            "note this",
            "pin this",
            "persist this",
            "keep this in memory",
            "write this down",
        )
        return any(marker in text for marker in memory_markers)

    def _looks_like_shell_request(self, task: str) -> bool:
        text = task.strip().lower()
        if not text:
            return False
        shell_markers = (
            "bash", "shell", "terminal", "command", "command line", "run ", "execute", "exec", "script", "scan", "nmap", "ssh", "scp", "sftp",
            "ping", "curl", "wget", "traceroute", "tracepath", "netstat", "route", "ip route", "ip addr", "tcpdump", "netcat", "nc ", "dig",
            "nslookup", "whoami", "ps ", "top ", "lsof", "df ", "du ",
        )
        if any(marker in text for marker in shell_markers):
            return True
        return bool(
            re.search(
                r"\b(run|execute|exec|launch|invoke|start|inspect|check)\b.*\b(command|shell|terminal|script|scan|nmap|port|ports|ssh|ping|curl|wget)\b",
                text,
            )
        )

    def looks_like_readonly_chat_request(self, task: str) -> bool:
        text = task.strip().lower()
        if not text:
            return False
        if self._looks_like_execution_followup(text):
            return False
        readonly_markers = (
            "what", "which", "show", "read", "find", "search", "grep", "list", "current", "status", "where", "how many", "inspect", "check",
            "look at", "can you see", "tell me", "summarize",
        )
        readonly_targets = (
            "file", "files", "folder", "directory", "repo", "repository", "cwd", "working directory", "log", "logs", "artifact", "artifacts",
            "process", "cpu", "ram", "memory", "host", "system", "status", "code", "source", "src",
        )
        has_readonly_marker = any(marker in text for marker in readonly_markers)
        has_target = any(target in text for target in readonly_targets)
        return has_readonly_marker and has_target

    def chat_mode_requires_tools(self, task: str) -> bool:
        if self._is_smalltalk(task):
            return False
        if self._needs_loop_for_content_lookup(task):
            return True
        return self.looks_like_readonly_chat_request(task) or self._looks_like_action_request(task)
