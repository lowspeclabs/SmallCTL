from __future__ import annotations
import asyncio
import inspect
import logging
import re
import time
from typing import Any

from textual.containers import Vertical, VerticalScroll

from ..graph.tool_model_rules_support import _raw_function_call_pattern
from ..models.events import UIEvent, UIEventType
from .bubbles import ArtifactBubbleWidget, AssistantTurnWidget, BubbleWidget, SystemInterruptWidget
from .display import _CRITICAL_EVENTS, format_test_time_scaling_event


logger = logging.getLogger("smallctl.ui.console")


class ConsolePane(VerticalScroll):
    def __init__(self, *args, verbose: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._active_assistant_turn: AssistantTurnWidget | None = None
        self._last_system_message: str = ""
        self._verbose = bool(verbose)
        self._stream_flush_handle: asyncio.Handle | None = None
        self._stream_buffer_groups: list[dict[str, Any]] = []
        self._stream_flush_interval = 0.05
        self._stream_flush_soon_once = False
        self._visible_transcript_limit = 250
        self._hidden_transcript_entries = 0
        self._retention_placeholder: BubbleWidget | None = None
        self._json_suppress_active = False
        self._json_suppress_buffer = ""
        self._json_suppress_max_len = 5000

    def set_verbose(self, verbose: bool) -> None:
        self._verbose = bool(verbose)

    async def on_mount(self) -> None:
        self.styles.scrollbar_size_vertical = 0
        self.styles.scrollbar_size_horizontal = 0
        self.styles.scrollbar_gutter = "auto"
        await self.mount(Vertical(id="bubble-stack"))

    async def on_unmount(self) -> None:
        await self.flush_stream_buffers()

    async def append_line(self, line: str, kind: str = "system") -> None:
        await self.flush_stream_buffers()
        await self._add_bubble(kind, line)
        self._last_system_message = line

    async def clear_bubbles(self) -> None:
        self._cancel_stream_flush()
        self._clear_stream_buffers()
        self._json_suppress_active = False
        self._json_suppress_buffer = ""
        stack = self.query_one("#bubble-stack", Vertical)
        await stack.remove_children()
        self._active_assistant_turn = None
        self._hidden_transcript_entries = 0
        self._retention_placeholder = None

    def has_active_assistant_text(self) -> bool:
        if self._active_assistant_turn is None:
            return False
        return self._active_assistant_turn.has_assistant_text()

    def get_active_assistant_text(self) -> str:
        if self._active_assistant_turn is None:
            return ""
        return self._active_assistant_turn.get_assistant_text()

    async def begin_assistant_turn(self) -> None:
        """Create the assistant turn container up front.

        This lets the UI show that the model is responding before the first
        assistant token arrives, which is useful when the model spends a while
        emitting thinking output first.
        """
        await self._ensure_assistant_turn()

    async def append_event(self, event: UIEvent) -> None:
        speaker = _coerce_speaker(event.data.get("speaker"))
        if event.event_type == UIEventType.ASSISTANT:
            if event.data.get("kind") == "replace":
                await self.flush_stream_buffers()
                await self._replace_assistant(event.content, speaker=speaker)
                return
            await self._ensure_assistant_turn(speaker=speaker)
            if event.data.get("kind") == "print":
                await self.flush_stream_buffers()
                await self._append_full_printout(
                    event.content,
                    artifact_id=event.data.get("artifact_id")
                )
            else:
                filtered = self._filter_tool_call_tokens_from_stream(event.content)
                if filtered:
                    self._append_stream_buffer("assistant", filtered)
                    self._schedule_stream_flush()
            return
        if event.event_type == UIEventType.THINKING:
            await self._ensure_assistant_turn(speaker=speaker)
            if event.data.get("kind") == "replace":
                await self.flush_stream_buffers()
                await self._replace_thinking(event.content)
            else:
                self._append_stream_buffer("thinking", event.content)
                self._schedule_stream_flush()
            return
        if event.event_type == UIEventType.SHELL_STREAM:
            await self._ensure_assistant_turn(speaker=speaker)
            tool_name = _coerce_str(event.data.get("tool_name"))
            tool_call_id = _coerce_str(event.data.get("tool_call_id"))
            self._append_stream_buffer(
                "shell",
                event.content,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            )
            self._schedule_stream_flush()
            return
        await self.flush_stream_buffers()
        if event.event_type == UIEventType.TOOL_CALL:
            await self._ensure_assistant_turn(speaker=speaker)
            await self._append_tool_call(
                str(event.data.get("display_text") or event.content),
                tool_name=event.content,
                tool_call_id=_coerce_str(event.data.get("tool_call_id")),
                args=event.data.get("args"),
            )
            return
        if event.event_type == UIEventType.TOOL_RESULT:
            tool_name = _coerce_str(event.data.get("tool_name"))
            if self._active_assistant_turn is not None:
                if speaker:
                    self._active_assistant_turn.set_speaker(speaker)
                nested = await self._append_tool_result(
                    str(event.data.get("display_text") or event.content),
                    tool_name=tool_name,
                    tool_call_id=_coerce_str(event.data.get("tool_call_id")),
                    data=event.data,
                )
                if nested:
                    return
            if tool_name in {"shell_exec", "ssh_exec", "file_read", "ssh_file_read"}:
                return
            self._active_assistant_turn = None
            await self._add_bubble("system", str(event.data.get("display_text") or event.content))
            return

        if event.event_type == UIEventType.SYSTEM and event.data.get("kind") == "test_time_scaling":
            await self._append_test_time_scaling_event(event)
            return

        if event.data.get("ui_kind") == "subtask_checklist":
            turn = await self._ensure_assistant_turn()
            title = event.data.get("checklist_title") or "Task Checklist"
            content = event.content or ""
            self.app._app_logger.debug(
                "ui_subtask_checklist: title=%r content_len=%d empty=%s",
                title, len(content), not content.strip(),
            )
            await turn.set_task_checklist(content, title=f"📋 {title}")
            return

        # P2.2: critical backend events render as compact inline interruptions
        # only when verbose mode is enabled (default off). Suppressed events must
        # not split the active assistant turn.
        is_critical = (
            event.event_type == UIEventType.SYSTEM
            and event.data.get("ui_kind") in _CRITICAL_EVENTS
        )
        if is_critical:
            self._last_system_message = str(event.data.get("display_text") or event.content)
            if self._verbose:
                await self._append_critical_interrupt(event)
            return

        kind_map = {
            UIEventType.USER: "user",
            UIEventType.ERROR: "error",
            UIEventType.SYSTEM: "system",
            UIEventType.ALERT: "system",
        }
        kind = kind_map.get(event.event_type)
        if kind is None:
            return
        text = str(event.data.get("display_text") or event.content)
        is_interactive_alert = (
            event.event_type == UIEventType.ALERT
            and (
                event.data.get("ui_kind") in {"approve_prompt", "sudo_password_prompt"}
                or "interrupt" in event.data
            )
        )
        if kind == "system" and not self._verbose and not is_interactive_alert:
            # Suppressed system events should not create a visible bubble, but
            # non-info system events still break the active assistant turn so that
            # subsequent assistant messages render as a new turn rather than
            # appending to the previous one. Flush any pending stream first;
            # otherwise the delayed batch can attach pre-boundary text to the
            # next assistant turn. Purely informational suppressed events preserve
            # the active turn to avoid interrupting the flow.
            if event.data.get("ui_kind") not in {"info", "status"}:
                await self.flush_stream_buffers()
                self._active_assistant_turn = None
                self._stream_flush_soon_once = True
            self._last_system_message = text
            return
        if event.event_type in {
            UIEventType.USER,
            UIEventType.ERROR,
            UIEventType.SYSTEM,
            UIEventType.METRICS,
            UIEventType.ALERT,
        }:
            self._active_assistant_turn = None
            self._stream_flush_soon_once = True
        await self._add_bubble(kind, text)
        if kind == "system":
            self._last_system_message = text

    async def _append_assistant(self, text: str) -> None:
        if not text:
            return
        turn = await self._ensure_assistant_turn()
        await turn.append_assistant_text(text)

    async def _replace_assistant(self, text: str, *, speaker: str | None = None) -> None:
        """Overwrite the streamed assistant text with the cleaned version.
        This is called after _extract_inline_tool_calls strips tool JSON."""
        turn = self._active_assistant_turn or self._latest_assistant_turn()
        if turn is None:
            return
        if speaker:
            turn.set_speaker(speaker)
        turn.replace_assistant_text(text)
        # If the turn was hidden because all content was blanked (e.g. degenerate
        # loop placeholder), drop the reference so the next model call creates a
        # fresh turn instead of appending to a hidden one.
        if not turn.display:
            if self._active_assistant_turn is turn:
                self._active_assistant_turn = None

    def _latest_assistant_turn(self) -> AssistantTurnWidget | None:
        try:
            stack = self.query_one("#bubble-stack", Vertical)
            for child in reversed(stack.children):
                if isinstance(child, AssistantTurnWidget):
                    return child
        except Exception:
            return None
        return None

    def _all_assistant_turns(self) -> list[AssistantTurnWidget]:
        try:
            stack = self.query_one("#bubble-stack", Vertical)
            return [
                child for child in stack.children if isinstance(child, AssistantTurnWidget)
            ]
        except Exception:
            return []

    async def _append_full_printout(self, text: str, *, artifact_id: str | None) -> None:
        turn = await self._ensure_assistant_turn()
        await turn.add_full_printout(text, artifact_id=artifact_id)

    async def _append_thinking(self, text: str) -> None:
        turn = await self._ensure_assistant_turn()
        await turn.append_thinking_text(text)

    async def _replace_thinking(self, text: str) -> None:
        turn = await self._ensure_assistant_turn()
        await turn.replace_thinking_text(text)

    async def _append_shell_stream(
        self,
        text: str,
        *,
        tool_name: str | None,
        tool_call_id: str | None,
    ) -> None:
        turn = await self._ensure_assistant_turn()
        await turn.append_shell_stream(text, tool_name=tool_name, tool_call_id=tool_call_id)

    async def _append_tool_call(
        self,
        text: str,
        *,
        tool_name: str,
        tool_call_id: str | None = None,
        args: dict[str, Any] | None = None,
    ) -> None:
        turn = await self._ensure_assistant_turn()
        await turn.add_tool_call(text, tool_name=tool_name, tool_call_id=tool_call_id, args=args)

    async def _append_tool_result(
        self,
        text: str,
        *,
        tool_name: str | None,
        tool_call_id: str | None,
        data: dict[str, Any] | None = None,
    ) -> bool:
        turn = self._active_assistant_turn
        if turn is not None and turn.find_tool_call_detail(
            tool_name=tool_name, tool_call_id=tool_call_id
        ):
            await turn.add_tool_result(
                text,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                data=data,
            )
            return True
        for turn in reversed(self._all_assistant_turns()):
            if turn.find_tool_call_detail(
                tool_name=tool_name, tool_call_id=tool_call_id
            ):
                await turn.add_tool_result(
                    text,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    data=data,
                )
                return True
        return False

    async def _add_bubble(self, kind: str, text: str) -> BubbleWidget:
        bubble = BubbleWidget(kind=kind, text=text)
        stack = self.query_one("#bubble-stack", Vertical)
        await stack.mount(bubble)
        await self._enforce_visible_retention()
        return bubble

    async def _append_test_time_scaling_event(self, event: UIEvent) -> None:
        text = format_test_time_scaling_event(event)
        title = "Test-Time Scaling"
        phase = str(event.data.get("phase") or "").strip()
        if phase:
            title += f" ({phase})"
        panel = ArtifactBubbleWidget(title=title, text=text, collapsed=False)
        panel.add_class("assistant-detail-test-time-scaling")
        stack = self.query_one("#bubble-stack", Vertical)
        await stack.mount(panel)
        await self._enforce_visible_retention()
        self._active_assistant_turn = None
        self._last_system_message = event.content

    def _strip_json_tool_call_blocks(self, text: str) -> str:
        """Remove markdown JSON tool-call blocks from combined stream text.

        This catches blocks whose opening fence and `json` language tag were
        split across stream chunks, which the per-chunk filter cannot detect
        until the tag arrives.
        """
        if "```" not in text:
            return text
        cleaned = text
        # Repeatedly remove ```json ... ``` blocks.
        while True:
            match = re.search(r"```json\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
            if not match:
                break
            block = match.group(1)
            # Only remove if the block looks like a tool call.
            lowered = block.lower()
            if any(key in lowered for key in ('"name"', '"tool_name"', '"tool_call"', '"tool"', '"action"', '"function"')):
                cleaned = cleaned[: match.start()] + cleaned[match.end() :]
            else:
                break
        return cleaned

    def _filter_tool_call_tokens_from_stream(self, text: str) -> str:
        """Strip inline tool-call syntax from assistant stream chunks.

        The inline parser will later extract these calls and emit a proper
        TOOL_CALL event, but the live token stream must not render the raw
        `tool_name(key=value)` signature or JSON tool-call blocks as user-visible
        assistant text.

        Only raw function calls with explicit key=value arguments (the harness
        inline format) and JSON objects that look like tool calls are removed.
        Ordinary prose, markdown code fences, and generic function-like text
        such as `print("hi")` are preserved.
        """
        if not text:
            return text

        # If we are already suppressing a JSON tool-call block, keep buffering
        # until the block terminates (closing fence or balanced braces).
        if self._json_suppress_active:
            self._json_suppress_buffer += text
            buf = self._json_suppress_buffer
            # Markdown JSON fence end
            if "\n```" in buf or buf.rstrip().endswith("```"):
                self._json_suppress_active = False
                self._json_suppress_buffer = ""
                return ""
            # Balanced braces for raw JSON objects
            brace_count = 0
            in_string = False
            escape = False
            closed_at = -1
            for i, ch in enumerate(buf):
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if ch == "{":
                        brace_count += 1
                    elif ch == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            closed_at = i
                            break
            if closed_at != -1:
                remainder = buf[closed_at + 1 :]
                # The JSON object may be inside a markdown fence; consume the
                # closing fence so it does not leak into the visible stream.
                remainder_stripped = remainder.lstrip()
                if remainder_stripped.startswith("```"):
                    remainder = remainder_stripped[len("```") :]
                self._json_suppress_active = False
                self._json_suppress_buffer = ""
                return self._filter_tool_call_tokens_from_stream(remainder)
            # Safety flush if the buffer grows too large without termination
            if len(self._json_suppress_buffer) > self._json_suppress_max_len:
                flushed = self._json_suppress_buffer
                self._json_suppress_active = False
                self._json_suppress_buffer = ""
                return flushed
            return ""

        # Detect the start of a markdown JSON fence. Split chunks can separate
        # the triple backticks from the `json` language tag, so look for either.
        lower = text.lower()
        fence_start_idx = -1
        fence_kind = ""
        for marker in ("```json", "```"):
            idx = lower.find(marker)
            if idx != -1 and (fence_start_idx == -1 or idx < fence_start_idx):
                fence_start_idx = idx
                fence_kind = marker
        if fence_kind == "```json":
            before = text[:fence_start_idx]
            after = text[fence_start_idx + len("```json") :]
            # If the block closes in the same chunk, suppress just the block.
            if "\n```" in after or after.rstrip().endswith("```"):
                return before
            self._json_suppress_active = True
            self._json_suppress_buffer = after
            return before
        if fence_kind == "```":
            # We saw an unlabeled fence start; only suppress if the next
            # non-whitespace content is the `json` language tag.
            after_fence = text[fence_start_idx + len("```") :]
            stripped = after_fence.lstrip()
            if stripped.lower().startswith("json"):
                after_lang = stripped[len("json") :]
                if "\n```" in after_lang or after_lang.rstrip().endswith("```"):
                    return text[:fence_start_idx]
                self._json_suppress_active = True
                self._json_suppress_buffer = after_lang
                return text[:fence_start_idx]

        # Detect raw JSON tool-call objects that are not inside fences.
        # Look for a leading `{` followed quickly by a tool-name key. This
        # catches {"name": "tool_name", ...} and {"tool_name": "tool_name", ...}
        # as well as the common "json\n{" prefix emitted by some models.
        stripped = text.lstrip()
        leading_ws = text[: len(text) - len(stripped)] if stripped else text
        json_start_match = re.match(
            r"(?:json\s*[\n\r]+)?\s*\{\s*\"(name|tool_name|tool_call|tool|action|function)\"",
            stripped,
            re.IGNORECASE,
        )
        if json_start_match:
            # If the previous buffered assistant text ends with an opening
            # markdown fence, that fence belongs to the JSON block we are
            # about to suppress. Remove it from the buffer so it does not leak.
            if self._stream_buffer_groups:
                last_group = self._stream_buffer_groups[-1]
                if last_group.get("kind") == "assistant" and last_group.get("parts"):
                    last_part = last_group["parts"][-1]
                    fence_match = re.search(r"```\s*$", last_part)
                    if fence_match:
                        last_group["parts"][-1] = last_part[: fence_match.start()]
            # Start suppressing from the JSON object onward.
            self._json_suppress_active = True
            self._json_suppress_buffer = stripped
            return leading_ws

        # Terminal tool calls: task_complete(...), task_fail(...)
        stripped = re.sub(
            r"(?s)\b(task_complete|task_fail)\s*\([^)]*\)",
            "",
            text,
        )
        # Generic registered tool calls using the harness inline format:
        # tool_name(key=value, ...)
        raw_call_regex = _raw_function_call_pattern()
        stripped = re.sub(raw_call_regex, "", stripped)
        return stripped

    def _schedule_stream_flush(self) -> None:
        handle = self._stream_flush_handle
        if handle is not None and not handle.cancelled():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        def callback() -> None:
            asyncio.create_task(self.flush_stream_buffers(finalize_assistant=False))

        if self._stream_flush_soon_once:
            self._stream_flush_soon_once = False
            self._stream_flush_handle = loop.call_soon(callback)
        else:
            self._stream_flush_handle = loop.call_later(
                self._stream_flush_interval,
                callback,
            )

    async def flush_stream_buffers(self, *, finalize_assistant: bool = True) -> None:
        started = time.perf_counter()
        self._cancel_stream_flush()
        # If we were suppressing a JSON tool-call block that never terminated,
        # flush the buffered text as ordinary assistant text so it is not lost.
        if self._json_suppress_active:
            buffered = self._json_suppress_buffer
            self._json_suppress_active = False
            self._json_suppress_buffer = ""
            if buffered:
                self._append_stream_buffer("assistant", buffered)
        groups = list(self._stream_buffer_groups)
        self._clear_stream_buffers()
        # Merge consecutive groups of the same kind so that JSON tool-call
        # fences split across chunk boundaries can be stripped from the full
        # assistant text before it is rendered.
        merged_groups: list[dict[str, Any]] = []
        for item in groups:
            if not item.get("parts"):
                continue
            key = (item.get("kind"), item.get("tool_name") or "", item.get("tool_call_id") or "")
            if merged_groups:
                last = merged_groups[-1]
                last_key = (last.get("kind"), last.get("tool_name") or "", last.get("tool_call_id") or "")
                if last_key == key:
                    last.setdefault("parts", []).extend(item.get("parts") or [])
                    continue
            merged_groups.append(dict(item))
        groups = merged_groups
        flushed_chars = 0
        for item in groups:
            text = "".join(item.get("parts") or [])
            if not text:
                continue
            flushed_chars += len(text)
            kind = item.get("kind")
            if kind == "assistant":
                text = self._strip_json_tool_call_blocks(text)
                if text:
                    await self._append_assistant(text)
            elif kind == "thinking":
                await self._append_thinking(text)
            elif kind == "shell":
                await self._append_shell_stream(
                    text,
                    tool_name=item.get("tool_name"),
                    tool_call_id=item.get("tool_call_id"),
                )
        if finalize_assistant:
            turn = self._active_assistant_turn
            if turn is not None and hasattr(turn, "finalize_assistant_render"):
                try:
                    turn.finalize_assistant_render()
                except Exception:
                    pass
        if groups and logger.isEnabledFor(logging.DEBUG):
            widget_count = self._bubble_stack_widget_count()
            segment_count = self._estimate_bubble_stack_segments()
            logger.debug(
                "ui_stream_flush %s",
                {
                    "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
                    "group_count": len(groups),
                    "flushed_chars": flushed_chars,
                    "bubble_widget_count": widget_count,
                    "estimated_segment_count": segment_count,
                },
            )

    def _bubble_stack_widget_count(self) -> int:
        try:
            stack = self.query_one("#bubble-stack", Vertical)
            return len(stack.children)
        except Exception:
            return 0

    def _estimate_bubble_stack_segments(self) -> int:
        """Rough segment count across nested rendered TextBlockWidget contents."""
        try:
            stack = self.query_one("#bubble-stack", Vertical)
        except Exception:
            return 0
        try:
            from rich.console import Console as RichConsole

            rc = RichConsole(width=80, no_color=True, force_terminal=False)
            options = rc.options
        except Exception:
            rc = None
            options = None

        def _count(widget: Any) -> int:
            rendered = getattr(widget, "_rendered_content", None)
            total = 0
            if rendered is not None:
                segs = getattr(rendered, "_segments", None) or getattr(rendered, "segments", None)
                if isinstance(segs, list):
                    total += len(segs)
                elif rc is not None and options is not None and hasattr(rendered, "__rich_console__"):
                    try:
                        total += len(list(rc.render(rendered, options)))
                    except Exception:
                        pass
            for child in getattr(widget, "children", ()):
                total += _count(child)
            return total

        return sum(_count(child) for child in stack.children)

    def _cancel_stream_flush(self) -> None:
        handle = self._stream_flush_handle
        if handle is not None and not handle.cancelled():
            handle.cancel()
        self._stream_flush_handle = None

    def _clear_stream_buffers(self) -> None:
        self._stream_buffer_groups = []

    def _append_stream_buffer(
        self,
        kind: str,
        text: str,
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        if not text:
            return
        key = (kind, tool_name or "", tool_call_id or "")
        if self._stream_buffer_groups:
            last = self._stream_buffer_groups[-1]
            last_key = (last.get("kind"), last.get("tool_name") or "", last.get("tool_call_id") or "")
            if last_key == key:
                last.setdefault("parts", []).append(text)
                return
        self._stream_buffer_groups.append(
            {
                "kind": kind,
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "parts": [text],
            }
        )

    async def _append_critical_interrupt(self, event: UIEvent) -> None:
        text = str(event.data.get("display_text") or event.content)
        interrupt = SystemInterruptWidget(text=text)
        if self._active_assistant_turn is not None:
            await self._active_assistant_turn.add_meta_widget(interrupt)
        else:
            stack = self.query_one("#bubble-stack", Vertical)
            await stack.mount(interrupt)
            await self._enforce_visible_retention()
        self._last_system_message = event.content

    async def _ensure_assistant_turn(self, *, speaker: str | None = None) -> AssistantTurnWidget:
        if self._active_assistant_turn is None:
            turn = AssistantTurnWidget(speaker=speaker or "assistant")
            stack = self.query_one("#bubble-stack", Vertical)
            await stack.mount(turn)
            self._active_assistant_turn = turn
            self._json_suppress_active = False
            self._json_suppress_buffer = ""
            await self._enforce_visible_retention()
        elif speaker:
            self._active_assistant_turn.set_speaker(speaker)
        return self._active_assistant_turn

    async def _enforce_visible_retention(self) -> None:
        try:
            limit = int(getattr(self, "_visible_transcript_limit", 250) or 0)
        except (TypeError, ValueError):
            limit = 250
        if limit <= 0:
            return
        try:
            stack = self.query_one("#bubble-stack", Vertical)
            children = list(stack.children)
        except Exception:
            return
        removals = _select_visible_retention_removals(
            children,
            active=self._active_assistant_turn,
            limit=limit,
        )
        if not removals:
            return
        self._hidden_transcript_entries += len(removals)
        for child in removals:
            if child is self._retention_placeholder:
                continue
            try:
                result = child.remove()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                continue
        await self._ensure_retention_placeholder()

    async def _ensure_retention_placeholder(self) -> None:
        if self._hidden_transcript_entries <= 0:
            return
        try:
            stack = self.query_one("#bubble-stack", Vertical)
        except Exception:
            return
        text = (
            f"{self._hidden_transcript_entries} older UI entries hidden to keep the TUI responsive. "
            "Use session restore or artifacts for full history."
        )
        placeholder = self._retention_placeholder
        if placeholder is not None:
            try:
                placeholder.set_text(text)
                return
            except Exception:
                self._retention_placeholder = None
        placeholder = BubbleWidget(kind="system", text=text)
        placeholder.add_class("bubble-retention-placeholder")
        await stack.mount(placeholder)
        self._retention_placeholder = placeholder

    async def update_thinking_indicator(self) -> None:
        if self._active_assistant_turn is not None:
            if self._active_assistant_turn._last_thinking_detail is not None:
                self._active_assistant_turn._last_thinking_detail.update_thinking_timer()
            if self._active_assistant_turn._current_tool_calls_container is not None:
                self._active_assistant_turn._current_tool_calls_container.update_timer()
            for detail in self._active_assistant_turn._tool_call_details:
                if not detail.has_result:
                    detail.update_timer()

    def get_last_system_message(self) -> str:
        return self._last_system_message


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_speaker(value: object) -> str:
    speaker = str(value or "assistant").strip().lower()
    return speaker or "assistant"


def _select_visible_retention_removals(
    children: list[Any],
    *,
    active: Any,
    limit: int,
) -> list[Any]:
    if limit <= 0 or len(children) <= limit:
        return []
    excess = len(children) - limit
    removals: list[Any] = []
    for child in children:
        if child is active:
            continue
        if getattr(child, "classes", None) and "bubble-retention-placeholder" in str(getattr(child, "classes", "")):
            continue
        removals.append(child)
        if len(removals) >= excess:
            break
    return removals
