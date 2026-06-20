import json
import inspect
import re
from typing import Any, TYPE_CHECKING
from textual.app import ComposeResult
from textual.containers import Vertical
from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from rich.markup import escape as markup_escape
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Collapsible, Static

if TYPE_CHECKING:
    pass


KIND_LABEL = {
    "user": "User",
    "thinking": "thinking",
    "assistant": "Assistant",
    "planner": "PLANNER",
    "tool_call": "Tool Calls",
    "tool_result": "RESULT",
    "error": "ERROR",
    "system": "SYSTEM",
    "alert": "ALERT",
    "cancel": "SYSTEM",
    "ready": "READY",
}

KIND_COLOR = {
    "user": "#ffffff",
    "thinking": "#ca8a04",
    "assistant": "#eab308",
    "planner": "#e879f9",
    "tool_call": "#16a34a",
    "tool_result": "#38bdf8",
    "error": "#ef4444",
    "system": "#eab308",
    "alert": "#ef4444",
    "shell": "#ca8a04",
    "cancel": "#ef4444",
    "ready": "#eab308",
}


def _looks_like_complete_pipe_table(text: str) -> bool:
    lines = [line.rstrip() for line in text.splitlines()]
    for index in range(len(lines) - 2):
        header = lines[index].strip()
        separator = lines[index + 1].strip()
        if "|" not in header or "|" not in separator:
            continue
        separator_cells = [cell.strip() for cell in separator.strip("|").split("|")]
        if not separator_cells or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator_cells):
            continue
        expected_cells = len(separator_cells)
        body_count = 0
        for row in lines[index + 2:]:
            stripped = row.strip()
            if not stripped:
                break
            if "|" not in stripped:
                return False
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if len(cells) != expected_cells:
                return False
            body_count += 1
        if body_count > 0:
            return True
    return False


def _markdown_render_ready(text: str) -> bool:
    if not text.strip():
        return False
    if text.count("```") % 2 == 1:
        return False
    if text.count("`") % 2 == 1:
        return False
    if _looks_like_complete_pipe_table(text):
        return True
    if re.search(r"(?m)^#{1,6}\s+\S", text):
        return True
    if re.search(r"(?m)^\s*[-*+]\s+\S", text):
        return True
    if re.search(r"(?m)^\s*\d+\.\s+\S", text):
        return True
    if "```" in text:
        return True
    return False


def _format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{int(seconds * 1000)}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m{secs:.0f}s"


def _trim_title_value(value: str, *, limit: int = 80) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _format_full_printout_text(text: str) -> str:
    if not text:
        return ""
    return "\n".join(f"  {line}" if line else "  " for line in text.splitlines())


def _build_full_printout_bubble(*, text: str, artifact_id: str | None) -> "ArtifactBubbleWidget":
    title = "Full Artifact Contents"
    if artifact_id:
        title += f" ({artifact_id})"
    return ArtifactBubbleWidget(title=title, text=_format_full_printout_text(text))


def _set_command_status_class(widget: Widget, success: bool | None) -> None:
    widget.remove_class("tool-status-success", "tool-status-error")
    widget.add_class("tool-status-error" if success is False else "tool-status-success")


class BubbleWidget(Static):
    text: reactive[str] = reactive("")

    def __init__(
        self,
        *,
        kind: str,
        text: str = "",
        id: str | None = None,
        nested: bool = False,
    ) -> None:
        self.kind = kind
        classes = f"bubble bubble-{kind}"
        if nested:
            classes += " bubble-nested"
        super().__init__("", id=id, classes=classes)
        self.styles.width = "100%"
        self.styles.max_width = "100%"
        self.text = text
        self._refresh_content()

    def set_text(self, value: str) -> None:
        self.text = value
        self._refresh_content()

    def append_text(self, value: str) -> None:
        self.text += value
        self._refresh_content()

    def _refresh_content(self) -> None:
        color = KIND_COLOR.get(self.kind, "#9ca3af")
        content = Text()
        if self.kind in ("user", "system"):
            label = KIND_LABEL.get(self.kind, self.kind.upper())
            content.append(f"{label}\n", style="bold #ffffff")
        content.append(self.text, style=color)
        self.update(content)


    def get_selection(self, selection: Any) -> Any:
        try:
            return super().get_selection(selection)
        except IndexError:
            # Fallback for wrapped text where UI rows > source lines
            return ([self.text], "\n") if self.text else None


class TextBlockWidget(Static):
    text: reactive[str] = reactive("")

    def __init__(
        self,
        text: str = "",
        id: str | None = None,
        classes: str | None = None,
        *,
        render_markdown: bool = False,
    ) -> None:
        super().__init__("", id=id, classes=classes)
        self.styles.width = "100%"
        self.styles.max_width = "100%"
        self.render_markdown = render_markdown
        self._markdown_finalized = False
        self._rendered_content: object | None = None
        self.text = text
        self._refresh_content()

    def set_text(self, value: str) -> None:
        self.text = value
        self._markdown_finalized = False
        self._refresh_content()

    def append_text(self, value: str) -> None:
        self.text += value
        self._markdown_finalized = False
        self._refresh_content()

    def finalize_markdown_render(self) -> None:
        if not self.render_markdown:
            return
        self._markdown_finalized = True
        self._refresh_content()

    def _refresh_content(self) -> None:
        if self.render_markdown and self._markdown_finalized and _markdown_render_ready(self.text):
            self._rendered_content = RichMarkdown(self.text)
            self.update(self._rendered_content)
            return
        # Use a plain Text object to avoid Rich markup parsing on arbitrary model output.
        # Model responses may contain brackets like [task_complete(...)] that Rich
        # would try to parse as markup tags, causing 'Expected markup value' crashes.
        self._rendered_content = Text(self.text)
        self.update(self._rendered_content)

    def get_selection(self, selection: Any) -> Any:
        try:
            return super().get_selection(selection)
        except IndexError:
            # Fallback for wrapped text where UI rows > source lines
            return ([self.text], "\n") if self.text else None

    def has_content(self) -> bool:
        return bool(self.text.strip())


class AssistantDetailWidget(Collapsible):
    PREVIEW_LIMIT = 96

    def __init__(self, *, kind: str, text: str = "", id: str | None = None) -> None:
        self.kind = kind
        self._body_widget = Vertical(classes="assistant-detail-body")
        self._content_widget = TextBlockWidget(text, classes="assistant-detail-content")
        self._text = text
        self._thinking_start_time: float | None = None
        self._thinking_done_time: float | None = None
        if kind == "thinking":
            import time
            self._thinking_start_time = time.monotonic()
        super().__init__(
            self._body_widget,
            title=self._build_title(text),
            collapsed=True,
            id=id,
            classes=f"assistant-detail assistant-detail-{kind}",
        )

    async def on_mount(self) -> None:
        if not self._body_widget.children:
            await self._body_widget.mount(self._content_widget)

    @property
    def text(self) -> str:
        return self._text

    def set_text(self, value: str) -> None:
        self._text = value
        self._content_widget.set_text(value)
        new_title = self._build_title(value)
        if self.title != new_title:
            self.title = new_title
            self.refresh(layout=True)

    def append_text(self, value: str) -> None:
        self.set_text(f"{self._text}{value}")

    def finalize_thinking(self) -> None:
        """Call when thinking is done to show final duration."""
        if self.kind == "thinking" and self._thinking_start_time is not None:
            import time
            self._thinking_done_time = time.monotonic()
            new_title = self._build_title(self._text)
            if self.title != new_title:
                self.title = new_title
                self.refresh(layout=True)

    def update_thinking_timer(self) -> None:
        """Update title with live timer while thinking."""
        if self.kind == "thinking" and self._thinking_done_time is None:
            new_title = self._build_title(self._text)
            if self.title != new_title:
                self.title = new_title
                self.refresh(layout=True)

    def _build_title(self, text: str) -> str:
        color = KIND_COLOR.get(self.kind, "#9ca3af")
        if self.kind == "thinking":
            return self._build_thinking_title()
        preview = " ".join(text.split())
        if len(preview) > self.PREVIEW_LIMIT:
            preview = preview[: self.PREVIEW_LIMIT - 3].rstrip() + "..."
        preview = markup_escape(preview)
        return f"[bold {color}]{preview}[/]" if preview else f"[bold {color}]{self.kind.upper()}[/]"

    def _build_thinking_title(self) -> str:
        import time
        color = KIND_COLOR.get("thinking", "#ca8a04")
        if self._thinking_done_time is not None and self._thinking_start_time is not None:
            duration = self._thinking_done_time - self._thinking_start_time
            return f"[bold {color}]Thought: {self._format_duration(duration)}[/]"
        if self._thinking_start_time is not None:
            elapsed = time.monotonic() - self._thinking_start_time
            return f"[bold {color}]thinking {self._format_duration(elapsed)}...[/]"
        return f"[bold {color}]THINK[/]"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        return _format_duration(seconds)

    async def add_child_widget(self, widget: Widget) -> None:
        await self._body_widget.mount(widget)
        self._body_widget.refresh(layout=True)


class ArtifactBubbleWidget(Collapsible):
    def __init__(
        self,
        *,
        title: str,
        path: str | None = None,
        text: str = "",
        collapsed: bool = True,
        id: str | None = None,
    ) -> None:
        self.path = path or ""
        self.text_content = text
        self._title_base = title
        self._content_widget = TextBlockWidget(text)

        super().__init__(
            self._content_widget,
            title=self._build_title(),
            collapsed=collapsed,
            id=id,
            classes="bubble bubble-artifact"
        )

    def _build_title(self) -> str:
        t = f"[bold #9ca3af]📦 {markup_escape(self._title_base)}[/]"
        if self.path:
            t += f" [{markup_escape(self.path)}]"
        return t

    def _refresh_content(self) -> None:
        self._content_widget.set_text(self.text_content)
        self.title = self._build_title()

    def append_text(self, text: str) -> None:
        self.text_content += text
        self._refresh_content()


class LiveOutputBubbleWidget(ArtifactBubbleWidget):
    def __init__(
        self,
        *,
        text: str = "",
        command: str | None = None,
        tool_name: str | None = None,
        success: bool | None = None,
        id: str | None = None,
    ) -> None:
        self.command = str(command or "").strip()
        self.tool_name = str(tool_name or "").strip()
        self.success = success
        super().__init__(
            title="Live Output",
            text=text,
            collapsed=True,
            id=id,
        )
        self._set_content_color()
        self.add_class("bubble-liveoutput")
        _set_command_status_class(self, self.success)

    def set_command(self, command: str | None) -> None:
        if not command:
            return
        self.command = str(command).strip()
        self.title = self._build_title()

    def set_success(self, success: bool | None) -> None:
        self.success = success
        self._set_content_color()
        _set_command_status_class(self, self.success)
        self.title = self._build_title()

    def _set_content_color(self) -> None:
        self._content_widget.styles.color = "#ef4444" if self.success is False else "#16a34a"

    def _build_title(self) -> str:
        if self.success is False:
            status_color = "#ef4444"
            status_suffix = " (failed)"
        elif self.success is True:
            status_color = "#16a34a"
            status_suffix = ""
        else:
            status_color = "#16a34a"
            status_suffix = ""
        command = self.command or "Command ran"
        display_cmd = markup_escape(_trim_title_value(command))
        if self.tool_name == "ssh_exec":
            if self.command:
                display_cmd = markup_escape(_trim_title_value(json.dumps(self.command)))
            return f"[bold #16a34a]command: [/][bold {status_color}]{display_cmd}{status_suffix}[/]"
        return f"[bold #0891b2]Live Output: [/][bold {status_color}]{display_cmd}{status_suffix}[/]"


class TaskChecklistWidget(Collapsible):
    def __init__(
        self,
        *,
        title: str,
        text: str = "",
        id: str | None = None,
    ) -> None:
        self._content_widget = TextBlockWidget(text)
        super().__init__(
            self._content_widget,
            title=f"[bold #22c55e]{title}[/]",
            collapsed=True,
            id=id,
            classes="assistant-detail assistant-detail-checklist",
        )

    def set_text(self, value: str) -> None:
        self._content_widget.set_text(value)

    def set_title(self, value: str) -> None:
        self.title = value


class ToolCallDetailWidget(AssistantDetailWidget):
    def __init__(
        self,
        *,
        text: str,
        tool_name: str,
        tool_call_id: str | None = None,
        args: dict[str, Any] | None = None,
        id: str | None = None,
    ) -> None:
        import time
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self._args: dict[str, Any] = dict(args) if args else {}
        self._success: bool | None = None
        self._result_widgets: list[Widget] = []
        self._pending_full_printouts: list[ArtifactBubbleWidget] = []
        self._start_time: float = time.monotonic()
        self._done_time: float | None = None
        super().__init__(kind="tool_call", text=text, id=id)
        if self.tool_name == "ssh_exec":
            self._text = ""
            self._content_widget.set_text("")

        if self.tool_name in {"shell_exec", "ssh_exec"}:
            _set_command_status_class(self, self._success)

    @property
    def has_result(self) -> bool:
        return bool(self._result_widgets)

    def _build_title(self, text: str) -> str:
        color = KIND_COLOR.get(self.kind, "#9ca3af")
        duration_str = ""
        if self._done_time is not None:
            duration = self._done_time - self._start_time
            duration_str = f" ({_format_duration(duration)})"
        elif self._start_time is not None:
            import time
            elapsed = time.monotonic() - self._start_time
            duration_str = f" ({_format_duration(elapsed)})"

        if self.tool_name in {"shell_exec", "ssh_exec"}:
            if self._done_time is not None:
                if self._success is True:
                    base = f"{self.tool_name} succeeded"
                elif self._success is False:
                    color = "#ef4444"
                    base = f"{self.tool_name} failed"
                else:
                    base = self.tool_name or self.kind.upper()
            else:
                base = self.tool_name or self.kind.upper()
            target = ""
            if self.tool_name == "ssh_exec" and isinstance(self._args, dict):
                target = str(self._args.get("target") or "").strip()
            title = f"[bold {color}]{base}{duration_str}[/]"
            if target:
                title += f" [bold #ca8a04]{markup_escape(_trim_title_value(target))}[/]"
            return title

        preview = " ".join(text.split())
        if len(preview) > self.PREVIEW_LIMIT:
            preview = preview[: self.PREVIEW_LIMIT - 3].rstrip() + "..."
        preview = markup_escape(preview)
        base = preview or self.tool_name or self.kind.upper()
        return f"[bold {color}]{base}{duration_str}[/]"

    def finalize(self) -> None:
        if self._done_time is None:
            import time
            self._done_time = time.monotonic()
            self.title = self._build_title(self._text)
            self.refresh(layout=True)

    def update_timer(self) -> None:
        if self._done_time is None:
            new_title = self._build_title(self._text)
            if self.title != new_title:
                self.title = new_title
                self.refresh(layout=True)

    async def add_result(self, text: str, *, tool_name: str | None = None, data: dict[str, Any] | None = None) -> Widget:
        if tool_name in {"shell_exec", "ssh_exec"}:
            self._success = bool(data.get("success")) if isinstance(data, dict) else None
            _set_command_status_class(self, self._success)
            command = self._args.get("command") if isinstance(self._args, dict) else None
            live_bubble = await self.get_or_create_live_output_bubble(command=command, tool_name=tool_name, success=self._success)
            current = live_bubble.text_content
            if current and text.startswith(current.rstrip("\n")):
                suffix = text[len(current.rstrip("\n")):]
                if suffix:
                    live_bubble.append_text(suffix)
            else:
                live_bubble.append_text(text)
            self.finalize()
            self._last_shell_stream = None
            self._last_assistant_block = None
            self._last_thinking_detail = None
            return live_bubble

        data = data or {}
        artifact_id = data.get("artifact_id")

        # If it's a known artifact-generating tool or has an artifact_id, use the new bubble
        # Otherwise fall back to the standard detail block
        if artifact_id or tool_name in {"file_read", "shell_exec", "artifact_read", "grep", "yaml_read"}:
            title = "File Content" if tool_name == "file_read" else "Command Output"
            if tool_name == "grep":
                title = "Grep Matches"
            if artifact_id:
                title += f" ({artifact_id})"

            path = data.get("source") or data.get("command") or data.get("path")

            bubble = ArtifactBubbleWidget(
                title=title,
                path=path,
                text=text
            )
            bubble.add_class("assistant-detail-nested")
            await self.add_child_widget(bubble)
            self._result_widgets.append(bubble)
            await self._flush_pending_full_printouts()
            self.finalize()
            return bubble

        detail = AssistantDetailWidget(kind="tool_result", text=text, id=None)
        detail.add_class("assistant-detail-nested")
        await self.add_child_widget(detail)
        self._result_widgets.append(detail)
        await self._flush_pending_full_printouts()
        self.finalize()
        return detail

    async def get_or_create_artifact_bubble(self, title: str, *, path: str | None = None) -> ArtifactBubbleWidget:
        for w in self._result_widgets:
            if isinstance(w, ArtifactBubbleWidget) and w._title_base == title:
                return w

        # Create a new one if not found
        bubble = ArtifactBubbleWidget(
            title=title,
            path=path,
            text="",
            collapsed=True
        )
        bubble.add_class("assistant-detail-nested")
        await self.add_child_widget(bubble)
        self._result_widgets.append(bubble)
        return bubble

    async def get_or_create_live_output_bubble(
        self,
        *,
        command: str | None = None,
        tool_name: str | None = None,
        success: bool | None = None,
    ) -> LiveOutputBubbleWidget:
        for w in self._result_widgets:
            if isinstance(w, LiveOutputBubbleWidget):
                if command:
                    w.set_command(command)
                if tool_name:
                    w.tool_name = tool_name
                    w.title = w._build_title()
                if success is not None:
                    w.set_success(success)
                return w

        bubble = LiveOutputBubbleWidget(text="", command=command, tool_name=tool_name, success=success)
        bubble.add_class("assistant-detail-nested")
        await self.add_child_widget(bubble)
        self._result_widgets.append(bubble)
        return bubble

    async def attach_or_queue_full_printout(self, text: str, *, artifact_id: str | None) -> None:
        bubble = _build_full_printout_bubble(text=text, artifact_id=artifact_id)
        bubble.add_class("assistant-detail-nested")
        if self._result_widgets:
            await self._attach_full_printout_widget(bubble)
            return
        self._pending_full_printouts.append(bubble)

    async def _flush_pending_full_printouts(self) -> None:
        while self._pending_full_printouts and self._result_widgets:
            bubble = self._pending_full_printouts.pop(0)
            await self._attach_full_printout_widget(bubble)

    async def _attach_full_printout_widget(self, bubble: ArtifactBubbleWidget) -> None:
        target = self._result_widgets[-1] if self._result_widgets else None
        if isinstance(target, AssistantDetailWidget):
            await target.add_child_widget(bubble)
            return
        await self.add_child_widget(bubble)

class ToolCallsContainerWidget(Collapsible):
    def __init__(self, *, id: str | None = None) -> None:
        self._body_widget = Vertical(classes="tool-calls-container")
        import time
        self._start_time: float = time.monotonic()
        self._done_time: float | None = None
        super().__init__(
            self._body_widget,
            title=self._build_title(),
            collapsed=True,
            id=id,
            classes="assistant-detail assistant-detail-tool_call bubble bubble-tool_call"
        )

    def _build_title(self) -> str:
        if self._done_time is not None:
            duration = self._done_time - self._start_time
            return f"[bold #16a34a]🛠️ Tool Calls ({_format_duration(duration)})[/]"
        import time
        elapsed = time.monotonic() - self._start_time
        return f"[bold #16a34a]🛠️ Tool Calls ({_format_duration(elapsed)})[/]"

    def finalize(self) -> None:
        if self._done_time is None:
            import time
            self._done_time = time.monotonic()
            self.title = self._build_title()
            self.refresh(layout=True)

    def update_timer(self) -> None:
        if self._done_time is None:
            new_title = self._build_title()
            if self.title != new_title:
                self.title = new_title
                self.refresh(layout=True)

    async def add_tool_call(self, detail: Widget) -> None:
        await self._body_widget.mount(detail)


class LiveOutputContainerWidget(Collapsible):
    def __init__(self, *, id: str | None = None) -> None:
        self._body_widget = Vertical(classes="liveoutput-container")
        super().__init__(
            self._body_widget,
            title="[bold #0891b2]Live Output[/]",
            collapsed=False,
            id=id,
            classes="assistant-detail assistant-detail-liveoutput bubble bubble-liveoutput"
        )

    async def add_stream_bubble(self, bubble: ArtifactBubbleWidget) -> None:
        await self._body_widget.mount(bubble)


class AssistantTurnWidget(Vertical):
    def __init__(self, *, id: str | None = None, speaker: str = "assistant") -> None:
        super().__init__(id=id, classes="assistant-turn bubble bubble-assistant")
        self.styles.height = "auto"
        self._speaker = _coerce_speaker(speaker)
        self._label_widget = Static(self._build_label_text(), classes="assistant-turn-label")
        self._body_widget = Vertical(classes="assistant-turn-body")
        self._body_widget.styles.height = "auto"
        self._content_widget = Vertical(classes="assistant-turn-content")
        self._content_widget.styles.height = "auto"
        self._last_assistant_block: TextBlockWidget | None = None
        self._last_thinking_detail: AssistantDetailWidget | None = None
        self._last_shell_stream: ArtifactBubbleWidget | None = None
        self._tool_call_details: list[ToolCallDetailWidget] = []
        self._current_tool_calls_container: ToolCallsContainerWidget | None = None
        self._live_output_container: LiveOutputContainerWidget | None = None
        self._task_checklist_widget: TaskChecklistWidget | None = None

    def has_assistant_text(self) -> bool:
        try:
            return any(
                isinstance(child, TextBlockWidget) and child.display and child.has_content()
                for child in self._content_body().children
            )
        except Exception:
            return False

    def get_assistant_text(self) -> str:
        parts: list[str] = []
        try:
            for child in self._content_body().children:
                if isinstance(child, TextBlockWidget) and child.display:
                    parts.append(child.text)
        except Exception:
            pass
        return "\n".join(parts)

    def has_meta_content(self) -> bool:
        try:
            return bool(list(self._content_body().children))
        except Exception:
            return False

    async def add_meta_widget(self, widget: Widget) -> None:
        await self._content_widget.mount(widget)

    async def set_task_checklist(self, text: str, *, title: str = "📋 Task Checklist") -> None:
        if self._task_checklist_widget is None:
            self._task_checklist_widget = TaskChecklistWidget(title=title, text=text)
            await self._content_widget.mount(self._task_checklist_widget)
            self._content_widget.refresh(layout=True)
        else:
            self._task_checklist_widget.set_text(text)
            self._task_checklist_widget.set_title(title)
            await self._ensure_checklist_at_bottom()

    async def _ensure_checklist_at_bottom(self) -> None:
        if self._task_checklist_widget is None:
            return
        try:
            children = self._content_widget.children
        except Exception:
            return
        if not children or children[-1] is self._task_checklist_widget:
            return
        # Move existing widget to the bottom instead of destroying it
        self._content_widget.move_child(self._task_checklist_widget, after=children[-1])
        self._content_widget.refresh(layout=True)

    def compose(self) -> ComposeResult:
        yield self._label_widget
        with self._body_widget:
            yield self._content_widget

    @property
    def speaker(self) -> str:
        return self._speaker

    def set_speaker(self, speaker: str | None) -> None:
        normalized = _coerce_speaker(speaker)
        if normalized == self._speaker:
            return
        self._speaker = normalized
        self._label_widget.update(self._build_label_text())

    def _build_label_text(self) -> str:
        label = KIND_LABEL.get(self._speaker, self._speaker.upper())
        color = KIND_COLOR.get(self._speaker, "#eab308")
        return f"[bold {color}]{label}[/]"

    async def append_assistant_text(self, text: str) -> None:
        if not text:
            return
        if self._last_thinking_detail is not None:
            self._last_thinking_detail.finalize_thinking()
        if self._current_tool_calls_container is not None:
            self._current_tool_calls_container.finalize()
            self._current_tool_calls_container = None
        if self._last_assistant_block is None:
            text = _trim_leading_blank_lines(text)
            if not text:
                return
            self._last_assistant_block = TextBlockWidget(
                classes="assistant-turn-text",
                render_markdown=True,
            )
            await self._content_body().mount(self._last_assistant_block)
            await self._ensure_checklist_at_bottom()
        self._last_assistant_block.append_text(text)
        self._last_thinking_detail = None

    def finalize_assistant_render(self) -> None:
        block = self._last_assistant_block or self._latest_assistant_text_block()
        if block is not None and block.display:
            block.finalize_markdown_render()

    def _latest_assistant_text_block(self) -> TextBlockWidget | None:
        try:
            for child in reversed(self._content_body().children):
                if isinstance(child, TextBlockWidget):
                    return child
        except Exception:
            return None
        return None

    def replace_assistant_text(self, text: str) -> None:
        """Overwrite the streamed assistant text with cleaned content.
        Called after the stream finishes to remove any inline tool call JSON."""
        block = self._last_assistant_block or self._latest_assistant_text_block()
        if block is None:
            return
        cleaned = _trim_leading_blank_lines(text)
        if cleaned.strip() == "[Previous assistant output was halted because it entered a repetition loop.]":
            cleaned = ""
        if not cleaned.strip():
            block.set_text("")
            block.display = False
            if block is self._last_assistant_block:
                self._last_assistant_block = None
            self._content_widget.refresh(layout=True)
            return
        block.display = True
        block.set_text(cleaned)
        block.finalize_markdown_render()
        self._last_assistant_block = block

    async def add_full_printout(self, text: str, *, artifact_id: str | None) -> None:
        target_detail = None
        for detail in reversed(self._tool_call_details):
            if detail.tool_name == "artifact_print":
                target_detail = detail
                break

        if target_detail is not None:
            await target_detail.attach_or_queue_full_printout(text, artifact_id=artifact_id)
        else:
            bubble = _build_full_printout_bubble(text=text, artifact_id=artifact_id)
            bubble.add_class("assistant-detail-nested")
            await self._content_body().mount(bubble)
            await self._ensure_checklist_at_bottom()
        self._last_assistant_block = None
        self._last_thinking_detail = None
        self._last_shell_stream = None

    async def append_thinking_text(self, text: str) -> None:
        if not text:
            return
        if self._current_tool_calls_container is not None:
            self._current_tool_calls_container.finalize()
            self._current_tool_calls_container = None
        if self._last_thinking_detail is None:
            self._last_thinking_detail = AssistantDetailWidget(kind="thinking", text="")
            await self._content_body().mount(self._last_thinking_detail)
            await self._ensure_checklist_at_bottom()
        self._last_thinking_detail.append_text(text)
        self._last_thinking_detail.update_thinking_timer()
        self._current_tool_calls_container = None
        self._last_assistant_block = None
        self._last_shell_stream = None

    async def replace_thinking_text(self, text: str) -> None:
        if self._current_tool_calls_container is not None:
            self._current_tool_calls_container.finalize()
            self._current_tool_calls_container = None
        if self._last_thinking_detail is None:
            if not text:
                return
            self._last_thinking_detail = AssistantDetailWidget(kind="thinking", text="")
            await self._content_body().mount(self._last_thinking_detail)
            await self._ensure_checklist_at_bottom()
        self._last_thinking_detail.set_text(text)
        self._last_thinking_detail.update_thinking_timer()
        self._current_tool_calls_container = None
        self._last_assistant_block = None
        self._last_shell_stream = None

    async def append_shell_stream(
        self,
        text: str,
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        if not text:
            return

        detail = self._match_tool_call_detail(tool_name=tool_name, tool_call_id=tool_call_id)
        if detail is not None:
            bubble = await detail.get_or_create_live_output_bubble()
            bubble.append_text(text)
            self._last_shell_stream = None
            self._last_assistant_block = None
            self._last_thinking_detail = None
            return

        if tool_name in {"shell_exec", "ssh_exec"}:
            self._last_assistant_block = None
            self._last_thinking_detail = None
            if self._last_shell_stream is None:
                self._last_shell_stream = LiveOutputBubbleWidget(text="")
                self._last_shell_stream.add_class("assistant-detail-nested")
            self._last_shell_stream.append_text(text)
            if self._current_tool_calls_container is not None:
                self._current_tool_calls_container.finalize()
                self._current_tool_calls_container = None
            return

        if self._live_output_container is None:
            self._live_output_container = LiveOutputContainerWidget()
            await self._content_body().mount(self._live_output_container)

        if self._last_shell_stream is None:
            self._last_shell_stream = LiveOutputBubbleWidget(
                text="",
            )
            self._last_shell_stream.add_class("assistant-detail-nested")
            await self._live_output_container.add_stream_bubble(self._last_shell_stream)
            await self._ensure_checklist_at_bottom()
        self._last_shell_stream.append_text(text)

        if self._current_tool_calls_container is not None:
            self._current_tool_calls_container.finalize()
            self._current_tool_calls_container = None
        self._last_assistant_block = None
        self._last_thinking_detail = None

    async def add_tool_call(
        self,
        text: str,
        *,
        tool_name: str,
        tool_call_id: str | None = None,
        args: dict[str, Any] | None = None,
    ) -> ToolCallDetailWidget:
        detail = ToolCallDetailWidget(text=text, tool_name=tool_name, tool_call_id=tool_call_id, args=args)
        container = self._current_tool_calls_container
        if container is None:
            container = ToolCallsContainerWidget()
            self._current_tool_calls_container = container
            await self._content_body().mount(container)
            await self._ensure_checklist_at_bottom()
        await container.add_tool_call(detail)
        self._tool_call_details.append(detail)
        if tool_name in {"shell_exec", "ssh_exec"}:
            await self._adopt_pending_live_output(detail)
        self._last_assistant_block = None
        self._last_thinking_detail = None
        return detail

    async def _adopt_pending_live_output(self, detail: ToolCallDetailWidget) -> None:
        bubble = self._last_shell_stream
        if bubble is None:
            return
        if isinstance(bubble, LiveOutputBubbleWidget):
            command = detail._args.get("command") if isinstance(detail._args, dict) else None
            bubble.tool_name = detail.tool_name
            bubble.set_command(command)
            bubble.title = bubble._build_title()
        try:
            result = bubble.remove()
            if inspect.isawaitable(result):
                await result
        except Exception:
            pass
        await detail.add_child_widget(bubble)
        detail._result_widgets.append(bubble)
        self._last_shell_stream = None
        container = self._live_output_container
        if container is not None and not list(container._body_widget.children):
            try:
                result = container.remove()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass
            self._live_output_container = None

    async def add_tool_result(
        self,
        text: str,
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> bool:
        detail = self._match_tool_call_detail(tool_name=tool_name, tool_call_id=tool_call_id)
        if detail is None:
            return False
        await detail.add_result(text, tool_name=tool_name, data=data)
        self._last_assistant_block = None
        self._last_thinking_detail = None
        return True

    def _match_tool_call_detail(
        self,
        *,
        tool_name: str | None,
        tool_call_id: str | None,
    ) -> ToolCallDetailWidget | None:
        if tool_call_id:
            for detail in self._tool_call_details:
                if detail.tool_call_id == tool_call_id:
                    return detail
        if tool_name:
            for detail in self._tool_call_details:
                if detail.tool_name == tool_name and not detail.has_result:
                    return detail
            for detail in reversed(self._tool_call_details):
                if detail.tool_name == tool_name:
                    return detail
        for detail in self._tool_call_details:
            if not detail.has_result:
                return detail
        return self._tool_call_details[-1] if self._tool_call_details else None

    def _body(self) -> Vertical:
        return self._body_widget

    def _content_body(self) -> Vertical:
        return self._content_widget

    def _main_body(self) -> Vertical:
        return self._content_widget

    def _meta_body(self) -> Vertical:
        return self._content_widget


def _coerce_speaker(value: str | None) -> str:
    speaker = str(value or "assistant").strip().lower()
    return speaker or "assistant"


def _trim_leading_blank_lines(text: str) -> str:
    return re.sub(r"^(?:[ \t]*\r?\n)+", "", text)


class SystemInterruptWidget(Static):
    def __init__(
        self,
        *,
        text: str = "",
        id: str | None = None,
    ) -> None:
        super().__init__("", id=id, classes="bubble bubble-system-interrupt")
        self.text = text
        self._refresh_content()

    def set_text(self, value: str) -> None:
        self.text = value
        self._refresh_content()

    def _refresh_content(self) -> None:
        line = self.text.replace("[harness] ", "").strip()
        self.update(Text(f"  ⚠ {line}", style="#eab308"))
