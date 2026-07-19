import json
import inspect
import logging
import re
import time
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


logger = logging.getLogger("smallctl.ui.bubbles")

_RICH_MARKDOWN_MAX_CHARS = 8192


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

_LITE_DIM_STYLE = "#9ca3af"
_LITE_MARKER_STYLE = "bold #9ca3af"
_LITE_HEADER_STYLE = "bold #ffffff"
_LITE_CODE_STYLE = "italic #93c5fd"
_LITE_QUOTE_STYLE = "#9ca3af"

_EMPH_TOKENS_RE = re.compile(r"(\*\*|\*|~~)")


def _make_style(styles: set[str]) -> str:
    return " ".join(sorted(styles))


def _style_emphasis(segment: str) -> Text:
    result = Text()
    active: set[str] = set()
    last = 0
    for match in _EMPH_TOKENS_RE.finditer(segment):
        result.append(segment[last : match.start()], _make_style(active))
        token = match.group(0)
        if token == "**":
            active.symmetric_difference_update({"bold"})
        elif token == "*":
            active.symmetric_difference_update({"italic"})
        elif token == "~~":
            active.symmetric_difference_update({"strike"})
        last = match.end()
    result.append(segment[last:], _make_style(active))
    return result


def _style_inline(line: str) -> Text:
    result = Text()
    parts: list[tuple[bool, str]] = []
    in_code = False
    code_run_len = 0
    last = 0
    for match in re.finditer(r"`+", line):
        if not in_code:
            parts.append((False, line[last : match.start()]))
            code_run_len = len(match.group(0))
            in_code = True
        elif len(match.group(0)) == code_run_len:
            parts.append((True, line[last : match.start()]))
            in_code = False
        last = match.end()
    if in_code:
        # Unclosed backtick: treat the whole line as plain text so we don't
        # accidentally hide the remainder of a long code block while streaming.
        return Text(line)
    parts.append((False, line[last:]))
    for is_code, segment in parts:
        if is_code:
            result.append(segment, _LITE_CODE_STYLE)
        else:
            result.append(_style_emphasis(segment))
    return result


def _style_line_lite(line: str, in_fence: bool, *, inline: bool = True) -> Text:
    if in_fence:
        return Text(line, style=_LITE_DIM_STYLE)
    stripped = line.lstrip()
    if stripped.startswith("```"):
        return Text(line, style=_LITE_DIM_STYLE)
    if re.fullmatch(r"\s*(?:---+|===+|\*\*\*+|___+)\s*", line):
        return Text(line, style=_LITE_DIM_STYLE)

    header_match = re.match(r"^(#{1,6})(\s+)(.*)$", line)
    if header_match:
        text = Text()
        text.append(header_match.group(1), _LITE_DIM_STYLE)
        text.append(header_match.group(2), "")
        if inline:
            text.append(_style_inline(header_match.group(3)))
        else:
            text.append(header_match.group(3))
        text.stylize(_LITE_HEADER_STYLE, len(header_match.group(1)) + len(header_match.group(2)), len(line))
        return text

    bullet_match = re.match(r"^(\s*)[-*+](\s)(.*)$", line)
    if bullet_match:
        text = Text()
        text.append(bullet_match.group(1), "")
        text.append("•", _LITE_MARKER_STYLE)
        text.append(bullet_match.group(2), "")
        if inline:
            text.append(_style_inline(bullet_match.group(3)))
        else:
            text.append(bullet_match.group(3))
        return text

    number_match = re.match(r"^(\s*)(\d+)\.(\s)(.*)$", line)
    if number_match:
        text = Text()
        text.append(number_match.group(1), "")
        text.append(number_match.group(2) + ".", _LITE_MARKER_STYLE)
        text.append(number_match.group(3), "")
        if inline:
            text.append(_style_inline(number_match.group(4)))
        else:
            text.append(number_match.group(4))
        return text

    quote_match = re.match(r"^(\s*)>(\s?)(.*)$", line)
    if quote_match:
        text = Text()
        text.append(quote_match.group(1) + ">", _LITE_QUOTE_STYLE)
        text.append(quote_match.group(2), "")
        if inline:
            text.append(_style_inline(quote_match.group(3)))
        else:
            text.append(quote_match.group(3))
        return text

    if inline:
        return _style_inline(line)
    return Text(line)


def _style_text_lite(text: str) -> Text:
    result = Text()
    in_fence = False
    lines = text.splitlines()
    ends_with_newline = text.endswith("\n")
    for index, line in enumerate(lines):
        is_last_line = index == len(lines) - 1
        # Only apply inline (emphasis/code) styling to complete lines. The
        # last line is incomplete if the input does not end with a newline;
        # styling it while tokens are still arriving causes chunk-boundary
        # artifacts such as `***CPU` being rendered as bold+italic.
        inline_ok = not (is_last_line and not ends_with_newline)
        result.append(_style_line_lite(line, in_fence, inline=inline_ok))
        if index < len(lines) - 1 or ends_with_newline:
            result.append("\n", "")
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
    return result


def _looks_like_complete_pipe_table(text: str) -> bool:
    lines = [line.rstrip() for line in text.splitlines()]
    for index in range(len(lines) - 2):
        header = lines[index].strip()
        separator = lines[index + 1].strip()
        if "|" not in header or "|" not in separator:
            continue
        separator_cells = [cell.strip() for cell in separator.strip("|").split("|")]
        if not separator_cells or not all(
            re.fullmatch(r":?-{3,}:?", cell) for cell in separator_cells
        ):
            continue
        expected_cells = len(separator_cells)
        body_count = 0
        for row in lines[index + 2 :]:
            stripped = row.strip()
            if not stripped:
                break
            if "|" not in stripped:
                break
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if len(cells) != expected_cells:
                if body_count == 0:
                    return False
                break
            body_count += 1
        if body_count > 0:
            return True
    return False


def _inline_code_backticks_balanced(text: str) -> bool:
    """Return True if every inline code backtick span is properly closed.

    This is more accurate than a global backtick count because it ignores
    triple-backtick code fences and checks each inline span independently.
    Markdown allows runs of one or more backticks to delimit inline code,
    so `` ` `` and ```` ``code`` ```` are both valid.

    The scan is single-pass over the input; it tracks fenced code blocks by
    triple+ backtick runs at the start of a line and uses a small stack to
    track open inline runs.  This avoids the previous O(n^2) backtracking
    regex and nested search.
    """
    stack: list[int] = []
    in_fence = False
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "`":
            i += 1
            continue
        start = i
        while i < n and text[i] == "`":
            i += 1
        run_len = i - start
        # Triple+ backticks at the start of a line toggle a fenced code block.
        at_line_start = start == 0 or text[start - 1] == "\n"
        if run_len >= 3 and at_line_start:
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        # Inline code span: a matching run of the same length closes the
        # current span; runs of a different length are content inside the
        # span (e.g. `` ` ``) and are ignored.
        if not stack:
            stack.append(run_len)
        elif stack[-1] == run_len:
            stack.pop()
    return not stack


def _markdown_render_ready(text: str) -> bool:
    if not text.strip():
        return False
    if text.count("```") % 2 == 1:
        return False
    if not _inline_code_backticks_balanced(text):
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
    # Emphasis / strikethrough spans also benefit from full rendering.
    if re.search(r"\*\*[^*\n]+\*\*", text):
        return True
    if re.search(r"(?<![*\w])\*[^*\n]+\*(?![*\w])", text):
        return True
    if re.search(r"~~[^~\n]+~~", text):
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


def _build_full_printout_bubble(
    *, text: str, artifact_id: str | None
) -> "ArtifactBubbleWidget":
    title = "Full Artifact Contents"
    if artifact_id:
        title += f" ({artifact_id})"
    return ArtifactBubbleWidget(title=title, text=_format_full_printout_text(text))


def _command_status(success: bool | None, *, command_ran: bool = False) -> str:
    if success is False and command_ran:
        return "warning"
    if success is False:
        return "error"
    return "success"


def _set_command_status_class(
    widget: Widget, success: bool | None, *, command_ran: bool = False
) -> None:
    status = _command_status(success, command_ran=command_ran)
    widget.remove_class(
        "tool-status-success", "tool-status-warning", "tool-status-error"
    )
    widget.add_class(f"tool-status-{status}")
    color_map = {"success": "#16a34a", "warning": "#eab308", "error": "#ef4444"}
    if hasattr(widget, "_title") and widget._title is not None:
        widget._title.styles.color = color_map[status]


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
            return (self.text, "\n") if self.text else None


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
        self._rich_markdown_renderable: object | None = None
        self._rich_markdown_text: str | None = None
        self._plain_text_renderable = Text(text)
        self.text = text
        self._refresh_content()

    def set_text(self, value: str) -> None:
        if value == self.text:
            return
        self.text = value
        self._markdown_finalized = False
        self._rich_markdown_renderable = None
        self._rich_markdown_text = None
        self._plain_text_renderable = (
            _style_text_lite(value) if self.render_markdown else Text(value)
        )
        self._refresh_content()

    def append_text(self, value: str) -> None:
        if not value:
            return
        self.text += value
        self._markdown_finalized = False
        self._rich_markdown_renderable = None
        self._rich_markdown_text = None
        self._plain_text_renderable = (
            _style_text_lite(self.text) if self.render_markdown else Text(self.text)
        )
        self._refresh_content()

    def finalize_markdown_render(self) -> None:
        if not self.render_markdown:
            return
        self._markdown_finalized = True
        self._refresh_content()

    def _refresh_content(self) -> None:
        started = time.perf_counter()
        text_len = len(self.text)
        try:
            if (
                self.render_markdown
                and self._markdown_finalized
                and text_len <= _RICH_MARKDOWN_MAX_CHARS
                and _markdown_render_ready(self.text)
            ):
                try:
                    if (
                        self._rich_markdown_renderable is None
                        or self._rich_markdown_text != self.text
                    ):
                        self._rich_markdown_renderable = RichMarkdown(self.text)
                        self._rich_markdown_text = self.text
                    self._rendered_content = self._rich_markdown_renderable
                    self.update(self._rendered_content)
                    return
                except Exception:
                    # Malformed markdown can occasionally crash Rich's parser.
                    # Fall back to plain text so the UI never goes blank.
                    self._rich_markdown_renderable = None
                    self._rich_markdown_text = None
                    pass
            # Prefer cheap, live markdown styling while streaming; fall back to a
            # plain Text object for non-markdown blocks or malformed content.
            self._plain_text_renderable = (
                _style_text_lite(self.text) if self.render_markdown else Text(self.text)
            )
            self._rendered_content = self._plain_text_renderable
            self.update(self._plain_text_renderable)
        finally:
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            if elapsed_ms > 5 or text_len > 10000:
                segment_count = 0
                rendered = self._rendered_content
                if rendered is not None:
                    segs = getattr(rendered, "_segments", None) or getattr(rendered, "segments", None)
                    if isinstance(segs, list):
                        segment_count = len(segs)
                logger.debug(
                    "text_block_refresh %s",
                    {
                        "elapsed_ms": elapsed_ms,
                        "text_len": text_len,
                        "markdown_finalized": self._markdown_finalized,
                        "render_markdown": self.render_markdown,
                        "segment_count": segment_count,
                    },
                )

    def get_selection(self, selection: Any) -> Any:
        try:
            return super().get_selection(selection)
        except IndexError:
            # Fallback for wrapped text where UI rows > source lines
            return (self.text, "\n") if self.text else None

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
        started = time.perf_counter()
        try:
            color = KIND_COLOR.get(self.kind, "#9ca3af")
            if self.kind == "thinking":
                return self._build_thinking_title()
            preview = " ".join(text.split())
            if len(preview) > self.PREVIEW_LIMIT:
                preview = preview[: self.PREVIEW_LIMIT - 3].rstrip() + "..."
            preview = markup_escape(preview)
            return (
                f"[bold {color}]{preview}[/]"
                if preview
                else f"[bold {color}]{self.kind.upper()}[/]"
            )
        finally:
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            if elapsed_ms > 5 or len(text) > 10000:
                logger.debug(
                    "assistant_detail_build_title %s",
                    {"elapsed_ms": elapsed_ms, "text_len": len(text), "kind": self.kind},
                )

    def _build_thinking_title(self) -> str:
        import time

        color = KIND_COLOR.get("thinking", "#ca8a04")
        if (
            self._thinking_done_time is not None
            and self._thinking_start_time is not None
        ):
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
            classes="bubble bubble-artifact",
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
        self.command_ran = False
        super().__init__(
            title="Live Output",
            text=text,
            collapsed=True,
            id=id,
        )
        self._set_content_color()
        self.add_class("bubble-liveoutput")
        _set_command_status_class(self, self.success, command_ran=self.command_ran)

    def set_command(self, command: str | None) -> None:
        if not command:
            return
        self.command = str(command).strip()
        self.title = self._build_title()

    def set_success(self, success: bool | None, *, command_ran: bool = False) -> None:
        self.success = success
        self.command_ran = command_ran
        self._set_content_color()
        _set_command_status_class(self, self.success, command_ran=self.command_ran)
        self.title = self._build_title()

    def _set_content_color(self) -> None:
        self._content_widget.styles.color = (
            "#ef4444" if self.success is False else "#16a34a"
        )

    def _build_title(self) -> str:
        if self.success is False and self.command_ran:
            label_color = "#eab308"
            status_color = "#ef4444"
            status_suffix = " (failed)"
        elif self.success is False:
            label_color = "#ef4444"
            status_color = "#ef4444"
            status_suffix = " (failed)"
        elif self.success is True:
            label_color = "#16a34a"
            status_color = "#16a34a"
            status_suffix = ""
        else:
            label_color = "#16a34a"
            status_color = "#16a34a"
            status_suffix = ""
        command = self.command or "Command ran"
        display_cmd = markup_escape(_trim_title_value(command))
        if self.tool_name == "ssh_exec":
            if self.command:
                display_cmd = markup_escape(_trim_title_value(json.dumps(self.command)))
            return f"[bold {label_color}]command: [/][bold {status_color}]{display_cmd}{status_suffix}[/]"
        return f"[bold {label_color}]Live Output: [/][bold {status_color}]{display_cmd}{status_suffix}[/]"


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
        self._command_ran: bool = False
        self._result_widgets: list[Widget] = []
        self._pending_full_printouts: list[ArtifactBubbleWidget] = []
        self._start_time: float = time.monotonic()
        self._done_time: float | None = None
        super().__init__(kind="tool_call", text=text, id=id)
        # The collapsible title already shows the tool call signature/status,
        # so keep the body reserved for nested result widgets and avoid
        # duplicating the signature below the title.
        self._content_widget.display = False
        if self.tool_name in {"shell_exec", "ssh_exec"}:
            _set_command_status_class(
                self, self._success, command_ran=self._command_ran
            )

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
                    color = "#eab308" if self._command_ran else "#ef4444"
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

    async def add_result(
        self,
        text: str,
        *,
        tool_name: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> Widget:
        if tool_name in {"shell_exec", "ssh_exec"}:
            output_dict = None
            if isinstance(data, dict):
                output_dict = data.get("output")
            if (
                not isinstance(output_dict, dict)
                and isinstance(text, str)
                and text.strip().startswith("{")
            ):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict) and "output" in parsed:
                        output_dict = parsed["output"]
                except Exception:
                    pass

            exit_code = (
                output_dict.get("exit_code") if isinstance(output_dict, dict) else None
            )
            self._success = (
                bool(data.get("success")) if isinstance(data, dict) else None
            )
            self._command_ran = self._success is False and exit_code is not None
            _set_command_status_class(
                self, self._success, command_ran=self._command_ran
            )
            command = (
                self._args.get("command") if isinstance(self._args, dict) else None
            )
            live_bubble = await self.get_or_create_live_output_bubble(
                command=command,
                tool_name=tool_name,
                success=self._success,
                command_ran=self._command_ran,
            )

            if isinstance(output_dict, dict):
                stdout = str(output_dict.get("stdout") or "")
                stderr = str(output_dict.get("stderr") or "")
                parts = []
                if stdout:
                    parts.append(stdout)
                if stderr:
                    parts.append(f"--- [STDERR] ---\n{stderr}")
                if exit_code is not None:
                    parts.append(f"--- [EXIT CODE: {exit_code}] ---")
                final_text = "\n\n".join(parts) or "ok"
                live_bubble.text_content = final_text
                live_bubble._refresh_content()
            else:
                current = live_bubble.text_content
                clean_text = text
                if isinstance(text, str) and "--- [PROGRESS] ---" in text:
                    parts = text.split("\n\n")
                    non_progress_parts = [
                        p for p in parts if not p.startswith("--- [PROGRESS] ---")
                    ]
                    clean_text = "\n\n".join(non_progress_parts)

                if current and clean_text.startswith(current.rstrip("\n")):
                    suffix = clean_text[len(current.rstrip("\n")) :]
                    if suffix:
                        live_bubble.append_text(suffix)
                elif not current:
                    live_bubble.append_text(text)
                else:
                    exit_code_match = re.search(
                        r"--- \[EXIT CODE: \d+\] ---", clean_text
                    )
                    if exit_code_match and "EXIT CODE" not in current:
                        live_bubble.append_text(f"\n\n{exit_code_match.group(0)}")

            self.finalize()
            self._last_shell_stream = None
            self._last_assistant_block = None
            self._last_thinking_detail = None
            return live_bubble

        data = data or {}
        artifact_id = data.get("artifact_id")
        output_dict = data.get("output") if isinstance(data, dict) else None
        if not isinstance(output_dict, dict):
            output_dict = None
        if not artifact_id and isinstance(output_dict, dict):
            artifact_id = output_dict.get("artifact_id") or output_dict.get(
                "body_artifact_id"
            )

        # If it's a known artifact-generating tool or has an artifact_id, use the new bubble
        # Otherwise fall back to the standard detail block
        if artifact_id or tool_name in {
            "file_read",
            "shell_exec",
            "artifact_read",
            "grep",
            "yaml_read",
            "web_search",
            "web_fetch",
        }:
            title = "File Content" if tool_name == "file_read" else "Command Output"
            if tool_name == "grep":
                title = "Grep Matches"
            elif tool_name == "web_search":
                title = "Web Search Results"
            elif tool_name == "web_fetch":
                title = "Web Fetch Result"
            if artifact_id:
                title += f" ({artifact_id})"

            path = data.get("source") or data.get("command") or data.get("path")
            if not path and tool_name == "web_fetch" and isinstance(output_dict, dict):
                path = output_dict.get("url") or output_dict.get("canonical_url")
            if not path and tool_name == "web_search" and isinstance(self._args, dict):
                path = self._args.get("query")

            bubble = ArtifactBubbleWidget(title=title, path=path, text=text)
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

    async def get_or_create_artifact_bubble(
        self, title: str, *, path: str | None = None
    ) -> ArtifactBubbleWidget:
        for w in self._result_widgets:
            if isinstance(w, ArtifactBubbleWidget) and w._title_base == title:
                return w

        # Create a new one if not found
        bubble = ArtifactBubbleWidget(title=title, path=path, text="", collapsed=True)
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
        command_ran: bool = False,
    ) -> LiveOutputBubbleWidget:
        for w in self._result_widgets:
            if isinstance(w, LiveOutputBubbleWidget):
                if command:
                    w.set_command(command)
                if tool_name:
                    w.tool_name = tool_name
                    w.title = w._build_title()
                if success is not None:
                    w.set_success(success, command_ran=command_ran)
                return w

        bubble = LiveOutputBubbleWidget(
            text="", command=command, tool_name=tool_name, success=success
        )
        bubble.set_success(success, command_ran=command_ran)
        bubble.add_class("assistant-detail-nested")
        await self.add_child_widget(bubble)
        self._result_widgets.append(bubble)
        return bubble

    async def attach_or_queue_full_printout(
        self, text: str, *, artifact_id: str | None
    ) -> None:
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
            classes="assistant-detail assistant-detail-tool_call bubble bubble-tool_call",
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
            classes="assistant-detail assistant-detail-liveoutput bubble bubble-liveoutput",
        )

    async def add_stream_bubble(self, bubble: ArtifactBubbleWidget) -> None:
        await self._body_widget.mount(bubble)


class AssistantTurnWidget(Vertical):
    def __init__(self, *, id: str | None = None, speaker: str = "assistant") -> None:
        super().__init__(id=id, classes="assistant-turn bubble bubble-assistant")
        self.styles.height = "auto"
        self._speaker = _coerce_speaker(speaker)
        self._label_widget = Static(
            self._build_label_text(), classes="assistant-turn-label"
        )
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
                isinstance(child, TextBlockWidget)
                and child.display
                and child.has_content()
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

    async def set_task_checklist(
        self, text: str, *, title: str = "📋 Task Checklist"
    ) -> None:
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
        started = time.perf_counter()
        text_len = len(text)
        try:
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
                body = self._content_body()
                if not body.is_mounted:
                    self._last_assistant_block = None
                    return
                await body.mount(self._last_assistant_block)
                await self._ensure_checklist_at_bottom()
            self._last_assistant_block.append_text(text)
            self._last_thinking_detail = None
        finally:
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            if elapsed_ms > 5 or text_len > 10000:
                logger.debug(
                    "assistant_turn_append_text %s",
                    {
                        "elapsed_ms": elapsed_ms,
                        "chunk_len": text_len,
                        "content_widget_children": len(self._content_widget.children),
                    },
                )

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

    def has_visible_content(self) -> bool:
        """Return True if this turn has any user-facing content to display."""
        try:
            for child in self._content_body().children:
                if not getattr(child, "display", True):
                    continue
                if isinstance(child, TextBlockWidget) and child.has_content():
                    return True
                if isinstance(
                    child,
                    (
                        AssistantDetailWidget,
                        ToolCallsContainerWidget,
                        ArtifactBubbleWidget,
                        LiveOutputContainerWidget,
                        TaskChecklistWidget,
                    ),
                ):
                    return True
        except Exception:
            pass
        return False

    def replace_assistant_text(self, text: str) -> None:
        """Overwrite the streamed assistant text with cleaned content.
        Called after the stream finishes to remove any inline tool call JSON."""
        block = self._last_assistant_block or self._latest_assistant_text_block()
        if block is None:
            return
        cleaned = _trim_leading_blank_lines(text)
        if (
            cleaned.strip()
            == "[Previous assistant output was halted because it entered a repetition loop.]"
        ):
            cleaned = ""
        if not cleaned.strip():
            block.set_text("")
            block.display = False
            if block is self._last_assistant_block:
                self._last_assistant_block = None
            self._content_widget.refresh(layout=True)
            # Hide the entire turn if nothing visible remains after blanking.
            if not self.has_visible_content():
                self.display = False
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
            await target_detail.attach_or_queue_full_printout(
                text, artifact_id=artifact_id
            )
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
            body = self._content_body()
            if not body.is_mounted:
                return
            self._last_thinking_detail = AssistantDetailWidget(kind="thinking", text="")
            await body.mount(self._last_thinking_detail)
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
            body = self._content_body()
            if not body.is_mounted:
                return
            self._last_thinking_detail = AssistantDetailWidget(kind="thinking", text="")
            await body.mount(self._last_thinking_detail)
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

        detail = self._match_tool_call_detail(
            tool_name=tool_name, tool_call_id=tool_call_id
        )
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
        detail = ToolCallDetailWidget(
            text=text, tool_name=tool_name, tool_call_id=tool_call_id, args=args
        )
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
            command = (
                detail._args.get("command") if isinstance(detail._args, dict) else None
            )
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
        detail = self._match_tool_call_detail(
            tool_name=tool_name, tool_call_id=tool_call_id
        )
        if detail is None:
            return False
        await detail.add_result(text, tool_name=tool_name, data=data)
        self._last_assistant_block = None
        self._last_thinking_detail = None
        return True

    def find_tool_call_detail(
        self,
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
    ) -> ToolCallDetailWidget | None:
        if not self._tool_call_details:
            return None
        if tool_call_id:
            for detail in self._tool_call_details:
                if detail.tool_call_id == tool_call_id:
                    return detail
        if tool_name:
            for detail in self._tool_call_details:
                if detail.tool_name == tool_name:
                    return detail
        return None

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
