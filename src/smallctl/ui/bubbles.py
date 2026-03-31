from typing import Any, TYPE_CHECKING
from textual.app import ComposeResult
from textual.containers import Vertical
from rich.text import Text
from rich.markup import escape as markup_escape
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Collapsible, Static

if TYPE_CHECKING:
    from .app import SmallctlApp


KIND_LABEL = {
    "user": "USER",
    "thinking": "THINK",
    "assistant": "ASSIST",
    "tool_call": "TOOL",
    "tool_result": "RESULT",
    "error": "ERROR",
    "ansible": "ANSIBLE",
    "system": "SYSTEM",
    "alert": "ALERT",
}


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
        self.text = text
        self._refresh_content()

    def set_text(self, value: str) -> None:
        self.text = value
        self._refresh_content()

    def append_text(self, value: str) -> None:
        self.text += value
        self._refresh_content()

    def _refresh_content(self) -> None:
        label = KIND_LABEL.get(self.kind, self.kind.upper())
        content = Text()
        content.append(label, style="bold")
        content.append("\n")
        content.append(self.text)
        self.update(content)

    def get_selection(self, selection: Any) -> Any:
        try:
            return super().get_selection(selection)
        except IndexError:
            # Fallback for wrapped text where UI rows > source lines
            return ([self.text], "\n") if self.text else None


class TextBlockWidget(Static):
    text: reactive[str] = reactive("")

    def __init__(self, text: str = "", id: str | None = None, classes: str | None = None) -> None:
        super().__init__("", id=id, classes=classes)
        self.text = text
        self._refresh_content()

    def set_text(self, value: str) -> None:
        self.text = value
        self._refresh_content()

    def append_text(self, value: str) -> None:
        self.text += value
        self._refresh_content()

    def _refresh_content(self) -> None:
        # Use a plain Text object to avoid Rich markup parsing on arbitrary model output.
        # Model responses may contain brackets like [task_complete(...)] that Rich
        # would try to parse as markup tags, causing 'Expected markup value' crashes.
        # would try to parse as markup tags, causing 'Expected markup value' crashes.
        self.update(Text(self.text))

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
        self.title = self._build_title(value)

    def append_text(self, value: str) -> None:
        self.set_text(f"{self._text}{value}")

    def _build_title(self, text: str) -> str:
        label = KIND_LABEL.get(self.kind, self.kind.upper())
        preview = " ".join(text.split())
        if len(preview) > self.PREVIEW_LIMIT:
            preview = preview[: self.PREVIEW_LIMIT - 3].rstrip() + "..."
        preview = markup_escape(preview)
        return f"{label}: {preview}" if preview else label

    async def add_child_widget(self, widget: Widget) -> None:
        await self._body_widget.mount(widget)


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
        t = f"📦 ARTIFACT: {markup_escape(self._title_base)}"
        if self.path:
            t += f" [{markup_escape(self.path)}]"
        return t

    def _refresh_content(self) -> None:
        self._content_widget.set_text(self.text_content)
        self.title = self._build_title()

    def append_text(self, text: str) -> None:
        self.text_content += text
        self._refresh_content()


class ToolCallDetailWidget(AssistantDetailWidget):
    def __init__(
        self,
        *,
        text: str,
        tool_name: str,
        tool_call_id: str | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(kind="tool_call", text=text, id=id)
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self._result_widgets: list[Widget] = []

    @property
    def has_result(self) -> bool:
        return bool(self._result_widgets)

    async def add_result(self, text: str, *, tool_name: str | None = None, data: dict[str, Any] | None = None) -> Widget:
        data = data or {}
        artifact_id = data.get("artifact_id")
        
        # If it's a known artifact-generating tool or has an artifact_id, use the new bubble
        # Otherwise fall back to the standard detail block
        if artifact_id or tool_name in {"file_read", "shell_exec", "artifact_read", "grep", "yaml_read"}:
            title = "File Content" if tool_name == "file_read" else "Command Output"
            if tool_name == "grep": title = "Grep Matches"
            if artifact_id: title += f" ({artifact_id})"
            
            path = data.get("source") or data.get("command") or data.get("path")
            
            bubble = ArtifactBubbleWidget(
                title=title,
                path=path,
                text=text
            )
            bubble.add_class("assistant-detail-nested")
            await self.add_child_widget(bubble)
            self._result_widgets.append(bubble)
            return bubble
            
        detail = AssistantDetailWidget(kind="tool_result", text=text, id=None)
        detail.add_class("assistant-detail-nested")
        await self.add_child_widget(detail)
        self._result_widgets.append(detail)
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


class AssistantTurnWidget(Vertical):
    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="assistant-turn bubble bubble-assistant")
        self._last_assistant_block: TextBlockWidget | None = None
        self._last_thinking_detail: AssistantDetailWidget | None = None
        self._last_shell_stream: ArtifactBubbleWidget | None = None
        self._tool_call_details: list[ToolCallDetailWidget] = []

    def has_assistant_text(self) -> bool:
        return self._last_assistant_block is not None and self._last_assistant_block.has_content()

    def get_assistant_text(self) -> str:
        parts: list[str] = []
        try:
            body = self._body()
            for child in body.children:
                if isinstance(child, TextBlockWidget):
                    parts.append(child.text)
        except Exception:
            pass
        return "\n".join(parts)

    def compose(self) -> ComposeResult:
        yield Static(KIND_LABEL["assistant"], classes="assistant-turn-label")
        yield Vertical(classes="assistant-turn-body")

    async def append_assistant_text(self, text: str) -> None:
        if not text:
            return
        if self._last_assistant_block is None:
            self._last_assistant_block = TextBlockWidget(classes="assistant-turn-text")
            await self._body().mount(self._last_assistant_block)
        self._last_assistant_block.append_text(text)
        self._last_thinking_detail = None

    def replace_assistant_text(self, text: str) -> None:
        """Overwrite the current assistant text block with cleaned content.
        Called after the stream finishes to remove any inline tool call JSON."""
        if self._last_assistant_block is not None:
            self._last_assistant_block.set_text(text)

    async def add_full_printout(self, text: str, *, artifact_id: str | None) -> None:
        title = "Full Artifact Contents"
        if artifact_id:
            title += f" ({artifact_id})"
        bubble = ArtifactBubbleWidget(title=title, text=text)
        bubble.add_class("assistant-detail-nested")
        await self._body().mount(bubble)
        self._last_assistant_block = None
        self._last_thinking_detail = None
        self._last_shell_stream = None

    async def append_thinking_text(self, text: str) -> None:
        if not text:
            return
        if self._last_thinking_detail is None:
            self._last_thinking_detail = AssistantDetailWidget(kind="thinking", text="")
            await self._body().mount(self._last_thinking_detail)
        self._last_thinking_detail.append_text(text)
        self._last_assistant_block = None
        self._last_shell_stream = None

    async def append_shell_stream(self, text: str) -> None:
        if not text:
            return
            
        # Find the active shell_exec call to nest the stream under
        shell_call = None
        for detail in reversed(self._tool_call_details):
            if detail.tool_name in {"shell_exec", "ssh_exec"}:
                shell_call = detail
                break
        
        if shell_call:
            bubble = await shell_call.get_or_create_artifact_bubble("Live Output")
            bubble.append_text(text)
        else:
            # Fallback to turn-level stream if no tool call found (unlikely)
            if self._last_shell_stream is None:
                self._last_shell_stream = ArtifactBubbleWidget(
                    title="Live Output",
                    text=""
                )
                self._last_shell_stream.add_class("assistant-detail-nested")
                await self._body().mount(self._last_shell_stream)
            self._last_shell_stream.append_text(text)
            
        self._last_assistant_block = None
        self._last_thinking_detail = None

    async def add_tool_call(
        self,
        text: str,
        *,
        tool_name: str,
        tool_call_id: str | None = None,
    ) -> ToolCallDetailWidget:
        detail = ToolCallDetailWidget(text=text, tool_name=tool_name, tool_call_id=tool_call_id)
        await self._body().mount(detail)
        self._tool_call_details.append(detail)
        self._last_assistant_block = None
        self._last_thinking_detail = None
        return detail

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
        return self.query_one(".assistant-turn-body", Vertical)
