from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models.tool_result import ToolEnvelope
from ..state import ArtifactRecord, json_safe_value
from .rendering import render_shell_failure, render_shell_output
from .messages import _LISTING_PREVIEW_ENTRY_LIMIT, format_compact_tool_message
from .policy import estimate_text_tokens


@dataclass
class ArtifactPolicy:
    inline_token_limit: int = 1000
    preview_char_limit: int = 4000
    force_artifact_tools: tuple[str, ...] = (
        "file_read",
        "grep",
        "http_get",
        "http_post",
        "yaml_read",
        "shell_exec",
        "find_files",
        "ansible_inventory",
        "ansible_task",
        "ansible_playbook",
    )

    def should_externalize(self, *, tool_name: str, serialized_output: str) -> bool:
        if tool_name == "artifact_read":
            return False
        return tool_name in self.force_artifact_tools or estimate_text_tokens(serialized_output) > self.inline_token_limit


class ArtifactStore:
    def __init__(
        self, 
        base_dir: Path, 
        run_id: str, 
        policy: ArtifactPolicy | None = None,
        artifact_start_index: int | None = None,
    ) -> None:
        self.policy = policy or ArtifactPolicy()
        self.run_id = run_id
        self.run_dir = base_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        discovered = self._discover_next_index()
        self._next_index = max(discovered, artifact_start_index or 1)

    def persist_tool_result(self, *, tool_name: str, result: ToolEnvelope) -> ArtifactRecord:
        if tool_name == "shell_exec" and isinstance(result.metadata, dict):
            result.metadata.setdefault("model_visible", False)
            result.metadata.setdefault("hidden", True)
        payload = result.to_dict()
        serialized = json.dumps(payload, ensure_ascii=True, default=str, indent=2)
        artifact_id = f"A{self._next_index:04d}"
        self._next_index += 1
        metadata = _coerce_metadata_payload(result.metadata)
        if tool_name == "shell_exec":
            metadata["model_visible"] = False
            metadata["hidden"] = True
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        txt_path = self.run_dir / f"{artifact_id}.txt"
        json_path = self.run_dir / f"{artifact_id}.json"

        # Write full envelope to .json for audit
        json_path.write_text(serialized, encoding="utf-8")

        # Write meaningful content to .txt for artifact_read/paging
        txt_content: str
        if isinstance(result.output, str):
            # Raw string allows proper line-paging
            txt_content = result.output
        elif tool_name == "shell_exec" and not result.success:
            failure_output = result.metadata.get("output") if isinstance(result.metadata, dict) else None
            txt_content = render_shell_failure(
                error=result.error,
                output=failure_output if isinstance(failure_output, dict) else None,
                preview_limit=None,
                strip_whitespace=False,
            )
        elif isinstance(result.output, dict) and self._is_shell_output(result.output):
            # Shell-like output should stay readable instead of being wrapped in JSON.
            txt_content = render_shell_output(result.output, preview_limit=None, strip_whitespace=False)
        elif isinstance(result.output, dict):
            plain_text = self._structured_plain_text(result.output)
            if plain_text is not None:
                txt_content = plain_text
            else:
                # Fallback to JSON-serialized output for structured data
                txt_content = json.dumps(result.output, ensure_ascii=True, default=str, indent=2)
        else:
            # Fallback to JSON-serialized output for structured data
            txt_content = json.dumps(result.output, ensure_ascii=True, default=str, indent=2)
        txt_path.write_text(txt_content, encoding="utf-8")

        source = self._source_from_metadata(tool_name=tool_name, metadata=metadata)
        summary = self._summarize(tool_name=tool_name, result=result, metadata=metadata)
        keywords = self._keywords(source=source, summary=summary, metadata=metadata)
        path_tags = self._path_tags(metadata)
        inline_content = None
        if not self.policy.should_externalize(tool_name=tool_name, serialized_output=serialized):
            inline_content = txt_content
        preview_text = self._preview_text(
            tool_name=tool_name,
            result=result,
            metadata=metadata,
            limit=self.policy.preview_char_limit,
        )
        return ArtifactRecord(
            artifact_id=artifact_id,
            kind=tool_name,
            source=source,
            created_at=created_at,
            size_bytes=len(serialized.encode("utf-8")),
            summary=summary,
            keywords=keywords,
            path_tags=path_tags,
            tool_name=tool_name,
            content_path=str(txt_path),
            inline_content=inline_content,
            preview_text=preview_text,
            metadata=metadata,
        )

    def persist_thinking(self, *, raw_thinking: str, summary: str, source: str = "assistant") -> ArtifactRecord:
        artifact_id = f"A{self._next_index:04d}"
        self._next_index += 1
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        txt_path = self.run_dir / f"{artifact_id}.txt"
        txt_path.write_text(raw_thinking, encoding="utf-8")

        return ArtifactRecord(
            artifact_id=artifact_id,
            kind="thought",
            source=source,
            created_at=created_at,
            size_bytes=len(raw_thinking.encode("utf-8")),
            summary=summary,
            tool_name="",
            content_path=str(txt_path),
            inline_content=None,
            preview_text=raw_thinking[:self.policy.preview_char_limit],
            metadata={"distilled": True},
        )

    def compact_tool_message(
        self,
        artifact: ArtifactRecord,
        result: ToolEnvelope,
        *,
        request_text: str | None = None,
    ) -> str:
        return format_compact_tool_message(artifact, result, request_text=request_text)

    @staticmethod
    def _is_shell_output(output: dict[str, Any]) -> bool:
        return "stdout" in output or "stderr" in output

    def _discover_next_index(self) -> int:
        indexes = []
        for path in self.run_dir.glob("A*.json"):
            try:
                indexes.append(int(path.stem[1:]))
            except ValueError:
                continue
        return (max(indexes) + 1) if indexes else 1

    @staticmethod
    def _source_from_metadata(*, tool_name: str, metadata: dict[str, Any]) -> str:
        for key in ("path", "url", "command", "name"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return tool_name

    @staticmethod
    def _summarize(*, tool_name: str, result: ToolEnvelope, metadata: dict[str, Any]) -> str:
        if not result.success and result.error:
            return result.error[:160]
        source = metadata.get("path") or metadata.get("url") or metadata.get("command")
        output = result.output
        if isinstance(output, str):
            if isinstance(source, str) and source:
                total_lines = metadata.get("total_lines")
                complete_file = bool(metadata.get("complete_file"))
                if "line_start" in metadata or "line_end" in metadata:
                    if complete_file and isinstance(total_lines, int) and total_lines > 0:
                        return f"{Path(source).name} full file ({total_lines} lines)"
                    return (
                        f"{Path(source).name} lines "
                        f"{metadata.get('line_start', '?')}-{metadata.get('line_end', '?')}"
                    )
                return f"{Path(source).name} text ({len(output)} chars)"
            compact = " ".join(output.split())
            if len(compact) > 120:
                return f"{tool_name} text ({len(output)} chars)"
            return compact[:160] or tool_name
        if isinstance(output, dict):
            keys = ", ".join(sorted(output.keys())[:5])
            return f"{tool_name} keys: {keys}" if keys else tool_name
        if isinstance(output, list):
            return f"{tool_name} returned {len(output)} items"
        if source:
            return str(source)
        return tool_name

    @staticmethod
    def _keywords(*, source: str, summary: str, metadata: dict[str, Any]) -> list[str]:
        chunks = [source, summary]
        for key in ("path", "url", "command", "content_type"):
            value = metadata.get(key)
            if isinstance(value, str):
                chunks.append(value)
        joined = " ".join(chunks).lower()
        tokens = re.findall(r"[a-z0-9_.:/\\-]+", joined)
        seen: list[str] = []
        for token in tokens:
            if token not in seen:
                seen.append(token)
        return seen[:20]

    @staticmethod
    def _path_tags(metadata: dict[str, Any]) -> list[str]:
        path = metadata.get("path")
        if not isinstance(path, str) or not path:
            return []
        target = Path(path)
        tags = [target.name]
        tags.extend(part for part in target.parts[-4:-1])
        deduped: list[str] = []
        for tag in tags:
            if tag and tag not in deduped:
                deduped.append(tag)
        return deduped

    @staticmethod
    def _preview_text(
        *,
        tool_name: str,
        result: ToolEnvelope,
        metadata: dict[str, Any],
        limit: int,
    ) -> str | None:
        if not result.success:
            return result.error[:limit] if result.error else None
        output = result.output
        if isinstance(output, str):
            preview = output[:limit].strip()
            return preview or None
        if isinstance(output, dict):
            plain_preview = ArtifactStore._structured_plain_text(output)
            if plain_preview is not None:
                preview = plain_preview[:limit].strip()
                return preview or None
            if ArtifactStore._is_shell_output(output):
                preview = render_shell_output(output, preview_limit=limit, strip_whitespace=False)
                return preview or None
        structured_preview = ArtifactStore._structured_preview(output=output, limit=min(limit, 400))
        if structured_preview:
            return structured_preview
        if tool_name == "yaml_read" and output is not None:
            rendered = json.dumps(output, ensure_ascii=True, default=str)
            return rendered[: min(limit, 400)]
        source = metadata.get("path") or metadata.get("url")
        if isinstance(source, str) and source:
            return source[:limit]
        return None

    @staticmethod
    def _structured_preview(*, output: Any, limit: int) -> str | None:
        if isinstance(output, dict):
            text_preview = ArtifactStore._structured_text_preview(output=output, limit=limit)
            if text_preview:
                return text_preview
            rendered = json.dumps(output, ensure_ascii=True, default=str)
            return rendered[:limit] if rendered else None
        if isinstance(output, list):
            lines: list[str] = []
            for item in output[:_LISTING_PREVIEW_ENTRY_LIMIT]:
                line = ArtifactStore._preview_line(item)
                if line:
                    lines.append(line)
            if len(output) > _LISTING_PREVIEW_ENTRY_LIMIT:
                lines.append(f"... {len(output) - _LISTING_PREVIEW_ENTRY_LIMIT} more items")
            preview = "\n".join(lines).strip()
            if preview:
                return preview[:limit]
            rendered = json.dumps(output[:_LISTING_PREVIEW_ENTRY_LIMIT], ensure_ascii=True, default=str)
            return rendered[:limit] if rendered else None
        return None

    @staticmethod
    def _structured_text_preview(*, output: dict[str, Any], limit: int) -> str | None:
        plain_text = ArtifactStore._structured_plain_text(output)
        if plain_text is not None:
            return plain_text[:limit]
        sections: list[str] = []
        for key in ("stdout", "stderr", "content", "message"):
            value = output.get(key)
            if isinstance(value, str) and value.strip():
                label = f"{key}:"
                sections.append(f"{label}\n{value.strip()}")
            if len("\n\n".join(sections)) >= limit:
                break
        preview = "\n\n".join(sections).strip()
        return preview[:limit] if preview else None

    @staticmethod
    def _structured_plain_text(output: dict[str, Any]) -> str | None:
        keys = set(output.keys())
        if keys <= {"status", "message"}:
            value = output.get("message")
        elif keys <= {"status", "question"}:
            value = output.get("question")
        else:
            return None
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    @staticmethod
    def _preview_line(item: Any) -> str:
        if isinstance(item, dict):
            name = item.get("name") or item.get("relative") or item.get("path") or item.get("id")
            if isinstance(name, str) and name:
                line = name
                item_type = item.get("type")
                if isinstance(item_type, str) and item_type:
                    line = f"{line} [{item_type}]"
                size = item.get("size")
                if isinstance(size, int) and size >= 0:
                    line = f"{line} ({size} bytes)"
                return line
            rendered = json.dumps(item, ensure_ascii=True, default=str)
            return rendered
        if item is None:
            return ""
        return str(item)


def _coerce_metadata_payload(value: Any) -> dict[str, Any]:
    normalized = json_safe_value(value or {})
    return normalized if isinstance(normalized, dict) else {}
