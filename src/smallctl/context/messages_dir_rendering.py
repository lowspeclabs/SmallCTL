from __future__ import annotations

from ..models.tool_result import ToolEnvelope
from ..state import ArtifactRecord

LISTING_PREVIEW_ENTRY_LIMIT = 50


def render_dir_list_tree(
    items: list[object],
    *,
    max_depth: int = 2,
    max_children: int = 8,
) -> str:
    lines: list[str] = []
    append_dir_tree_lines(
        lines,
        items,
        depth=0,
        max_depth=max_depth,
        max_children=max_children,
    )
    return "\n".join(lines).strip()


def render_dir_list_result(
    items: list[object],
    *,
    metadata: dict[str, object] | None = None,
    max_depth: int = 2,
    max_children: int = 8,
    max_items: int | None = None,
) -> str:
    lines: list[str] = []
    listing_metadata = metadata if isinstance(metadata, dict) else {}

    path = listing_metadata.get("path")
    total_items = listing_metadata.get("total_items")
    count = total_items if isinstance(total_items, int) and total_items >= 0 else listing_metadata.get("count")
    if not isinstance(count, int):
        count = len(items)

    if isinstance(path, str) and path.strip():
        if count >= 0:
            lines.append(f"{path.strip()} ({count} items)")
        else:
            lines.append(path.strip())
    elif count >= 0:
        lines.append(f"{count} items")

    rendered_items = items if max_items is None else items[:max_items]
    tree_preview = render_dir_list_tree(
        rendered_items,
        max_depth=max_depth,
        max_children=max_children,
    )
    if tree_preview:
        lines.append(tree_preview)

    if max_items is not None:
        remaining = len(items) - len(rendered_items)
        if remaining > 0:
            lines.append(f"... {remaining} more items")

    return "\n".join(lines).strip() or "directory listed"


def dir_list_preview_is_incomplete(
    artifact: ArtifactRecord,
    *,
    result: ToolEnvelope | None = None,
) -> bool:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    total_items = metadata.get("total_items")
    if total_items is None:
        total_items = metadata.get("count")
    if total_items is None and result and result.output:
        if isinstance(result.output, list):
            total_items = len(result.output)

    if total_items is not None and total_items > LISTING_PREVIEW_ENTRY_LIMIT:
        return True
    if result and isinstance(result.output, list) and dir_list_tree_has_truncation(result.output):
        return True
    return False


def dir_list_tree_has_truncation(items: list[object]) -> bool:
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("children_truncated"):
            return True
        children = item.get("children")
        if isinstance(children, list) and dir_list_tree_has_truncation(children):
            return True
    return False


def listing_preview_is_incomplete(
    artifact: ArtifactRecord,
    *,
    result: ToolEnvelope | None = None,
) -> bool:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    if metadata.get("truncated"):
        return True
    return dir_list_preview_is_incomplete(artifact, result=result)


def append_dir_tree_lines(
    lines: list[str],
    items: list[object],
    *,
    depth: int,
    max_depth: int,
    max_children: int,
) -> None:
    indent = "  " * depth
    preview_items = items[:max_children]
    for item in preview_items:
        append_dir_tree_line(
            lines,
            item,
            depth=depth,
            max_depth=max_depth,
            max_children=max_children,
        )
    if len(items) > len(preview_items):
        lines.append(f"{indent}... {len(items) - len(preview_items)} more items")


def append_dir_tree_line(
    lines: list[str],
    item: object,
    *,
    depth: int,
    max_depth: int,
    max_children: int,
) -> None:
    indent = "  " * depth
    if not isinstance(item, dict):
        text = str(item or "").strip()
        if text:
            lines.append(f"{indent}{text}")
        return

    name = str(item.get("name") or item.get("path") or "").strip()
    if not name:
        return

    parts = [name]
    item_type = str(item.get("type") or "").strip()
    if item_type:
        parts.append(f"[{item_type}]")

    size = item.get("size")
    if isinstance(size, int) and size >= 0:
        parts.append(f"({size} bytes)")

    children = item.get("children")
    children_count = item.get("children_count")
    if item_type == "dir" and isinstance(children_count, int) and children_count >= 0:
        parts.append(f"({children_count} children)")

    lines.append(f"{indent}{' '.join(parts).strip()}")

    if depth >= max_depth:
        if isinstance(children_count, int) and children_count > 0 and isinstance(children, list):
            lines.append(f"{indent}  ... more nested items")
        return

    if isinstance(children, list) and children:
        append_dir_tree_lines(
            lines,
            children,
            depth=depth + 1,
            max_depth=max_depth,
            max_children=max_children,
        )
        if item.get("children_truncated"):
            lines.append(f"{indent}  ... more nested items")
