from __future__ import annotations

import asyncio

from smallctl.tools.fs import file_write


def test_file_write_rejects_truncated_tool_json_payload(tmp_path) -> None:
    target = tmp_path / "rogue-grid-defense.html"

    result = asyncio.run(file_write(str(target), '{"tool_name'))

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "truncated_write_payload"
    assert not target.exists()


def test_file_write_rejects_truncated_html_starter_payload(tmp_path) -> None:
    target = tmp_path / "rogue-grid-defense.html"

    result = asyncio.run(file_write(str(target), "<!DOCTYPE"))

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "truncated_write_payload"
    assert not target.exists()


def test_file_write_allows_full_content_over_poisoned_truncated_payload(tmp_path) -> None:
    target = tmp_path / "rogue-grid-defense.html"
    target.write_text('{"tool_name', encoding="utf-8")
    full_html = """<!DOCTYPE html>
<html>
<head><title>Rogue Grid Defense</title></head>
<body><main id="game"></main><script>const grid = Array.from({length: 12});</script></body>
</html>
"""

    result = asyncio.run(file_write(str(target), full_html))

    assert result["success"] is True
    assert target.read_text(encoding="utf-8") == full_html


def test_file_write_allows_full_content_over_truncated_html_stub(tmp_path) -> None:
    target = tmp_path / "rogue-grid-defense.html"
    target.write_text("<!DOCTYPE", encoding="utf-8")
    full_html = """<!DOCTYPE html>
<html>
<head><title>Rogue Grid Defense</title></head>
<body><main id="game"></main><script>const grid = Array.from({length: 12});</script></body>
</html>
"""

    result = asyncio.run(file_write(str(target), full_html))

    assert result["success"] is True
    assert target.read_text(encoding="utf-8") == full_html
