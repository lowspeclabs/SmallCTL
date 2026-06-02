from __future__ import annotations

from smallctl.ui.input import InputPane, _format_paste_preview


def test_input_pane_supports_terminal_paste_shortcuts() -> None:
    pane = InputPane()

    for key in ("ctrl+v", "ctrl+shift+v", "shift+insert"):
        bindings = pane._bindings.get_bindings_for_key(key)
        assert any(binding.action == "paste" for binding in bindings)


def test_input_pane_paste_shows_preview_but_submits_full_text(monkeypatch) -> None:
    pane = InputPane()
    full_text = "x" * 541
    posted = []

    monkeypatch.setattr(pane, "post_message", posted.append)

    pane._paste_text(full_text)

    assert pane.text == "[pasted ~541 chars]"

    pane.action_submit()

    assert posted[-1].value == full_text
    assert posted[-1].display_value == "[pasted ~541 chars]"
    assert pane._raw_text_override is None


def test_input_pane_multiline_paste_preview() -> None:
    assert _format_paste_preview("a\nb\nc\nd") == "[pasted ~4 lines]"
