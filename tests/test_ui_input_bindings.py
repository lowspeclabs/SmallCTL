from __future__ import annotations

from smallctl.ui.input import InputPane


def test_input_pane_supports_terminal_paste_shortcuts() -> None:
    pane = InputPane()

    for key in ("ctrl+v", "ctrl+shift+v", "shift+insert"):
        bindings = pane._bindings.get_bindings_for_key(key)
        assert any(binding.action == "paste" for binding in bindings)
