from __future__ import annotations

from pathlib import Path

from smallctl.config import resolve_config


def test_preset_applies_defaults(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = resolve_config({"preset": "safe-small-model"})
    assert cfg.preset == "safe-small-model"
    assert cfg.max_prompt_tokens == 4096
    assert cfg.reasoning_mode == "off"


def test_cli_override_wins_over_preset(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = resolve_config({"preset": "safe-small-model", "max_prompt_tokens": 6000})
    assert cfg.max_prompt_tokens == 6000


def test_local_yaml_override_wins_over_preset(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".smallctl.yaml").write_text("max_prompt_tokens: 7777\n", encoding="utf-8")
    cfg = resolve_config({"preset": "safe-small-model"})
    assert cfg.max_prompt_tokens == 7777


def test_lmstudio_preset_sets_provider_profile(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = resolve_config({"preset": "lmstudio-small-model"})
    assert cfg.provider_profile == "lmstudio"
    assert cfg.reasoning_mode == "tags"


def test_unknown_preset_reports_compatibility_warning(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = resolve_config({"preset": "does-not-exist"})
    assert any("Unknown preset" in item for item in cfg.compatibility_warnings)
