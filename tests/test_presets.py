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
    assert cfg.first_token_timeout_sec == 45


def test_staged_reasoning_flag_parses_from_cli_and_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SMALLCTL_STAGED_REASONING", "true")
    env_config = resolve_config({})
    assert env_config.staged_reasoning is True

    cli_config = resolve_config({"staged_reasoning": False})
    assert cli_config.staged_reasoning is False


def test_unknown_preset_reports_compatibility_warning(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = resolve_config({"preset": "does-not-exist"})
    assert any("Unknown preset" in item for item in cfg.compatibility_warnings)


def test_lmstudio_warns_when_restart_fallback_is_unconfigured(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = resolve_config({"provider_profile": "lmstudio"})
    assert any("LM Studio native API unload will be attempted automatically" in item for item in cfg.compatibility_warnings)


def test_lmstudio_warns_when_first_token_timeout_is_too_low(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = resolve_config({"provider_profile": "lmstudio", "first_token_timeout_sec": 25})
    assert any("first_token_timeout_sec below 45s" in item for item in cfg.compatibility_warnings)
