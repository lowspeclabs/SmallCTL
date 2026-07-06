from __future__ import annotations

import pytest

from smallctl.harness import Harness


def test_gemma_4_with_llamacpp_logs_reasoning_budget_warning(tmp_path: pytest.TempPathFactory, monkeypatch, caplog) -> None:
    monkeypatch.chdir(tmp_path)

    with caplog.at_level("WARNING", logger="smallctl.harness.initialization"):
        Harness(
            endpoint="http://192.168.1.9:8080/v1",
            model="Gemma 4 12b",
            provider_profile="llamacpp",
            phase="explore",
            api_key="test-key",
        )

    assert any("--reasoning-budget" in record.message for record in caplog.records)
    assert any("Gemma-4 model with llama.cpp" in record.message for record in caplog.records)
    assert any("--swa-full" in record.message for record in caplog.records)
    assert any("--ctx-size" in record.message for record in caplog.records)


def test_gemma_4_with_lmstudio_does_not_log_reasoning_budget_warning(tmp_path: pytest.TempPathFactory, monkeypatch, caplog) -> None:
    monkeypatch.chdir(tmp_path)

    with caplog.at_level("WARNING", logger="smallctl.harness.initialization"):
        Harness(
            endpoint="http://example.test/v1",
            model="Gemma 4 12b",
            provider_profile="lmstudio",
            phase="explore",
            api_key="test-key",
        )

    assert not any("--reasoning-budget" in record.message for record in caplog.records)


def test_non_gemma_model_with_llamacpp_does_not_log_reasoning_budget_warning(tmp_path: pytest.TempPathFactory, monkeypatch, caplog) -> None:
    monkeypatch.chdir(tmp_path)

    with caplog.at_level("WARNING", logger="smallctl.harness.initialization"):
        Harness(
            endpoint="http://192.168.1.9:8080/v1",
            model="qwen2.5-coder-7b-instruct",
            provider_profile="llamacpp",
            phase="explore",
            api_key="test-key",
        )

    assert not any("--reasoning-budget" in record.message for record in caplog.records)
