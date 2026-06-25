from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_client_package_import_surface_stable() -> None:
    pkg = importlib.import_module("smallctl.client")
    assert hasattr(pkg, "OpenAICompatClient")
    assert hasattr(pkg, "SSEStreamer")
    assert hasattr(pkg, "get_provider_adapter")


def test_legacy_src_import_path_still_resolves() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    legacy = importlib.import_module("src.smallctl.client")
    assert hasattr(legacy, "OpenAICompatClient")
    assert hasattr(legacy, "SSEStreamer")


def test_normalize_backend_model_name_strips_provider_prefixes() -> None:
    from smallctl.client import OpenAICompatClient

    assert OpenAICompatClient.normalize_backend_model_name("openrouter/google/gemma-4-e4b") == "gemma-4b"
    assert OpenAICompatClient.normalize_backend_model_name("google/gemma-4-it") == "gemma-4"
    assert OpenAICompatClient.normalize_backend_model_name("local/qwen2.5-4b-instruct-q4_k_m") == "qwen2.5-4b"


def test_normalize_backend_model_name_handles_windows_gguf_paths() -> None:
    from smallctl.client import OpenAICompatClient

    windows_path = (
        r"C:\Users\svaye\.lmstudio\models\lmstudio-community"
        r"\gemma-4-E4B-it-GGUF\gemma-4-E4B-it-Q4_K_M.gguf"
    )
    assert OpenAICompatClient.normalize_backend_model_name(windows_path) == "gemma-4b"
    assert OpenAICompatClient.normalize_backend_model_name("/home/user/models/gemma-4-e4b-it-q4_k_m.gguf") == "gemma-4b"
    assert OpenAICompatClient.normalize_backend_model_name("gemma-4-e4b-it-q4_k_m.gguf") == "gemma-4b"


def test_apply_backend_model_profile_only_when_context_below_32k() -> None:
    from smallctl.client import OpenAICompatClient

    client = OpenAICompatClient("http://localhost", "gemma")
    assert client.apply_backend_model_profile("openrouter/google/gemma-4-e4b", 16384) == "gemma-4b"
    assert client.is_small_model is True

    client2 = OpenAICompatClient("http://localhost", "gemma")
    assert client2.apply_backend_model_profile("openrouter/google/gemma-4-e4b", 32768) is None
    assert client2.is_small_model is False


def test_apply_backend_model_profile_ignores_non_small_models() -> None:
    from smallctl.client import OpenAICompatClient

    client = OpenAICompatClient("http://localhost", "unknown")
    assert client.apply_backend_model_profile("openrouter/anthropic/claude-3-5-sonnet", 16384) is None
    assert client.is_small_model is False
