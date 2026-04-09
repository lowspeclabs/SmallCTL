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
