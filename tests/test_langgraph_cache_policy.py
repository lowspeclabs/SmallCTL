"""Phase 6: LangGraph native CachePolicy evaluation.

Caching is rejected for this codebase. No existing graph node is pure for a single
run with a provably correct cache key:

- ``prepare_indexer_prompt`` was the closest candidate, but it calls
  ``harness._build_prompt_messages()``, whose output depends on the active
  conversation state, retrieval state, available tools, and model configuration.
  A complete cache key would need to include every one of those inputs, and any
  omission would silently serve stale prompts.
- The actual source-indexing work is performed by ``model_call`` and mutating
  tool calls (``index_batch_write``, ``index_finalize``), which are explicitly
  excluded from caching by the safety rules.

Consequently the feature flag exists only for future experiments and must stay
off. These tests document that rejection.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from smallctl.config import SmallctlConfig, resolve_config
from smallctl.graph.runtime import ChatGraphRuntime, LoopGraphRuntime
from smallctl.graph.runtime_specialized import (
    IndexerGraphRuntime,
    PlanningGraphRuntime,
    ToolPlanRuntime,
)
from smallctl.graph.runtime_staged import StagedExecutionRuntime
from smallctl.harness import HarnessConfig


RUNTIME_GRAPH_CLASSES: list[Any] = [
    LoopGraphRuntime,
    ChatGraphRuntime,
    PlanningGraphRuntime,
    IndexerGraphRuntime,
    ToolPlanRuntime,
    StagedExecutionRuntime,
]


def test_config_flag_defaults_to_false() -> None:
    assert SmallctlConfig().langgraph_cache_policy_enabled is False


def test_harness_config_flag_defaults_to_false() -> None:
    config = HarnessConfig(endpoint="http://example.test/v1", model="test-model")
    assert config.langgraph_cache_policy_enabled is False


def test_cli_flag_can_enable() -> None:
    config = resolve_config({"langgraph_cache_policy_enabled": True})
    assert config.langgraph_cache_policy_enabled is True


def test_env_var_string_is_parsed_as_bool() -> None:
    config = resolve_config({"langgraph_cache_policy_enabled": "1"})
    assert config.langgraph_cache_policy_enabled is True


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("runtime_cls", RUNTIME_GRAPH_CLASSES)
def test_no_compiled_graph_node_has_cache_policy(
    runtime_cls: type,
    enabled: bool,
) -> None:
    """CachePolicy is not attached to any node because no candidate was approved."""
    harness = SimpleNamespace(
        config=SimpleNamespace(langgraph_cache_policy_enabled=enabled),
        graph_checkpointer="memory",
        graph_checkpoint_path=None,
        _runlog=lambda *args, **kwargs: None,
    )
    runtime = runtime_cls.from_harness(harness)
    compiled = runtime._build_compiled_graph()

    for node_name, spec in compiled.builder.nodes.items():
        cache_policy = getattr(spec, "cache_policy", None)
        assert cache_policy is None, f"{node_name} unexpectedly has a cache_policy"
