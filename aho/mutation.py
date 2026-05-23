"""
aho/mutation.py
---------------
Mutation proposal and aggregation logic for the researcher module.

This module extracts the LLM-based mutation proposal system from
researcher.py to keep the main research flow clean.
"""

from __future__ import annotations

import json
from typing import Any

from aho.logging_aho import AHOLogger


# The researcher LLM system prompt
RESEARCHER_SYSTEM_PROMPT = """\
You are an autonomous research agent optimizing the `strategy` block of a
prompt configuration for Qwen 2.5 7B (an SLM with a 8k context window).

Your goal is to maximize the mean_harness_score by proposing ONE targeted
mutation to the strategy JSON.

Input you will receive
----------------------
  current_strategy    — the strategy dict from harness_config.json
  recent_scores       — list of recent mean_harness_score values (newest last)
  failure_modes       — list of aggregated failure mode strings with frequencies
  bugs                — list of bug strings from the last run

Output rules
------------
- Reply ONLY with a single valid JSON object (no markdown, no explanation).
- The object must contain ONLY keys that exist in the strategy dict.
- Change EXACTLY ONE key per response.
- Base your mutation on the most frequent failure mode.
- Do not propose a mutation identical to the previous one.

Allowed strategy keys and their valid values
--------------------------------------------
  thought_architecture: "think_before_every_tool_call" | "silent" | "chain_of_thought" | "reflection_after_tool" | "reflection_plus" | "multi_phase_discovery"
  delimiter_style:      "xml" | "json" | "markdown"
  error_handling:       "retry_with_hint" | "fail_fast" | "ignore_and_continue"
  tool_call_format:     "strict_xml" | "relaxed_json"
  system_prompt_addendum: any string (keep under 200 chars)
  forbidden_patterns:   list of strings
  max_steps:            integer 4..20

Failure mode → suggested fix quick reference
-------------------------------------------
  json_inside_think_block           → add "```json inside <think>" to forbidden_patterns
  hallucinated_tool                 → change delimiter_style to "xml"
  missing_required_tool             → change thought_architecture to "multi_phase_discovery"
  accuracy: answer missing keywords → change thought_architecture to "reflection_plus"
  latency: token_usage is high      → reduce max_steps; add length warning to system_prompt_addendum
  markdown_code_fence_in_tool_args  → change tool_call_format to "strict_xml"
  tool_parse_error                  → change delimiter_style to "xml"
"""


async def propose_mutation(
    cfg: dict[str, Any],
    recent_results: list[dict[str, Any]],
    logger: AHOLogger,
) -> dict[str, Any]:
    """
    Ask the LLM (same local endpoint) to propose a single strategy patch.
    Returns a dict of { strategy_key: new_value }.
    """
    from src.smallctl.client import OpenAICompatClient

    client = OpenAICompatClient(
        base_url=cfg["endpoint"],
        model=cfg["model"],
        api_key=cfg.get("api_key", "local-dev-key"),
        chat_endpoint="/chat/completions",
    )

    context: dict[str, Any] = {
        "current_strategy": cfg.get("strategy", {}),
        "recent_scores": [r.get("mean_harness_score", 0) for r in recent_results],
        "failure_modes": aggregate_failure_modes(recent_results),
        "bugs": aggregate_bugs(recent_results),
    }

    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(context, indent=2)},
    ]

    chunks: list[dict[str, Any]] = []
    async for event in client.stream_chat(messages=messages, tools=[]):
        chunks.append(event)

    response = OpenAICompatClient.collect_stream(
        chunks,
        reasoning_mode="off",
        thinking_start_tag="<thinking>",
        thinking_end_tag="</thinking>",
    )

    raw_text = response.assistant_text.strip()

    # Strip markdown fences if model wraps its response
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]

    patch: dict[str, Any] = json.loads(raw_text)
    logger.info("researcher", "mutation_proposed",
                f"proposed patch: {json.dumps(patch)}", patch=patch)
    return patch


def aggregate_failure_modes(results: list[dict[str, Any]]) -> list[str]:
    """Aggregate failure modes from recent results with counts."""
    from collections import Counter

    counts: Counter[str] = Counter()
    for r in results:
        for m in r.get("failure_modes", []):
            counts[m] += 1
    return [f"{m} (x{c})" for m, c in counts.most_common(5)]


def aggregate_bugs(results: list[dict[str, Any]]) -> list[str]:
    """Aggregate unique bugs from recent results (up to 5)."""
    bugs: list[str] = []
    for r in results:
        bugs.extend(r.get("bugs", []))
    return list(dict.fromkeys(bugs))[:5]  # deduplicated, up to 5


def apply_patch(cfg: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Apply a mutation patch to the config. Returns a new config dict."""
    mutated: dict[str, Any] = json.loads(json.dumps(cfg))
    allowed_keys = {
        "thought_architecture", "delimiter_style", "error_handling",
        "tool_call_format", "system_prompt_addendum", "forbidden_patterns", "max_steps",
    }
    for key, value in patch.items():
        if key in allowed_keys:
            mutated["strategy"][key] = value
    mutated["version"] = cfg.get("version", 0) + 1
    mutated["strategy_id"] = f"v{mutated['version']}"
    return mutated
