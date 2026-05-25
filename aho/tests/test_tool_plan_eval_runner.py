"""
tests/test_tool_plan_eval_runner.py
-----------------------------------
Integration tests for the AHO harness scorer and runner pipeline.
Validates that trial results are correctly scored and that the evaluation
pipeline handles failure modes, stagnation, and tool history as expected.
"""

from __future__ import annotations

import pytest
from aho.eval import score_trial, score_strategy


class TestScoreTrial:
    def test_perfect_trial(self):
        trial = {
            "result": {"status": "completed", "summary": "The weather is sunny."},
            "token_usage": 500,
            "step_count": 5,
            "tool_records": {
                "op_1": {"tool_name": "weather_lookup", "args": {"city": "Paris"}, "result": {}},
                "op_2": {"tool_name": "clothing_suggest", "args": {"condition": "sunny"}, "result": {}},
            },
            "failure_events": [],
            "stagnation_counters": {},
            "tool_history": ["weather_lookup", "clothing_suggest", "task_complete"],
            "error": None,
        }
        cfg = {
            "scoring": {
                "w_accuracy": 0.6,
                "w_format": 0.3,
                "w_latency": 0.05,
                "w_stagnation": 0.05,
                "latency_penalty_per_100_tokens": 0.01,
            },
            "strategy": {"tool_call_format": "strict_xml"},
        }
        ground_truth = {"expected_keywords": ["sunny"], "required_tool_calls": ["weather_lookup"]}

        result = score_trial(trial, cfg, ground_truth)
        assert result["accuracy"] == 1.0
        assert result["format_score"] > 0.8
        assert result["harness_score"] > 0.8
        assert result["stagnation_penalty"] == 0.0

    def test_crash_trial(self):
        trial = {
            "result": {"status": "failed", "reason": "AttributeError"},
            "token_usage": 0,
            "step_count": 0,
            "tool_records": {},
            "failure_events": [],
            "stagnation_counters": {},
            "tool_history": [],
            "error": "AttributeError: 'NoneType' object has no attribute 'split'",
        }
        cfg = {"scoring": {}, "strategy": {}}
        ground_truth = {"expected_keywords": ["sunny"]}

        result = score_trial(trial, cfg, ground_truth)
        assert result["accuracy"] == 0.0
        assert result["harness_score"] == 0.0
        assert "trial crashed" in result["bugs"][0]

    def test_stagnation_penalty(self):
        trial = {
            "result": {"status": "completed", "summary": "done"},
            "token_usage": 200,
            "step_count": 10,
            "tool_records": {},
            "failure_events": [
                {"failure_class": "loop_guard", "message": "Repeated tool call", "severity": "warning"},
                {"failure_class": "loop_guard", "message": "Repeated tool call", "severity": "warning"},
            ],
            "stagnation_counters": {"no_actionable_progress": 3},
            "tool_history": ["file_read", "file_read", "file_read", "file_read"],
            "error": None,
        }
        cfg = {
            "scoring": {
                "w_accuracy": 0.6,
                "w_format": 0.3,
                "w_latency": 0.05,
                "w_stagnation": 0.05,
            },
            "strategy": {"tool_call_format": "strict_xml"},
        }
        ground_truth = {"expected_keywords": ["done"]}

        result = score_trial(trial, cfg, ground_truth)
        assert result["stagnation_penalty"] > 0.0
        assert any("stagnation" in fm for fm in result["failure_modes"])

    def test_hallucinated_tool_penalty(self):
        trial = {
            "result": {"status": "completed", "summary": "answer"},
            "token_usage": 100,
            "step_count": 3,
            "tool_records": {
                "op_1": {"tool_name": "fake_tool_9000", "args": {}, "result": {}},
            },
            "failure_events": [],
            "stagnation_counters": {},
            "tool_history": ["fake_tool_9000"],
            "error": None,
        }
        cfg = {
            "scoring": {},
            "strategy": {"tool_call_format": "strict_xml"},
        }
        ground_truth = {"expected_keywords": ["answer"]}

        result = score_trial(trial, cfg, ground_truth)
        assert result["format_score"] < 1.0
        assert any("hallucinated_tool" in fm for fm in result["failure_modes"])


class TestScoreStrategy:
    def test_empty_trials(self):
        result = score_strategy([], {}, {})
        assert result["pass_at_n"] == 0.0
        assert result["mean_harness_score"] == 0.0
        assert "no_trials_returned" in result["failure_modes"]

    def test_mixed_trials(self):
        cfg = {
            "scoring": {
                "w_accuracy": 0.6,
                "w_format": 0.3,
                "w_latency": 0.05,
                "w_stagnation": 0.05,
            },
            "strategy": {"tool_call_format": "strict_xml"},
        }
        ground_truth = {"expected_keywords": ["sunny"]}

        trials = [
            {
                "result": {"status": "completed", "summary": "It is sunny."},
                "token_usage": 300,
                "step_count": 4,
                "tool_records": {},
                "failure_events": [],
                "stagnation_counters": {},
                "tool_history": [],
                "error": None,
            },
            {
                "result": {"status": "completed", "summary": "It is rainy."},
                "token_usage": 400,
                "step_count": 5,
                "tool_records": {},
                "failure_events": [],
                "stagnation_counters": {},
                "tool_history": [],
                "error": None,
            },
        ]

        result = score_strategy(trials, cfg, ground_truth)
        assert result["pass_at_n"] == 0.5
        assert result["n_bugs"] == 0
        assert len(result["trial_scores"]) == 2


class TestPipelineIntegration:
    def test_aboracle_extract_json_array(self):
        from aho.validation.pipeline import ABOracle

        # Whole text is JSON
        assert ABOracle._extract_json_array('[{"a": 1}]') == [{"a": 1}]

        # Mixed output with trailing text
        text = 'some log\n[{"a": 1}, {"b": 2}]\nDone'
        assert ABOracle._extract_json_array(text) == [{"a": 1}, {"b": 2}]

        # No array
        assert ABOracle._extract_json_array("just text") is None

    def test_std_dev_helper(self):
        from aho.validation.pipeline import _std_dev

        assert _std_dev([1.0, 1.0, 1.0]) == 0.0
        assert _std_dev([]) == 0.0
        # population std dev of [0, 1] = 0.5
        assert _std_dev([0.0, 1.0]) == 0.5
