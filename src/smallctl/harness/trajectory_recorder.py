from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..redaction import redact_sensitive_data


class TrajectoryRecorder:
    """Records clean tool_plan trajectories for future fine-tuning data."""

    def __init__(self, base_dir: str | Path = ".smallctl/traces") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def record_tool_plan_trajectory(
        self,
        harness: Any,
        result: dict[str, Any],
    ) -> Path | None:
        state = getattr(harness, "state", None)
        if state is None:
            return None
        scratchpad = getattr(state, "scratchpad", None)
        if not isinstance(scratchpad, dict):
            return None

        plan = scratchpad.get("_tool_plan")
        observations_text = str(scratchpad.get("_tool_plan_observations_text") or "").strip()
        refine_passes = int(scratchpad.get("_tool_plan_refine_passes", 0) or 0)
        refine_verdict = str(
            scratchpad.get("_tool_plan_refine_verdict")
            or getattr(state, "scratchpad", {}).get("_recovery_metrics", {}).get("tool_plan_refine_verdict", "pass")
        ).strip() or "pass"

        session_id = str(getattr(state, "thread_id", "") or getattr(harness, "conversation_id", "") or "unknown")
        task = str(
            getattr(state, "run_brief", None) and getattr(state.run_brief, "original_task", "")
            or getattr(state, "run_brief", None) and getattr(state.run_brief, "effective_task", "")
            or ""
        ).strip()

        metrics = getattr(state, "scratchpad", {}).get("_recovery_metrics")
        metrics = metrics if isinstance(metrics, dict) else {}

        tool_plan_steps: list[dict[str, Any]] = []
        if isinstance(plan, dict):
            raw_steps = plan.get("steps")
            if isinstance(raw_steps, list):
                for step in raw_steps:
                    if isinstance(step, dict):
                        tool_plan_steps.append({
                            "id": step.get("id"),
                            "tool": step.get("tool"),
                            "args": step.get("args"),
                        })

        payload: dict[str, Any] = {
            "task": task,
            "runtime": "tool_plan",
            "session_id": session_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tool_plan": tool_plan_steps,
            "observations": observations_text.splitlines() if observations_text else [],
            "solver_draft": str(scratchpad.get("_last_solver_draft") or "").strip(),
            "refine_verdict": refine_verdict,
            "final_answer": str(result.get("reason") or result.get("message") or "").strip(),
            "success": str(result.get("status") or "").strip().lower() in {"completed", "success", "ok"},
            "metrics": {
                "tool_plan_invocations": metrics.get("tool_plan_invocations", 0),
                "tool_plan_planner_tokens": metrics.get("tool_plan_planner_tokens", 0),
                "tool_plan_solver_tokens": metrics.get("tool_plan_solver_tokens", 0),
                "tool_plan_total_tokens": metrics.get("tool_plan_total_tokens", 0),
                "tool_plan_steps_requested": metrics.get("tool_plan_steps_requested", 0),
                "tool_plan_steps_executed": metrics.get("tool_plan_steps_executed", 0),
                "tool_plan_step_failures": metrics.get("tool_plan_step_failures", 0),
                "wall_clock_sec": round(result.get("latency_metrics", {}).get("tool_execution_duration_sec", 0.0) or 0.0, 3),
            },
        }

        out_path = self.base_dir / f"{session_id}.jsonl"
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(redact_sensitive_data(payload), ensure_ascii=True, sort_keys=True) + "\n")
        return out_path

    def record_escalation(
        self,
        harness: Any,
        advisory: dict[str, Any],
    ) -> Path | None:
        state = getattr(harness, "state", None)
        if state is None:
            return None
        scratchpad = getattr(state, "scratchpad", None)
        if not isinstance(scratchpad, dict):
            return None

        session_id = str(getattr(state, "thread_id", "") or getattr(harness, "conversation_id", "") or "unknown")
        run_brief = getattr(state, "run_brief", None)
        task = str(
            getattr(run_brief, "original_task", "")
            or getattr(run_brief, "effective_task", "")
            or ""
        ).strip()
        last_escalation = scratchpad.get("_last_escalation")
        last_escalation = last_escalation if isinstance(last_escalation, dict) else {}
        metrics = scratchpad.get("_recovery_metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        config = getattr(harness, "config", None)

        payload: dict[str, Any] = {
            "type": "escalation",
            "task": task,
            "runtime": "escalation",
            "session_id": session_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "escalation_id": advisory.get("escalation_id"),
            "trigger": last_escalation.get("trigger"),
            "small_model": scratchpad.get("_model_name") or getattr(config, "model", ""),
            "big_model": getattr(config, "escalation_model", ""),
            "provider_profile": getattr(config, "escalation_provider_profile", ""),
            "packet_chars": metrics.get("escalation_prompt_chars", 0),
            "verdict": advisory.get("verdict"),
            "confidence": advisory.get("confidence"),
            "recommended_next_action": advisory.get("recommended_next_action"),
            "accepted_by_harness": bool(advisory.get("success")),
            "validator_result": "pass" if advisory.get("success") else "fail",
            "metrics": {
                "escalation_prompt_chars": metrics.get("escalation_prompt_chars", 0),
                "escalation_response_chars": metrics.get("escalation_response_chars", 0),
                "escalation_wall_clock_sec": metrics.get("escalation_wall_clock_sec", 0.0),
            },
        }

        out_path = self.base_dir / f"{session_id}.jsonl"
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(redact_sensitive_data(payload), ensure_ascii=True, sort_keys=True) + "\n")
        return out_path
