from __future__ import annotations

from typing import Any


def config_value(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default) if config is not None else default


def fama_enabled(config: Any) -> bool:
    if not bool(config_value(config, "fama_enabled", True)):
        return False
    return str(config_value(config, "fama_mode", "lite") or "lite").strip().lower() != "off"


def default_ttl_steps(config: Any) -> int:
    return max(1, _to_int(config_value(config, "fama_default_ttl_steps", 2), 2))


def max_active_mitigations(config: Any) -> int:
    return max(1, _to_int(config_value(config, "fama_max_active_mitigations", 2), 2))


def signal_window(config: Any) -> int:
    return max(1, _to_int(config_value(config, "fama_signal_window", 8), 8))


def done_gate_on_failure(config: Any) -> bool:
    return bool(config_value(config, "fama_done_gate_on_failure", True))


def llm_judge_enabled(config: Any) -> bool:
    return bool(config_value(config, "fama_llm_judge_enabled", False))


def llm_judge_min_severity(config: Any) -> int:
    return max(1, min(3, _to_int(config_value(config, "fama_llm_judge_min_severity", 3), 3)))


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
