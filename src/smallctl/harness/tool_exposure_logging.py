from __future__ import annotations

from typing import Any


def log_fama_tool_exposure(
    harness: Any,
    *,
    hidden_tools: set[str],
    mode: str,
    batch_duplicates: bool = False,
) -> None:
    runlog = getattr(harness, "_runlog", None)
    if not callable(runlog):
        return
    try:
        from ..fama.state import active_mitigation_names

        active = sorted(active_mitigation_names(harness.state))
    except Exception:
        active = []
    try:
        from ..fama.tool_policy import fama_hidden_tool_reasons_for_exposure

        schemas = harness.registry.export_openai_tools(
            phase=harness.state.current_phase,
            mode=mode,
            profiles=set(harness.state.active_tool_profiles),
        )
        hidden_tool_reasons = fama_hidden_tool_reasons_for_exposure(
            schemas,
            state=harness.state,
            mode=mode,
            config=getattr(harness, "config", None),
        )
    except Exception:
        hidden_tool_reasons = {}

    signature = (tuple(sorted(hidden_tools)), tuple(active), mode)
    if batch_duplicates:
        batch_state = getattr(harness.state, "_fama_log_batch", None)
        if batch_state is not None:
            last_sig, count, last_logged = batch_state
            if last_sig == signature:
                harness.state._fama_log_batch = (signature, count + 1, last_logged)
                return
            if count > 1 and not last_logged:
                runlog(
                    "fama_tool_exposure_applied",
                    f"FAMA tool exposure policy applied: count={count} (suppressed {count - 1} duplicates)",
                    hidden_tools=list(last_sig[0]),
                    hidden_tool_reasons=hidden_tool_reasons,
                    active_mitigations=list(last_sig[1]),
                    mode=last_sig[2],
                    batched_count=count,
                )
                harness.state._fama_log_batch = (signature, 1, True)
                return
        harness.state._fama_log_batch = (signature, 1, False)

    runlog(
        "fama_tool_exposure_applied",
        "FAMA tool exposure policy applied",
        level="debug",
        subsystem="fama",
        hidden_tools=sorted(hidden_tools),
        hidden_tool_reasons=hidden_tool_reasons,
        active_mitigations=active,
        mode=mode,
    )


def log_tool_profile_exposure(
    harness: Any,
    *,
    mode: str,
    phase: str,
    profiles: set[str],
    exposed_tools: list[str],
    hidden_tools: list[str],
    reasons: dict[str, str],
) -> None:
    runlog = getattr(harness, "_runlog", None)
    if not callable(runlog):
        return
    runlog(
        "tool_profile_exposure",
        "tool profile exposure resolved",
        level="debug",
        subsystem="graph",
        mode=mode,
        phase=phase,
        profile_filter=sorted(profiles),
        exposed_tools=sorted(exposed_tools)[:50],
        hidden_tools=sorted(hidden_tools)[:50],
        reason=reasons,
    )
