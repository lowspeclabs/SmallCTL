from __future__ import annotations

from typing import Any


def is_weather_lookup_task(state: Any) -> bool:
    task_text = str(getattr(getattr(state, "run_brief", None), "original_task", "") or "").strip().lower()
    if not task_text:
        return False
    return any(marker in task_text for marker in ("weather", "forecast", "temperature"))


def has_specific_weather_answer(message: str) -> bool:
    text = " ".join(str(message or "").strip().lower().split())
    if not text:
        return False

    explicit_unavailable_markers = (
        "could not verify the exact",
        "couldn't verify the exact",
        "unable to verify the exact",
        "exact weather could not be verified",
        "exact current weather could not be verified",
        "exact temperature could not be verified",
        "i could not verify the exact",
    )
    if any(marker in text for marker in explicit_unavailable_markers):
        return True

    temperature_markers = ("°f", "°c", " fahrenheit", " celsius", " degree", " degrees")
    weather_markers = (
        "temperature",
        "temp",
        "forecast",
        "high",
        "low",
        "today",
        "currently",
        "weather",
        "sunny",
        "cloudy",
        "clear",
        "rain",
        "showers",
        "storm",
        "snow",
        "windy",
        "humid",
        "overcast",
        "drizzle",
        "thunder",
    )
    has_temperature = any(marker in text for marker in temperature_markers)
    has_weather_detail = any(marker in text for marker in weather_markers)
    return has_temperature and has_weather_detail


def looks_like_weather_search_meta_completion(message: str) -> bool:
    text = " ".join(str(message or "").strip().lower().split())
    if not text:
        return False
    if has_specific_weather_answer(text):
        return False
    meta_markers = (
        "web search completed",
        "search completed",
        "found ",
        "returned ",
    )
    return any(marker in text for marker in meta_markers) and "result" in text
