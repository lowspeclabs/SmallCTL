from __future__ import annotations

from dataclasses import fields

import pytest

from smallctl.config import SmallctlConfig
from smallctl.config_projection import (
    HARNESS_ONLY_FIELDS,
    LOCAL_ONLY_FIELDS,
    project_config_to_harness_kwargs,
)
from smallctl.config_support import _env_raw_config
from smallctl.harness.config import HarnessConfig


def test_no_harness_field_is_orphaned() -> None:
    """Every HarnessConfig field is either in SmallctlConfig or in HARNESS_ONLY_FIELDS."""
    harness_names = {f.name for f in fields(HarnessConfig)}
    smallctl_names = {f.name for f in fields(SmallctlConfig)}
    orphans = harness_names - smallctl_names - HARNESS_ONLY_FIELDS
    assert not orphans, (
        f"HarnessConfig has fields with no mapping to SmallctlConfig: {orphans}. "
        f"Add them to SmallctlConfig or to HARNESS_ONLY_FIELDS."
    )


def test_no_smallctl_field_is_silently_dropped() -> None:
    """Every SmallctlConfig field is either in HarnessConfig or in LOCAL_ONLY_FIELDS."""
    harness_names = {f.name for f in fields(HarnessConfig)}
    smallctl_names = {f.name for f in fields(SmallctlConfig)}
    dropped = smallctl_names - harness_names - LOCAL_ONLY_FIELDS
    assert not dropped, (
        f"SmallctlConfig has fields that would be silently dropped by the projection: {dropped}. "
        f"Add them to HarnessConfig or to LOCAL_ONLY_FIELDS."
    )


def test_projection_is_exhaustive() -> None:
    """project_config_to_harness_kwargs maps every overlapping field."""
    config = SmallctlConfig()
    kwargs = project_config_to_harness_kwargs(config)
    harness_names = {f.name for f in fields(HarnessConfig)}
    for name in harness_names:
        if name in HARNESS_ONLY_FIELDS:
            continue
        assert name in kwargs, f"Projection missing field: {name}"


def test_projection_produces_valid_harness_config() -> None:
    """The kwargs produced by the projection can construct a HarnessConfig."""
    config = SmallctlConfig()
    kwargs = project_config_to_harness_kwargs(config, run_logger=None)
    cfg = HarnessConfig(**kwargs)
    assert cfg.endpoint == config.endpoint
    assert cfg.model == config.model


def test_env_config_covers_all_smallctl_fields() -> None:
    """_env_raw_config has an entry for every SmallctlConfig field."""
    smallctl_names = {f.name for f in fields(SmallctlConfig)}
    env_names = set(_env_raw_config(lambda _key: None).keys())
    missing = smallctl_names - env_names - LOCAL_ONLY_FIELDS
    assert not missing, (
        f"_env_raw_config is missing keys for SmallctlConfig fields: {missing}. "
        f"Add them to config_support._env_raw_config()."
    )
