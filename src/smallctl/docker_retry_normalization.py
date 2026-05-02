from __future__ import annotations

import re
import shlex

DOCKER_REGISTRY_FAILURE_CLASSES = frozenset({
    "docker_image_not_found",
    "docker_registry_denied",
})

_DOCKER_PULL_VALUE_FLAGS = frozenset({
    "--config",
    "--platform",
})

_DOCKER_RUN_VALUE_FLAGS = frozenset({
    "-a",
    "-c",
    "-e",
    "-h",
    "-l",
    "-m",
    "-p",
    "-u",
    "-v",
    "-w",
    "--add-host",
    "--annotation",
    "--blkio-weight",
    "--blkio-weight-device",
    "--cap-add",
    "--cap-drop",
    "--cgroup-parent",
    "--cidfile",
    "--cpu-count",
    "--cpu-percent",
    "--cpu-period",
    "--cpu-quota",
    "--cpu-rt-period",
    "--cpu-rt-runtime",
    "--cpu-shares",
    "--cpus",
    "--cpuset-cpus",
    "--cpuset-mems",
    "--detach-keys",
    "--device",
    "--device-cgroup-rule",
    "--device-read-bps",
    "--device-read-iops",
    "--device-write-bps",
    "--device-write-iops",
    "--dns",
    "--dns-option",
    "--dns-search",
    "--domainname",
    "--entrypoint",
    "--env",
    "--env-file",
    "--expose",
    "--gpus",
    "--group-add",
    "--health-cmd",
    "--health-interval",
    "--health-retries",
    "--health-start-interval",
    "--health-start-period",
    "--health-timeout",
    "--hostname",
    "--init-path",
    "--io-maxbandwidth",
    "--io-maxiops",
    "--ip",
    "--ip6",
    "--ipc",
    "--isolation",
    "--label",
    "--label-file",
    "--link",
    "--link-local-ip",
    "--log-driver",
    "--log-opt",
    "--mac-address",
    "--memory",
    "--memory-reservation",
    "--memory-swap",
    "--memory-swappiness",
    "--mount",
    "--name",
    "--network",
    "--network-alias",
    "--oom-score-adj",
    "--pid",
    "--pids-limit",
    "--platform",
    "--publish",
    "--publish-all",
    "--pull",
    "--restart",
    "--runtime",
    "--shm-size",
    "--signal",
    "--stop-signal",
    "--stop-timeout",
    "--storage-opt",
    "--tmpfs",
    "--ulimit",
    "--user",
    "--userns",
    "--uts",
    "--volume",
    "--volumes-from",
    "--workdir",
})


def classify_docker_failure(text: str) -> str:
    lowered = str(text or "").lower()
    if not lowered:
        return ""
    if "manifest unknown" in lowered:
        return "docker_image_not_found"
    if "manifest for " in lowered and " not found" in lowered:
        return "docker_image_not_found"
    if "pull access denied" in lowered:
        return "docker_registry_denied"
    if "repository does not exist or may require 'docker login'" in lowered:
        return "docker_registry_denied"
    if "denied: requested access to the resource is denied" in lowered:
        return "docker_registry_denied"
    if "requested access to the resource is denied" in lowered:
        return "docker_registry_denied"
    return ""


def docker_failure_is_registry_resolution(failure_class: str) -> bool:
    return str(failure_class or "").strip() in DOCKER_REGISTRY_FAILURE_CLASSES


def extract_docker_command_target(command: str) -> tuple[str, str] | None:
    tokens = _split_command(command)
    if len(tokens) < 3:
        return None
    if tokens[0].lower() not in {"docker", "podman"}:
        return None

    subcommand = tokens[1].lower()
    if subcommand == "pull":
        image_ref = _extract_first_non_option_argument(tokens[2:], value_flags=_DOCKER_PULL_VALUE_FLAGS)
        if image_ref:
            return "docker_pull", normalize_docker_image_ref(image_ref)
        return None
    if subcommand == "run":
        image_ref = _extract_first_non_option_argument(tokens[2:], value_flags=_DOCKER_RUN_VALUE_FLAGS)
        if image_ref:
            return "docker_run_pull_resolution", normalize_docker_image_ref(image_ref)
        return None
    return None


def normalize_docker_image_ref(image_ref: str) -> str:
    image = str(image_ref or "").strip().strip("\"'")
    if not image:
        return ""
    image = re.sub(r"\s+", "", image).lower()
    tail = image.rsplit("/", 1)[-1]
    if "@" not in image and ":" not in tail:
        image = f"{image}:latest"
    return image


def docker_retry_family(command: str) -> str:
    parsed = extract_docker_command_target(command)
    if parsed is None:
        return ""
    _command_kind, image_ref = parsed
    return f"docker_registry_family::{image_ref}"


def docker_retry_key(command: str, failure_class: str) -> str:
    if not docker_failure_is_registry_resolution(failure_class):
        return ""
    parsed = extract_docker_command_target(command)
    if parsed is None:
        return ""
    _command_kind, image_ref = parsed
    return f"docker_registry_error::{str(failure_class or '').strip()}::{image_ref}"


def _extract_first_non_option_argument(tokens: list[str], *, value_flags: frozenset[str]) -> str:
    index = 0
    while index < len(tokens):
        token = str(tokens[index] or "").strip()
        if not token:
            index += 1
            continue
        if token == "--":
            return tokens[index + 1] if index + 1 < len(tokens) else ""
        if token.startswith("--"):
            flag_name = token.split("=", 1)[0]
            if "=" in token or flag_name not in value_flags:
                index += 1
            else:
                index += 2
            continue
        if token.startswith("-") and token != "-":
            short_flag = token[:2]
            if token in value_flags:
                index += 2
            elif short_flag in value_flags:
                index += 1
            else:
                index += 1
            continue
        return token
    return ""


def _split_command(command: str) -> list[str]:
    text = str(command or "").strip()
    if not text:
        return []
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()
