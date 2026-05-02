from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit


class WebSecurityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ValidatedWebUrl:
    url: str
    scheme: str
    host: str
    port: int
    domain: str
    resolved_addresses: tuple[str, ...] = ()


_BLOCKED_HOSTS = {"localhost"}


def validate_public_web_url(
    url: str,
    *,
    allowed_ports: set[int] | tuple[int, ...] | None = None,
    allow_private_targets: tuple[str, ...] | list[str] | None = None,
    resolver=socket.getaddrinfo,
) -> ValidatedWebUrl:
    return _validate_web_url(
        url,
        allowed_ports=allowed_ports,
        allow_private_targets=allow_private_targets,
        resolver=resolver,
    )


def validate_redirect_target(
    url: str,
    *,
    allowed_ports: set[int] | tuple[int, ...] | None = None,
    allow_private_targets: tuple[str, ...] | list[str] | None = None,
    resolver=socket.getaddrinfo,
) -> ValidatedWebUrl:
    return _validate_web_url(
        url,
        allowed_ports=allowed_ports,
        allow_private_targets=allow_private_targets,
        resolver=resolver,
    )


def resolve_and_validate_host(
    host: str,
    *,
    port: int,
    allow_private_targets: tuple[str, ...] | list[str] | None = None,
    resolver=socket.getaddrinfo,
) -> tuple[str, ...]:
    try:
        infos = resolver(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise WebSecurityError(f"Unable to resolve host: {host}") from exc
    addresses: list[str] = []
    normalized_allowlist = _normalize_allow_private_targets(allow_private_targets)
    for info in infos:
        sockaddr = info[4]
        address = str(sockaddr[0])
        _reject_private_address(address, host=host, allow_private_targets=normalized_allowlist)
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise WebSecurityError(f"No usable addresses resolved for host: {host}")
    return tuple(addresses)


def _validate_web_url(
    url: str,
    *,
    allowed_ports: set[int] | tuple[int, ...] | None = None,
    allow_private_targets: tuple[str, ...] | list[str] | None = None,
    resolver=socket.getaddrinfo,
) -> ValidatedWebUrl:
    try:
        parsed = urlsplit(str(url or "").strip())
    except ValueError as exc:
        raise WebSecurityError(f"Invalid URL: {url}") from exc

    if parsed.scheme.lower() not in {"http", "https"}:
        raise WebSecurityError("Only http and https URLs are allowed.")
    if parsed.username or parsed.password:
        raise WebSecurityError("URLs with embedded credentials are not allowed.")
    if not parsed.hostname:
        raise WebSecurityError("URL host is required.")

    host = parsed.hostname.strip().lower().rstrip(".")
    if not host or host in _BLOCKED_HOSTS:
        raise WebSecurityError("Localhost targets are not allowed.")
    if any(ch.isspace() for ch in host) or "%" in host:
        raise WebSecurityError("Invalid host encoding.")
    if _looks_like_obfuscated_ipv4_host(host):
        raise WebSecurityError("Obfuscated IP host forms are not allowed.")

    try:
        port = parsed.port
    except ValueError as exc:
        raise WebSecurityError("Invalid port in URL.") from exc
    if port is None:
        port = 443 if parsed.scheme.lower() == "https" else 80
    if allowed_ports and port not in set(int(item) for item in allowed_ports):
        raise WebSecurityError("Port is not allowed for web research.")

    resolved_addresses = resolve_and_validate_host(
        host,
        port=port,
        allow_private_targets=allow_private_targets,
        resolver=resolver,
    )
    netloc = host
    if parsed.port is not None and parsed.port not in {80, 443}:
        netloc = f"{host}:{parsed.port}"
    normalized = urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", parsed.query, ""))
    return ValidatedWebUrl(
        url=normalized,
        scheme=parsed.scheme.lower(),
        host=host,
        port=port,
        domain=host,
        resolved_addresses=resolved_addresses,
    )


def _reject_private_address(
    address: str,
    *,
    host: str | None = None,
    allow_private_targets: tuple[str, ...] | list[str] | None = None,
) -> None:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError as exc:
        raise WebSecurityError(f"Invalid resolved address: {address}") from exc
    if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast or ip.is_reserved:
        if _is_allowlisted_private_target(ip=ip, host=host, allow_private_targets=allow_private_targets):
            return
        raise WebSecurityError(f"Blocked unsafe target address: {address}")


def _normalize_allow_private_targets(
    allow_private_targets: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if not allow_private_targets:
        return ()
    normalized: list[str] = []
    for entry in allow_private_targets:
        token = str(entry or "").strip().lower().rstrip(".")
        if token:
            normalized.append(token)
    return tuple(normalized)


def _is_allowlisted_private_target(
    *,
    ip: ipaddress._BaseAddress,
    host: str | None,
    allow_private_targets: tuple[str, ...] | list[str] | None,
) -> bool:
    normalized_host = str(host or "").strip().lower().rstrip(".")
    for entry in _normalize_allow_private_targets(allow_private_targets):
        if _allow_private_target_entry_matches_ip(entry, ip):
            return True
        if normalized_host and _allow_private_target_entry_matches_host(entry, normalized_host):
            return True
    return False


def _allow_private_target_entry_matches_ip(entry: str, ip: ipaddress._BaseAddress) -> bool:
    try:
        return ip == ipaddress.ip_address(entry)
    except ValueError:
        pass
    try:
        return ip in ipaddress.ip_network(entry, strict=False)
    except ValueError:
        return False


def _allow_private_target_entry_matches_host(entry: str, host: str) -> bool:
    return host == entry or host.endswith("." + entry)


def _looks_like_obfuscated_ipv4_host(host: str) -> bool:
    if re.fullmatch(r"\d+", host):
        return True
    if re.fullmatch(r"0x[0-9a-f]+", host):
        return True
    labels = host.split(".")
    if len(labels) != 4:
        return False
    numeric_like = all(re.fullmatch(r"(?:0x[0-9a-f]+|0[0-7]+|\d+)", label or "") for label in labels)
    if not numeric_like:
        return False
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return True
    return False
