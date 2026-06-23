from __future__ import annotations

import ast
import shlex
from typing import Any


_DEBIAN_READINESS_MARKER = "__PREFLIGHT_DEBIAN_READINESS__"
_DEBIAN_READINESS_DONE = "__PREFLIGHT_DEBIAN_READINESS_DONE__"


def build_debian_readiness_probe_script() -> str:
    """Return a read-only shell snippet that probes Debian installer readiness.

    The script is designed to be appended to the standard installer preflight
    probe command.  It reports back structured data about the target system's
    apt/keyring state so the harness can surface Debian 13 footguns before the
    model spends turns reacting to them.
    """
    python_code = (
        "import os, re, sys; "
        "from pathlib import Path; "
        "result = {"
        " 'is_debian': False, "
        " 'debian_version_id': '', "
        " 'debian_codename': '', "
        " 'is_debian_13': False, "
        " 'keyring_path': '/etc/apt/keyrings/debian-archive-keyring.gpg', "
        " 'keyring_exists': False, "
        " 'keyring_size': 0, "
        " 'apt_key_available': False, "
        " 'debian_sources_path': '/etc/apt/sources.list.d/debian.sources', "
        " 'deb822_valid': False, "
        " 'deb822_missing_fields': [], "
        " 'trixie_security_present': False, "
        " 'sources_list_d_files': [] "
        "}; "
        "os_release = Path('/etc/os-release'); "
        "if os_release.exists(): "
        "    text = os_release.read_text(); "
        "    result['is_debian'] = 'ID=debian' in text or 'ID_LIKE=debian' in text; "
        "    m = re.search(r'VERSION_ID=\"?([^\"\\n]+)\"?', text); "
        "    result['debian_version_id'] = m.group(1) if m else ''; "
        "    m = re.search(r'VERSION_CODENAME=\"?([^\"\\n]+)\"?', text); "
        "    result['debian_codename'] = m.group(1) if m else ''; "
        "    result['is_debian_13'] = (result['is_debian'] and result['debian_version_id'].startswith('13')); "
        "keyring = Path(result['keyring_path']); "
        "result['keyring_exists'] = keyring.exists(); "
        "result['keyring_size'] = keyring.stat().st_size if keyring.exists() else 0; "
        "result['apt_key_available'] = os.system('command -v apt-key >/dev/null 2>&1') == 0; "
        "sources = Path(result['debian_sources_path']); "
        "if sources.exists(): "
        "    s = sources.read_text(); "
        "    required = ['Types:', 'URIs:', 'Suites:', 'Components:']; "
        "    missing = [f for f in required if f not in s]; "
        "    result['deb822_valid'] = not missing; "
        "    result['deb822_missing_fields'] = missing; "
        "    result['trixie_security_present'] = 'trixie-security' in s; "
        "    result['trixie_release_present'] = 'trixie ' in s or 'Suites: trixie' in s; "
        "sources_d = Path('/etc/apt/sources.list.d'); "
        "if sources_d.is_dir(): "
        "    try: result['sources_list_d_files'] = [str(p) for p in sources_d.iterdir() if p.is_file()]; "
        "    except Exception: pass; "
        "print(repr(result))"
    )
    parts = [
        f'echo "{_DEBIAN_READINESS_MARKER}"',
        f"python3 -c {shlex.quote(python_code)}",
        f'echo "{_DEBIAN_READINESS_DONE}"',
    ]
    return "; ".join(parts)


def parse_debian_readiness_probe_output(combined_output: str) -> dict[str, Any]:
    """Parse the debian readiness probe section from combined stdout/stderr."""
    probes: dict[str, Any] = {
        "is_debian": False,
        "debian_version_id": "",
        "debian_codename": "",
        "is_debian_13": False,
        "keyring_path": "/etc/apt/keyrings/debian-archive-keyring.gpg",
        "keyring_exists": False,
        "keyring_size": 0,
        "apt_key_available": False,
        "debian_sources_path": "/etc/apt/sources.list.d/debian.sources",
        "deb822_valid": False,
        "deb822_missing_fields": [],
        "trixie_security_present": False,
        "trixie_release_present": False,
        "sources_list_d_files": [],
        "issues": [],
        "ready": True,
    }
    if _DEBIAN_READINESS_MARKER not in combined_output:
        return probes

    section = combined_output.split(_DEBIAN_READINESS_MARKER, 1)[1]
    if _DEBIAN_READINESS_DONE in section:
        section = section.split(_DEBIAN_READINESS_DONE, 1)[0]

    # Find the repr(dict) emitted by python3 -c "print(repr(result))"
    for line in section.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = ast.literal_eval(line)
                if isinstance(parsed, dict):
                    probes.update(parsed)
                    break
            except (ValueError, SyntaxError):
                continue

    probes["issues"] = _debian_readiness_issues(probes)
    probes["ready"] = not bool(probes["issues"])
    return probes


def _debian_readiness_issues(probes: dict[str, Any]) -> list[str]:
    """Return a list of human-readable readiness issues."""
    issues: list[str] = []
    if not probes.get("is_debian"):
        return issues

    codename = str(probes.get("debian_codename") or "").strip()
    if codename == "trixie":
        if not probes.get("keyring_exists"):
            issues.append(
                f"Debian 13 (trixie) detected but {probes['keyring_path']} is missing. "
                "Install the debian-archive-keyring package or copy the keyring before apt operations."
            )
        elif not int(probes.get("keyring_size") or 0):
            issues.append(
                f"Debian 13 (trixie) keyring at {probes['keyring_path']} is empty."
            )
        if probes.get("apt_key_available"):
            issues.append(
                "apt-key is still present; Debian 13 deprecated apt-key. "
                "Prefer /etc/apt/keyrings/ and signed-by= in deb822 or .list sources."
            )
        if not probes.get("deb822_valid"):
            missing = probes.get("deb822_missing_fields") or []
            issues.append(
                f"/etc/apt/sources.list.d/debian.sources is missing deb822 fields: {', '.join(missing)}. "
                "Repair sources before apt-get update."
            )
        if probes.get("trixie_security_present"):
            issues.append(
                "trixie-security repository is configured but Debian 13 testing/security often 404s. "
                "Verify the URL or use trixie-updates / stable-security instead."
            )
    else:
        if not probes.get("keyring_exists"):
            issues.append(
                f"Debian keyring at {probes['keyring_path']} is missing. "
                "This path is required on Debian 12+ for signed-by=."
            )
    return issues


def debian_readiness_summary(probes: dict[str, Any]) -> str:
    """Return a concise human-readable summary of Debian readiness."""
    if not probes.get("is_debian"):
        return "Target does not appear to be Debian; no Debian-specific readiness checks apply."

    parts = [
        f"Debian readiness scan: {probes.get('debian_codename') or 'unknown'} "
        f"(version {probes.get('debian_version_id') or 'unknown'})."
    ]
    if probes.get("is_debian_13"):
        parts.append("Debian 13 (trixie) detected.")

    keyring = probes.get("keyring_path")
    if probes.get("keyring_exists"):
        parts.append(f"Keyring {keyring} exists ({probes.get('keyring_size', 0)} bytes).")
    else:
        parts.append(f"Keyring {keyring} MISSING.")

    if probes.get("apt_key_available"):
        parts.append("apt-key is available (deprecated on Debian 13).")
    else:
        parts.append("apt-key is not available (expected on Debian 13).")

    if probes.get("deb822_valid"):
        parts.append("deb822 sources look valid.")
    else:
        missing = probes.get("deb822_missing_fields") or []
        parts.append(f"deb822 sources INVALID (missing {', '.join(missing)}).")

    if probes.get("trixie_security_present"):
        parts.append("trixie-security repository is configured (may 404).")

    issues = probes.get("issues") or []
    if issues:
        parts.append("Issues:")
        for issue in issues:
            parts.append(f"  - {issue}")
    else:
        parts.append("No Debian installer readiness issues detected.")

    return "\n".join(parts)
