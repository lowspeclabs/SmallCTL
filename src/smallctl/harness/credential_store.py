from __future__ import annotations

import hashlib


class CredentialStore:
    """Ephemeral, in-memory store for sensitive credentials.

    Plaintext credentials are never written to ``LoopState`` or checkpoint JSON.
    Callers store secrets here and persist only fingerprints/hashes in durable
    state. The store is intentionally not serializable and is discarded when the
    harness process exits.
    """

    def __init__(self) -> None:
        self._sudo_password: str | None = None
        self._ssh_passwords: dict[str, str] = {}
        self._by_fingerprint: dict[str, str] = {}

    @staticmethod
    def fingerprint(value: str | None) -> str:
        """Return a stable, non-reversible fingerprint for a credential."""
        value = str(value or "").strip()
        if not value:
            return ""
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _ssh_key(host: str, user: str | None) -> str:
        host = str(host or "").strip().lower()
        user = str(user or "").strip().lower()
        return f"{user}@{host}" if user else host

    def set_sudo_password(self, password: str | None) -> str:
        """Store a sudo password and return its fingerprint."""
        password = str(password or "").strip() or None
        self._sudo_password = password
        return self.fingerprint(password) if password else ""

    def get_sudo_password(self) -> str | None:
        """Return the stored sudo password, if any."""
        return self._sudo_password

    def set_ssh_password(self, host: str, user: str | None, password: str) -> str:
        """Store an SSH password and return its fingerprint."""
        password = str(password or "").strip()
        if not password:
            return ""
        fingerprint = self.fingerprint(password)
        key = self._ssh_key(host, user)
        self._ssh_passwords[key] = password
        self._by_fingerprint[fingerprint] = password
        return fingerprint

    def get_ssh_password(self, host: str, user: str | None) -> str | None:
        """Return the SSH password for a host/user pair, if known."""
        key = self._ssh_key(host, user)
        return self._ssh_passwords.get(key)

    def get_ssh_password_by_fingerprint(self, fingerprint: str) -> str | None:
        """Return an SSH password by its fingerprint, if known."""
        return self._by_fingerprint.get(fingerprint)
