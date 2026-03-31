from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class SessionInventory:
    hosts: dict[str, dict[str, Any]] = field(default_factory=dict)
    groups: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def localhost_default(cls) -> "SessionInventory":
        inv = cls()
        inv.add_host("localhost", {"ansible_connection": "local"})
        inv.add_group("all", ["localhost"])
        return inv

    def to_ansible_inventory(self) -> dict[str, Any]:
        all_hosts = {name: vars_ for name, vars_ in self.hosts.items()}
        children = {
            group: {"hosts": {h: {} for h in hosts}} for group, hosts in self.groups.items()
        }
        return {"all": {"hosts": all_hosts, "children": children}}

    def list(self) -> dict[str, Any]:
        return {"hosts": self.hosts, "groups": self.groups}

    def add_host(self, name: str, variables: dict[str, Any] | None = None) -> None:
        self.hosts[name] = variables or {}

    def add_group(self, name: str, hosts: list[str] | None = None) -> None:
        self.groups.setdefault(name, [])
        for host in hosts or []:
            if host not in self.groups[name]:
                self.groups[name].append(host)

    def remove_host(self, name: str) -> None:
        self.hosts.pop(name, None)
        for group_hosts in self.groups.values():
            if name in group_hosts:
                group_hosts.remove(name)

    def merge_inventory_file(self, path: str) -> None:
        if yaml is None:
            raise RuntimeError("Dependency missing: pyyaml")
        inv_path = Path(path).resolve()
        if not inv_path.exists():
            raise FileNotFoundError(f"Inventory path not found: {inv_path}")
        data = yaml.safe_load(inv_path.read_text(encoding="utf-8")) or {}

        all_node = data.get("all", {})
        for host_name, host_vars in (all_node.get("hosts") or {}).items():
            self.add_host(host_name, host_vars or {})

        children = all_node.get("children") or {}
        for group_name, group_node in children.items():
            group_hosts = list((group_node or {}).get("hosts", {}).keys())
            self.add_group(group_name, group_hosts)

