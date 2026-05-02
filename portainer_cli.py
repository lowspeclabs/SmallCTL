#!/usr/bin/env python3
"""Portainer API CLI - Basic functionality for interacting with Portainer API."""

import json
import sys
import argparse
import requests
from typing import Optional, Dict, Any


class PortainerClient:
    """Client for interacting with Portainer REST API."""

    def __init__(self, base_url: str = None, token: str = None):
        """Initialize Portainer client.

        Args:
            base_url: Portainer API base URL (default: http://localhost:8000)
            token: Authentication token (default: None)
        """
        self.base_url = base_url or "http://localhost:8000"
        self.token = token
        self.headers = {}

        if token:
            self.headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }

    def login(self) -> bool:
        """Login to Portainer and get authentication token.

        Returns:
            True if login successful, False otherwise
        """
        try:
            response = requests.post(
                f"{self.base_url}/auth",
                headers={"Content-Type": "application/json"},
                json={
                    "Username": self.token.get("username", ""),
                    "Password": self.token.get("password", "")
                },
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            self.token = {
                "username": data.get("Username"),
                "password": data.get("Password")
            }

            # Set authorization header for subsequent requests
            if "jwtToken" in data:
                self.headers = {
                    "Authorization": f"Bearer {data['jwtToken']}",
                    "Content-Type": "application/json"
                }

            return True
        except requests.exceptions.RequestException as e:
            print(f"Login failed: {e}", file=sys.stderr)
            return False

    def health_check(self) -> bool:
        """Check if Portainer API is healthy.

        Returns:
            True if API is healthy, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/version",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            print(f"Portainer version: {response.json().get('version', 'unknown')}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Health check failed: {e}", file=sys.stderr)
            return False

    def list_containers(self) -> Optional[Dict[str, Any]]:
        """List all containers.

        Returns:
            JSON response if successful, None otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/docker/containers/json",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to list containers: {e}", file=sys.stderr)
            return None

    def create_container(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a new container.

        Args:
            config: Container configuration

        Returns:
            Container info if successful, None otherwise
        """
        try:
            response = requests.post(
                f"{self.base_url}/docker/containers/create",
                headers=self.headers,
                json=config,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to create container: {e}", file=sys.stderr)
            return None

    def start_container(self, container_id: str) -> bool:
        """Start a container.

        Args:
            container_id: Container ID or name

        Returns:
            True if started successfully, False otherwise
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/toby/fromid/{container_id}",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Failed to start container: {e}", file=sys.stderr)
            return False

    def stop_container(self, container_id: str) -> bool:
        """Stop a container.

        Args:
            container_id: Container ID or name

        Returns:
            True if stopped successfully, False otherwise
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/fromid/{container_id}",
                headers=self.headers,
                json={"keep": True},
                timeout=10
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Failed to stop container: {e}", file=sys.stderr)
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Portainer API CLI - Interact with Portainer API",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "-u", "--url",
        type=str,
        default="http://localhost:8000",
        help="Portainer API base URL"
    )

    parser.add_argument(
        "-t", "--token",
        type=str,
        help="Portainer authentication credentials (format: username:password)"
    )

    parser.add_argument(
        "-f", "--file",
        type=str,
        help="JSON file with authentication token (JWT)"
    )

    parser.add_argument(
        "-c", "--command",
        type=str,
        required=True,
        choices=["health", "login", "list", "create", "start", "stop", "logout"],
        help="Command to execute"
    )

    parser.add_argument(
        "args",
        nargs="*",
        help="Additional arguments for the command"
    )

    args = parser.parse_args()

    # Initialize client
    client = PortainerClient(base_url=args.url)

    # Parse token if provided
    token = client.token
    if not token:
        token = {}
    if args.token:
        parts = args.token.split(":", 1)
        if len(parts) == 2:
            token = {"username": parts[0], "password": parts[1]}
    elif args.file:
        try:
            with open(args.file, "r") as f:
                token_raw = f.read().strip()
                # Try to parse as JSON first (if it's a JWT-like structure)
                if token_raw.startswith("{"):
                    token = json.loads(token_raw)
                else:
                    # Assume it's a JWT token
                    token = {"jwt": token_raw}
        except Exception as e:
            print(f"Failed to read token file: {e}", file=sys.stderr)
            sys.exit(1)

    # Execute command
    if args.command == "health":
        if not token and not client.health_check:
            print("Health check requires no authentication, but API might be unreachable.", file=sys.stderr)
        client.health_check()

    elif args.command == "login":
        token["username"] = args.token
        if token["username"]:
            client.token = token
            if client.login():
                print("Login successful!")
            else:
                sys.exit(1)
        else:
            print("Provide credentials with -t or -f option", file=sys.stderr)
            sys.exit(1)

    elif args.command == "list":
        containers = client.list_containers()
        if containers:
            print(json.dumps(containers, indent=2))
        else:
            sys.exit(1)

    elif args.command == "create":
        if not args.args:
            print("Provide container configuration via stdin or --config-file", file=sys.stderr)
            sys.exit(1)
        # Implementation would go here
        pass

    elif args.command == "start":
        if not args.args:
            print("Provide container ID/name after the command", file=sys.stderr)
            sys.exit(1)
        if client.start_container(args.args[0]):
            print("Container started successfully")
        else:
            sys.exit(1)

    elif args.command == "stop":
        if not args.args:
            print("Provide container ID/name after the command", file=sys.stderr)
            sys.exit(1)
        if client.stop_container(args.args[0]):
            print("Container stopped successfully")
        else:
            sys.exit(1)

    elif args.command == "logout":
        client.headers = {}
        client.token = None
        print("Logged out successfully")


if __name__ == "__main__":
    main()