#!/usr/bin/env python3
"""Audit the observable DeepSeek Harness boundaries without network or tools.

This standard-library toy turns the official Harness documentation into checks
for secret redaction, provider/session identity, plugin cleanup, MCP reconnect
and webhook admission. It is not the DeepSeek Harness implementation.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass, field
from typing import Mapping


class HarnessContractError(ValueError):
    """A toy Harness contract was violated."""


SECRET_NAME = re.compile(r"(KEY|TOKEN|PASSWORD|SECRET|CREDENTIAL)", re.I)


@dataclass(frozen=True)
class Provider:
    provider_id: str
    display_name: str
    protocol: str
    base_url: str
    models: tuple[str, ...]


class ProviderRegistry:
    """Keep stable provider identity and never return the literal secret."""

    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}
        self._secret_digests: dict[str, str] = {}

    def add(self, provider: Provider, secret: str) -> Mapping[str, str]:
        if not provider.provider_id or provider.provider_id in self._providers:
            raise HarnessContractError("provider id must be non-empty and unique")
        if not secret:
            raise HarnessContractError("provider secret is required")
        self._providers[provider.provider_id] = provider
        self._secret_digests[provider.provider_id] = hashlib.sha256(
            secret.encode("utf-8")
        ).hexdigest()
        return {
            "provider_id": provider.provider_id,
            "display_name": provider.display_name,
            "protocol": provider.protocol,
            "credential": "<redacted>",
        }

    def rename(self, provider_id: str, new_id: str) -> None:
        if provider_id not in self._providers or new_id in self._providers:
            raise HarnessContractError("provider identity cannot be renamed in place")
        raise HarnessContractError("create a new provider and delete the old one")


@dataclass
class Session:
    session_id: str
    provider_id: str
    model_id: str
    request_count: int = 0

    def send(self) -> None:
        self.request_count += 1

    def change_model(self, provider_id: str, model_id: str) -> None:
        if self.request_count:
            raise HarnessContractError(
                "an existing session keeps the model recorded in its log"
            )
        self.provider_id = provider_id
        self.model_id = model_id


@dataclass
class Plugin:
    name: str
    requires: tuple[str, ...] = ()
    effects: list[str] = field(default_factory=list)


class PluginRuntime:
    def __init__(self, services: set[str]) -> None:
        self.services = services
        self.loaded: dict[str, Plugin] = {}

    def load(self, plugin: Plugin) -> None:
        missing = set(plugin.requires) - self.services
        if missing:
            raise HarnessContractError(f"missing plugin services: {sorted(missing)}")
        self.loaded[plugin.name] = plugin

    def unload(self, name: str) -> tuple[str, ...]:
        plugin = self.loaded.pop(name)
        cleaned = tuple(reversed(plugin.effects))
        plugin.effects.clear()
        return cleaned


class MCPBridge:
    """Model namespace, reconnect budget and credential environment filtering."""

    def __init__(self, server_name: str, reconnect_budget: int = 2) -> None:
        self.server_name = server_name
        self.reconnect_budget = reconnect_budget
        self.connected = False
        self.registered_tools: set[str] = set()

    def child_environment(self, env: Mapping[str, str]) -> dict[str, str]:
        return {
            key: value
            for key, value in env.items()
            if not key.startswith("DSH_") and not SECRET_NAME.search(key)
        }

    def connect(self, tools: list[str]) -> None:
        self.connected = True
        self.registered_tools = {
            f"mcp__{self.server_name}__{tool}" for tool in tools
        }

    def outage(self) -> None:
        self.connected = False

    def reconnect(self, tools: list[str]) -> bool:
        if self.reconnect_budget <= 0:
            self.registered_tools.clear()
            return False
        self.reconnect_budget -= 1
        self.connect(tools)
        return True

    def call(self, qualified_tool: str) -> None:
        if not self.connected or qualified_tool not in self.registered_tools:
            raise HarnessContractError("MCP call is unavailable during outage")


class GitHubWebhook:
    """Admission is authenticated and asynchronous; it is not task success."""

    def __init__(self, secret: bytes) -> None:
        self.secret = secret
        self.admitted_sessions: list[str] = []

    def signature(self, body: bytes) -> str:
        digest = hmac.new(self.secret, body, hashlib.sha256).hexdigest()
        return f"sha256={digest}"

    def admit(self, body: bytes, signature: str, delivery_id: str) -> int:
        expected = self.signature(body)
        if not hmac.compare_digest(expected, signature):
            raise HarnessContractError("invalid webhook signature")
        # The documented webhook runtime has no delivery/execution dedupe state.
        self.admitted_sessions.append(f"session-{delivery_id}-{len(self.admitted_sessions)}")
        return 202


def main() -> None:
    registry = ProviderRegistry()
    descriptor = registry.add(
        Provider(
            provider_id="deepseek-v41",
            display_name="DeepSeek V4.1",
            protocol="openai-responses",
            base_url="https://api.deepseek.com",
            models=("deepseek-flash",),
        ),
        secret="sk-synthetic",
    )
    assert descriptor["credential"] == "<redacted>"
    try:
        registry.rename("deepseek-v41", "renamed")
    except HarnessContractError:
        pass
    else:
        raise AssertionError("provider identity must remain stable")

    session = Session("session-a", "deepseek-v41", "deepseek-flash")
    session.send()
    try:
        session.change_model("other-provider", "other-model")
    except HarnessContractError:
        pass
    else:
        raise AssertionError("model change after first request must be rejected")

    runtime = PluginRuntime({"llm", "tools"})
    plugin = Plugin("audit-plugin", requires=("llm",), effects=["subscription", "timer"])
    runtime.load(plugin)
    assert runtime.unload("audit-plugin") == ("timer", "subscription")
    try:
        runtime.load(Plugin("bad-plugin", requires=("missing",)))
    except HarnessContractError:
        pass
    else:
        raise AssertionError("missing plugin dependency must fail")

    bridge = MCPBridge("memory", reconnect_budget=1)
    child_env = bridge.child_environment(
        {"PATH": "/bin", "DSH_HOME": "/tmp/dsh", "API_KEY": "secret"}
    )
    assert child_env == {"PATH": "/bin"}
    bridge.connect(["search"])
    qualified_tool = "mcp__memory__search"
    bridge.call(qualified_tool)
    bridge.outage()
    try:
        bridge.call(qualified_tool)
    except HarnessContractError:
        pass
    else:
        raise AssertionError("MCP calls must fail while disconnected")
    assert bridge.reconnect(["search"])
    bridge.call(qualified_tool)
    bridge.reconnect_budget = 0
    bridge.outage()
    assert not bridge.reconnect(["search"])
    assert not bridge.registered_tools

    webhook = GitHubWebhook(b"webhook-secret")
    body = b'{"action":"ready_for_review"}'
    signature = webhook.signature(body)
    assert webhook.admit(body, signature, "delivery-1") == 202
    assert webhook.admit(body, signature, "delivery-1") == 202
    assert len(webhook.admitted_sessions) == 2

    print(
        json.dumps(
            {
                "ok": True,
                "provider_descriptor": descriptor,
                "session_model": session.model_id,
                "mcp_tools_after_budget": sorted(bridge.registered_tools),
                "duplicate_webhook_admissions": len(webhook.admitted_sessions),
                "network_called": False,
                "evidence": "local_protocol_toy",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
