#!/usr/bin/env python3
"""Audit observable Qwen3.8 Max (0902) cache and request boundaries.

The QwenCloud Context Cache and text-generation documents expose request,
cache, and usage contracts that can be checked without a live API call. This
standard-library toy checks those host-side invariants. It does not reproduce
QwenCloud's cache scheduler, billing system, model, or provider internals.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping


class ContractError(ValueError):
    """A request or cache operation violates the documented toy contract."""


MODEL_ID = "qwen3.8-max-0902"
MIN_CACHE_TOKENS = 1024
CACHE_TTL_SECONDS = 300
MAX_MARKERS = 4
MAX_LOOKBACK_BLOCKS = 20
MAX_FOLLOWUP_MESSAGES = 20


def cache_prefix_eligible(token_count: int) -> bool:
    """The documented size threshold is eligibility, not a hit guarantee."""

    return token_count >= MIN_CACHE_TOKENS


def explicit_followup_reuses_cached_block(other_message_count: int) -> bool:
    """Apply the docs' message-count rule, separate from block lookback."""

    if other_message_count < 0:
        raise ContractError("message count cannot be negative")
    return other_message_count <= MAX_FOLLOWUP_MESSAGES


@dataclass(frozen=True)
class Request:
    model: str
    transport: str
    thinking: bool
    reasoning_effort: str | None = None
    thinking_budget: int | None = None
    tool_choice: str = "auto"
    headers: Mapping[str, str] = None  # type: ignore[assignment]
    previous_response_id: str | None = None

    def validate(self) -> None:
        if self.model != MODEL_ID:
            raise ContractError("model identity is not pinned to the 0902 alias")
        if self.transport not in {"chat", "responses"}:
            raise ContractError("unknown transport")
        if self.reasoning_effort is not None and self.thinking_budget is not None:
            raise ContractError("reasoning_effort and thinking_budget are mutually exclusive")
        if self.reasoning_effort is not None and self.reasoning_effort not in {
            "low",
            "medium",
            "xhigh",
        }:
            raise ContractError("unsupported reasoning_effort")
        if self.thinking and self.tool_choice not in {"auto", "none"}:
            raise ContractError("thinking mode only accepts auto or none tool_choice")

        headers = self.headers or {}
        session_cache = headers.get("x-dashscope-session-cache") == "enable"
        if session_cache and self.transport != "responses":
            raise ContractError("session cache requires the Responses API")


@dataclass(frozen=True)
class CacheMarker:
    token_count: int
    marker_type: str = "ephemeral"


def effective_markers(markers: list[CacheMarker]) -> tuple[CacheMarker, ...]:
    """The documented contract keeps only the last four markers."""

    for marker in markers:
        if marker.marker_type != "ephemeral":
            raise ContractError("cache_control.type must be ephemeral")
    return tuple(markers[-MAX_MARKERS:])


@dataclass
class CacheBlock:
    account: str
    model: str
    prefix: tuple[str, ...]
    last_used: int
    mode: str


class PrefixCache:
    """Small deterministic model of explicit/implicit prefix reuse."""

    def __init__(self) -> None:
        self.blocks: list[CacheBlock] = []

    @staticmethod
    def _check_prefix(prefix: tuple[str, ...]) -> None:
        if not cache_prefix_eligible(len(prefix)):
            raise ContractError("a cacheable prefix must contain at least 1024 tokens")

    def create_explicit(
        self,
        account: str,
        model: str,
        prefix: tuple[str, ...],
        now: int,
        marker_count: int = 1,
    ) -> CacheBlock:
        self._check_prefix(prefix)
        if marker_count > MAX_MARKERS:
            raise ContractError("marker count must be reduced before cache creation")
        block = CacheBlock(account, model, prefix, now, "explicit")
        self.blocks.append(block)
        return block

    def lookup_explicit(
        self,
        account: str,
        model: str,
        prefix: tuple[str, ...],
        now: int,
        blocks_after_marker: int = 0,
    ) -> int:
        self._check_prefix(prefix)
        for block in reversed(self.blocks):
            if block.mode != "explicit" or block.account != account or block.model != model:
                continue
            if now - block.last_used >= CACHE_TTL_SECONDS:
                continue
            if blocks_after_marker > MAX_LOOKBACK_BLOCKS:
                continue
            if tuple(prefix[: len(block.prefix)]) == block.prefix:
                block.last_used = now
                return len(block.prefix)
        return 0

    def lookup_implicit(
        self,
        account: str,
        model: str,
        prefix: tuple[str, ...],
        available: bool,
    ) -> int:
        """Return a hit only when the provider scheduler makes one available."""

        self._check_prefix(prefix)
        if not available:
            return 0
        for block in reversed(self.blocks):
            if block.mode == "implicit" and block.account == account and block.model == model:
                if tuple(prefix[: len(block.prefix)]) == block.prefix:
                    return len(block.prefix)
        self.blocks.append(CacheBlock(account, model, prefix, 0, "implicit"))
        return 0

    def create_implicit(self, account: str, model: str, prefix: tuple[str, ...]) -> None:
        self._check_prefix(prefix)
        self.blocks.append(CacheBlock(account, model, prefix, 0, "implicit"))


class SessionCache:
    """Model the Responses/session header and previous-response lineage."""

    def __init__(self) -> None:
        self._responses: dict[str, tuple[str, str, tuple[str, ...], int]] = {}

    def first_turn(
        self,
        request: Request,
        account: str,
        prefix: tuple[str, ...],
        response_id: str,
        now: int,
    ) -> None:
        request.validate()
        if request.transport != "responses":
            raise ContractError("session cache first turn requires Responses")
        if (request.headers or {}).get("x-dashscope-session-cache") != "enable":
            raise ContractError("session cache header is missing")
        if len(prefix) < MIN_CACHE_TOKENS:
            raise ContractError("session cache prefix is shorter than 1024 tokens")
        self._responses[response_id] = (account, request.model, prefix, now)

    def continue_turn(
        self,
        request: Request,
        account: str,
        response_id: str,
        now: int,
    ) -> int:
        request.validate()
        if request.previous_response_id not in self._responses:
            raise ContractError("previous_response_id is not available")
        old_account, old_model, prefix, last_used = self._responses[request.previous_response_id]
        if (old_account, old_model) != (account, request.model):
            raise ContractError("session cache is isolated by account and model")
        if now - last_used >= CACHE_TTL_SECONDS:
            raise ContractError("session cache expired")
        self._responses[response_id] = (account, request.model, prefix, now)
        self._responses[request.previous_response_id] = (account, request.model, prefix, now)
        return len(prefix)


def cached_tokens(usage: Mapping[str, Any]) -> int:
    """Read the QwenCloud Responses API usage field."""

    details = usage.get("input_tokens_details")
    if not isinstance(details, Mapping):
        raise ContractError("usage.input_tokens_details is missing")
    value = details.get("cached_tokens")
    if not isinstance(value, int) or value < 0:
        raise ContractError("usage.input_tokens_details.cached_tokens is invalid")
    return value


def chat_completion_cached_tokens(usage: Mapping[str, Any]) -> int:
    """Read the separate Chat Completions usage field from the cache docs."""

    details = usage.get("prompt_tokens_details")
    if not isinstance(details, Mapping):
        raise ContractError("usage.prompt_tokens_details is missing")
    value = details.get("cached_tokens")
    if not isinstance(value, int) or value < 0:
        raise ContractError("usage.prompt_tokens_details.cached_tokens is invalid")
    return value


def billing_factor(mode: str, event: str) -> float:
    factors = {
        ("explicit", "create"): 1.25,
        ("explicit", "hit"): 0.10,
        ("implicit", "create"): 1.00,
        ("implicit", "hit"): 0.20,
        ("session", "create"): 1.25,
        ("session", "hit"): 0.10,
    }
    try:
        return factors[(mode, event)]
    except KeyError as exc:
        raise ContractError("unknown cache billing event") from exc


def expect_error(action: Any, fragment: str) -> None:
    try:
        action()
    except ContractError as exc:
        if fragment not in str(exc):
            raise AssertionError(f"unexpected error: {exc}") from exc
    else:
        raise AssertionError(f"expected ContractError containing {fragment!r}")


def main() -> None:
    prefix = tuple(f"token-{index}" for index in range(MIN_CACHE_TOKENS))

    assert not cache_prefix_eligible(MIN_CACHE_TOKENS - 1)
    assert cache_prefix_eligible(MIN_CACHE_TOKENS)
    assert explicit_followup_reuses_cached_block(MAX_FOLLOWUP_MESSAGES)
    assert not explicit_followup_reuses_cached_block(MAX_FOLLOWUP_MESSAGES + 1)
    markers = effective_markers([CacheMarker(1024) for _ in range(5)])
    assert len(markers) == MAX_MARKERS
    expect_error(lambda: effective_markers([CacheMarker(1024, "persistent")]), "ephemeral")

    cache = PrefixCache()
    cache.create_explicit("acct-a", MODEL_ID, prefix, now=0)
    assert cache.lookup_explicit("acct-a", MODEL_ID, prefix + ("tail",), now=1) == 1024
    assert cache.lookup_explicit("acct-a", MODEL_ID, prefix, now=302) == 0
    cache.create_explicit("acct-a", MODEL_ID, prefix, now=400)
    assert cache.lookup_explicit("acct-a", MODEL_ID, prefix, now=401, blocks_after_marker=21) == 0
    assert cache.lookup_explicit("acct-b", MODEL_ID, prefix, now=401) == 0
    assert cache.lookup_explicit("acct-a", "qwen3.8-max", prefix, now=401) == 0

    cache.create_implicit("acct-a", MODEL_ID, prefix)
    assert cache.lookup_implicit("acct-a", MODEL_ID, prefix, available=False) == 0
    assert cache.lookup_implicit("acct-a", MODEL_ID, prefix, available=True) == 1024

    thinking = Request(
        MODEL_ID,
        "responses",
        thinking=True,
        reasoning_effort="xhigh",
        tool_choice="auto",
        headers={"x-dashscope-session-cache": "enable"},
    )
    thinking.validate()
    expect_error(
        lambda: Request(
            MODEL_ID,
            "responses",
            thinking=True,
            reasoning_effort="low",
            tool_choice="required",
        ).validate(),
        "only accepts auto or none",
    )
    expect_error(
        lambda: Request(
            MODEL_ID,
            "responses",
            thinking=True,
            reasoning_effort="medium",
            thinking_budget=2048,
        ).validate(),
        "mutually exclusive",
    )

    session = SessionCache()
    session.first_turn(thinking, "acct-a", prefix, "resp-1", now=10)
    continuation = Request(
        MODEL_ID,
        "responses",
        thinking=True,
        reasoning_effort="xhigh",
        headers={"x-dashscope-session-cache": "enable"},
        previous_response_id="resp-1",
    )
    assert session.continue_turn(continuation, "acct-a", "resp-2", now=11) == 1024
    responses_cached = cached_tokens({"input_tokens_details": {"cached_tokens": 1024}})
    chat_cached = chat_completion_cached_tokens(
        {"prompt_tokens_details": {"cached_tokens": 768}}
    )
    assert responses_cached == 1024
    assert chat_cached == 768
    expect_error(
        lambda: session.continue_turn(continuation, "acct-b", "resp-3", now=12),
        "isolated",
    )

    output = {
        "ok": True,
        "model": MODEL_ID,
        "capability_gate": {
            "thinking_forced_tool_rejected": True,
            "reasoning_budget_conflict_rejected": True,
        },
        "cache": {
            "effective_marker_count": len(markers),
            "minimum_size_is_eligibility_only": True,
            "explicit_example_boundary_requires_live_probe": True,
            "explicit_hit_tokens": 1024,
            "explicit_expired_hit_tokens": 0,
            "lookback_miss": True,
            "lookback_window_content_blocks": MAX_LOOKBACK_BLOCKS,
            "followup_message_window": MAX_FOLLOWUP_MESSAGES,
            "twenty_windows_have_distinct_units": True,
            "account_and_model_isolation": True,
            "implicit_hit_not_guaranteed": True,
            "responses_cached_tokens": responses_cached,
            "chat_completion_cached_tokens": chat_cached,
            "billing_factors": {
                "explicit_create": billing_factor("explicit", "create"),
                "explicit_hit": billing_factor("explicit", "hit"),
                "implicit_create": billing_factor("implicit", "create"),
                "implicit_hit": billing_factor("implicit", "hit"),
                "session_create": billing_factor("session", "create"),
                "session_hit": billing_factor("session", "hit"),
            },
        },
        "network_called": False,
        "evidence": "local_protocol_toy",
    }
    print(json.dumps(output, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
