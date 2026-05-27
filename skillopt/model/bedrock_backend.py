"""AWS Bedrock Converse API backend for SkillOpt.

Uses boto3 bedrock-runtime client with the Converse API, which supports
Claude, Llama, Mistral, and other Bedrock-hosted models with a unified
message format and native SigV4 authentication.

Configuration via environment variables:
    AWS_BEDROCK_REGION          - AWS region (default: us-east-1)
    AWS_BEDROCK_PROFILE         - AWS CLI profile name (optional)
    BEDROCK_OPTIMIZER_MODEL     - Model ID for optimizer (default: us.anthropic.claude-sonnet-4-6-20250514)
    BEDROCK_TARGET_MODEL        - Model ID for target (default: us.anthropic.claude-sonnet-4-6-20250514)

Supports cross-region inference IDs (us.anthropic.*, eu.anthropic.*, etc.)
and standard model IDs (anthropic.claude-3-5-sonnet-20241022-v2:0, etc.)
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

import boto3

from skillopt.model.common import (
    CompatAssistantMessage,
    CompatToolCall,
    CompatToolFunction,
    TokenTracker,
)

# -- Configuration -----------------------------------------------------------

REGION = os.environ.get("AWS_BEDROCK_REGION", "us-east-1")
PROFILE = os.environ.get("AWS_BEDROCK_PROFILE", "").strip() or None

OPTIMIZER_DEPLOYMENT = os.environ.get(
    "BEDROCK_OPTIMIZER_MODEL", "us.anthropic.claude-sonnet-4-6-20250514"
)
TARGET_DEPLOYMENT = os.environ.get(
    "BEDROCK_TARGET_MODEL", "us.anthropic.claude-sonnet-4-6-20250514"
)

REASONING_EFFORT: str | None = None

# -- Internals ---------------------------------------------------------------

tracker = TokenTracker()

_client_lock = threading.Lock()
_client: Any = None


def _get_client():
    global _client
    with _client_lock:
        if _client is None:
            session_kwargs: dict[str, Any] = {}
            if PROFILE:
                session_kwargs["profile_name"] = PROFILE
            session = boto3.Session(**session_kwargs)
            _client = session.client("bedrock-runtime", region_name=REGION)
        return _client


def _reset_client() -> None:
    global _client
    with _client_lock:
        _client = None


# -- Message conversion ------------------------------------------------------


def _messages_from_system_user(system: str, user: str) -> tuple[list[dict], list[dict]]:
    system_messages = [{"text": system}] if system else []
    messages = [{"role": "user", "content": [{"text": user}]}]
    return system_messages, messages


def _convert_chat_messages(messages: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Convert OpenAI-style chat messages to Bedrock Converse format."""
    system_parts: list[dict] = []
    converse_messages: list[dict] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        if role == "system":
            if content:
                system_parts.append({"text": str(content)})
            continue

        if role == "user":
            converse_messages.append({
                "role": "user",
                "content": [{"text": str(content)}],
            })
            continue

        if role == "assistant":
            parts: list[dict] = []
            if content:
                parts.append({"text": str(content)})
            for tool_call in msg.get("tool_calls") or []:
                function = tool_call.get("function", {}) or {}
                args_str = function.get("arguments", "{}") or "{}"
                try:
                    args_json = json.loads(args_str)
                except (json.JSONDecodeError, TypeError):
                    args_json = {"raw": args_str}
                parts.append({
                    "toolUse": {
                        "toolUseId": tool_call.get("id", ""),
                        "name": function.get("name", ""),
                        "input": args_json,
                    }
                })
            if parts:
                converse_messages.append({"role": "assistant", "content": parts})
            continue

        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            result_content = str(content)
            converse_messages.append({
                "role": "user",
                "content": [{
                    "toolResult": {
                        "toolUseId": tool_call_id,
                        "content": [{"text": result_content}],
                    }
                }],
            })
            continue

    return system_parts, converse_messages


def _convert_tools(tools: list[dict[str, Any]] | None) -> list[dict] | None:
    """Convert OpenAI-style tool definitions to Bedrock toolConfig format."""
    if not tools:
        return None
    bedrock_tools: list[dict] = []
    for tool in tools:
        function = tool.get("function", tool)
        params = function.get("parameters", {"type": "object", "properties": {}})
        bedrock_tools.append({
            "toolSpec": {
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "inputSchema": {"json": params},
            }
        })
    return bedrock_tools


def _extract_response(response: dict) -> tuple[str, list[dict], dict[str, int]]:
    """Extract text, tool calls, and usage from Converse response."""
    output = response.get("output", {})
    message = output.get("message", {})
    content_blocks = message.get("content", [])

    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for block in content_blocks:
        if "text" in block:
            text_parts.append(block["text"])
        elif "toolUse" in block:
            tool_use = block["toolUse"]
            tool_calls.append({
                "id": tool_use.get("toolUseId", ""),
                "type": "function",
                "function": {
                    "name": tool_use.get("name", ""),
                    "arguments": json.dumps(tool_use.get("input", {})),
                },
            })

    usage = response.get("usage", {})
    usage_info = {
        "prompt_tokens": usage.get("inputTokens", 0),
        "completion_tokens": usage.get("outputTokens", 0),
        "total_tokens": usage.get("inputTokens", 0) + usage.get("outputTokens", 0),
    }

    return "".join(text_parts), tool_calls, usage_info


# -- Core call functions -----------------------------------------------------


def _converse(
    model_id: str,
    system: list[dict],
    messages: list[dict],
    max_tokens: int,
    retries: int,
    stage: str,
    tools: list[dict] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> tuple[str, list[dict], dict[str, int]]:
    """Call Bedrock Converse API with retries."""
    client = _get_client()
    last_err = None

    for attempt in range(retries):
        try:
            kwargs: dict[str, Any] = {
                "modelId": model_id,
                "messages": messages,
                "inferenceConfig": {"maxTokens": max_tokens},
            }
            if system:
                kwargs["system"] = system
            if tools:
                tool_config: dict[str, Any] = {"tools": tools}
                if tool_choice is not None:
                    if isinstance(tool_choice, str):
                        if tool_choice == "auto":
                            tool_config["toolChoice"] = {"auto": {}}
                        elif tool_choice == "required":
                            tool_config["toolChoice"] = {"any": {}}
                        elif tool_choice == "none":
                            pass
                    elif isinstance(tool_choice, dict):
                        fn_name = tool_choice.get("function", {}).get("name")
                        if fn_name:
                            tool_config["toolChoice"] = {"tool": {"name": fn_name}}
                kwargs["toolConfig"] = tool_config

            response = client.converse(**kwargs)
            text, tool_call_list, usage_info = _extract_response(response)

            tracker.record(stage, usage_info["prompt_tokens"], usage_info["completion_tokens"])
            return text, tool_call_list, usage_info

        except Exception as e:
            last_err = e
            sleep = min(2 ** attempt, 30)
            time.sleep(sleep)

    raise RuntimeError(f"Bedrock Converse call failed after {retries} retries: {last_err}")


# -- Public API (matches interface of azure_openai.py) -----------------------


def chat_optimizer(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    reasoning_effort: str | None = None,
    timeout: int | None = None,
) -> tuple[str, dict]:
    system_parts, messages = _messages_from_system_user(system, user)
    text, _, usage = _converse(
        OPTIMIZER_DEPLOYMENT, system_parts, messages, max_completion_tokens, retries, stage,
    )
    return text, usage


def chat_target(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "target",
    reasoning_effort: str | None = None,
    timeout: int | None = None,
) -> tuple[str, dict]:
    system_parts, messages = _messages_from_system_user(system, user)
    text, _, usage = _converse(
        TARGET_DEPLOYMENT, system_parts, messages, max_completion_tokens, retries, stage,
    )
    return text, usage


def chat_with_deployment(
    deployment: str,
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "custom",
    reasoning_effort: str | None = None,
    timeout: int | None = None,
) -> tuple[str, dict]:
    system_parts, messages = _messages_from_system_user(system, user)
    text, _, usage = _converse(
        deployment, system_parts, messages, max_completion_tokens, retries, stage,
    )
    return text, usage


def chat_optimizer_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    reasoning_effort: str | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
) -> tuple[Any, dict]:
    system_parts, converse_msgs = _convert_chat_messages(messages)
    bedrock_tools = _convert_tools(tools)
    text, tool_calls, usage = _converse(
        OPTIMIZER_DEPLOYMENT, system_parts, converse_msgs,
        max_completion_tokens, retries, stage, bedrock_tools, tool_choice,
    )
    if return_message:
        msg = CompatAssistantMessage(
            content=text,
            tool_calls=[
                CompatToolCall(
                    id=tc["id"],
                    function=CompatToolFunction(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                for tc in tool_calls
            ],
        )
        return msg, usage
    return text, usage


def chat_target_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "target",
    reasoning_effort: str | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
) -> tuple[Any, dict]:
    system_parts, converse_msgs = _convert_chat_messages(messages)
    bedrock_tools = _convert_tools(tools)
    text, tool_calls, usage = _converse(
        TARGET_DEPLOYMENT, system_parts, converse_msgs,
        max_completion_tokens, retries, stage, bedrock_tools, tool_choice,
    )
    if return_message:
        msg = CompatAssistantMessage(
            content=text,
            tool_calls=[
                CompatToolCall(
                    id=tc["id"],
                    function=CompatToolFunction(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                for tc in tool_calls
            ],
        )
        return msg, usage
    return text, usage


def chat_messages_with_deployment(
    deployment: str,
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "custom",
    reasoning_effort: str | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
) -> tuple[Any, dict]:
    system_parts, converse_msgs = _convert_chat_messages(messages)
    bedrock_tools = _convert_tools(tools)
    text, tool_calls, usage = _converse(
        deployment, system_parts, converse_msgs,
        max_completion_tokens, retries, stage, bedrock_tools, tool_choice,
    )
    if return_message:
        msg = CompatAssistantMessage(
            content=text,
            tool_calls=[
                CompatToolCall(
                    id=tc["id"],
                    function=CompatToolFunction(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                for tc in tool_calls
            ],
        )
        return msg, usage
    return text, usage


def get_token_summary() -> dict:
    return tracker.summary()


def reset_token_tracker() -> None:
    tracker.reset()


def set_reasoning_effort(effort: str | None) -> None:
    global REASONING_EFFORT
    REASONING_EFFORT = effort if effort else None


def set_target_deployment(deployment: str) -> None:
    global TARGET_DEPLOYMENT
    TARGET_DEPLOYMENT = deployment
    os.environ["BEDROCK_TARGET_MODEL"] = deployment


def set_optimizer_deployment(deployment: str) -> None:
    global OPTIMIZER_DEPLOYMENT
    OPTIMIZER_DEPLOYMENT = deployment
    os.environ["BEDROCK_OPTIMIZER_MODEL"] = deployment


def configure_bedrock(
    *,
    region: str | None = None,
    profile: str | None = None,
    optimizer_model: str | None = None,
    target_model: str | None = None,
) -> None:
    global REGION, PROFILE, OPTIMIZER_DEPLOYMENT, TARGET_DEPLOYMENT
    if region is not None:
        REGION = region.strip()
        os.environ["AWS_BEDROCK_REGION"] = REGION
    if profile is not None:
        PROFILE = profile.strip() or None
        if PROFILE:
            os.environ["AWS_BEDROCK_PROFILE"] = PROFILE
        else:
            os.environ.pop("AWS_BEDROCK_PROFILE", None)
    if optimizer_model is not None:
        OPTIMIZER_DEPLOYMENT = optimizer_model.strip()
        os.environ["BEDROCK_OPTIMIZER_MODEL"] = OPTIMIZER_DEPLOYMENT
    if target_model is not None:
        TARGET_DEPLOYMENT = target_model.strip()
        os.environ["BEDROCK_TARGET_MODEL"] = TARGET_DEPLOYMENT
    _reset_client()
