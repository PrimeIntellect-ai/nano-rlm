"""Native Anthropic Messages transport for the engine's chat-message history."""

from copy import deepcopy
import json
import time
from typing import Any

from anthropic import AsyncAnthropic
from openai.types.chat import ChatCompletion

from rlm.config import ProviderConfig


def _content_blocks(content: str | list | None) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return deepcopy(content or [])


def _messages(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    system = []
    conversation = []
    for message in messages:
        role = message["role"]
        if role == "system":
            system.extend(_content_blocks(message.get("content")))
            continue
        if role == "tool":
            role = "user"
            blocks = [
                {
                    "type": "tool_result",
                    "tool_use_id": message["tool_call_id"],
                    "content": deepcopy(message.get("content") or ""),
                }
            ]
        elif role == "assistant" and "anthropic_content" in message:
            blocks = deepcopy(message["anthropic_content"])
        else:
            blocks = _content_blocks(message.get("content"))
            for tool in message.get("tool_calls") or []:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool["id"],
                        "name": tool["function"]["name"],
                        "input": json.loads(tool["function"]["arguments"]),
                    }
                )
        if not blocks:
            continue
        if conversation and conversation[-1]["role"] == role:
            conversation[-1]["content"].extend(blocks)
        else:
            conversation.append({"role": role, "content": blocks})
    return system, conversation


async def create_anthropic_completion(
    client: AsyncAnthropic,
    *,
    provider: ProviderConfig,
    model: str,
    messages: list[dict],
    extra_headers: dict[str, str],
    tools: list[dict] | None = None,
    tool_choice: str | None = None,
    parallel_tool_calls: bool = False,
) -> ChatCompletion:
    system, conversation = _messages(messages)
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": provider.max_output_tokens,
        "messages": conversation,
        "extra_headers": extra_headers,
    }
    if system:
        request["system"] = system
    if provider.prompt_cache != "off":
        control = {"type": "ephemeral"}
        if provider.prompt_cache == "1h":
            control["ttl"] = "1h"
        request["cache_control"] = control
        # The system prefix can remain cached when compaction changes the history
        # or switches tool_choice to none for a checkpoint.
        if system:
            system[-1]["cache_control"] = dict(control)
    if tools:
        request["tools"] = [
            {
                "name": tool["function"]["name"],
                "description": tool["function"]["description"],
                "input_schema": deepcopy(tool["function"]["parameters"]),
            }
            for tool in tools
        ]
        request["tool_choice"] = (
            {"type": "none"}
            if tool_choice == "none"
            else {"type": "auto", "disable_parallel_tool_use": not parallel_tool_calls}
        )

    response = await client.messages.create(**request)
    blocks = [block.model_dump(exclude_none=True) for block in response.content]
    tool_calls = [
        {
            "id": block["id"],
            "type": "function",
            "function": {
                "name": block["name"],
                "arguments": json.dumps(block["input"], ensure_ascii=False),
            },
        }
        for block in blocks
        if block["type"] == "tool_use"
    ]
    usage = response.usage
    cache_read = usage.cache_read_input_tokens or 0
    cache_write = usage.cache_creation_input_tokens or 0
    prompt_tokens = usage.input_tokens + cache_read + cache_write
    return ChatCompletion(
        id=response.id,
        object="chat.completion",
        created=int(time.time()),
        model=response.model,
        choices=[
            {
                "index": 0,
                "finish_reason": {
                    "tool_use": "tool_calls",
                    "max_tokens": "length",
                    "model_context_window_exceeded": "length",
                }.get(response.stop_reason, "stop"),
                "message": {
                    "role": "assistant",
                    "content": "".join(
                        block["text"] for block in blocks if block["type"] == "text"
                    )
                    or None,
                    "tool_calls": tool_calls or None,
                    "anthropic_content": blocks,
                },
            }
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": prompt_tokens + usage.output_tokens,
            "prompt_tokens_details": {
                "cached_tokens": cache_read,
                "cache_write_tokens": cache_write,
            },
        },
    )
