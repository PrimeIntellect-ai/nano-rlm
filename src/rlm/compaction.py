"""Context checkpoint helpers."""

import re
from collections.abc import Mapping
from typing import Any, cast

from openai import APIError, AsyncOpenAI, BadRequestError

CHECKPOINT_ATTEMPTS = 3

CHECKPOINT_PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue

Be concise, structured, and focused on helping the next LLM seamlessly continue the work.

Reply with the summary as plain text. Do not call any tools - summarize from the conversation as it stands."""

REPL_NOTE = (
    "\n\n"
    "Note: the IPython kernel stays running across this compaction. "
    "All variables, imports, and in-memory data are preserved. "
    "Mention important variable names and what they contain so the "
    "next LLM knows what's available."
)

SUMMARY_FRAMING = """Another language model started to solve this problem and produced \
a summary of its thinking process. Use this to build on the work \
that has already been done and avoid duplicating work. Here is \
the summary produced by the other language model, use the \
information in this summary to assist with your own analysis:"""

DROPPED_TOOL_RESULT = "[tool output dropped because it exceeded the context limit]"

_CONTEXT_FIELDS = (
    "max_model_len",
    "context_length",
    "context_window",
    "max_context_length",
)
_OVERFLOW_MARKERS = (
    "request entity too large",
    "context_length",
    "context length",
    "context window",
    "prompt is too long",
    "too many tokens",
    "token limit exceeded",
)
_WINDOW_PATTERNS = (
    re.compile(
        r"(?:maximum|max(?:imum)?)[^.\n]{0,40}(?:context length|context window)"
        r"[^\d]{0,20}([\d,]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"[\"']?(?:max_model_len|context_length)[\"']?\s*[:=]\s*([\d,]+)",
        re.IGNORECASE,
    ),
)
_window_cache: dict[tuple[str, str], int | None] = {}


def context_error(error: BadRequestError) -> tuple[bool, int | None]:
    details = f"{error} {error.body or ''}"
    overflow = any(marker in details.casefold() for marker in _OVERFLOW_MARKERS)
    for pattern in _WINDOW_PATTERNS:
        if match := pattern.search(details):
            window = int(match.group(1).replace(",", ""))
            return overflow, default_threshold(window)
    return overflow, None


def default_threshold(context_window: int) -> int:
    return max(1, context_window * 9 // 10)


def _model_context_window(payload: Mapping[str, Any], model: str) -> int | None:
    card = next(
        (
            item
            for item in payload.get("data") or []
            if isinstance(item, Mapping) and item.get("id") == model
        ),
        None,
    )
    if card is None:
        return None
    for field in _CONTEXT_FIELDS:
        value = card.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


async def discover_threshold(client: AsyncOpenAI, model: str) -> int | None:
    key = (str(client.base_url), model)
    if key not in _window_cache:
        try:
            # The SDK needs a parameterized mapping type to parse into (bare `dict` fails).
            payload = await client.get("/models", cast_to=cast(Any, dict[str, Any]))
            _window_cache[key] = _model_context_window(payload, model)
        except (APIError, AttributeError):
            _window_cache[key] = None
    window = _window_cache[key]
    return default_threshold(window) if window is not None else None


def estimated_tokens(text: str) -> int:
    return (len(text) + 3) // 4


def drop_latest_tool_result(messages: list[dict]) -> bool:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "tool":
            continue
        if message.get("content") == DROPPED_TOOL_RESULT:
            continue
        messages[index] = {**message, "content": DROPPED_TOOL_RESULT}
        return True
    return False
