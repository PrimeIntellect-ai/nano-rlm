"""Context checkpoint helpers."""

from collections.abc import Mapping
from typing import Any

from openai import APIError, APIStatusError, AsyncOpenAI

CHECKPOINT_PROMPT = """Create a concise continuation summary for the current task.
Preserve what is needed to resume accurately:
- The user's objective, exact requirements, constraints, and unresolved decisions.
- Completed work, evidence/results, important paths or sources, and remaining next steps.
- Ongoing orchestration: child names/IDs, assignments and pending follow-ups; Bash job IDs,
  commands and last-known outcomes; subscriptions/targets; unread or retrieved events still
  requiring action; output cursors and useful history message/window references.
- Any interrupted or uncertain operation and side effects that must be inspected before retrying.

Include runnable commands, test filters, or concrete edits when relevant and known.
Use only APIs/tools actually available; do not invent edits, results, resource IDs, or state.
Distinguish last-observed state from assumptions: background work may progress during compaction.
If resources are no longer needed, note that they can be cancelled; do not imply they were cancelled.

Summarize from the existing conversation. Do not call tools. Reply with the summary as plain text."""

REPL_NOTE = (
    "\n\nCompaction itself preserves the IPython kernel and supervisor-owned resources. "
    "Record useful Python variable names and what they contain, but account for any recovery "
    "notice: variables lost in a restart must not be described as still available. "
    "Conversation history remains queryable. Preserve resource names/IDs so handles can be "
    "recovered through the registries. After compaction, refresh resource metadata and inbox "
    "state before deciding whether to repeat work or wait."
)

SUMMARY_FRAMING = (
    "The earlier conversation was compacted. The summary below preserves task progress, "
    "constraints, and references for continuing. Compaction does not finish or restart "
    "background work. Treat resource statuses as last-observed: refresh metadata and inbox "
    "state with the available tools. Consult original history for exact instructions or "
    "missing evidence, and do not duplicate work merely because its full conversation is absent."
)

RESERVE_TOKENS = 16_384
"""Compact when this many tokens remain below the model context window."""

TOOL_OUTPUT_MAX_BYTES = 20_000
"""Middle-out truncation budget for one tool result before it enters the conversation."""

_CONTEXT_FIELDS = (
    "max_model_len",
    "context_length",
    "context_window",
    "max_context_length",
)
_OVERFLOW_MARKERS = (
    # OpenAI error code "context_length_exceeded"; OpenRouter relays the raw body.
    "context_length_exceeded",
    # OpenAI Responses/Completions: "Your input exceeds the context window of this model".
    "exceeds the context window",
    # OpenAI chat: "Input tokens exceed the configured limit of N tokens. Please reduce
    # the length of the messages."; Groq words it the same way.
    "reduce the length of the messages",
    # vLLM: "This model's maximum context length is N tokens"; the renderers pre-flight:
    # "Prompt length (N) exceeds maximum context length (M)"; Mistral uses the same words.
    "maximum context length",
    # Anthropic: "prompt is too long: N tokens > M maximum".
    "prompt is too long",
    # Anthropic byte-size overflow: HTTP 413 {"type": "request_too_large"}.
    "request_too_large",
    # HTTP proxies reject an oversized body with 413 "Request Entity Too Large".
    "request entity too large",
    # Google: "The input token count (N) exceeds the maximum number of tokens allowed (M)".
    "exceeds the maximum number of tokens",
    # xAI: "This model's maximum prompt length is N but the request contains M tokens".
    "maximum prompt length is",
)
_window_cache: dict[tuple[str, str], int | None] = {}


class CompactionFailed(Exception):
    """Every checkpoint attempt failed - the caller ends the run cleanly instead."""


def is_context_overflow(error: APIStatusError) -> bool:
    details = f"{error} {error.body or ''}"
    # An overflow is deterministic: a 400, or a 413 for a byte-size cap.
    return error.status_code in (400, 413) and any(
        marker in details.casefold() for marker in _OVERFLOW_MARKERS
    )


def default_threshold(context_window: int) -> int:
    """Leave a fixed reserve below the window; small windows keep at least half."""
    return max(context_window - RESERVE_TOKENS, context_window // 2)


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
    key = (str(getattr(client, "base_url", None)), model)
    if key not in _window_cache:
        try:
            # Bound startup discovery latency when the client supports options.
            lister = (
                client.with_options(max_retries=0, timeout=10.0)
                if hasattr(client, "with_options")
                else client
            )
            page = await lister.models.list()
            # Context-window fields are provider extensions in model_extra.
            payload = {
                "data": [
                    {"id": card.id, **(card.model_extra or {})} for card in page.data
                ]
            }
        except (APIError, AttributeError):
            # A transient listing failure must not disable compaction for the
            # rest of the process - leave the cache empty so the next engine retries.
            return None
        _window_cache[key] = _model_context_window(payload, model)
    window = _window_cache[key]
    return default_threshold(window) if window is not None else None


def truncate_tool_output(text: str, max_bytes: int = TOOL_OUTPUT_MAX_BYTES) -> str:
    """Keep the head and tail of an oversized tool result and say what was cut."""
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    keep = max_bytes // 2
    head = data[:keep].decode("utf-8", errors="ignore")
    tail = data[-keep:].decode("utf-8", errors="ignore")
    return (
        f"Warning: truncated output (original token count: {estimated_tokens(text)})\n"
        f"Total output lines: {text.count(chr(10)) + 1}\n\n"
        f"{head}\n[... {len(data) - 2 * keep} bytes truncated ...]\n{tail}"
    )


def estimated_tokens(chars: str) -> int:
    """Rough token count at four characters per token."""
    return (len(chars) + 3) // 4


def compactable(messages: list[dict]) -> bool:
    """Whether compaction can reclaim anything - some history beyond the task exists."""
    first_user = next(
        (i for i, m in enumerate(messages) if m.get("role") == "user"), None
    )
    return any(
        m.get("role") != "system" and i != first_user for i, m in enumerate(messages)
    )


OMITTED_CONTEXT = (
    "Some middle context has been omitted to make room for summarization. "
    "Summarize only the information provided. Do not call tools."
)


def hollow_middle(
    messages: list[dict], *, prompt_tokens: int | None, remove_tokens: int
) -> list[dict]:
    """Shorten summary input, keeping its beginning and end and whole tool exchanges.

    Measured input usage calibrates the serialized-size estimate. Provider tokenization
    and tool-schema overhead can differ; further truncated attempts shrink again.
    """
    import json

    system = []
    groups: list[list[dict]] = []
    for message in messages:
        if message.get("role") == "system":
            system.append(message)
        elif message.get("content") == OMITTED_CONTEXT:
            continue
        elif message.get("role") == "tool" and groups:
            groups[-1].append(message)
        else:
            groups.append([message])
    if len(groups) < 3:
        return messages
    sizes = [len(json.dumps(group, ensure_ascii=False)) for group in groups]
    total = len(json.dumps(messages, ensure_ascii=False))
    measured = prompt_tokens or max(1, (total + 3) // 4)
    marker = {"role": "user", "content": OMITTED_CONTEXT}
    needed = total * remove_tokens / measured + len(json.dumps(marker))
    # Grow a contiguous hole around the middle while retaining both endpoints.
    left = right = len(groups) // 2
    removed = sizes[left]
    while removed < needed and (left > 1 or right < len(groups) - 2):
        if left > 1 and (
            right == len(groups) - 2 or sum(sizes[:left]) > sum(sizes[right + 1 :])
        ):
            left -= 1
            removed += sizes[left]
        else:
            right += 1
            removed += sizes[right]
    return [
        *system,
        *(message for group in groups[:left] for message in group),
        marker,
        *(message for group in groups[right + 1 :] for message in group),
    ]
