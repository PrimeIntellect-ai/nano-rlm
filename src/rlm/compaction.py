"""Context checkpoint helpers."""

import json
import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from openai import APIError, APIStatusError, AsyncOpenAI

from rlm.tools.output import KEPT_PER_END, OutputSize

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

CUT_REPLY_NOTE = (
    "\n\nYour last reply reached the context cutoff and was cut off mid-thought. "
    "Record where that reply left off and what you were about to do next, so the work "
    "continues from there."
)

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
_window_cache: dict[tuple[str, str], tuple[int | None, bool]] = {}


class CompactionFailed(Exception):
    """Every checkpoint attempt failed - the caller ends the run cleanly instead."""


class CompactionFormatViolation(CompactionFailed):
    """The model exhausted its attempts without producing a usable summary."""


def is_context_overflow(error: APIStatusError) -> bool:
    details = f"{error} {error.body or ''}"
    # An overflow is deterministic: a 400, or a 413 for a byte-size cap.
    return error.status_code in (400, 413) and any(
        marker in details.casefold() for marker in _OVERFLOW_MARKERS
    )


def default_threshold(context_window: int, summary_budget: int) -> int:
    """Reserve room for the summary request and reply; small windows keep half."""
    return max(context_window - summary_budget, context_window // 2)


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


async def discover_context_limits(
    client: AsyncOpenAI, model: str
) -> tuple[int | None, bool]:
    """Discover the real window and support for an exact per-request context cap."""
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
            return None, False
        supports_limit = any(
            card.get("id") == model and card.get("supports_max_context_tokens") is True
            for card in payload["data"]
        )
        _window_cache[key] = _model_context_window(payload, model), supports_limit
    return _window_cache[key]


def truncate_tool_output(
    text: str,
    max_bytes: int = TOOL_OUTPUT_MAX_BYTES,
    omitted: OutputSize | None = None,
) -> str:
    """Keep the head and tail of an oversized tool result and say what was cut.
    `omitted` is how much more the whole result measures than `text`, which kept only
    its start and end; the note describes the whole result."""
    data = text.encode("utf-8")
    size = OutputSize.of(text) + (omitted or OutputSize())
    if size.nbytes <= max_bytes:
        return text
    keep = max_bytes // 2
    if omitted is not None:
        keep = min(keep, KEPT_PER_END)
    head = data[:keep].decode("utf-8", errors="ignore")
    tail = data[-keep:].decode("utf-8", errors="ignore")
    return (
        f"Warning: truncated output (original token count: {estimated_tokens(size.nchars)})\n"
        f"Total output lines: {size.newlines + 1}\n\n"
        f"{head}\n[... {size.nbytes - 2 * keep} bytes truncated ...]\n{tail}"
    )


def estimated_tokens(chars: str | int) -> int:
    """Rough token count at four characters per token, of text or a character count."""
    return ((chars if isinstance(chars, int) else len(chars)) + 3) // 4


def compactable(messages: list[dict]) -> bool:
    """Whether compaction can reclaim anything - some history beyond the task exists."""
    first_user = next(
        (i for i, m in enumerate(messages) if m.get("role") == "user"), None
    )
    return any(
        m.get("role") != "system" and i != first_user for i, m in enumerate(messages)
    )


TOOL_OUTPUT_STUB = "[tool output omitted to make room for the summary]"

KEEP_REPLY_TOKENS = (4_096, 16_384)
"""Maximum beginning/end budgets shared by an oversized newest reply's text fields."""

_REASONING_FIELDS = ("reasoning_content", "reasoning")
_OMISSION = re.compile(
    r"^\[(\d+) earlier messages were omitted to make room for the summary\."
)


def omission_marker(count: int) -> dict:
    return {
        "role": "user",
        "content": (
            f"[{count} earlier messages were omitted to make room for the summary. "
            "The full history remains available through rlm.history. Summarize from "
            "what is shown and do not call tools.]"
        ),
    }


def _omitted_count(message: dict) -> int | None:
    content = message.get("content")
    if message.get("role") != "user" or not isinstance(content, str):
        return None
    match = _OMISSION.match(content)
    return int(match.group(1)) if match else None


def _size(item: Any) -> int:
    return len(json.dumps(item, ensure_ascii=False))


def _keep_ends(text: str, head: int, tail: int) -> str:
    """Keep the ends of `text` only when the omission note still makes it shorter."""
    omitted = len(text) - head - tail
    note = f"\n[... {omitted} characters omitted ...]\n"
    if omitted <= len(note):
        return text
    return text[:head] + note + (text[-tail:] if tail else "")


def trim_summary_input(
    messages: list[dict], *, context_tokens: int, remove_tokens: int
) -> list[dict]:
    """Shrink a summary request by about `remove_tokens`, cheapest context first.

    System messages, the first message after them (the task or the carried-over summary)
    and the newest exchange are kept. Between them, oldest first: tool outputs give way to
    a stub, then assistant reasoning is dropped (tool calls stay as the record of what was
    done), then whole exchanges are replaced by one omission marker. If that is still not
    enough, the newest assistant reply keeps its beginning and end, sized to the
    remaining space and shared across content and reasoning fields.

    `context_tokens` measures `messages` as a prompt and calibrates the serialized-size
    estimate. The measured prompt also covers tool schemas, which are not in `messages`,
    so the chars-per-token ratio runs low; over-remove a little rather than spend another
    attempt. `messages` itself is left unchanged.
    """
    system: list[dict] = []
    groups: list[list[dict]] = []
    omitted = 0
    for message in deepcopy(messages):
        if message.get("role") == "system":
            system.append(message)
        elif (count := _omitted_count(message)) is not None:
            omitted += count
        elif message.get("role") == "tool" and groups:
            groups[-1].append(message)
        else:
            groups.append([message])
    total = _size(messages)
    needed = 1.25 * total * remove_tokens / max(context_tokens, 1) + _size(
        omission_marker(len(messages))
    )
    middle = groups[1:-1]
    newest = groups[-1] if len(groups) > 1 else []
    removed = 0
    for message in (message for group in middle for message in group):
        if removed >= needed:
            break
        if message.get("role") == "tool" and _size(message.get("content")) > _size(
            TOOL_OUTPUT_STUB
        ):
            removed += _size(message)
            message["content"] = TOOL_OUTPUT_STUB
            removed -= _size(message)
    for message in (message for group in middle for message in group):
        if removed >= needed:
            break
        if message.get("role") == "assistant":
            removed += _size(message)
            for field in _REASONING_FIELDS:
                message.pop(field, None)
            removed -= _size(message)
    while middle and removed < needed:
        group = middle.pop(0)
        removed += sum(_size(message) for message in group)
        omitted += len(group)
    if removed < needed:
        fields = [
            (message, field, text)
            for message in newest
            if message.get("role") == "assistant"
            for field in ("content", *_REASONING_FIELDS)
            if isinstance(text := message.get(field), str) and text
        ]
        chars = sum(len(text) for _, _, text in fields)
        size = sum(_size(text) for _, _, text in fields)
        head, tail = KEEP_REPLY_TOKENS
        keep = min(
            4 * (head + tail),
            max(0, int(chars * (1 - (needed - removed) / max(size, 1)))),
        )
        for message, field, text in fields:
            share = max(2, keep * len(text) // max(chars, 1))
            start = max(1, share * head // (head + tail))
            message[field] = _keep_ends(text, start, share - start)
    return [
        *system,
        *(message for group in groups[:1] for message in group),
        *([omission_marker(omitted)] if omitted else []),
        *(message for group in middle for message in group),
        *newest,
    ]
