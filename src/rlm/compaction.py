"""Context checkpoint helpers."""

from collections.abc import Mapping
from typing import Any

from openai import APIError, APIStatusError, AsyncOpenAI

CHECKPOINT_PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary another LLM can ACT on immediately to resume the task.

The harness preserves the caller's requests verbatim alongside your summary. Do not reconstruct or repeat the task; focus on the state needed to continue.

Include:
- Current progress, established facts with source URLs or file paths, and key decisions
- Unresolved questions, contradictory evidence, and a NUMBERED list of remaining next steps
- Important tool state that the next model can reuse

For coding tasks, include exact reproduction/test commands and concrete pending edits in fenced code blocks when applicable. For research tasks, preserve the evidence needed to support or reject candidates. Distinguish verified facts from hypotheses. If the task is already solved, state the answer and its supporting evidence.

Be concise and concrete. Reply with the summary as plain text. Do not call tools or emit tool-call markup; summarize from the conversation as it stands."""

REPL_NOTE = (
    "\n\n"
    "Note: the IPython kernel stays running across this compaction. "
    "All variables, imports, and in-memory data are preserved. "
    "Mention important variable names and what they contain so the "
    "next LLM knows what's available."
)

SUMMARY_FRAMING = """Another language model started to solve this problem and produced \
a summary of its thinking process. You also have access to the state of the tools that \
were used by that language model. Use this to build on the work \
that has already been done and avoid duplicating work. Here is \
the summary produced by the other language model, use the \
information in this summary to assist with your own analysis:"""

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
    # getattr: a client without a base_url (or models API, caught below) simply
    # yields no threshold — compaction then stays overflow-reactive.
    key = (str(getattr(client, "base_url", None)), model)
    if key not in _window_cache:
        try:
            # `models.list()` keeps provider extensions in each card's `model_extra`;
            # a raw `cast_to` parse breaks on one Python version or another.
            # One quick attempt only: discovery runs at engine start, so a dead or
            # slow /models endpoint must not stall the session behind the
            # transport's retry budget. (Duck-typed clients without `with_options`
            # are queried as-is.)
            lister = (
                client.with_options(max_retries=0, timeout=10.0)
                if hasattr(client, "with_options")
                else client
            )
            page = await lister.models.list()
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
