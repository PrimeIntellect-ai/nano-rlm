"""Context Language Model (CLM) mode: the model edits its own context through a file.

Follows "Context Language Models" (arXiv 2609.37725) and its released harness. The
editable region (everything after the system prompt and the first user message) is
mirrored to ``LIVE_CTX_MAIN.txt`` before each tool call::

    [[CTX_TURN 1 role=assistant]]
    <reasoning>
    <content>
    ipython {"code": "..."}

    [[CTX_TURN 2 role=tool]]
    <output>

and read back after it. Turns the model left unchanged keep their original message
(same ledger index, identical tokens), so an edit invalidates the prefix cache only
from its first changed turn. Changed or new turns become plain assistant/user text,
as in the paper, except that an edited tool result stays a tool result while its call
is kept. Later user requests are pinned: their header carries ``pinned``
and an edit that changes or drops one is rejected.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

CONTEXT_FILE_NAME = "LIVE_CTX_MAIN.txt"
PROTECTED = 2
RESERVE_TOKENS = 2048
NUDGE_RATIOS = (0.25, 0.5, 0.75)
ADAPTIVE_OBS_WINDOW = 3
ADAPTIVE_OBS_MULT = 2.0
ADAPTIVE_FLOOR = 0.5
ADAPTIVE_MIN_BAND = 0.10

_HEADER_RE = re.compile(
    r"^\[\[CTX_TURN\s+(\d+)\s+role=([A-Za-z]+)(\s+pinned)?\]\]\s*$", re.M
)

CONTEXT_PROMPT = """## Managing your context

**Goal: maximize task success** — keep going until the task is done or you run out of
budget (no penalty for extra turns). Your context budget is {budget}; each
result shows your current size.

Your conversation is mirrored to `{path}` (refreshed before
every tool call). **Free up context by editing that file** — replace stale regions (big
outputs, dead ends, superseded notes) with a concise, specific summary. A bloated
transcript wastes budget and dulls your reasoning, so compact as it grows.

**Locate text with code — never paste or retype it** (context is long: retyping wastes
tokens and usually mis-matches). Turns are numbered `[[CTX_TURN 1 …]]`, `[[CTX_TURN 2 …]]`,
… in order from the top of the editable region (the hidden system/task prefix is NOT
counted, so the first turn you can edit is 1). Match a turn by its `[[CTX_TURN <i> role=…]]`
header, a block by a short unique first/last line, or slice on the headers:
    import re; p="{path}"; s=open(p).read()
    # collapse turn 7 (its header -> the next header); its body is never retyped:
    s=re.sub(r"(\\[\\[CTX_TURN 7 [^\\]]*\\]\\]).*?(?=\\n\\[\\[CTX_TURN|\\Z)",
             r"\\1\\n[grep done: parser.py:142 drops quoted commas; fix=csv.reader]", s, flags=re.S)
    open(p,"w").write(s)

**Rules:** don't print this file (its text is already in your context). Keep the
`[[CTX_TURN …]]` header of any turn you keep — emptying its text drops that turn; the
system/task prefix is protected for you. Turns marked `pinned` are user requests: keep
them unchanged. Each result says whether the file changed or matched nothing.

**Compact cheaply** — an edit forces everything *after* it to be re-read, so cost grows
with how much text FOLLOWS the edit:
- **Batch**: one large compaction beats many small edits.
- **Mind what's below your edit** — it all gets re-read, so don't compact a small early
  region while a long, still-useful tail sits beneath it (that re-reads the whole tail
  for little gain). Keep a useful tail; if it's much larger than what you'd compact,
  wait and compact head + tail together — UNLESS you expect to hit the context limit
  soon, then compact now.
- **Be generous in the summary**: the tail is re-read regardless, so a detailed
  replacement is essentially free."""

_NOTE_CONTRACT = (
    "When you write a replacement note, COPY facts forward from the text you are "
    "replacing (quote them): every RULED OUT candidate with its reason and the words "
    "'do not retry'; the exact queries/commands already tried; exact values marked "
    "VERIFIED or UNVERIFIED; and a NEXT line."
)
_OVERFLOW_CONSEQUENCE = (
    "if you cross the limit your context is replaced by an automatic summary and its "
    "details are lost"
)


def rendered_text(message: dict) -> str:
    """The text the model saw for a turn: reasoning, content, then the issued tool call."""
    parts = []
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        parts.append(reasoning)
    if isinstance(message.get("content"), str) and message["content"]:
        parts.append(message["content"])
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        piece = f"{function.get('name', '')} {function.get('arguments') or ''}".strip()
        if piece:
            parts.append(piece)
    return "\n".join(parts)


@dataclass(frozen=True)
class _Turn:
    message: dict
    role: str
    body: str
    pinned: bool


@dataclass(frozen=True)
class EditResult:
    messages: list[dict] | None
    """The new context (protected prefix included), or None when nothing is applied."""
    note: str


class ContextFile:
    """The mirror file, its read-back, and the context budget around it."""

    def __init__(self, directory: Path, budget: int | None):
        self.path = Path(directory) / CONTEXT_FILE_NAME
        self.budget = budget
        self.strict_target = budget - RESERVE_TOKENS if budget else 0
        self.tokens_per_char = 0.25
        self._turns: list[_Turn] = []
        self._rendered = ""
        self._nudged: set[float] = set()

    def prompt(self) -> str:
        budget = f"{self.budget} tokens" if self.budget else "not fixed"
        return CONTEXT_PROMPT.format(budget=budget, path=self.path)

    # ---------------------------------------------------------------- tokens
    def count(self, messages: list[dict]) -> int:
        return int(_chars(messages) * self.tokens_per_char)

    def calibrate(self, prompt_tokens: int, messages: list[dict]) -> None:
        """Fit the char-based count to the server's ``prompt_tokens`` for ``messages``."""
        chars = _chars(messages)
        if prompt_tokens > 0 and chars > 0:
            self.tokens_per_char = prompt_tokens / chars

    # ---------------------------------------------------------------- mirror
    def write(self, messages: list[dict], pinned: list[dict]) -> None:
        """Mirror ``messages[PROTECTED:]``; ``pinned`` holds later user requests."""
        self._turns = [
            _Turn(
                message,
                message.get("role", "user"),
                rendered_text(message),
                any(message is p for p in pinned),
            )
            for message in messages[PROTECTED:]
        ]
        self._rendered = "\n\n".join(
            f"[[CTX_TURN {n} role={t.role}{' pinned' if t.pinned else ''}]]\n{t.body}"
            for n, t in enumerate(self._turns, start=1)
        )
        self.path.write_text(self._rendered, encoding="utf-8")

    def sync(self, messages: list[dict], code: str) -> EditResult:
        """Read the mirror back after a tool call; ``messages`` is the context it was
        written from and ``code`` is the tool input (for the "matched nothing" note)."""
        if not self.path.exists():
            return EditResult(None, "")
        text = self.path.read_text(encoding="utf-8", errors="replace")
        before = self.count(messages)
        if text.strip() == self._rendered.strip():
            if CONTEXT_FILE_NAME in code:
                return EditResult(
                    None,
                    f"\n[{CONTEXT_FILE_NAME}: NO change — your edit matched nothing, so "
                    f"context is still ~{before} tokens. Match text you have already "
                    f"seen, or target the real turn headers, which look like "
                    f"`[[CTX_TURN 12 role=assistant]]` (turn index first, then role).]",
                )
            return EditResult(None, "")
        candidate, error = self._parse(text, messages[:PROTECTED])
        if error:
            return EditResult(
                None,
                f"\n[{CONTEXT_FILE_NAME}: edit REJECTED — {error}, so it was NOT "
                f"applied (still ~{before} tokens).]",
            )
        after = self.count(candidate)
        limit = self.strict_target
        if after > before and not (limit and after <= limit):
            rule = (
                f"An edit must FIT the {limit}-token limit"
                if limit
                else "A compaction must SHRINK context"
            )
            return EditResult(
                None,
                f"\n[{CONTEXT_FILE_NAME}: edit REJECTED — it GREW context "
                f"~{before}->{after} tokens, so it was NOT applied (still ~{before}). "
                f"{rule}: you likely duplicated/appended "
                f"content — replace stale text with a SHORTER summary instead.]",
            )
        if after > before:
            note = (
                f"\n[{CONTEXT_FILE_NAME}: edit applied but it GREW context "
                f"~{before}->{after} tokens (it fits, so it was kept). If you "
                f"meant to condense, you likely duplicated content instead of "
                f"replacing it.]"
            )
        elif limit and after > limit:
            note = (
                f"\n[{CONTEXT_FILE_NAME}: edit applied — context ~{before}->{after} "
                f"tokens, but STILL OVER the ~{limit}-token limit. Compact more "
                f"NOW (delete stale turns/outputs) or your context will be summarized "
                f"automatically.]"
            )
        else:
            note = (
                f"\n[{CONTEXT_FILE_NAME}: edit applied — context ~{before}->{after} "
                f"tokens, {len(candidate)} turns]"
            )
        return EditResult(candidate, note)

    def _parse(self, text: str, protected: list[dict]) -> tuple[list[dict], str | None]:
        matches = list(_HEADER_RE.finditer(text))
        sections: list[tuple[int | None, str, str]] = []
        lead = text[: matches[0].start()] if matches else text
        if lead.strip():
            sections.append((None, "user", lead.strip()))
        for k, match in enumerate(matches):
            end = matches[k + 1].start() if k + 1 < len(matches) else len(text)
            body = text[match.end() : end].strip()
            if body:
                sections.append((int(match.group(1)), match.group(2).lower(), body))

        out: list[dict] = []
        kept: set[int] = set()
        new: set[int] = set()
        for n, role, body in sections:
            turn = self._turns[n - 1] if n and 1 <= n <= len(self._turns) else None
            if (
                turn is not None
                and n not in kept
                and role == turn.role
                and body == turn.body.strip()
            ):
                kept.add(n)
                out.append(turn.message)
                continue
            if turn is not None and turn.pinned:
                return [], f"pinned turn {n} (a user request) was changed"
            new.add(len(out))
            if turn is not None and role == turn.role == "tool":
                # An edited result stays the answer to its call, so the call turn
                # before it keeps its tokens.
                out.append({**turn.message, "content": body})
                continue
            out.append(
                {
                    "role": "assistant" if role == "assistant" else "user",
                    "content": body,
                }
            )
        for n, turn in enumerate(self._turns, start=1):
            if turn.pinned and n not in kept:
                return [], f"pinned turn {n} (a user request) was removed"
        out, new = _repair_tool_pairs(out, new)
        return [*protected, *_merge_new(out, new)], None

    # ---------------------------------------------------------------- readout
    def readout(self, tokens: int) -> str:
        if not self.budget:
            return ""
        over = (
            "" if tokens <= self.strict_target else f" — OVER; compact {self.path} now"
        )
        return f"\n[context: ~{tokens}/{self.strict_target} tokens{over}]"

    def nudge(self, messages: list[dict]) -> str | None:
        """The paper's escalating budget nudges: a one-shot note per crossed tier, and an
        urgent note on every turn once the headroom is smaller than recent outputs."""
        if not self.budget:
            return None
        tokens = self.count(messages)
        self._nudged = {f for f in self._nudged if tokens >= int(self.budget * f)}
        need = self._adaptive_headroom(messages)
        if self.strict_target - tokens < need:
            ratio = (self.strict_target - need) / self.strict_target
            return (
                "CONTEXT BUDGET NUDGE (URGENT): "
                f"You are at {tokens}/{self.strict_target} tokens — over "
                f"{int(round(ratio * 100))}% of your hard context limit and about "
                "to be cut off. Compact your context THIS TURN (do nothing else): remove "
                f"stale regions now. If you cross the limit your context is replaced by an "
                f"automatic summary and its details are lost. {self._hint()}"
            )
        crossed = [f for f in NUDGE_RATIOS if tokens >= int(self.budget * f)]
        to_fire = [f for f in crossed if f not in self._nudged]
        if not to_fire:
            return None
        tier = max(to_fire)
        self._nudged |= set(crossed)
        pct = int(round(tier * 100))
        if tier <= 0.25:
            return (
                "CONTEXT BUDGET NUDGE: "
                f"context is at ~{pct}% of your {self.budget}-token budget ({tokens} tokens). "
                "No action needed. Before your next few searches, make sure your notes "
                "record which queries and documents you already tried."
            )
        if tier <= 0.5:
            body = (
                f"context is at ~{pct}% of your {self.budget}-token budget ({tokens} tokens). "
                "Finish the unit of work in flight, then tidy ONCE. " + _NOTE_CONTRACT
            )
        else:
            body = (
                f"context is at ~{pct}% of your {self.budget}-token budget ({tokens} tokens) — "
                "close to the limit. Compact settled spans now — but do NOT wipe: edits "
                "keeping under 25% of the region they touch are usually followed by "
                "re-doing the deleted work. "
                + _NOTE_CONTRACT
                + f" {_OVERFLOW_CONSEQUENCE}."
            )
        return f"CONTEXT BUDGET NUDGE: {body} {self._hint()}"

    def _adaptive_headroom(self, messages: list[dict]) -> int:
        tools = [m for m in messages[PROTECTED:] if m.get("role") == "tool"]
        obs = max((self.count([m]) for m in tools[-ADAPTIVE_OBS_WINDOW:]), default=0)
        limit = self.strict_target
        by_ratio = int(ADAPTIVE_MIN_BAND * limit)
        by_obs = int(ADAPTIVE_OBS_MULT * obs)
        return min(max(by_ratio, by_obs), int((1.0 - ADAPTIVE_FLOOR) * limit))

    def _hint(self) -> str:
        return (
            f"Compact {self.path} by locating stale regions with code (match a turn by its "
            "[[CTX_TURN i ...]] header or a block by short start/end anchors) and replacing "
            "them with summaries — do not retype the text you remove. Compact settled spans; "
            "keep anything you have not finished using."
        )


def _chars(messages: list[dict]) -> int:
    return len(json.dumps(messages, ensure_ascii=False, default=str))


def _repair_tool_pairs(
    messages: list[dict], new: set[int]
) -> tuple[list[dict], set[int]]:
    """Keep a tool-call turn structured only while all its results directly follow it;
    otherwise both sides become plain text, as for any edited turn."""
    out = list(messages)
    converted: set[int] = set()
    claimed: set[int] = set()
    for i, message in enumerate(out):
        calls = (
            message.get("tool_calls") if message.get("role") == "assistant" else None
        )
        if not calls:
            continue
        ids = {call.get("id") for call in calls}
        j = i + 1
        while j < len(out) and out[j].get("role") == "tool":
            j += 1
        if {
            out[k].get("tool_call_id") for k in range(i + 1, j)
        } == ids and j - i - 1 == len(ids):
            claimed.update(range(i + 1, j))
        else:
            out[i] = {"role": "assistant", "content": rendered_text(message)}
            converted.add(i)
    for i, message in enumerate(out):
        if message.get("role") == "tool" and i not in claimed:
            out[i] = {"role": "user", "content": message.get("content") or ""}
            converted.add(i)
    return out, new | converted


def _merge_new(messages: list[dict], new: set[int]) -> list[dict]:
    """Merge consecutive same-role turns that are both new; untouched turns stay as-is."""
    merged: list[dict] = []
    previous_new = False
    for i, message in enumerate(messages):
        is_new = i in new
        if (
            is_new
            and previous_new
            and message["role"] in ("user", "assistant")
            and merged[-1]["role"] == message["role"]
        ):
            merged[-1] = {
                "role": message["role"],
                "content": f"{merged[-1]['content']}\n\n{message['content']}".strip(),
            }
        else:
            merged.append(message)
        previous_new = is_new
    return merged
