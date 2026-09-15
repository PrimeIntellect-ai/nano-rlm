"""Delimiters for harness-generated messages that travel in the user role.

Everything the runtime says to the model (inbox notices, hints, nudges, kernel
recovery notices, compaction summaries) and everything a parent agent sends arrives as
a user-role message, the same transport a human request uses. These helpers wrap such
messages in a tag that names their origin and attach the same facts as structured
provenance on the ledger entry, so a recorded conversation can be converted to a chat
template with distinct roles without guessing from the wording.

    <runtime_event kind="notice" unread="2" hints="env-prefix">...</runtime_event>
    <agent_input from="parent" agent="a1b2c3" kind="instruction">...</agent_input>

The human task prompt is never wrapped. Tool output keeps its own role.
"""

from __future__ import annotations

RUNTIME_EVENT = "runtime_event"
AGENT_INPUT = "agent_input"


def _attrs(attrs: dict[str, object]) -> str:
    parts = []
    for key, value in attrs.items():
        if value is None or value == "" or value == []:
            continue
        if isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        text = str(value).replace("&", "&amp;").replace('"', "&quot;")
        parts.append(f' {key}="{text}"')
    return "".join(parts)


def runtime_event(kind: str, text: str, **attrs: object) -> tuple[dict, dict]:
    """A user-role message for a runtime event, plus its provenance record.

    kind: notice (inbox count and hints), recovery (kernel restart), nudge (empty or
    plan-like reply), compaction (context summary)."""
    provenance = {
        "source": "runtime",
        "kind": kind,
        **{k: v for k, v in attrs.items() if v not in (None, "", [])},
    }
    content = (
        f'<{RUNTIME_EVENT} kind="{kind}"{_attrs(attrs)}>\n{text}\n</{RUNTIME_EVENT}>'
    )
    return {"role": "user", "content": content}, provenance


def agent_input(text: str, *, agent: str | None, kind: str) -> tuple[dict, dict]:
    """A user-role message carrying a parent agent's instruction (kind: instruction or
    steer), plus its provenance record."""
    provenance = {"source": "agent", "from": "parent", "kind": kind}
    if agent:
        provenance["agent"] = agent
    content = (
        f'<{AGENT_INPUT} from="parent"{_attrs({"agent": agent, "kind": kind})}>\n'
        f"{text}\n</{AGENT_INPUT}>"
    )
    return {"role": "user", "content": content}, provenance
