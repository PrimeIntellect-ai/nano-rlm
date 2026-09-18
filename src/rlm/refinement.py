"""Continual-harness refinement: model-proposed edits to harness state.

A refinement pass is a side model call, like compaction: the live conversation is
extended with one user message that asks for a JSON proposal of create/update/delete
edits, the proposal is validated and applied to one store under its lock, and the full
result (with before/after snapshots per edit) is appended to that store's
``refinements.jsonl`` so it can be rolled back later. The engine drives the model calls
and retries; this module holds the prompts, the JSON contract and the apply logic.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from rlm.harness import (
    KINDS,
    HarnessEntry,
    HarnessKind,
    HarnessScope,
    HarnessStore,
    HarnessView,
    RefinementEvent,
    slug,
)

RefinementAction = Literal["create", "update", "delete"]

REFINE_PROMPT = """Review this conversation and improve the continual harness: the persistent, editable
set of prompt notes, memories, skill descriptions and subagent specs shown below. This is
similar to context compaction, but instead of summarizing you emit precise create, update
or delete edits to reusable state.

Components:
- prompt: supplemental prompt notes only. The base system prompt is immutable and MUST NOT be rewritten.
- memory: durable facts, decisions, failures, preferences and outcomes.
- skill: a description of an importable Python module that is already available in the
  kernel. Skill create/update edits MUST include `reference` = {"type": "python",
  "import": "<module>", "callable": "run", "call_pattern": "await <module>(...)"} and an
  `arguments` object describing accepted inputs, required fields, defaults and constraints
  (`{}` only when the callable takes nothing). Importable modules: %(importable)s. Never
  invent a module: a skill entry describes how to use existing code, it does not create it.
- subagent: a reusable delegation spec (purpose, instructions, when to invoke). Include the
  call form: compose a concise task prompt and `child = await rlm.agent.spawn(task,
  name="worker")`; collect the answer with `await child.result()`.

Scope and persistence policy:
- %(scope_policy)s
- Entry ids above carry a display-only `local:`, `ancestor:` or `global:` prefix; use the
  bare id in edits. Ancestor entries are read-only context: never propose update or delete
  edits for them; create a local entry when an override is genuinely needed.
- Use memory for declarative facts and preferences, skill for repeatable procedures exposed
  as Python calls, prompt for narrow behavioural policy addendums, and subagent for reusable
  delegation roles. Create or update the smallest relevant component.
- If a prior refinement caused problems, replace or delete the faulty entries.

Use the conversation, the current harness state and the refinement history. Prefer small,
evidence-backed edits; if no useful edit is justified, return an empty edits array with a
rationale. Never edit source files. Do not call tools. Reply with JSON only, in exactly
this shape:

{
  "summary": "one sentence",
  "rationale": "why these edits are justified by conversation evidence",
  "expected_outcome": "what should improve and how to validate it",
  "edits": [
    {
      "action": "create|update|delete",
      "kind": "prompt|memory|skill|subagent",
      "id": "stable id for update/delete, optional for create",
      "title": "required for create/update",
      "content": "required for create/update",
      "path": "optional grouping path",
      "reference": {"type": "python", "import": "module", "callable": "run", "call_pattern": "await module(...)"},
      "arguments": {"name": {"type": "string", "required": true, "description": "accepted input"}},
      "metadata": {},
      "reason": "why this edit is useful"
    }
  ]
}"""

LOCAL_SCOPE_POLICY = (
    "Requested scope: local. Prefer local edits for current task progress, temporary "
    "blockers, current-run coordination and project facts that are not clearly reusable "
    "across sessions. Global entries are read-only context in a local refinement: do not "
    "propose update or delete edits for them; create a local entry instead if an override "
    "is needed."
)
GLOBAL_SCOPE_POLICY = (
    "Requested scope: global. Only propose stable cross-session lessons, durable user "
    "preferences, reusable skills/subagents or explicitly project-qualified facts that "
    "should affect future sessions. Do not persist session-only progress, temporary "
    "blockers or current-run coordination globally."
)

REVIEW_PROMPT = """Decide whether this checkpoint should run a continual-harness refinement (trigger:
%(trigger)s; %(turns)d work turns since the last review). A refinement writes local
harness state by default: approve when the conversation since the last review contains
evidence useful to this session's future turns (a repeated failure, a reusable tactic, a
repeated delegation role, a durable fact or preference, a user correction). Reject one-off
noise, unsupported hypotheses and transient tool output. Do not call tools. Reply with JSON
only:

{"should_refine": true|false, "rationale": "short reason", "instructions": "optional focus for the refinement"}"""

HISTORY_LIMIT = 5
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")


class RefinementRejected(ValueError):
    """The reply is not a usable proposal; the engine may resample."""


class RefinementFailed(RuntimeError):
    """No usable proposal within the attempt budget, or an unknown rollback target."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RefinementEdit:
    action: str
    kind: str
    id: str | None = None
    title: str | None = None
    content: str | None = None
    path: str | None = None
    reference: dict[str, Any] | None = None
    arguments: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    reason: str | None = None


@dataclass
class RefinementProposal:
    summary: str
    rationale: str
    expected_outcome: str
    edits: list[RefinementEdit]


@dataclass
class AppliedEdit:
    action: str
    kind: str
    id: str
    applied: bool
    error: str | None = None
    title: str | None = None
    content: str | None = None
    reason: str | None = None
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None


@dataclass
class RefinementResult:
    """One line of ``refinements.jsonl``: everything needed to explain or undo a pass."""

    id: str
    trigger: str
    scope: HarnessScope
    summary: str
    rationale: str
    expected_outcome: str
    applied_edits: list[AppliedEdit]
    rollback_of: str | None = None
    usage: dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0}
    )
    request_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)

    @property
    def changes(self) -> list[str]:
        return [f"{e.action} {e.kind}:{e.id}" for e in self.applied_edits if e.applied]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RefinementResult:
        edits = [AppliedEdit(**edit) for edit in data.get("applied_edits", [])]
        return cls(**{**data, "applied_edits": edits})


# --- prompts ---------------------------------------------------------------


def refine_prompt(
    view: HarnessView,
    history: list[RefinementResult],
    *,
    scope: HarnessScope,
    instructions: str | None,
    importable_names: list[str],
) -> str:
    """The user message appended to the live conversation for a planning call."""
    parts = [
        REFINE_PROMPT
        % {
            "importable": ", ".join(sorted(importable_names)) or "(none)",
            "scope_policy": GLOBAL_SCOPE_POLICY
            if scope == "global"
            else LOCAL_SCOPE_POLICY,
        },
        f"<current_harness_state>\n{view.overview(max_entries_per_kind=40, max_content_chars=240)}\n</current_harness_state>",
        f"<refinement_history>\n{history_for_prompt(history)}\n</refinement_history>",
    ]
    if instructions:
        parts.append(f"<refine_instructions>\n{instructions}\n</refine_instructions>")
    return "\n\n".join(parts)


def review_prompt(
    view: HarnessView,
    history: list[RefinementResult],
    *,
    trigger: str,
    turns_since_review: int,
) -> str:
    return "\n\n".join(
        [
            REVIEW_PROMPT % {"trigger": trigger, "turns": turns_since_review},
            f"<current_harness_state>\n{view.overview(max_entries_per_kind=20)}\n</current_harness_state>",
            f"<refinement_history>\n{history_for_prompt(history)}\n</refinement_history>",
        ]
    )


def history_for_prompt(
    history: list[RefinementResult], limit: int = HISTORY_LIMIT
) -> str:
    if not history:
        return "No prior refinements."
    lines = []
    for item in history[-limit:]:
        rollback = f" rollback_of={item.rollback_of}" if item.rollback_of else ""
        changes = ", ".join(item.changes) or "no applied edits"
        lines.append(
            f"- [{item.id}] ({item.scope}{rollback}) {item.summary}: {changes}"
        )
    if len(history) > limit:
        lines.append(f"- +{len(history) - limit} older refinements")
    return "\n".join(lines)


# --- parsing ---------------------------------------------------------------


def _incomplete_json(text: str) -> bool:
    depth = 0
    in_string = False
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
        elif in_string:
            if ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
    return in_string or depth > 0


def extract_json_object(text: str) -> dict[str, Any]:
    """The JSON object in a reply: bare, fenced, or wrapped in prose."""
    trimmed = text.strip()
    candidates = []
    if trimmed.startswith("{") and trimmed.endswith("}"):
        candidates.append(trimmed)
    if fenced := _FENCE_RE.search(trimmed):
        candidates.append(fenced.group(1).strip())
    start, end = trimmed.find("{"), trimmed.rfind("}")
    if start != -1 and end > start:
        candidates.append(trimmed[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    if _incomplete_json(trimmed[start:] if start != -1 else trimmed):
        raise RefinementRejected("the reply's JSON object is truncated")
    raise RefinementRejected("the reply does not contain a JSON object")


def _record(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def parse_proposal(text: str) -> RefinementProposal:
    """Normalize an untrusted reply into a proposal; malformed edit fields survive so
    ``apply_proposal`` can report them individually."""
    value = extract_json_object(text)
    raw_edits = value.get("edits")
    if not isinstance(raw_edits, list):
        raise RefinementRejected("the proposal has no edits array")
    edits = [
        RefinementEdit(
            action=str(edit.get("action")),
            kind=str(edit.get("kind")),
            id=_text(edit.get("id")),
            title=_text(edit.get("title")),
            content=_text(edit.get("content")),
            path=_text(edit.get("path")),
            reference=_record(edit.get("reference")),
            arguments=_record(edit.get("arguments")),
            metadata=_record(edit.get("metadata")),
            reason=_text(edit.get("reason")),
        )
        for edit in raw_edits
        if isinstance(edit, dict)
    ]
    return RefinementProposal(
        summary=_text(value.get("summary")) or "Refined continual harness state",
        rationale=_text(value.get("rationale")) or "",
        expected_outcome=_text(value.get("expected_outcome")) or "",
        edits=edits,
    )


def parse_review(text: str) -> tuple[bool, str, str | None]:
    """``(should_refine, rationale, instructions)`` from an auto-refine review reply."""
    value = extract_json_object(text)
    return (
        value.get("should_refine") is True,
        _text(value.get("rationale")) or "no rationale",
        _text(value.get("instructions")),
    )


# --- apply -----------------------------------------------------------------


def validate_edit(
    edit: RefinementEdit, entry_id: str, importable_names: set[str]
) -> str | None:
    if edit.action not in ("create", "update", "delete"):
        return f"unsupported action {edit.action!r}"
    if edit.kind not in KINDS:
        return f"unsupported kind {edit.kind!r}"
    if edit.kind == "prompt" and entry_id == "base_system_prompt":
        return "base system prompt is not editable"
    if edit.action != "create" and not edit.id:
        return f"{edit.action} requires id"
    if edit.action == "delete":
        return None
    if not edit.title or not edit.content:
        return f"{edit.action} requires title and content"
    if edit.kind != "skill":
        return None
    if edit.arguments is None:
        return f"{edit.action} skill requires arguments"
    reference = edit.reference
    if not reference or reference.get("type") != "python":
        return f"{edit.action} skill requires a python reference"
    module = reference.get("import")
    if not isinstance(module, str) or not module:
        return f"{edit.action} skill requires a python import"
    if not any(
        isinstance(reference.get(key), str) and reference[key]
        for key in ("callable", "call_pattern")
    ):
        return f"{edit.action} skill requires callable or call_pattern"
    if module.split(".")[0] not in importable_names:
        return f"{edit.action} skill references unknown module {module!r}"
    return None


def apply_proposal(
    store: HarnessStore,
    proposal: RefinementProposal,
    *,
    trigger: str,
    importable_names: set[str],
    baseline: dict[str, dict[str, dict[str, Any]]] | None = None,
    rollback_of: str | None = None,
    result_id: str | None = None,
    usage: dict[str, int] | None = None,
    request_ids: list[str] | None = None,
) -> RefinementResult:
    """Apply a proposal to one store under its lock and record the result.

    Invalid edits are recorded with ``applied=False`` and an ``error``; the rest still
    apply. ``baseline`` is the store's entries as snapshotted before planning: an entry
    that changed meanwhile (another writer) is left alone and its edit rejected.
    """
    applied: list[AppliedEdit] = []
    touched: set[str] = set()
    with store.transaction():
        for edit in proposal.edits:
            entry_id = edit.id or (
                slug(edit.title or edit.kind, edit.kind)
                if edit.action == "create"
                else ""
            )
            outcome = AppliedEdit(
                action=edit.action,
                kind=edit.kind,
                id=entry_id,
                applied=False,
                title=edit.title,
                content=edit.content,
                reason=edit.reason,
            )
            applied.append(outcome)
            if error := validate_edit(edit, entry_id, importable_names):
                outcome.error = error
                continue
            kind: HarnessKind = edit.kind  # type: ignore[assignment]
            records = store.entries[kind]
            before = records.get(entry_id)
            outcome.before = asdict(before) if before is not None else None
            key = f"{kind}:{entry_id}"
            if baseline is not None and key not in touched:
                if outcome.before != baseline.get(kind, {}).get(entry_id):
                    outcome.error = "entry changed during refinement planning"
                    continue
            if edit.action == "delete":
                if before is None:
                    outcome.error = "entry not found"
                    continue
                del records[entry_id]
            elif edit.action == "create" and before is not None:
                outcome.error = "entry already exists"
                continue
            elif edit.action == "update" and before is None:
                outcome.error = "entry not found"
                continue
            else:
                after = HarnessEntry(
                    id=entry_id,
                    kind=kind,
                    title=edit.title or (before.title if before else entry_id),
                    content=edit.content or (before.content if before else ""),
                    path=edit.path or (before.path if before else "general"),
                    scope=before.scope if before else store.scope,
                    reference=edit.reference
                    if edit.reference is not None
                    else (before.reference if before else {}),
                    arguments=edit.arguments
                    if edit.arguments is not None
                    else (before.arguments if before else {}),
                    metadata=edit.metadata
                    if edit.metadata is not None
                    else (before.metadata if before else {}),
                    source="refinement",
                    created_at=before.created_at if before else _now(),
                    version=before.version + 1 if before else 1,
                )
                records[entry_id] = after
                outcome.after = asdict(after)
            touched.add(key)
            outcome.applied = True
        result = RefinementResult(
            id=result_id or uuid.uuid4().hex,
            trigger=trigger,
            scope=store.scope,
            summary=proposal.summary,
            rationale=proposal.rationale,
            expected_outcome=proposal.expected_outcome,
            applied_edits=applied,
            rollback_of=rollback_of,
            usage=usage or {"prompt_tokens": 0, "completion_tokens": 0},
            request_ids=list(request_ids or []),
        )
        store.refinements.append(
            RefinementEvent(
                id=result.id,
                trigger=trigger,
                changes=result.changes,
                evidence=proposal.rationale,
                outcome=proposal.expected_outcome,
            )
        )
    store.append_result(result.to_dict())
    return result


def rollback_proposal(target: RefinementResult) -> RefinementProposal:
    """Edits that restore every ``before`` snapshot of ``target``, newest first."""
    edits: list[RefinementEdit] = []
    for edit in reversed(target.applied_edits):
        if not edit.applied:
            continue
        if edit.before is not None:
            edits.append(
                RefinementEdit(
                    action="update" if edit.after is not None else "create",
                    kind=edit.kind,
                    id=edit.id,
                    title=edit.before["title"],
                    content=edit.before["content"],
                    path=edit.before["path"],
                    reference=edit.before["reference"],
                    arguments=edit.before["arguments"],
                    metadata=edit.before["metadata"],
                    reason=f"Rollback {target.id}",
                )
            )
        elif edit.after is not None:
            edits.append(
                RefinementEdit(
                    action="delete",
                    kind=edit.kind,
                    id=edit.id,
                    reason=f"Rollback {target.id}",
                )
            )
    return RefinementProposal(
        summary=f"Rollback refinement {target.id}",
        rationale=f"Restores the harness snapshots recorded by refinement {target.id}.",
        expected_outcome="The faulty refinement's edits are reverted.",
        edits=edits,
    )


def load_history(store: HarnessStore) -> list[RefinementResult]:
    return [RefinementResult.from_dict(record) for record in store.results()]


def find_result(store: HarnessStore, result_id: str) -> RefinementResult:
    for item in load_history(store):
        if item.id == result_id:
            return item
    raise RefinementFailed(f"refinement {result_id!r} not found in {store.dir}")


def baseline_of(store: HarnessStore) -> dict[str, dict[str, dict[str, Any]]]:
    """Entries as they are now, for conflict detection at apply time."""
    store.load()
    return {
        kind: {entry_id: asdict(entry) for entry_id, entry in records.items()}
        for kind, records in store.entries.items()
    }


def notice_text(result: RefinementResult) -> str:
    """One-paragraph runtime notice describing an applied refinement."""
    applied = [e for e in result.applied_edits if e.applied]
    rejected = [e for e in result.applied_edits if not e.applied]
    head = (
        f"Continual harness refined ({result.scope}, id {result.id}): {result.summary}"
    )
    lines = [head]
    if applied:
        lines.append(
            "Applied: " + "; ".join(f"{e.action} {e.kind}:{e.id}" for e in applied)
        )
    else:
        lines.append("No edits applied.")
    if rejected:
        lines.append(
            "Rejected: "
            + "; ".join(f"{e.action} {e.kind}:{e.id} ({e.error})" for e in rejected)
        )
    lines.append(
        "The system prompt now reflects the updated harness. Roll back with "
        f"`await rlm.refine.run(rollback_id={result.id!r})` if these edits prove wrong."
    )
    return "\n".join(lines)


__all__ = [
    "AppliedEdit",
    "RefinementEdit",
    "RefinementFailed",
    "RefinementProposal",
    "RefinementRejected",
    "RefinementResult",
    "apply_proposal",
    "baseline_of",
    "extract_json_object",
    "find_result",
    "history_for_prompt",
    "load_history",
    "notice_text",
    "parse_proposal",
    "parse_review",
    "refine_prompt",
    "review_prompt",
    "rollback_proposal",
    "validate_edit",
]
