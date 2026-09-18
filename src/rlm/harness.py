"""Continual harness state: durable prompt notes, memories, skill descriptions and
sub-agent specs that supplement the immutable system prompt.

A ``HarnessStore`` is one ``harness_state.json`` file. A ``HarnessView`` composes the
stores an agent can see: its own session-local store (read/write), the local stores of
its ancestors (read-only) and, when the runtime contract names one, a global store shared
across sessions (read/write). Inside the kernel, ``harness()`` builds the view from the
``RLM_HARNESS_*`` environment the engine sets at kernel start.

File format (shared with prime-agent)::

    {"schema": 1,
     "entries": {"prompt": {id: entry}, "memory": {...}, "skill": {...}, "subagent": {...}},
     "refinements": [event, ...]}
"""

from __future__ import annotations

import builtins
import fcntl
import json
import os
import re
import unicodedata
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

HarnessKind = Literal["prompt", "memory", "skill", "subagent"]
HarnessScope = Literal["local", "global"]
EntryLayer = Literal["local", "ancestor", "global"]

KINDS: tuple[HarnessKind, ...] = ("prompt", "memory", "skill", "subagent")
STATE_FILE_NAME = "harness_state.json"
RESULTS_FILE_NAME = "refinements.jsonl"
HARNESS_DIR_NAME = "harness"

LOCAL_DIR_ENV = "RLM_HARNESS_LOCAL_DIR"
GLOBAL_DIR_ENV = "RLM_HARNESS_GLOBAL_DIR"
ANCESTOR_DIRS_ENV = "RLM_HARNESS_ANCESTOR_DIRS"
SKILLS_DIR_ENV = "RLM_HARNESS_SKILLS_DIR"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(raw: str, fallback: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in raw.strip())
    normalized = "_".join(part for part in normalized.split("_") if part)
    return (normalized or fallback)[:80]


def local_dir(session_dir: str | Path) -> Path:
    return Path(session_dir) / HARNESS_DIR_NAME


# --- query terms -----------------------------------------------------------

_CJK_TERM_CHARS = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af"
    r"\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f"
    r"\U0002b820-\U0002ceaf\U0002ceb0-\U0002ebef\U0002ebf0-\U0002ee5f"
    r"\U0002f800-\U0002fa1f\U00030000-\U0003134f\U00031350-\U000323af"
    r"\U000323b0-\U0003347f]"
)


def _query_runs(text: str) -> list[str]:
    """Split lowercase text into word runs; punctuation ends a run, and a run also
    breaks at CJK boundaries so spacing-free CJK is cut apart from adjacent words."""
    runs: list[str] = []
    run: list[str] = []
    run_is_cjk = False
    for ch in text:
        if unicodedata.category(ch).startswith("M") or ch.isalnum():
            ch_is_cjk = bool(_CJK_TERM_CHARS.match(ch))
            if run and ch_is_cjk != run_is_cjk:
                runs.append("".join(run))
                run = []
            run_is_cjk = ch_is_cjk
            run.append(ch)
        elif run:
            runs.append("".join(run))
            run = []
    if run:
        runs.append("".join(run))
    return runs


def query_terms(query: str) -> list[str]:
    """Tokenize a query into lowercase substring terms: ASCII runs of 3+ characters,
    other-script runs of 2+, and overlapping bigrams for CJK runs."""
    terms: list[str] = []
    seen: set[str] = set()
    for run in _query_runs(query.lower()):
        if _CJK_TERM_CHARS.search(run):
            candidates = [run[i : i + 2] for i in range(len(run) - 1)] or [run]
        elif run.isascii():
            candidates = [run] if len(run) >= 3 else []
        else:
            candidates = [run] if len(run) >= 2 else []
        for term in candidates:
            if term not in seen:
                seen.add(term)
                terms.append(term)
    return terms


def score_entry(entry: HarnessEntry, terms: list[str]) -> float:
    """Weighted term overlap across title, content and path/id; matches in more
    distinct fields count more."""
    title = entry.title.lower()
    content = entry.content.lower()
    path_and_id = f"{entry.path} {entry.id}".lower()
    total = 0.0
    for term in terms:
        hits = (term in title) + (term in content) + (term in path_and_id)
        if hits:
            total += 1 + (hits - 1) * 0.5
    return total


# --- records ---------------------------------------------------------------


@dataclass
class HarnessEntry:
    """A reusable prompt note, memory, skill description or sub-agent spec."""

    id: str
    kind: HarnessKind
    title: str
    content: str
    path: str = "general"
    scope: HarnessScope = "local"
    reference: dict[str, Any] = field(default_factory=dict)
    arguments: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "agent"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    version: int = 1


@dataclass
class RefinementEvent:
    """Compact record of one refinement pass, kept with the state it changed."""

    id: str
    trigger: str
    changes: list[str]
    evidence: str = ""
    outcome: str = ""
    created_at: str = field(default_factory=_now)


_ENTRY_FIELDS = {f.name for f in fields(HarnessEntry)}
_EVENT_FIELDS = {f.name for f in fields(RefinementEvent)}


def validate_skill_reference(reference: dict[str, Any] | None) -> dict[str, Any]:
    """A skill entry describes an importable Python call: it needs a module and a
    callable or call pattern."""
    if not isinstance(reference, dict):
        raise ValueError("skill entries require a Python reference")
    normalized = dict(reference)
    if normalized.get("type") != "python":
        raise ValueError("skill reference.type must be 'python'")
    if not isinstance(normalized.get("import"), str) or not normalized["import"]:
        raise ValueError("skill reference requires a Python import")
    if not any(
        isinstance(normalized.get(key), str) and normalized[key]
        for key in ("callable", "call_pattern")
    ):
        raise ValueError("skill reference requires a callable or call_pattern")
    return normalized


def _check_kind(kind: str) -> HarnessKind:
    if kind not in KINDS:
        raise ValueError(f"unknown harness kind {kind!r}; expected one of {KINDS}")
    return kind  # type: ignore[return-value]


# --- one file --------------------------------------------------------------


class HarnessStore:
    """CRUD over one ``harness_state.json``.

    Mutations take a file lock, reload, apply and rewrite the file atomically, so the
    engine and the kernel (and other agents sharing a global store) never clobber each
    other. Reads reload whenever the on-disk mtime moved since the last load.
    """

    def __init__(self, directory: str | Path, *, scope: HarnessScope = "local"):
        self.dir = Path(directory).expanduser().resolve()
        self.path = self.dir / STATE_FILE_NAME
        self.scope: HarnessScope = scope
        self.entries: dict[HarnessKind, dict[str, HarnessEntry]] = {
            k: {} for k in KINDS
        }
        self.refinements: list[RefinementEvent] = []
        self._loaded_mtime: int | None = None
        self.load()

    # -- persistence

    def _disk_mtime(self) -> int | None:
        try:
            return self.path.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    def _sync(self) -> None:
        if self._disk_mtime() != self._loaded_mtime:
            self.load()

    def load(self) -> HarnessStore:
        mtime = self._disk_mtime()
        if mtime is None:
            self.entries = {k: {} for k in KINDS}
            self.refinements = []
            self._loaded_mtime = None
            return self
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{self.path}: harness state must be a JSON object")
        entries: dict[HarnessKind, dict[str, HarnessEntry]] = {k: {} for k in KINDS}
        for kind in KINDS:
            for entry_id, raw in (data.get("entries", {}).get(kind) or {}).items():
                values = {
                    key: value for key, value in raw.items() if key in _ENTRY_FIELDS
                }
                values["id"] = str(entry_id)
                values["kind"] = kind
                values.setdefault("scope", self.scope)
                entries[kind][str(entry_id)] = HarnessEntry(**values)
        self.entries = entries
        self.refinements = [
            RefinementEvent(**{k: v for k, v in raw.items() if k in _EVENT_FIELDS})
            for raw in data.get("refinements") or []
        ]
        self._loaded_mtime = mtime
        return self

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        data = {
            "schema": 1,
            "entries": {
                kind: {entry_id: asdict(e) for entry_id, e in records.items()}
                for kind, records in self.entries.items()
            },
            "refinements": [asdict(event) for event in self.refinements],
        }
        temp = self.path.with_name(
            f"{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temp.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            os.replace(temp, self.path)
        finally:
            temp.unlink(missing_ok=True)
        self._loaded_mtime = self._disk_mtime()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Exclusive lock for a load-modify-save cycle."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / f"{STATE_FILE_NAME}.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                self.load()
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @contextmanager
    def transaction(self) -> Iterator[HarnessStore]:
        """Lock, reload, let the caller edit ``entries``/``refinements`` directly, then
        save once. The lock is not re-entrant: use plain attribute edits inside."""
        with self._locked():
            yield self
            self.save()

    def append_result(self, record: dict[str, Any]) -> None:
        """Append one full refinement result to this store's ``refinements.jsonl``."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / RESULTS_FILE_NAME, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def results(self) -> builtins.list[dict[str, Any]]:
        """Every refinement result recorded for this store, oldest first."""
        path = self.dir / RESULTS_FILE_NAME
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    # -- reads

    def get(self, kind: HarnessKind, id: str) -> HarnessEntry | None:
        self._sync()
        return self.entries[_check_kind(kind)].get(id)

    def list(self, kind: HarnessKind | None = None) -> builtins.list[HarnessEntry]:
        self._sync()
        kinds = [_check_kind(kind)] if kind else builtins.list(KINDS)
        records = [e for k in kinds for e in self.entries[k].values()]
        return sorted(records, key=lambda e: (e.kind, e.path, e.title, e.id))

    def count(self, kind: HarnessKind) -> int:
        self._sync()
        return len(self.entries[_check_kind(kind)])

    def list_refinements(self) -> builtins.list[RefinementEvent]:
        self._sync()
        return builtins.list(self.refinements)

    # -- writes

    def _put(
        self,
        kind: HarnessKind,
        title: str,
        content: str,
        *,
        id: str,
        path: str | None,
        reference: dict[str, Any] | None,
        arguments: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
        source: str,
    ) -> HarnessEntry:
        existing = self.entries[kind].get(id)
        if existing is not None:
            existing.title = title
            existing.content = content
            # None keeps the stored value so a title/content-only update does not
            # reset the grouping path or wipe a skill's reference/argument contract.
            if path is not None:
                existing.path = path
            if reference is not None:
                existing.reference = dict(reference)
            if arguments is not None:
                existing.arguments = dict(arguments)
            if metadata is not None:
                existing.metadata = dict(metadata)
            existing.source = source
            existing.updated_at = _now()
            existing.version += 1
            entry = existing
        else:
            entry = HarnessEntry(
                id=id,
                kind=kind,
                title=title,
                content=content,
                path=path if path is not None else "general",
                scope=self.scope,
                reference=dict(reference or {}),
                arguments=dict(arguments or {}),
                metadata=dict(metadata or {}),
                source=source,
            )
            self.entries[kind][id] = entry
        self.save()
        return entry

    def create(
        self,
        kind: HarnessKind,
        title: str,
        content: str,
        *,
        id: str | None = None,
        path: str = "general",
        reference: dict[str, Any] | None = None,
        arguments: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "agent",
    ) -> HarnessEntry:
        kind = _check_kind(kind)
        if kind == "skill":
            reference = validate_skill_reference(reference)
        entry_id = id or slug(title, kind)
        with self._locked():
            if entry_id in self.entries[kind]:
                raise ValueError(f"{kind} entry {entry_id!r} already exists")
            return self._put(
                kind,
                title,
                content,
                id=entry_id,
                path=path,
                reference=reference,
                arguments=arguments,
                metadata=metadata,
                source=source,
            )

    def update(
        self,
        kind: HarnessKind,
        id: str,
        title: str,
        content: str,
        *,
        path: str | None = None,
        reference: dict[str, Any] | None = None,
        arguments: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "agent",
    ) -> HarnessEntry:
        kind = _check_kind(kind)
        if kind == "skill" and reference is not None:
            reference = validate_skill_reference(reference)
        with self._locked():
            if id not in self.entries[kind]:
                raise ValueError(f"{kind} entry {id!r} does not exist")
            return self._put(
                kind,
                title,
                content,
                id=id,
                path=path,
                reference=reference,
                arguments=arguments,
                metadata=metadata,
                source=source,
            )

    def upsert(
        self,
        kind: HarnessKind,
        title: str,
        content: str,
        *,
        id: str | None = None,
        path: str | None = None,
        reference: dict[str, Any] | None = None,
        arguments: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "agent",
    ) -> HarnessEntry:
        kind = _check_kind(kind)
        if kind == "skill" and (reference is not None or id is None):
            reference = validate_skill_reference(reference)
        with self._locked():
            return self._put(
                kind,
                title,
                content,
                id=id or slug(title, kind),
                path=path,
                reference=reference,
                arguments=arguments,
                metadata=metadata,
                source=source,
            )

    def restore(self, kind: HarnessKind, id: str, entry: HarnessEntry | None) -> None:
        """Put back an exact prior snapshot (``None`` removes the entry); for rollback."""
        kind = _check_kind(kind)
        with self._locked():
            if entry is None:
                self.entries[kind].pop(id, None)
            else:
                self.entries[kind][id] = HarnessEntry(
                    **{**asdict(entry), "id": id, "kind": kind}
                )
            self.save()

    def delete(self, kind: HarnessKind, id: str) -> bool:
        kind = _check_kind(kind)
        with self._locked():
            if id not in self.entries[kind]:
                return False
            del self.entries[kind][id]
            self.save()
            return True

    def record_refinement(
        self,
        trigger: str,
        changes: builtins.list[str] | str,
        *,
        evidence: str = "",
        outcome: str = "",
        id: str | None = None,
    ) -> RefinementEvent:
        with self._locked():
            event = RefinementEvent(
                id=id or uuid.uuid4().hex,
                trigger=trigger,
                changes=[changes]
                if isinstance(changes, str)
                else builtins.list(changes),
                evidence=evidence,
                outcome=outcome,
            )
            self.refinements.append(event)
            self.save()
            return event

    def snapshot(self) -> dict[str, Any]:
        self._sync()
        return {
            "path": str(self.path),
            "scope": self.scope,
            "entries": {
                kind: {entry_id: asdict(e) for entry_id, e in records.items()}
                for kind, records in self.entries.items()
            },
            "refinements": [asdict(event) for event in self.refinements],
        }


# --- what one agent sees ---------------------------------------------------


def _split_layer(id: str) -> tuple[EntryLayer | None, str]:
    """Accept ids as ``overview()`` displays them (``local:x``, ``global:x``,
    ``ancestor:x``); a bare id has no layer."""
    layer, sep, rest = id.partition(":")
    if sep and rest and layer in ("local", "ancestor", "global"):
        return layer, rest  # type: ignore[return-value]
    return None, id


class HarnessView:
    """The harness layers visible to one agent.

    Reads merge ``local`` (own session), ``ancestors`` (nearest first, read-only) and
    ``global_``. Writes go to ``local`` unless ``global_=True``; a write aimed at an
    ancestor layer raises ``PermissionError``.
    """

    def __init__(
        self,
        local: HarnessStore,
        *,
        global_: HarnessStore | None = None,
        ancestors: builtins.list[HarnessStore] | None = None,
    ):
        self.local = local
        self.global_ = global_
        self.ancestors = builtins.list(ancestors or [])

    def layers(self) -> builtins.list[tuple[EntryLayer, HarnessStore]]:
        result: builtins.list[tuple[EntryLayer, HarnessStore]] = [("local", self.local)]
        result.extend(("ancestor", store) for store in self.ancestors)
        if self.global_ is not None:
            result.append(("global", self.global_))
        return result

    def _target(
        self, global_: bool, id: str | None = None
    ) -> tuple[HarnessStore, str | None]:
        layer, bare = _split_layer(id) if id is not None else (None, None)
        if layer == "ancestor":
            raise PermissionError("ancestor harness entries are read-only")
        if global_ or layer == "global":
            if self.global_ is None:
                raise RuntimeError(
                    "no global harness store is configured for this session"
                )
            return self.global_, bare
        return self.local, bare

    # -- reads

    def entries(
        self, kind: HarnessKind | None = None
    ) -> builtins.list[tuple[EntryLayer, HarnessEntry]]:
        """Every visible entry with the layer it comes from, local first."""
        return [(layer, e) for layer, store in self.layers() for e in store.list(kind)]

    def list(self, kind: HarnessKind | None = None) -> builtins.list[HarnessEntry]:
        return [entry for _, entry in self.entries(kind)]

    def get(self, kind: HarnessKind, id: str) -> HarnessEntry | None:
        layer, bare = _split_layer(id)
        for store_layer, store in self.layers():
            if layer is not None and store_layer != layer:
                continue
            if (entry := store.get(kind, bare)) is not None:
                return entry
        return None

    def search(
        self, query: str, kind: HarnessKind | None = None, limit: int = 10
    ) -> builtins.list[HarnessEntry]:
        """Visible entries ranked by weighted term overlap with ``query``."""
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise TypeError("limit must be a positive int")
        terms = query_terms(query)
        if not terms:
            return []
        scored = [(score_entry(e, terms), e) for e in self.list(kind)]
        ranked = sorted(
            (item for item in scored if item[0] > 0),
            key=lambda item: (item[0], item[1].updated_at),
            reverse=True,
        )
        return [entry for _, entry in ranked[:limit]]

    def refinements(self) -> builtins.list[RefinementEvent]:
        events = self.local.list_refinements()
        if self.global_ is not None:
            events.extend(self.global_.list_refinements())
        return sorted(events, key=lambda e: e.created_at)

    def counts(self) -> dict[str, Any]:
        return {
            "local": {kind: self.local.count(kind) for kind in KINDS},
            "global": {kind: self.global_.count(kind) for kind in KINDS}
            if self.global_ is not None
            else None,
            "ancestors": len(self.ancestors),
            "refinements": len(self.refinements()),
        }

    def overview(
        self, *, max_entries_per_kind: int = 20, max_content_chars: int = 120
    ) -> str:
        """Human-readable summary of every visible layer."""
        lines = [f"Harness state: local={self.local.path}"]
        if self.ancestors:
            lines.append(
                "ancestors (read-only): "
                + ", ".join(str(s.path) for s in self.ancestors)
            )
        if self.global_ is not None:
            lines.append(f"global: {self.global_.path}")
        for kind in KINDS:
            records = self.entries(kind)
            lines.append(f"{kind}: {len(records)}")
            for layer, entry in records[:max_entries_per_kind]:
                lines.append("  - " + format_entry(layer, entry, max_content_chars))
            if len(records) > max_entries_per_kind:
                lines.append(f"  - +{len(records) - max_entries_per_kind} more")
        events = self.refinements()
        lines.append(f"refinements: {len(events)}")
        for event in events[-5:]:
            lines.append(
                f"  - [{event.id}] {event.trigger}: {', '.join(event.changes)}"
            )
        return "\n".join(lines)

    def snapshot(self) -> dict[str, Any]:
        return {
            "local": self.local.snapshot(),
            "ancestors": [store.snapshot() for store in self.ancestors],
            "global": self.global_.snapshot() if self.global_ is not None else None,
        }

    # -- writes

    def create(
        self,
        kind: HarnessKind,
        title: str,
        content: str,
        *,
        id: str | None = None,
        path: str = "general",
        reference: dict[str, Any] | None = None,
        arguments: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "agent",
        global_: bool = False,
    ) -> HarnessEntry:
        store, bare = self._target(global_, id)
        return store.create(
            kind,
            title,
            content,
            id=bare,
            path=path,
            reference=reference,
            arguments=arguments,
            metadata=metadata,
            source=source,
        )

    def update(
        self,
        kind: HarnessKind,
        id: str,
        title: str,
        content: str,
        *,
        path: str | None = None,
        reference: dict[str, Any] | None = None,
        arguments: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "agent",
        global_: bool = False,
    ) -> HarnessEntry:
        store, bare = self._target(global_, id)
        assert bare is not None
        return store.update(
            kind,
            bare,
            title,
            content,
            path=path,
            reference=reference,
            arguments=arguments,
            metadata=metadata,
            source=source,
        )

    def delete(self, kind: HarnessKind, id: str, *, global_: bool = False) -> bool:
        store, bare = self._target(global_, id)
        assert bare is not None
        return store.delete(kind, bare)

    def record_refinement(
        self,
        trigger: str,
        changes: builtins.list[str] | str,
        *,
        evidence: str = "",
        outcome: str = "",
        id: str | None = None,
        global_: bool = False,
    ) -> RefinementEvent:
        store, _ = self._target(global_)
        return store.record_refinement(
            trigger, changes, evidence=evidence, outcome=outcome, id=id
        )

    # -- per-kind conveniences

    def create_memory(self, title: str, content: str, **kwargs: Any) -> HarnessEntry:
        return self.create("memory", title, content, **kwargs)

    def update_memory(
        self, id: str, title: str, content: str, **kwargs: Any
    ) -> HarnessEntry:
        return self.update("memory", id, title, content, **kwargs)

    def delete_memory(self, id: str, **kwargs: Any) -> bool:
        return self.delete("memory", id, **kwargs)

    def create_prompt_note(
        self, title: str, content: str, **kwargs: Any
    ) -> HarnessEntry:
        kwargs.setdefault("path", "policy")
        return self.create("prompt", title, content, **kwargs)

    def update_prompt_note(
        self, id: str, title: str, content: str, **kwargs: Any
    ) -> HarnessEntry:
        return self.update("prompt", id, title, content, **kwargs)

    def delete_prompt_note(self, id: str, **kwargs: Any) -> bool:
        return self.delete("prompt", id, **kwargs)

    def create_skill(
        self, title: str, content: str, *, reference: dict[str, Any], **kwargs: Any
    ) -> HarnessEntry:
        return self.create("skill", title, content, reference=reference, **kwargs)

    def update_skill(
        self, id: str, title: str, content: str, **kwargs: Any
    ) -> HarnessEntry:
        return self.update("skill", id, title, content, **kwargs)

    def delete_skill(self, id: str, **kwargs: Any) -> bool:
        return self.delete("skill", id, **kwargs)

    def create_subagent(self, title: str, content: str, **kwargs: Any) -> HarnessEntry:
        return self.create("subagent", title, content, **kwargs)

    def update_subagent(
        self, id: str, title: str, content: str, **kwargs: Any
    ) -> HarnessEntry:
        return self.update("subagent", id, title, content, **kwargs)

    def delete_subagent(self, id: str, **kwargs: Any) -> bool:
        return self.delete("subagent", id, **kwargs)


def format_entry(layer: EntryLayer, entry: HarnessEntry, max_content_chars: int) -> str:
    """One-line rendering shared by ``overview()`` and the system prompt."""
    extras = ""
    if entry.kind == "skill":
        if entry.reference:
            extras += " ref=" + compact(
                json.dumps(entry.reference, ensure_ascii=False, sort_keys=True),
                max_content_chars,
            )
        if entry.arguments:
            extras += " args=" + compact(
                json.dumps(entry.arguments, ensure_ascii=False, sort_keys=True),
                max_content_chars,
            )
    return (
        f"[{layer}:{entry.id}] {entry.title} ({entry.path}, v{entry.version}){extras}: "
        + compact(entry.content, max_content_chars)
    )


def compact(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


# --- construction ----------------------------------------------------------


def build_view(
    local: str | Path,
    *,
    global_dir: str | Path | None = None,
    ancestor_dirs: builtins.list[str | Path] | tuple[str | Path, ...] = (),
) -> HarnessView:
    return HarnessView(
        HarnessStore(local, scope="local"),
        global_=HarnessStore(global_dir, scope="global") if global_dir else None,
        ancestors=[HarnessStore(d, scope="local") for d in ancestor_dirs],
    )


def harness(session_dir: str | Path | None = None) -> HarnessView:
    """The harness visible to this kernel.

    With no argument the view follows the ``RLM_HARNESS_*`` environment the engine set
    at kernel start; a session directory loads that session's local store alone.
    """
    if session_dir is not None:
        return build_view(local_dir(session_dir))
    local = os.environ.get(LOCAL_DIR_ENV)
    if not local:
        raise RuntimeError("the continual harness is disabled for this session")
    ancestors = [
        d for d in os.environ.get(ANCESTOR_DIRS_ENV, "").split(os.pathsep) if d
    ]
    return build_view(
        local,
        global_dir=os.environ.get(GLOBAL_DIR_ENV) or None,
        ancestor_dirs=ancestors,
    )


__all__ = [
    "KINDS",
    "HarnessEntry",
    "HarnessKind",
    "HarnessScope",
    "HarnessStore",
    "HarnessView",
    "RefinementEvent",
    "build_view",
    "format_entry",
    "harness",
    "local_dir",
    "query_terms",
    "score_entry",
    "validate_skill_reference",
]
