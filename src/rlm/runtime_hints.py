"""Per-agent hint delivery and lifetime budgets."""

from dataclasses import dataclass, field


@dataclass
class RuntimeHints:
    pending: list[tuple[str, str]] = field(default_factory=list)
    muted: set[str] = field(default_factory=set)
    counts: dict[str, int] = field(default_factory=dict)
    env_prefixes: dict[str, int] = field(default_factory=dict)

    def add(self, tag: str, text: str, *, limit: int | None = None) -> None:
        if tag in self.muted:
            return
        if limit is not None:
            count = self.counts.get(tag, 0)
            if count >= limit:
                return
            self.counts[tag] = count + 1
        self.pending.append(
            (tag, f'{text} (Mute this hint with await rlm.hints.mute("{tag}").)')
        )

    def take(self) -> list[tuple[str, str]]:
        """Drain pending hints, excluding tags muted since they were queued."""
        hints = [(tag, text) for tag, text in self.pending if tag not in self.muted]
        self.pending.clear()
        return hints

    def observe_env_prefix(self, name: str) -> int:
        count = self.env_prefixes.get(name, 0) + 1
        self.env_prefixes[name] = count
        return count
