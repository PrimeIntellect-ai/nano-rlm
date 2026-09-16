from rlm.runtime_hints import RuntimeHints


def test_budget_counts_queued_hints_but_not_muted_attempts():
    hints = RuntimeHints()
    hints.muted.add("run-detach")
    hints.add("run-detach", "muted", limit=2)
    assert hints.counts == {}
    hints.muted.clear()
    hints.add("run-detach", "first", limit=2)
    hints.add("run-detach", "second", limit=2)
    hints.add("run-detach", "over budget", limit=2)
    delivered = hints.take()
    assert [text.split(" (Mute")[0] for _, text in delivered] == ["first", "second"]
    assert hints.take() == []
    hints.add("run-detach", "still over budget", limit=2)
    assert hints.take() == []
    hints.add("wait-held-job", "independent budget", limit=2)
    assert [tag for tag, _ in hints.take()] == ["wait-held-job"]


def test_muting_before_delivery_drops_pending_hints_without_refunding_budget():
    hints = RuntimeHints()
    hints.add("run-detach", "queued", limit=1)
    hints.add("quote-nesting", "visible")
    hints.muted.add("run-detach")
    assert [tag for tag, _ in hints.take()] == ["quote-nesting"]
    hints.muted.clear()
    hints.add("run-detach", "over budget", limit=1)
    assert hints.take() == []
    other = RuntimeHints()
    other.add("run-detach", "separate agent", limit=1)
    assert len(other.take()) == 1
