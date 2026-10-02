"""Run-stable ids: the n-th id of a scope under one root is the same on every run, so a
replayed session sees the same identifiers it saw the first time."""

import hashlib
from collections import Counter


class Ids:
    def __init__(self, root: str):
        self.root = root
        self._counts: Counter[str] = Counter()

    def next(self, scope: str) -> str:
        n = self._counts[scope]
        self._counts[scope] += 1
        return hashlib.sha256(f"{self.root}/{scope}/{n}".encode()).hexdigest()[:32]
