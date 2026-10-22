"""Episodic memory: the agent's record of past tasks, what it did, and how it went.

Each finished task is appended to a JSON-lines file. Before a new task, the most similar past
episodes (TF-IDF cosine similarity over task text) are retrieved and shown to the model as
"relevant past experience", including lessons from failures. This changes behaviour without
changing any weights. It is IN-CONTEXT learning (retrieval), which is different from the
trained router in learning/router.py, and the README is careful to call it that.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..text.tfidf import TfidfIndex


@dataclass
class Episode:
    task: str
    answer: str
    tools_used: list[str]
    success: bool | None = None
    lesson: str = ""
    timestamp: float = field(default_factory=time.time)


class EpisodicMemory:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self.episodes: list[Episode] = []
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self.episodes.append(Episode(**json.loads(line)))

    def add(self, ep: Episode) -> None:
        self.episodes.append(ep)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(asdict(ep)) + "\n")

    def similar(self, task: str, k: int = 3, min_score: float = 0.2) -> list[Episode]:
        if not self.episodes:
            return []
        index = TfidfIndex().fit([(str(i), e.task) for i, e in enumerate(self.episodes)])
        return [self.episodes[int(i)] for i, score in index.search(task, k) if score >= min_score]

    def as_prompt(self, task: str) -> str:
        eps = self.similar(task)
        if not eps:
            return ""
        lines = ["Relevant past experience:"]
        for e in eps:
            outcome = {True: "succeeded", False: "FAILED", None: "outcome unknown"}[e.success]
            lines.append(f"- Task: {e.task!r} -> used {e.tools_used or 'no tools'}, {outcome}."
                         + (f" Lesson: {e.lesson}" if e.lesson else ""))
        return "\n".join(lines)
