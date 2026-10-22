"""A tool router that LEARNS from feedback: a contextual bandit over task features.

Problem: given a task description, which tool should the agent try? Nobody hand-writes rules.
The router sees features of the task (its words, hashed into a fixed-size vector), picks a tool,
the agent runs it, and the environment says whether the final answer was right (reward 1/0).
The router then updates only the chosen tool's weights.

This is a contextual bandit with one logistic-regression model per tool:
    p(success | tool, task) = σ(w_tool · x_task)
Choice: the tool with the highest predicted success (plus an exploration bonus while learning).
Update: one step of gradient ascent on the log-likelihood of the observed reward.

This is genuine learning: the weights change with experience, are saved to disk, and the
measured success rate rises (see tests/test_learning.py and examples/learn_tool_routing.py).
"""

from __future__ import annotations

import json
import math
import random
import re
import zlib
from pathlib import Path

DIM = 512


def features(text: str) -> dict[int, float]:
    """Feature hashing: each word and each word pair maps to one of DIM slots (a sparse vector).
    The vector size stays fixed however many different words appear."""
    words = re.findall(r"[a-z]+|\d+|[-+*/^%]", text.lower())
    words = ["<num>" if w.isdigit() else w for w in words]
    feats: dict[int, float] = {0: 1.0}  # slot 0 is a bias term
    grams = words + [a + "_" + b for a, b in zip(words, words[1:])]
    for g in grams:
        slot = 1 + zlib.crc32(g.encode()) % (DIM - 1)
        feats[slot] = feats.get(slot, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in feats.values()))
    return {k: v / norm for k, v in feats.items()}


class ToolRouter:
    def __init__(self, tools: list[str], lr: float = 0.5, epsilon: float = 0.1, seed: int = 0):
        self.tools = list(tools)
        self.weights = {t: [0.0] * DIM for t in self.tools}
        self.counts = {t: 0 for t in self.tools}
        self.lr, self.epsilon = lr, epsilon
        self.rng = random.Random(seed)

    def predict(self, text: str) -> dict[str, float]:
        x = features(text)
        return {t: 1 / (1 + math.exp(-sum(self.weights[t][i] * v for i, v in x.items()))) for t in self.tools}

    def choose(self, text: str, explore: bool = False) -> str:
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(self.tools)  # try something else now and then
        p = self.predict(text)
        best = max(p.values())
        return self.rng.choice([t for t, v in p.items() if abs(v - best) < 1e-12])  # random among ties

    def update(self, text: str, tool: str, reward: float) -> None:
        x = features(text)
        w = self.weights[tool]
        pred = 1 / (1 + math.exp(-sum(w[i] * v for i, v in x.items())))
        for i, v in x.items():
            w[i] += self.lr * (reward - pred) * v  # gradient of the log-likelihood
        self.counts[tool] += 1

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"tools": self.tools, "weights": self.weights, "counts": self.counts}))

    @classmethod
    def load(cls, path: Path) -> "ToolRouter":
        d = json.loads(Path(path).read_text())
        r = cls(d["tools"])
        r.weights, r.counts = d["weights"], d["counts"]
        return r
