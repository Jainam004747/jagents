"""TF-IDF vectors and cosine similarity: the classic way to turn text into comparable vectors.

Each document becomes a sparse vector with one dimension per word:
    weight(word) = (1 + log tf) * log((1 + N) / (1 + df)) + 1-ish smoothing
so words that are frequent in THIS text but rare overall get high weight. Vectors are
normalised to length 1, so the cosine similarity of two texts is just their dot product.

Used by the agent's document search tool and by episodic memory ("have I seen a task like this?").
"""

from __future__ import annotations

import math
import re
from collections import Counter

STOP = set("""a an and are as at be but by for if in into is it no not of on or such that the their then
there these they this to was were will with i you he she we its our your from has have had been so than too
what which who how when where does do did can could would should""".split())

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in STOP]


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(t, 0.0) for t, w in a.items())


class TfidfIndex:
    def __init__(self):
        self.docs: list[tuple[str, dict]] = []  # (id, raw metadata)
        self.vectors: list[dict[str, float]] = []
        self.df: Counter = Counter()

    def fit(self, docs: list[tuple[str, str]]) -> "TfidfIndex":
        """docs: list of (id, text)."""
        self.docs = [(d, {"text": t}) for d, t in docs]
        tokenized = [tokenize(t) for _, t in docs]
        self.df = Counter()
        for toks in tokenized:
            self.df.update(set(toks))
        self.vectors = [self.vectorize_tokens(t) for t in tokenized]
        return self

    def vectorize_tokens(self, tokens: list[str]) -> dict[str, float]:
        n = max(1, len(self.docs))
        tf = Counter(tokens)
        vec = {w: (1 + math.log(c)) * math.log((1 + n) / (1 + self.df.get(w, 0)) + 1) for w, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {w: v / norm for w, v in vec.items()}

    def vectorize(self, text: str) -> dict[str, float]:
        return self.vectorize_tokens(tokenize(text))

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        q = self.vectorize(query)
        scored = [(self.docs[i][0], cosine(q, v)) for i, v in enumerate(self.vectors)]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda s: -s[1])
        return scored[:k]
