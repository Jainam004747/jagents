"""Word embeddings learned from scratch: skip-gram with negative sampling (word2vec, 2013).

Idea: a word is known by the company it keeps. For every word in the text, the model learns to
tell its real neighbours (within `window` words) apart from random words. Words that appear in
similar contexts ("espresso" and "latte") end up with similar vectors.

For a centre word c and a context word o, with vectors v_c and u_o:
    real pair:   maximise  log σ(u_o · v_c)
    k fakes:     maximise  log σ(−u_n · v_c)   for random words n
Gradients of these two terms update both vector tables with plain SGD.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .tfidf import tokenize


class Word2Vec:
    def __init__(self, dim: int = 32, window: int = 3, negatives: int = 5, min_count: int = 2, seed: int = 0):
        self.dim, self.window, self.negatives, self.min_count = dim, window, negatives, min_count
        self.rng = np.random.default_rng(seed)
        self.vocab: dict[str, int] = {}
        self.words: list[str] = []

    def fit(self, texts: list[str], epochs: int = 5, lr: float = 0.05) -> list[float]:
        tokens = [tokenize(t) for t in texts]
        counts = Counter(w for toks in tokens for w in toks)
        self.words = [w for w, c in counts.most_common() if c >= self.min_count]
        self.vocab = {w: i for i, w in enumerate(self.words)}
        v = len(self.words)
        self.center = self.rng.normal(0, 0.1, (v, self.dim))
        self.context = np.zeros((v, self.dim))
        # Negative samples are drawn with probability proportional to count^0.75 (from the paper):
        # common words are sampled more, but not overwhelmingly.
        freq = np.array([counts[w] for w in self.words], dtype=float) ** 0.75
        self.neg_p = freq / freq.sum()
        ids = [[self.vocab[w] for w in toks if w in self.vocab] for toks in tokens]
        losses = []
        for _ in range(epochs):
            total, n = 0.0, 0
            for sent in ids:
                for pos, c in enumerate(sent):
                    lo, hi = max(0, pos - self.window), min(len(sent), pos + self.window + 1)
                    for o in sent[lo:pos] + sent[pos + 1:hi]:
                        total += self._train_pair(c, o, lr)
                        n += 1
            losses.append(total / max(1, n))
        return losses

    def _train_pair(self, c: int, o: int, lr: float) -> float:
        negs = self.rng.choice(len(self.words), size=self.negatives, p=self.neg_p)
        targets = np.concatenate(([o], negs))
        labels = np.zeros(len(targets))
        labels[0] = 1.0
        vc = self.center[c]
        u = self.context[targets]
        scores = 1 / (1 + np.exp(-(u @ vc)))  # σ(u · v)
        err = scores - labels                # gradient of the logistic loss
        self.context[targets] -= lr * np.outer(err, vc)
        self.center[c] -= lr * (err @ u)
        return float(-np.log(scores[0] + 1e-9) - np.log(1 - scores[1:] + 1e-9).sum())

    def vector(self, word: str) -> np.ndarray | None:
        i = self.vocab.get(word)
        return None if i is None else self.center[i]

    def most_similar(self, word: str, k: int = 5) -> list[tuple[str, float]]:
        v = self.vector(word)
        if v is None:
            return []
        m = self.center / (np.linalg.norm(self.center, axis=1, keepdims=True) + 1e-9)
        sims = m @ (v / (np.linalg.norm(v) + 1e-9))
        order = np.argsort(-sims)
        return [(self.words[i], float(sims[i])) for i in order if self.words[i] != word][:k]
