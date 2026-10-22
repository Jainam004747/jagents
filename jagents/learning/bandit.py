"""Multi-armed bandits: the simplest "learn from rewards" problem.

There are K slot machines ("arms") with unknown payout probabilities. Each pull gives reward 1
or 0. The goal is to win as much as possible, which means balancing EXPLORATION (trying arms to
learn about them) against EXPLOITATION (pulling the arm that looks best so far).

REGRET measures the cost of learning: how much less was earned than by always pulling the best
arm. A good strategy's regret grows slower and slower (logarithmically) over time.
"""

from __future__ import annotations

import math

import numpy as np


class EpsilonGreedy:
    """With probability epsilon pull a random arm, otherwise the best average so far."""

    def __init__(self, k: int, epsilon: float = 0.1, seed: int = 0):
        self.counts = np.zeros(k)
        self.values = np.zeros(k)
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def select(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(self.counts)))
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]  # running mean


class UCB1:
    """Optimism in the face of uncertainty: pick the arm with the highest UPPER CONFIDENCE BOUND,
    mean + sqrt(2 ln t / n). Rarely tried arms get a big bonus, so they are explored, and the
    bonus shrinks as evidence accumulates. No random exploration parameter is needed."""

    def __init__(self, k: int):
        self.counts = np.zeros(k)
        self.values = np.zeros(k)
        self.t = 0

    def select(self) -> int:
        self.t += 1
        for arm, n in enumerate(self.counts):
            if n == 0:
                return arm  # try every arm once first
        bonus = np.sqrt(2 * math.log(self.t) / self.counts)
        return int(np.argmax(self.values + bonus))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


def run(agent, probabilities: list[float], steps: int, seed: int = 1) -> np.ndarray:
    """Plays `steps` rounds; returns cumulative regret after each round."""
    rng = np.random.default_rng(seed)
    best = max(probabilities)
    regret = np.zeros(steps)
    total = 0.0
    for t in range(steps):
        arm = agent.select()
        reward = float(rng.random() < probabilities[arm])
        agent.update(arm, reward)
        total += best - probabilities[arm]
        regret[t] = total
    return regret
