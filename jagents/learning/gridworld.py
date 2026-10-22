"""Reinforcement learning with tabular Q-learning in a small grid world.

    S . . . #
    . # . . .        S start   G goal (+1)   X pit (-1)   # wall
    . # . X .        every move costs -0.01, so shorter paths are better
    . . . # .
    # . . . G

The agent does not know the map. It learns Q(state, action): the expected total future reward
of taking `action` in `state` and acting well afterwards, from experience alone:

    Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))

The term in brackets is the TEMPORAL-DIFFERENCE error: how much better or worse things went than
expected. Rewards at the goal propagate backwards through the grid one update at a time, and
the greedy policy argmax_a Q(s, a) becomes a shortest safe path.
"""

from __future__ import annotations

import numpy as np

LAYOUT = [
    "S...#",
    ".#...",
    ".#.X.",
    "...#.",
    "#...G",
]
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
ARROWS = "↑↓←→"


class GridWorld:
    def __init__(self, layout: list[str] = LAYOUT, step_cost: float = -0.01, slip: float = 0.0, seed: int = 0):
        self.grid = layout
        self.rows, self.cols = len(layout), len(layout[0])
        self.start = next((r, c) for r in range(self.rows) for c in range(self.cols) if layout[r][c] == "S")
        self.step_cost = step_cost
        self.slip = slip  # chance that a random action happens instead (a stochastic world)
        self.rng = np.random.default_rng(seed)

    @property
    def n_states(self) -> int:
        return self.rows * self.cols

    def index(self, pos: tuple[int, int]) -> int:
        return pos[0] * self.cols + pos[1]

    def reset(self) -> int:
        self.pos = self.start
        return self.index(self.pos)

    def step(self, action: int) -> tuple[int, float, bool]:
        if self.slip and self.rng.random() < self.slip:
            action = int(self.rng.integers(4))
        dr, dc = ACTIONS[action]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        if 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != "#":
            self.pos = (r, c)  # walls and edges: stay in place
        cell = self.grid[self.pos[0]][self.pos[1]]
        if cell == "G":
            return self.index(self.pos), 1.0, True
        if cell == "X":
            return self.index(self.pos), -1.0, True
        return self.index(self.pos), self.step_cost, False


class QLearner:
    def __init__(self, n_states: int, alpha: float = 0.5, gamma: float = 0.95, epsilon: float = 1.0,
                 epsilon_min: float = 0.05, decay: float = 0.995, seed: int = 0):
        self.q = np.zeros((n_states, 4))
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_min, self.decay = epsilon, epsilon_min, decay
        self.rng = np.random.default_rng(seed)

    def act(self, s: int, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(4))  # explore
        best = np.flatnonzero(self.q[s] == self.q[s].max())
        return int(self.rng.choice(best))  # exploit (random among ties)

    def learn(self, s: int, a: int, r: float, s2: int, done: bool) -> None:
        target = r if done else r + self.gamma * self.q[s2].max()
        self.q[s, a] += self.alpha * (target - self.q[s, a])

    def train(self, env: GridWorld, episodes: int, max_steps: int = 100) -> list[float]:
        returns = []
        for _ in range(episodes):
            s, total = env.reset(), 0.0
            for _ in range(max_steps):
                a = self.act(s)
                s2, r, done = env.step(a)
                self.learn(s, a, r, s2, done)
                s, total = s2, total + r
                if done:
                    break
            self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)
            returns.append(total)
        return returns


def evaluate(env: GridWorld, policy, episodes: int = 100, max_steps: int = 100) -> float:
    """Fraction of episodes that reach the goal following `policy(state) -> action`."""
    wins = 0
    for _ in range(episodes):
        s = env.reset()
        for _ in range(max_steps):
            s, r, done = env.step(policy(s))
            if done:
                wins += r > 0
                break
    return wins / episodes


def render_policy(env: GridWorld, q: np.ndarray) -> str:
    rows = []
    for r in range(env.rows):
        line = ""
        for c in range(env.cols):
            ch = env.grid[r][c]
            line += ch if ch in "#GX" else ARROWS[int(np.argmax(q[env.index((r, c))]))]
        rows.append(line)
    return "\n".join(rows)
