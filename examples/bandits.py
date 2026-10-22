"""Exploration vs exploitation: epsilon-greedy vs UCB1 on a 3-armed bandit. Run: python examples/bandits.py"""
from jagents.learning.bandit import UCB1, EpsilonGreedy, run

probs = [0.20, 0.50, 0.75]
for name, agent in [("epsilon-greedy (0.1)", EpsilonGreedy(3, 0.1)), ("UCB1", UCB1(3))]:
    regret = run(agent, probs, 5000)
    print(f"{name:22s} regret after 100: {regret[99]:6.1f}   1000: {regret[999]:6.1f}   5000: {regret[-1]:6.1f}"
          f"   pulls per arm: {agent.counts.astype(int).tolist()}")
