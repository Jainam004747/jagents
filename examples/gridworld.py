"""Tabular Q-learning finds a safe shortest path. Run: python examples/gridworld.py"""
import numpy as np

from jagents.learning.gridworld import GridWorld, QLearner, evaluate, render_policy

env = GridWorld()
rng = np.random.default_rng(1)
print("random policy reaches the goal:", f"{evaluate(env, lambda s: int(rng.integers(4))):.0%}")
agent = QLearner(env.n_states)
returns = agent.train(env, 600)
print("average return, first 50 episodes:", f"{np.mean(returns[:50]):+.2f}", " last 50:", f"{np.mean(returns[-50:]):+.2f}")
print("learned policy reaches the goal:", f"{evaluate(env, lambda s: agent.act(s, greedy=True)):.0%}")
print(render_policy(env, agent.q))
