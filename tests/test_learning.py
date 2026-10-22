import numpy as np

from jagents.agent.agent import Agent
from jagents.agent.llm import OfflinePolicy
from jagents.agent.tools import default_tools
from jagents.learning.bandit import UCB1, EpsilonGreedy, run
from jagents.learning.feedback import train_router
from jagents.learning.gridworld import GridWorld, QLearner, evaluate as grid_eval
from jagents.learning.router import ToolRouter
from jagents.learning.tasks import evaluate, generate


def test_ucb_has_less_regret_than_random_play():
    probs = [0.2, 0.5, 0.7]
    ucb = run(UCB1(3), probs, 3000)[-1]
    random_play = run(EpsilonGreedy(3, epsilon=1.0), probs, 3000)[-1]
    assert ucb < random_play / 3


def test_q_learning_finds_the_goal():
    env = GridWorld()
    q = QLearner(env.n_states, seed=0)
    q.train(env, 600)
    assert grid_eval(env, lambda s: q.act(s, greedy=True), episodes=20) == 1.0


def test_router_learns_from_rewards():
    router = ToolRouter(["calculator", "sql", "search", "read_file"], seed=1)
    policy = OfflinePolicy(router)
    agent = Agent(policy, default_tools())
    held_out = generate(60, seed=99)
    before = evaluate(agent, held_out).success_rate
    train_router(agent, policy, router, generate(300, seed=1))
    after = evaluate(agent, held_out).success_rate
    assert after >= 0.9 and after > before + 0.3
