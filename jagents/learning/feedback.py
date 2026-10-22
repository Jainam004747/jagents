"""Learning from feedback: run the agent, score its answers, and train the tool router online.

    for each task:
        tool  = router.choose(task)               # with some exploration
        answer = agent.run(task)                   # real tools, real outputs
        reward = 1 if answer is correct else 0     # automatic grading
        router.update(task, tool, reward)          # the policy changes
        memory.add(episode, lesson if it failed)   # retrieval context for future tasks

The learning curve (success rate per block of tasks) is the evidence that the agent improves
from experience, rather than just running a fixed script.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..agent.agent import Agent
from ..agent.llm import OfflinePolicy
from .router import ToolRouter
from .tasks import Task


@dataclass
class TrainingLog:
    rewards: list[float]

    def curve(self, block: int = 50) -> list[float]:
        return [sum(self.rewards[i:i + block]) / len(self.rewards[i:i + block])
                for i in range(0, len(self.rewards), block)]


def train_router(agent: Agent, policy: OfflinePolicy, router: ToolRouter, tasks: list[Task]) -> TrainingLog:
    rewards = []
    policy.explore = True
    for t in tasks:
        traj = agent.run(t.text)
        reward = 1.0 if t.check(traj.answer) else 0.0
        if policy.last_choice is not None:
            router.update(t.text, policy.last_choice, reward)
        lesson = "" if reward else f"'{policy.last_choice}' did not produce the right answer for this kind of task"
        agent.remember(traj, bool(reward), lesson)
        rewards.append(reward)
    policy.explore = False
    return TrainingLog(rewards)
