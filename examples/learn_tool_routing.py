"""The agent learns which tool to use from rewards alone. Run: python examples/learn_tool_routing.py

Before training, the router has no idea (random choice). After 400 graded tasks it picks the
right tool for unseen tasks. The learned weights are saved to data/router.json and used by
`python -m jagents ask --offline`.
"""
from pathlib import Path

from jagents.agent.agent import Agent
from jagents.agent.llm import OfflinePolicy
from jagents.agent.memory import EpisodicMemory
from jagents.agent.tools import default_tools
from jagents.learning.feedback import train_router
from jagents.learning.router import ToolRouter
from jagents.learning.tasks import evaluate, generate

router = ToolRouter(["calculator", "sql", "search", "read_file"], seed=1)
policy = OfflinePolicy(router)
agent = Agent(policy, default_tools(), memory=EpisodicMemory())
held_out = generate(150, seed=99)

before = evaluate(agent, held_out)
log = train_router(agent, policy, router, generate(400, seed=1))
after = evaluate(agent, held_out)
print(f"held-out success before training: {before.success_rate:.0%}  {before.by_family}")
print("success per block of 50 training tasks (with 10% exploration):", [f"{x:.0%}" for x in log.curve(50)])
print(f"held-out success after training:  {after.success_rate:.0%}  {after.by_family}")
path = Path(__file__).parent.parent / "data" / "router.json"
router.save(path)
print(f"saved learned weights to {path}")
