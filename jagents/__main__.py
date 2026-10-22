"""Command line: python -m jagents ask "How many calories are in a muffin?" [--offline] [-v]
                 python -m jagents train-router
                 python -m jagents eval [--offline]"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .agent.agent import Agent
from .agent.llm import AnthropicLLM, OfflinePolicy
from .agent.memory import EpisodicMemory
from .agent.tools import DATA, default_tools
from .learning.router import ToolRouter
from .learning.tasks import evaluate, generate

ROUTER = DATA / "router.json"
TOOLS = ["calculator", "sql", "search", "read_file"]


def make_agent(offline: bool, verbose: bool) -> Agent:
    memory = EpisodicMemory(DATA / "memory.jsonl")
    if offline or not os.environ.get("ANTHROPIC_API_KEY"):
        router = ToolRouter.load(ROUTER) if ROUTER.exists() else ToolRouter(TOOLS)
        llm = OfflinePolicy(router)
        if verbose:
            print("(offline policy" + (" with learned router)" if ROUTER.exists() else ", untrained router)"))
    else:
        llm = AnthropicLLM()

    def confirm(name, args):
        return input(f"Allow {name}({args})? [y/N] ").strip().lower() == "y"

    return Agent(llm, default_tools(), memory=memory, confirm=confirm, verbose=verbose)


def main() -> None:
    ap = argparse.ArgumentParser(prog="jagents")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--offline", action="store_true")
    ask.add_argument("-v", "--verbose", action="store_true")
    sub.add_parser("train-router")
    ev = sub.add_parser("eval")
    ev.add_argument("--offline", action="store_true")
    ev.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    if args.cmd == "ask":
        agent = make_agent(args.offline, args.verbose)
        traj = agent.run(args.question)
        print(traj.answer)
        agent.remember(traj, None)
    elif args.cmd == "train-router":
        import runpy
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples" / "learn_tool_routing.py"))
    elif args.cmd == "eval":
        agent = make_agent(args.offline, False)
        agent.memory = None
        r = evaluate(agent, generate(args.n, seed=1234))
        print(f"success {r.success_rate:.0%}  by family {r.by_family}  avg tool calls {r.avg_steps:.2f}  "
              f"tool errors {r.tool_errors}")


if __name__ == "__main__":
    main()
