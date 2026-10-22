"""The agent loop, written by hand with no framework.

    task -> model -> wants a tool? -> run it, send the result back -> model -> ... -> final answer

An "agent" is just this loop around a model plus tools. Everything that makes it trustworthy
lives in the loop, not in the model:
  * a step limit (no infinite loops, bounded cost)
  * only registered tools can run; arguments are checked
  * tools with side effects need explicit confirmation (a human-in-the-loop hook)
  * every step is recorded in a trajectory, for debugging and evaluation
  * tool errors are returned to the model as text, so it can recover, instead of crashing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .llm import LLM
from .memory import Episode, EpisodicMemory
from .tools import ToolRegistry, run_tool_safely

SYSTEM_PROMPT = """You are a careful assistant for a small cafe and a library of documents.
Use tools when they help: calculator for arithmetic, sql for menu prices (in cents) and calories,
search for facts in the documents. Prefer one precise tool call over guessing.
When you have the answer, reply with just the answer and one short sentence of explanation."""


@dataclass
class Step:
    tool: str
    arguments: dict
    output: str
    is_error: bool


@dataclass
class Trajectory:
    task: str
    steps: list[Step] = field(default_factory=list)
    answer: str = ""
    stopped_by_limit: bool = False

    @property
    def tools_used(self) -> list[str]:
        return [s.tool for s in self.steps]


class Agent:
    def __init__(self, llm: LLM, tools: ToolRegistry, memory: EpisodicMemory | None = None, max_steps: int = 6,
                 confirm: Callable[[str, dict], bool] | None = None, verbose: bool = False,
                 system_prompt: str = SYSTEM_PROMPT):
        self.llm, self.tools, self.memory = llm, tools, memory
        self.max_steps = max_steps
        self.confirm = confirm or (lambda name, args: False)  # side effects are denied unless approved
        self.verbose = verbose
        self.system_prompt = system_prompt

    def run(self, task: str) -> Trajectory:
        traj = Trajectory(task)
        content = task
        if self.memory:
            past = self.memory.as_prompt(task)
            if past:
                content = f"{task}\n\n{past}"
        messages: list[dict] = [{"role": "user", "content": content}]

        for _ in range(self.max_steps):
            resp = self.llm.complete(self.system_prompt, messages, self.tools.schemas())
            messages.append({"role": "assistant", "content": resp.raw_content or [{"type": "text", "text": resp.text}]})
            if not resp.tool_calls:
                traj.answer = resp.text.strip()
                break
            results = []
            for call in resp.tool_calls:
                tool = self.tools.tools.get(call.name)
                if tool is not None and tool.side_effects and not self.confirm(call.name, call.arguments):
                    output, err = f"error: '{call.name}' changes data and was not approved by the user", True
                else:
                    output, err = run_tool_safely(self.tools, call.name, call.arguments)
                traj.steps.append(Step(call.name, call.arguments, output, err))
                if self.verbose:
                    print(f"  -> {call.name}({call.arguments})\n     {output[:200]!r}")
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": output, "is_error": err})
            messages.append({"role": "user", "content": results})
        else:
            traj.stopped_by_limit = True
            traj.answer = "Stopped: reached the step limit before finishing."
        return traj

    def remember(self, traj: Trajectory, success: bool | None, lesson: str = "") -> None:
        if self.memory:
            self.memory.add(Episode(traj.task, traj.answer, traj.tools_used, success, lesson))
