"""Language-model back ends for the agent.

AnthropicLLM calls Claude through the Messages API with tool definitions. The model decides
which tool to call; the agent loop (agent.py) executes it.

OfflinePolicy needs no API key or network. It is NOT a language model: it is a transparent,
scripted policy whose tool choice comes from a learned router (learning/router.py), so the whole
agent loop, the safety checks, and learning from rewards can be run and tested deterministically.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from ..learning.router import ToolRouter


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_content: list[dict] = field(default_factory=list)  # blocks to append to the conversation


class LLM(Protocol):
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class AnthropicLLM:
    """Claude via the Messages API (plain HTTPS, no SDK, to show what goes over the wire)."""

    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str | None = None, api_key: str | None = None, max_tokens: int = 1024):
        self.model = model or os.environ.get("JAGENTS_MODEL")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.model or not self.api_key:
            raise RuntimeError("set ANTHROPIC_API_KEY and JAGENTS_MODEL (a Claude model id from "
                               "https://docs.anthropic.com/en/docs/about-claude/models), or use --offline")
        self.max_tokens = max_tokens

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        body = json.dumps({"model": self.model, "max_tokens": self.max_tokens, "system": system,
                           "messages": messages, "tools": tools}).encode()
        req = urllib.request.Request(self.URL, data=body, headers={
            "x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        text = "".join(b.get("text", "") for b in data["content"] if b["type"] == "text")
        calls = [ToolCall(b["id"], b["name"], b["input"]) for b in data["content"] if b["type"] == "tool_use"]
        return LLMResponse(text=text, tool_calls=calls, raw_content=data["content"])


MENU_ITEMS = ["hot chocolate", "cappuccino", "croissant", "espresso", "timbits", "coffee", "latte", "donut",
              "bagel", "muffin", "cookie", "tea"]


def extract_math(task: str) -> str | None:
    """Finds the longest run of digits, operators and parentheses that contains an operator."""
    best = None
    for cand in re.findall(r"[\d\s+\-*/^%().]+", task):
        cand = cand.strip().rstrip(".?").strip()
        if re.search(r"\d\s*[-+*/^%]\s*\(?\s*\d", cand) and cand.count("(") == cand.count(")"):
            if best is None or len(cand) > len(best):
                best = cand
    return best


class OfflinePolicy:
    """A scripted stand-in for an LLM.

    Turn 1: ask the router which tool fits the task, build arguments with simple rules, and call it.
    Turn 2: answer with the tool's output (first line of a SQL result, first passage of a search).
    The quality of this agent therefore depends entirely on the router, which LEARNS from rewards.
    """

    def __init__(self, router: ToolRouter, explore: bool = False):
        self.router = router
        self.explore = explore
        self.last_choice: str | None = None

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        task = messages[0]["content"] if isinstance(messages[0]["content"], str) else messages[0]["content"][0]["text"]
        task = task.split("\n\nRelevant past experience:")[0]
        results = [b for m in messages if m["role"] == "user" and isinstance(m["content"], list)
                   for b in m["content"] if b.get("type") == "tool_result"]
        if not results:
            tool = self.router.choose(task, explore=self.explore)
            self.last_choice = tool
            args = self._arguments(tool, task)
            block = {"type": "tool_use", "id": "call_1", "name": tool, "input": args}
            return LLMResponse(text="", tool_calls=[ToolCall("call_1", tool, args)], raw_content=[block])
        output = str(results[-1]["content"])
        answer = self._answer(output)
        return LLMResponse(text=answer, raw_content=[{"type": "text", "text": answer}])

    @staticmethod
    def _arguments(tool: str, task: str) -> dict:
        if tool == "calculator":
            return {"expression": extract_math(task) or "0"}
        if tool == "sql":
            item = next((i for i in MENU_ITEMS if i in task.lower()), "coffee")
            column = "calories" if "calorie" in task.lower() else "price_cents"
            return {"query": f"SELECT {column} FROM menu WHERE name = '{item}'"}
        return {"query": task}

    @staticmethod
    def _answer(output: str) -> str:
        if output.startswith("error"):
            return f"I could not complete the task ({output})."
        lines = [ln for ln in output.splitlines() if ln.strip()]
        if len(lines) >= 2 and "|" not in lines[1] and not lines[0].startswith("- "):
            return lines[1]  # SQL result: header line, then the value
        return lines[0] if lines else output
