from jagents.agent.agent import Agent
from jagents.agent.llm import LLMResponse, OfflinePolicy, ToolCall, extract_math
from jagents.agent.memory import Episode, EpisodicMemory
from jagents.agent.tools import SqlDatabase, ToolRegistry, calculate, default_tools, read_file, run_tool_safely
from jagents.learning.router import ToolRouter

import pytest


def test_calculator_is_safe():
    assert calculate("17 * 23 + 4") == "395"
    assert calculate("(2 + 3) * 4") == "20"
    with pytest.raises(Exception):
        calculate("__import__('os').system('echo hacked')")


def test_extract_math():
    assert extract_math("What is 17 * 23 + 4?") == "17 * 23 + 4"
    assert extract_math("Compute (3 + 4) * 5.") == "(3 + 4) * 5"
    assert extract_math("no numbers here") is None


def test_read_file_is_sandboxed():
    assert "tea" in read_file("tea.md").lower()
    with pytest.raises(PermissionError):
        read_file("../../../etc/passwd")


def test_sql_rejects_writes():
    db = SqlDatabase()
    assert "410" in db("SELECT calories FROM menu WHERE name = 'muffin'")
    for bad in ["DELETE FROM menu", "SELECT 1; DROP TABLE menu"]:
        with pytest.raises(PermissionError):
            db(bad)
    # through the registry the refusal becomes an error message for the model
    out, err = default_tools().run("sql", {"query": "DROP TABLE menu"})
    assert err and "SELECT" in out


def test_registry_errors_are_returned_not_raised():
    reg = default_tools()
    out, err = reg.run("nope", {})
    assert err and "unknown tool" in out
    out, err = reg.run("calculator", {})
    assert err and "missing" in out
    out, err = run_tool_safely(reg, "calculator", "not a dict")
    assert err


class Scripted:
    """A fake LLM that replays a fixed list of responses."""
    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, system, messages, tools):
        return self.responses.pop(0) if self.responses else LLMResponse("done")


def call(name, args):
    return LLMResponse("", [ToolCall("c", name, args)], [{"type": "tool_use", "id": "c", "name": name, "input": args}])


def test_step_limit_stops_a_looping_model():
    looping = Scripted([call("calculator", {"expression": "1+1"})] * 50)
    traj = Agent(looping, default_tools(), max_steps=3).run("loop forever")
    assert traj.stopped_by_limit and len(traj.steps) == 3


def test_side_effects_need_confirmation():
    llm = Scripted([call("kv_set", {"key": "a", "value": "1"}), LLMResponse("ok")])
    traj = Agent(llm, default_tools()).run("store a=1")
    assert traj.steps[0].is_error and "not approved" in traj.steps[0].output

    llm = Scripted([call("kv_set", {"key": "a", "value": "1"}), LLMResponse("ok")])
    traj = Agent(llm, default_tools(), confirm=lambda n, a: True).run("store a=1")
    assert not traj.steps[0].is_error


def test_offline_agent_answers_with_a_trained_router(tmp_path):
    router = ToolRouter(["calculator", "sql", "search", "read_file"], seed=0)
    for _ in range(30):
        router.update("What is 3 * 4?", "calculator", 1.0)
        router.update("What is 3 * 4?", "search", 0.0)
    agent = Agent(OfflinePolicy(router), default_tools())
    traj = agent.run("What is 6 * 7?")
    assert traj.tools_used == ["calculator"] and "42" in traj.answer


def test_memory_retrieves_similar_episodes(tmp_path):
    mem = EpisodicMemory(tmp_path / "m.jsonl")
    mem.add(Episode("How many calories are in a muffin?", "350", ["sql"], True, ""))
    mem.add(Episode("Which plant does tea come from?", "camellia", ["search"], True, ""))
    hits = mem.similar("How many calories in a bagel?")
    assert hits and hits[0].tools_used == ["sql"]
    assert len(EpisodicMemory(tmp_path / "m.jsonl").similar("calories muffin")) >= 1  # persisted
