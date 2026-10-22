"""Task suite and evaluation harness for the agent.

Every task has a checkable answer, so an episode can be scored automatically. That score is the
REWARD that trains the tool router, and the success rate is how agents are compared.

Three task families, each best served by a different tool:
  arithmetic         "What is 17 * 23 + 4?"                 -> calculator
  menu questions     "How many calories are in a muffin?"   -> sql
  document facts     "Which plant does tea come from?"      -> search
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass

from ..agent.tools import DATA

FACTS = [
    ("Which plant does all true tea come from?", "camellia sinensis"),
    ("What pressure is espresso brewed at?", "9 bars"),
    ("How long does cold brew coffee steep?", "12 to 24 hours"),
    ("Where is Greater Sudbury located?", "northern ontario"),
    ("What packets make up the TCP three-way handshake?", "syn-ack"),
    ("What happens to bagels before they are baked?", "boiled"),
    ("What kind of system is Git?", "version control"),
    ("What does a compiler's lexer split text into?", "tokens"),
    ("What kind of protocol is HTTP?", "request-response"),
    ("At what temperature does first crack happen when roasting coffee?", "196"),
    ("What is Redis commonly used as?", "cache"),
    ("What does garbage collection free automatically?", "memory"),
]

MENU_TEMPLATES = [
    ("How much does a {item} cost in cents?", "price_cents"),
    ("What is the price of a {item}?", "price_cents"),
    ("How many calories are in a {item}?", "calories"),
    ("Tell me the calorie count of one {item}.", "calories"),
]

MATH_TEMPLATES = [
    "What is {a} * {b} + {c}?",
    "Compute ({a} + {b}) * {c}.",
    "Please calculate {a} - {b} * {c}",
    "If I buy {a} items that cost {b} cents each, what is {a} * {b}?",
]


@dataclass
class Task:
    text: str
    expected: str
    family: str  # for analysis only; the agent never sees it

    def check(self, answer: str) -> bool:
        return self.expected.lower() in answer.lower()


def generate(n: int, seed: int = 0) -> list[Task]:
    rng = random.Random(seed)
    db = sqlite3.connect(":memory:")
    db.executescript((DATA / "menu.sql").read_text())
    items = [r[0] for r in db.execute("SELECT name FROM menu")]
    from ..agent.tools import calculate

    tasks = []
    for _ in range(n):
        family = rng.choice(["math", "menu", "fact"])
        if family == "math":
            t = rng.choice(MATH_TEMPLATES).format(a=rng.randint(2, 60), b=rng.randint(2, 60), c=rng.randint(2, 9))
            from ..agent.llm import extract_math
            tasks.append(Task(t, calculate(extract_math(t)), family))
        elif family == "menu":
            template, column = rng.choice(MENU_TEMPLATES)
            item = rng.choice(items)
            value = db.execute(f"SELECT {column} FROM menu WHERE name = ?", (item,)).fetchone()[0]
            tasks.append(Task(template.format(item=item), str(value), family))
        else:
            q, a = rng.choice(FACTS)
            tasks.append(Task(q, a, family))
    return tasks


@dataclass
class Report:
    success_rate: float
    by_family: dict[str, float]
    avg_steps: float
    tool_errors: int


def evaluate(agent, tasks: list[Task]) -> Report:
    ok, steps, errors = 0, 0, 0
    fam_total: dict[str, int] = {}
    fam_ok: dict[str, int] = {}
    for t in tasks:
        traj = agent.run(t.text)
        good = t.check(traj.answer)
        ok += good
        steps += len(traj.steps)
        errors += sum(s.is_error for s in traj.steps)
        fam_total[t.family] = fam_total.get(t.family, 0) + 1
        fam_ok[t.family] = fam_ok.get(t.family, 0) + good
    return Report(ok / len(tasks), {f: fam_ok[f] / fam_total[f] for f in fam_total}, steps / len(tasks), errors)
