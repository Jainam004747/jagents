"""Tools: functions the agent may call, each described by a name, a description and a JSON schema.

The model never runs code itself. It asks for a tool by name with arguments; the agent loop
checks the request, runs the Python function, and sends the result back as text. That boundary
is where safety lives: only registered tools exist, arguments are validated, read-only tools
cannot write, and side-effecting tools can require human confirmation.

Tools that talk to the other projects:
  search  -> jsearch's HTTP API when JSEARCH_URL is set (else a local TF-IDF index over data/corpus)
  sql     -> the jdb binary when JDB_BIN is set (else Python's built-in sqlite3, same SQL)
  kv_*    -> jredis (or Redis) over RESP when JREDIS_PORT is set (else an in-memory dict)
"""

from __future__ import annotations

import ast
import json
import operator
import os
import socket
import sqlite3
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..text.tfidf import TfidfIndex

DATA = Path(__file__).resolve().parents[2] / "data"
MAX_OUTPUT = 2000


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema of the arguments
    fn: Callable[..., str]
    side_effects: bool = False

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.parameters}


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def run(self, name: str, args: dict) -> tuple[str, bool]:
        """Returns (output, is_error). Never raises: errors go back to the model as text."""
        tool = self.tools.get(name)
        if tool is None:
            return f"error: unknown tool '{name}'. Available: {', '.join(self.tools)}", True
        missing = [k for k in tool.parameters.get("required", []) if k not in args]
        if missing:
            return f"error: missing argument(s) {missing} for {name}", True
        try:
            out = str(tool.fn(**args))
        except Exception as e:  # a failing tool must not crash the agent
            return f"error: {type(e).__name__}: {e}", True
        return out[:MAX_OUTPUT], False


# ---- calculator ---------------------------------------------------------------------------

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}


def calculate(expression: str) -> str:
    """Evaluates arithmetic safely by walking the syntax tree. Never use eval() on model output:
    eval("__import__('os').system('rm -rf ~')") would run."""
    if len(expression) > 200:
        raise ValueError("expression too long")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        raise ValueError(f"unsupported syntax: {ast.dump(node)[:40]}")

    result = ev(ast.parse(expression.replace("^", "**"), mode="eval"))
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(round(result, 6) if isinstance(result, float) else result)


# ---- document search --------------------------------------------------------------------

class DocumentSearch:
    def __init__(self, url: str | None = None, corpus: Path = DATA / "corpus"):
        self.url = url
        self.local = None
        if not url:
            docs = [(p.stem, p.read_text(encoding="utf-8")) for p in sorted(corpus.glob("*.md"))]
            self.local = TfidfIndex().fit(docs)

    def __call__(self, query: str) -> str:
        if self.url:  # jsearch's JSON API
            with urllib.request.urlopen(f"{self.url}/api/search?q={urllib.parse.quote(query)}", timeout=10) as r:
                hits = json.load(r)["results"][:3]
            return "\n".join(f"- {h['title']}: {h['snippet'].replace('<b>', '').replace('</b>', '')}" for h in hits) \
                or "no results"
        hits = self.local.search(query, k=3)
        lines = []
        for doc_id, score in hits:
            text = dict(self.local.docs)[doc_id]["text"]
            body = " ".join(line for line in text.splitlines() if not line.startswith("#"))
            lines.append(f"- {doc_id} ({score:.2f}): {body[:400]}")
        return "\n".join(lines) or "no results"


# ---- SQL (read-only) ------------------------------------------------------------------------

class SqlDatabase:
    """Read-only SQL over the menu database. Uses jdb if JDB_BIN is set, else sqlite3."""

    def __init__(self, jdb_bin: str | None = None, seed: Path = DATA / "menu.sql"):
        self.jdb_bin = jdb_bin
        self.seed_sql = seed.read_text()
        if jdb_bin:
            self.path = str(DATA / "menu.jdb")
            if not Path(self.path).exists():
                subprocess.run([jdb_bin, self.path], input=self.seed_sql, text=True, check=True)
        else:
            self.conn = sqlite3.connect(":memory:")
            self.conn.executescript(self.seed_sql)

    def __call__(self, query: str) -> str:
        q = query.strip().rstrip(";")
        if not q.lower().startswith("select") or ";" in q:
            raise PermissionError("only a single SELECT statement is allowed")
        if self.jdb_bin:
            r = subprocess.run([self.jdb_bin, self.path, "-header", "-c", q + ";"], capture_output=True, text=True,
                               timeout=10)
            if r.returncode != 0:
                raise RuntimeError(r.stderr.strip())
            return r.stdout.strip() or "(no rows)"
        cur = self.conn.execute(q)
        names = [d[0] for d in cur.description]
        rows = cur.fetchmany(50)
        return "\n".join(["|".join(names)] + ["|".join("" if v is None else str(v) for v in r) for r in rows])


# ---- key-value store (jredis / Redis) ---------------------------------------------------------

class KeyValue:
    """GET/SET over the RESP protocol (jredis or Redis), or an in-memory dict when offline."""

    def __init__(self, port: int | None = None, host: str = "localhost"):
        self.addr = (host, port) if port else None
        self.mem: dict[str, str] = {}

    def _call(self, *args: str) -> str | None:
        payload = f"*{len(args)}\r\n" + "".join(f"${len(a.encode())}\r\n{a}\r\n" for a in args)
        with socket.create_connection(self.addr, timeout=5) as s:
            s.sendall(payload.encode())
            f = s.makefile("rb")
            line = f.readline().decode().rstrip("\r\n")
            if line.startswith("$"):
                n = int(line[1:])
                return None if n < 0 else f.read(n + 2)[:-2].decode()
            if line.startswith("-"):
                raise RuntimeError(line[1:])
            return line[1:]

    def get(self, key: str) -> str:
        v = self._call("GET", key) if self.addr else self.mem.get(key)
        return "(nil)" if v is None else v

    def set(self, key: str, value: str) -> str:
        if self.addr:
            return self._call("SET", key, value) or "OK"
        self.mem[key] = value
        return "OK"


def read_file(name: str) -> str:
    """Reads a file from data/corpus only. Paths like ../../etc/passwd are refused."""
    base = (DATA / "corpus").resolve()
    target = (base / name).resolve()
    if base not in target.parents:
        raise PermissionError("access outside data/corpus is not allowed")
    return target.read_text(encoding="utf-8")[:MAX_OUTPUT]


def default_tools() -> ToolRegistry:
    reg = ToolRegistry()
    search = DocumentSearch(os.environ.get("JSEARCH_URL"))
    sql = SqlDatabase(os.environ.get("JDB_BIN"))
    kv = KeyValue(int(os.environ["JREDIS_PORT"]) if os.environ.get("JREDIS_PORT") else None)
    reg.register(Tool("calculator", "Evaluate an arithmetic expression such as '17 * 23 + 4'.",
                      {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
                      calculate))
    reg.register(Tool("search", "Search the document collection (coffee, food, computing, Sudbury). Returns top passages.",
                      {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}, search))
    reg.register(Tool("sql", "Run ONE read-only SELECT on the cafe database. Table menu(id, name, price_cents, "
                      "category, calories).",
                      {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}, sql))
    reg.register(Tool("kv_get", "Read a value from the key-value store.",
                      {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}, kv.get))
    reg.register(Tool("kv_set", "Store a value in the key-value store (changes data).",
                      {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                       "required": ["key", "value"]}, kv.set, side_effects=True))
    reg.register(Tool("read_file", "Read one document from the collection by file name, e.g. 'tea.md'.",
                      {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}, read_file))
    return reg


def run_tool_safely(registry: ToolRegistry, name: str, args: Any) -> tuple[str, bool]:
    if not isinstance(args, dict):
        return "error: arguments must be a JSON object", True
    return registry.run(name, args)
