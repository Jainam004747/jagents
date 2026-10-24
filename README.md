# jagents

AI agents and the learning algorithms behind them, built from scratch.

[![ci](https://github.com/Jainam004747/jagents/actions/workflows/ci.yml/badge.svg)](https://github.com/Jainam004747/jagents/actions/workflows/ci.yml)

A tool-using AI agent and the machine-learning pieces underneath it, written in plain Python + NumPy
with **no ML frameworks**: automatic differentiation, a neural network, a small GPT-style transformer,
word embeddings, a hand-written agent loop with real tools, and reinforcement learning that makes the
agent improve from rewards.

The agent's tools are my other projects: [jsearch](https://github.com/Jainam004747/jsearch) (document
search), [jdb](https://github.com/Jainam004747/jdb) (SQL) and [jredis](https://github.com/Jainam004747/jredis)
(key-value store). Each has a built-in fallback so this repo runs on its own.

## Results (all reproducible, see `examples/`)

| What | Before | After |
|---|---|---|
| MLP on a 3-class spiral dataset | 33% (chance) | **99.3%** accuracy |
| Q-learning in a grid world | 19% reach goal (random) | **100%** (shortest safe path) |
| Agent tool router, held-out tasks | 25% correct | **100%** after 400 graded tasks |
| Bandits, regret after 5000 pulls | ε-greedy 146 | **UCB1 63** |
| Char transformer (300 steps, CPU) | loss 4.95 | loss 2.06 |

## Quick start

```bash
git clone https://github.com/Jainam004747/jagents && cd jagents
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"        # installs numpy + pytest, and makes `jagents` importable
pytest -q                      # 18 tests: gradient checks, agent safety, learning works

python -m jagents ask --offline -v "How many calories are in a croissant?"
python -m jagents ask --offline "What is (12 + 8) * 3?"
python -m jagents eval --offline          # success rate over 150 generated tasks
```

With a Claude API key the same agent loop uses a real language model:

```bash
export ANTHROPIC_API_KEY=...  JAGENTS_MODEL=<a model id from docs.anthropic.com>
python -m jagents ask -v "Which plant does tea come from, and what does a latte cost?"
```

Connect the other projects instead of the fallbacks:

```bash
export JSEARCH_URL=http://localhost:8080   # jsearch web server
export JDB_BIN=../jdb/jdb                  # jdb binary
export JREDIS_PORT=6380                    # jredis server
```

## Output

The agent picking a tool and answering:

```text
$ python -m jagents ask --offline -v "How many calories are in a croissant?"
(offline policy with learned router)
  -> sql({'query': "SELECT calories FROM menu WHERE name = 'croissant'"})
     'calories\n300'
300

$ python -m jagents ask --offline -v "What is (12 + 8) * 3?"
(offline policy with learned router)
  -> calculator({'expression': '(12 + 8) * 3'})
     '60'
60

$ python -m jagents eval --offline
success 100%  by family {'menu': 1.0, 'math': 1.0, 'fact': 1.0}  avg tool calls 1.00  tool errors 0
```

The router learning from rewards:

```text
$ python examples/learn_tool_routing.py
held-out success before training: 25%
success per block of 50 training tasks (with 10% exploration): ['58%', '86%', '94%', '96%', '92%', '88%', '90%', '100%']
held-out success after training:  100%  {'menu': 1.0, 'fact': 1.0, 'math': 1.0}
```

Q-learning in a grid world (S start, G goal, X pit, # wall). The arrows are the learned policy:

```text
$ python examples/gridworld.py
random policy reaches the goal: 19%
average return, first 50 episodes: -0.28  last 50: +0.93
learned policy reaches the goal: 100%
↓→↓←#
↓#↓←←
↓#↓X↓
→↓↓#↓
#→→→G
```

Bandits and the small neural network:

```text
$ python examples/bandits.py
epsilon-greedy (0.1)   regret after 100:    7.9   1000:   31.5   5000:  146.5
UCB1                   regret after 100:   11.9   1000:   50.2   5000:   63.4

$ python examples/train_mlp.py
...
final accuracy 99.33% with 4547 parameters
```

## What is inside

```
jagents/nn/scalar.py      autograd on single numbers (the chain rule, visibly)
jagents/nn/tensor.py      the same for NumPy arrays: matmul, softmax, fused cross-entropy, broadcasting
jagents/nn/layers.py      Linear, MLP, Embedding, LayerNorm, causal self-attention, TinyTransformer
jagents/nn/optim.py       SGD and Adam
jagents/text/             TF-IDF + cosine similarity, word2vec (skip-gram, negative sampling)
jagents/agent/tools.py    tool registry, safe calculator (AST, never eval), sandboxed file reads,
                          SELECT-only SQL, side-effecting tools flagged
jagents/agent/llm.py      Claude over plain HTTPS (Messages API, tool use) + an offline policy
jagents/agent/agent.py    the agent loop: model -> tool calls -> results -> model, with a step limit
                          and user confirmation for tools that change data
jagents/agent/memory.py   episodic memory: past tasks retrieved by similarity into the prompt
jagents/learning/         bandits, Q-learning, a contextual-bandit tool router, task suite + grader
examples/                 one runnable script per idea
```

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the parts fit, [docs/DECISIONS.md](docs/DECISIONS.md)
for why, and [docs/DEV-LOG.md](docs/DEV-LOG.md) for the commands used to build and verify it.

## Honest scope

The offline policy is **not** a language model: it is a transparent scripted policy whose only learned
part is the tool router. It exists so the agent loop, safety checks and learning-from-rewards can be
tested deterministically without an API key. The transformer is tiny and trains on ~13 KB of text on a
CPU, so it learns spelling-level patterns, not knowledge.
