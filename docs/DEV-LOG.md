# Dev log — commands used and why

| Command | Why |
|---|---|
| `python -m venv .venv` | Isolated environment so project packages don't touch the system Python |
| `pip install -e ".[dev]"` | Editable install: code changes apply without reinstalling; `[dev]` adds pytest |
| `pytest -q` | Runs all tests; `-q` prints one dot per test |
| `python examples/autograd_demo.py` | Watch gradients computed by hand-written backprop |
| `python examples/train_mlp.py` | 3-class spiral classification, 99.3% accuracy |
| `python examples/train_char_transformer.py 300` | Train the tiny transformer for 300 steps and sample text |
| `python examples/learn_tool_routing.py` | Router learns from rewards (25% → 100%), saves `data/router.json` |
| `python examples/gridworld.py` | Q-learning: prints the learned policy as arrows |
| `python examples/bandits.py` | Regret of ε-greedy vs UCB1 |
| `python -m jagents ask --offline -v "..."` | Run the agent and print each tool call |
| `python -m jagents eval --offline` | Success rate on 150 generated tasks |

## Milestones

1. **Autograd.** Scalar `Value`, then `Tensor` with broadcasting. Verified with numerical gradients.
2. **Networks.** Linear/MLP + Adam; spirals 99.3%. Transformer: loss 4.95 → 2.06 in 300 CPU steps.
3. **Text.** TF-IDF search (fallback for jsearch) and word2vec with negative sampling.
4. **Agent.** Tool registry with schemas, safe tools, the loop with a step limit and confirmations,
   Claude via HTTPS, episodic memory.
5. **Learning agent.** Task generator + grader, contextual-bandit router, training loop. Held-out success
   25% → 100%.
6. **RL basics.** Bandits (UCB1 regret 63 vs 146) and Q-learning (19% → 100%).

## Bugs worth remembering

- `extract_math` missed expressions with parentheses like `(3 + 4) * 5` because the regex stopped at `(`.
  Rewrote it to take the longest balanced run of digits/operators/brackets that contains an operator.
- Examples failed with `ModuleNotFoundError: jagents` when run as scripts: the package must be installed
  (`pip install -e .`) or run with `PYTHONPATH=.`. Added `pyproject.toml`.
