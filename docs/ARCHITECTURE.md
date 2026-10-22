# Architecture

```
            task ("How many calories are in a muffin?")
              │
              ▼
      ┌──────────────┐  similar past episodes   ┌─────────────────┐
      │    Agent     │◄─────────────────────────│ EpisodicMemory  │
      │  run(task)   │──── remember(result) ───►│ JSONL + TF-IDF  │
      └──────┬───────┘                          └─────────────────┘
   messages  │ ▲ text / tool calls
             ▼ │
      ┌──────────────┐        ┌───────────────────────────────┐
      │     LLM      │        │ OfflinePolicy                 │
      │ AnthropicLLM │   or   │   ToolRouter (learned) picks  │
      └──────────────┘        │   the tool, rules fill args   │
             │                └───────────────────────────────┘
             ▼ tool calls
      ┌──────────────────────────────────────────────────────┐
      │ ToolRegistry: calculator · search · sql · read_file │
      │               kv_get · kv_set (side effect)          │
      └──────────────────────────────────────────────────────┘
          │ jsearch HTTP      │ jdb binary      │ jredis RESP     (fallbacks: TF-IDF, sqlite3, dict)
```

## The agent loop (agent.py)

1. Build the first user message: the task, plus "Relevant past experience" from memory if any.
2. Ask the model for the next turn, passing the JSON schemas of all tools.
3. No tool calls in the reply → that text is the answer. Stop.
4. Otherwise run each call. A tool marked `side_effects` runs only if `confirm(name, args)` says yes.
   Errors (unknown tool, missing argument, exception) come back as text with `is_error=True`, so the
   model can recover instead of the program crashing.
5. Append the results as a `tool_result` message and go to 2, at most `max_steps` times.

This is exactly the message shape the Anthropic Messages API uses, so the offline policy and Claude
are interchangeable behind the `LLM` protocol.

## Learning from rewards (learning/)

- `tasks.py` generates tasks with known answers in three families (math, menu, facts). A task's
  `check(answer)` is the reward.
- `router.py` is a contextual bandit: features are hashed words and word pairs of the task (512
  dims), one logistic model per tool predicts "this tool will succeed". `choose` takes the argmax (or
  explores with probability ε), `update` does one SGD step on the observed reward.
- `feedback.py` runs the loop: agent acts → grader scores → router updates → episode stored.
- `bandit.py` and `gridworld.py` are the textbook versions of the same idea: bandits have no state,
  the router has a state (the task text) but one decision, Q-learning has many decisions in a row.

## Neural networks (nn/)

`Tensor` stores `data`, `grad` and a closure `_backward` that adds its gradient to its parents.
`backward()` topologically sorts the graph and calls each closure in reverse. `unbroadcast` sums
gradients over dimensions that were broadcast in the forward pass. Every layer is built only from
Tensor operations, so the transformer's gradients come for free and are checked numerically in
`tests/test_nn.py`.

TinyTransformer = token embedding + position embedding → N × [LayerNorm → causal multi-head
attention → residual → LayerNorm → MLP → residual] → LayerNorm → linear to vocabulary. The causal
mask sets future positions to −1e9 before softmax; a test proves later tokens cannot affect
earlier outputs.
