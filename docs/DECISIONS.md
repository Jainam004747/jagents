# Design decisions

**NumPy only, no PyTorch.** The point is to show I understand what a framework does: the chain rule,
broadcasting gradients, why softmax and cross-entropy are fused. Gradient checks against finite
differences prove it is correct.

**Fused softmax + cross-entropy.** The gradient is just `(probs − one_hot) / N`. Chaining softmax, log and
indexing separately is slower and overflows for large logits; subtracting the row max first avoids that.

**Plain HTTPS to the Claude API instead of the SDK.** Shows exactly what goes over the wire: system prompt,
messages, tool schemas, `tool_use` and `tool_result` blocks. Model id comes from `JAGENTS_MODEL` so nothing
is hard-coded that will go stale.

**An offline policy behind the same interface.** Tests and CI must not need an API key or network, and must
be deterministic. Labelled clearly as not-an-LLM in the code and README.

**Tool errors are returned, not raised.** A real model makes bad calls. Returning `error: ...` lets it fix
the call on the next turn; raising would end the run.

**Safety in the tools, not in the prompt.** The calculator walks the AST (never `eval`), `read_file`
resolves paths and refuses anything outside `data/corpus`, SQL accepts one SELECT only, `kv_set` is marked
`side_effects` and needs confirmation, and `max_steps` bounds cost. A prompt can be ignored; code cannot.

**Contextual bandit for routing, not full RL.** Choosing a tool is one decision with an immediate reward,
which is exactly a contextual bandit. Logistic regression on hashed features is tiny, fast, explainable
(look at the weights) and learns in a few hundred tasks.

**Feature hashing.** No vocabulary to build or store; unseen words still map to a slot. Collisions at 512
dims barely matter for three task families.

**Episodic memory in the prompt, not in the weights.** Retrieval of similar past tasks (with lessons from
failures) helps an LLM immediately with no training, and is easy to inspect: it is a JSONL file.

**UCB1 next to ε-greedy.** Shows why exploration strategy matters: UCB's regret grows logarithmically,
ε-greedy keeps paying for random pulls forever.
