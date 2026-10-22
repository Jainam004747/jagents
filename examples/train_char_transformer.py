"""Train a tiny GPT-style transformer, written from scratch, on the document corpus.

Run: python examples/train_char_transformer.py [steps]
It learns to predict the next character, then writes new text one character at a time.
"""
import sys
import time
from pathlib import Path

import numpy as np

from jagents.nn.layers import TinyTransformer
from jagents.nn.optim import Adam

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 600
text = "\n".join(p.read_text() for p in sorted((Path(__file__).parent.parent / "data" / "corpus").glob("*.md")))
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = np.array([stoi[c] for c in text])
split = int(0.9 * len(data))
train, val = data[:split], data[split:]

model = TinyTransformer(len(chars), dim=64, heads=4, layers=2, block_size=48)
opt = Adam(model.parameters(), lr=3e-3)
rng = np.random.default_rng(0)
print(f"{len(text):,} characters, vocabulary {len(chars)}, {model.num_parameters():,} parameters")


def batch(source, size=24):
    ix = rng.integers(0, len(source) - model.block_size - 1, size)
    x = np.stack([source[i:i + model.block_size] for i in ix])
    y = np.stack([source[i + 1:i + model.block_size + 1] for i in ix])
    return x, y


t0 = time.time()
for step in range(steps + 1):
    x, y = batch(train)
    model.zero_grad()
    loss = model.loss(x, y)
    loss.backward()
    opt.step()
    if step % 100 == 0:
        vx, vy = batch(val)
        print(f"step {step:4d}  train loss {loss.data:.3f}  val loss {model.loss(vx, vy).data:.3f}  "
              f"({time.time() - t0:.0f}s)")

prompt = "Espresso is"
out = model.generate([stoi[c] for c in prompt], 200, rng)
print("\nSample:\n" + "".join(chars[i] for i in out))
