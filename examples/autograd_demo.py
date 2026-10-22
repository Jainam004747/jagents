"""Backpropagation by hand, on single numbers. Run: python examples/autograd_demo.py"""
from jagents.nn.scalar import Value

# A single neuron: out = tanh(w1*x1 + w2*x2 + b)
x1, x2 = Value(2.0), Value(0.0)
w1, w2, b = Value(-3.0), Value(1.0), Value(6.8813735870195432)
out = (x1 * w1 + x2 * w2 + b).tanh()
out.backward()
print(f"out = {out.data:.4f}")
for name, v in [("x1", x1), ("w1", w1), ("x2", x2), ("w2", w2), ("b", b)]:
    print(f"d(out)/d({name}) = {v.grad:+.4f}")

# Gradient descent on that neuron: push the output towards -1
for step in range(20):
    for p in (w1, w2, b):
        p.grad = 0.0
    out = (x1 * w1 + x2 * w2 + b).tanh()
    loss = (out - (-1.0)) ** 2
    loss.backward()
    for p in (w1, w2, b):
        p.data -= 0.1 * p.grad
print(f"after 20 steps: out = {out.data:.4f}, loss = {loss.data:.5f}")
