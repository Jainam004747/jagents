"""Automatic differentiation on single numbers: the smallest possible "how learning works".

Every Value remembers which operation produced it and from which inputs. backward() walks that
graph from the output to the inputs and applies the chain rule, so every Value learns
d(output)/d(itself). Training a model is then: compute a loss, call backward(), and nudge each
parameter a little against its gradient.

    a = Value(2.0); b = Value(-3.0)
    c = a * b + a          # c = -4
    c.backward()
    a.grad  -> -2.0        # dc/da = b + 1
    b.grad  ->  2.0        # dc/db = a

jagents.nn.tensor does exactly the same thing for whole arrays at once, which is what makes
real networks fast enough to train.
"""

from __future__ import annotations

import math


class Value:
    def __init__(self, data: float, _children: tuple = (), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # kept for printing/debugging the graph

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1 and d(a+b)/db = 1: the gradient flows through unchanged
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b and d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k: float):
        out = Value(self.data**k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        # Topological order: a node's gradient is complete only after every node that uses it
        # has passed its contribution back, so visit the graph from the output backwards.
        order, seen = [], set()

        def visit(v):
            if v not in seen:
                seen.add(v)
                for child in v._prev:
                    visit(child)
                order.append(v)

        visit(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()

    # Conveniences so expressions read naturally
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other**-1 if isinstance(other, Value) else 1.0 / other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Value(other) - self

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
