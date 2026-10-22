"""A small reverse-mode autodiff engine over NumPy arrays (a tiny PyTorch, written from scratch).

The idea is the same as jagents.nn.scalar.Value, but each node holds an array, so one operation
covers thousands of numbers. Each operation defines:

  * forward:  compute the output array
  * backward: given d(loss)/d(output), add d(loss)/d(input) to each input's .grad

Broadcasting needs care: if a (3,) bias was added to a (32, 3) batch, the bias's gradient is the
SUM over the 32 rows (it was used 32 times). unbroadcast() does that bookkeeping.
"""

from __future__ import annotations

import numpy as np


def unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sums `grad` down to `shape`, undoing NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, requires_grad: bool = False, _parents: tuple = ()):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._parents = _parents
        self._backward = None

    # ---- graph plumbing ----------------------------------------------------------------

    def _make(self, data, parents, backward):
        needs = any(p.requires_grad for p in parents)
        out = Tensor(data, requires_grad=needs, _parents=parents if needs else ())
        if needs:
            out._backward = backward
        return out

    @staticmethod
    def _wrap(x) -> "Tensor":
        return x if isinstance(x, Tensor) else Tensor(x)

    def _accumulate(self, g: np.ndarray):
        if self.requires_grad:
            self.grad = self.grad + unbroadcast(g, self.data.shape)

    def backward(self):
        """Backpropagation from a scalar loss through the whole graph."""
        order, seen = [], set()

        def visit(t):
            if id(t) not in seen:
                seen.add(id(t))
                for p in t._parents:
                    visit(p)
                order.append(t)

        visit(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(order):
            if t._backward is not None:
                t._backward(t.grad)

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    # ---- elementwise -----------------------------------------------------------------

    def __add__(self, other):
        other = Tensor._wrap(other)

        def bw(g):
            self._accumulate(g)
            other._accumulate(g)

        return self._make(self.data + other.data, (self, other), bw)

    __radd__ = __add__

    def __mul__(self, other):
        other = Tensor._wrap(other)

        def bw(g):
            self._accumulate(g * other.data)
            other._accumulate(g * self.data)

        return self._make(self.data * other.data, (self, other), bw)

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-Tensor._wrap(other))

    def __rsub__(self, other):
        return Tensor._wrap(other) + (-self)

    def __truediv__(self, other):
        return self * Tensor._wrap(other) ** -1.0

    def __pow__(self, k: float):
        def bw(g):
            self._accumulate(g * k * self.data ** (k - 1))

        return self._make(self.data**k, (self,), bw)

    def relu(self):
        def bw(g):
            self._accumulate(g * (self.data > 0))

        return self._make(np.maximum(self.data, 0), (self,), bw)

    def tanh(self):
        t = np.tanh(self.data)

        def bw(g):
            self._accumulate(g * (1 - t * t))

        return self._make(t, (self,), bw)

    def exp(self):
        e = np.exp(self.data)

        def bw(g):
            self._accumulate(g * e)

        return self._make(e, (self,), bw)

    def log(self):
        def bw(g):
            self._accumulate(g / self.data)

        return self._make(np.log(self.data), (self,), bw)

    # ---- reductions and shape ------------------------------------------------------------

    def sum(self, axis=None, keepdims=False):
        def bw(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accumulate(np.broadcast_to(g, self.data.shape))

        return self._make(self.data.sum(axis=axis, keepdims=keepdims), (self,), bw)

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def reshape(self, *shape):
        def bw(g):
            self._accumulate(g.reshape(self.data.shape))

        return self._make(self.data.reshape(*shape), (self,), bw)

    def transpose(self, *axes):
        inverse = np.argsort(axes)

        def bw(g):
            self._accumulate(g.transpose(*inverse))

        return self._make(self.data.transpose(*axes), (self,), bw)

    def __getitem__(self, idx):
        """Indexing, e.g. embedding lookup table[token_ids]. Gradients scatter-add back."""

        def bw(g):
            if self.requires_grad:
                full = np.zeros_like(self.data)
                np.add.at(full, idx, g)  # add.at handles repeated indices correctly
                self.grad = self.grad + full

        return self._make(self.data[idx], (self,), bw)

    # ---- linear algebra and probability ------------------------------------------------

    def __matmul__(self, other):
        other = Tensor._wrap(other)

        def bw(g):
            # C = A @ B  =>  dA = dC @ B^T,  dB = A^T @ dC  (batched over leading dimensions)
            self._accumulate(g @ np.swapaxes(other.data, -1, -2))
            other._accumulate(np.swapaxes(self.data, -1, -2) @ g)

        return self._make(self.data @ other.data, (self, other), bw)

    def softmax(self, axis=-1):
        shifted = self.data - self.data.max(axis=axis, keepdims=True)  # avoids overflow in exp
        e = np.exp(shifted)
        s = e / e.sum(axis=axis, keepdims=True)

        def bw(g):
            # Jacobian-vector product of softmax: s * (g - sum(g * s))
            self._accumulate(s * (g - (g * s).sum(axis=axis, keepdims=True)))

        return self._make(s, (self,), bw)

    def cross_entropy(self, targets: np.ndarray):
        """Mean cross-entropy between logits (N, V) and integer targets (N,).

        Fused with softmax: the gradient is simply (probabilities - one_hot) / N, which is
        both faster and numerically safer than chaining softmax, log and indexing.
        """
        logits = self.data
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        n = logits.shape[0]
        loss = -log_probs[np.arange(n), targets].mean()

        def bw(g):
            probs = np.exp(log_probs)
            probs[np.arange(n), targets] -= 1
            self._accumulate(g * probs / n)

        return self._make(loss, (self,), bw)
