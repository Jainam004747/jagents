"""Optimizers: how parameters move once their gradients are known."""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


class SGD:
    """Plain gradient descent: p -= lr * grad."""

    def __init__(self, params: list[Tensor], lr: float = 0.1):
        self.params, self.lr = params, lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad


class Adam:
    """Adam: gradient descent with momentum (m) and a per-parameter step size (v).

    m smooths the gradient over recent steps; v tracks how large gradients usually are, so
    parameters with consistently large gradients take smaller steps. The bias correction fixes
    the fact that m and v start at zero.
    """

    def __init__(self, params: list[Tensor], lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8):
        self.params, self.lr, self.eps = params, lr, eps
        self.b1, self.b2 = betas
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / (1 - self.b1**self.t)
            v_hat = self.v[i] / (1 - self.b2**self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
