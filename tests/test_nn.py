"""Autograd is checked against numerical gradients: (f(x+h) - f(x-h)) / 2h."""
import numpy as np

from jagents.nn.layers import MLP, TinyTransformer
from jagents.nn.optim import Adam
from jagents.nn.scalar import Value
from jagents.nn.tensor import Tensor


def test_scalar_chain_rule():
    a, b = Value(2.0), Value(-3.0)
    c = a * b + a
    c.backward()
    assert c.data == -4.0 and a.grad == -2.0 and b.grad == 2.0


def test_scalar_tanh_matches_numerical():
    x = Value(0.7)
    y = (x * x + x).tanh()
    y.backward()
    f = lambda v: np.tanh(v * v + v)
    assert abs(x.grad - (f(0.7 + 1e-6) - f(0.7 - 1e-6)) / 2e-6) < 1e-6


def numerical_grad(f, x, h=1e-5):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + h; up = f()
        x[i] = old - h; down = f()
        x[i] = old
        g[i] = (up - down) / (2 * h)
    return g


def test_tensor_gradients_match_numerical():
    rng = np.random.default_rng(0)
    w = Tensor(rng.normal(size=(4, 3)), requires_grad=True)
    x = Tensor(rng.normal(size=(5, 4)))
    targets = np.array([0, 2, 1, 1, 0])

    def loss():
        return ((x @ w).tanh() * 2.0).cross_entropy(targets)

    out = loss()
    out.backward()
    num = numerical_grad(lambda: float(loss().data), w.data)
    assert np.allclose(w.grad, num, atol=1e-6)


def test_softmax_and_broadcast_gradients():
    rng = np.random.default_rng(1)
    a = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    b = Tensor(rng.normal(size=(4,)), requires_grad=True)
    f = lambda: float(((a + b).softmax(axis=-1) * Tensor(np.arange(4.0))).sum().data)
    ((a + b).softmax(axis=-1) * Tensor(np.arange(4.0))).sum().backward()
    assert np.allclose(a.grad, numerical_grad(f, a.data), atol=1e-6)
    assert np.allclose(b.grad, numerical_grad(f, b.data), atol=1e-6)


def test_mlp_learns_xor():
    rng = np.random.default_rng(0)
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    model = MLP([2, 16, 2], rng)
    opt = Adam(model.parameters(), lr=0.05)
    for _ in range(300):
        model.zero_grad()
        loss = model(Tensor(x)).cross_entropy(y)
        loss.backward()
        opt.step()
    assert (model(Tensor(x)).data.argmax(axis=1) == y).all()


def test_transformer_loss_decreases_and_is_causal():
    text = "abcabcabcabcabcabcabcabcabcabc" * 4
    data = np.array([ord(c) - 97 for c in text])
    model = TinyTransformer(3, dim=16, heads=2, layers=1, block_size=8, seed=0)
    opt = Adam(model.parameters(), lr=1e-2)
    x = np.stack([data[i:i + 8] for i in range(0, 40, 4)])
    y = np.stack([data[i + 1:i + 9] for i in range(0, 40, 4)])
    first = float(model.loss(x, y).data)
    for _ in range(60):
        model.zero_grad()
        loss = model.loss(x, y)
        loss.backward()
        opt.step()
    assert float(loss.data) < first * 0.5

    # causality: changing a later token must not change the output at earlier positions
    a = model(x[:1]).data
    x2 = x[:1].copy(); x2[0, -1] = (x2[0, -1] + 1) % 3
    b = model(x2).data
    assert np.allclose(a.reshape(8, -1)[:-1], b.reshape(8, -1)[:-1])
