"""Neural-network building blocks on top of jagents.nn.tensor, up to a small GPT-style transformer."""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Module:
    def parameters(self) -> list[Tensor]:
        params = []
        for value in self.__dict__.values():
            if isinstance(value, Tensor) and value.requires_grad:
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def num_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters())


def _init(rng: np.random.Generator, fan_in: int, shape: tuple) -> Tensor:
    # Scale by 1/sqrt(fan_in) so activations neither explode nor vanish as layers stack up.
    return Tensor(rng.normal(0, 1 / np.sqrt(fan_in), shape), requires_grad=True)


class Linear(Module):
    """y = x @ W + b"""

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator, bias: bool = True):
        self.w = _init(rng, n_in, (n_in, n_out))
        self.b = Tensor(np.zeros(n_out), requires_grad=True) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        y = x @ self.w
        return y + self.b if self.b is not None else y


class MLP(Module):
    """Linear -> ReLU -> Linear -> ReLU -> ... -> Linear"""

    def __init__(self, sizes: list[int], rng: np.random.Generator):
        self.layers = [Linear(a, b, rng) for a, b in zip(sizes[:-1], sizes[1:])]

    def __call__(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = x.relu()
        return x


class Embedding(Module):
    """A learnable lookup table: token id -> vector."""

    def __init__(self, n: int, dim: int, rng: np.random.Generator):
        self.table = Tensor(rng.normal(0, 0.02, (n, dim)), requires_grad=True)

    def __call__(self, ids: np.ndarray) -> Tensor:
        return self.table[ids]


class LayerNorm(Module):
    """Normalises each vector to mean 0 / variance 1, then rescales. Stabilises deep networks."""

    def __init__(self, dim: int, eps: float = 1e-5):
        self.gain = Tensor(np.ones(dim), requires_grad=True)
        self.bias = Tensor(np.zeros(dim), requires_grad=True)
        self.eps = eps

    def __call__(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        var = (centered * centered).mean(axis=-1, keepdims=True)
        return centered * (var + self.eps) ** -0.5 * self.gain + self.bias


class CausalSelfAttention(Module):
    """Multi-head self-attention where each position may only look at itself and earlier ones.

    For every position, a query vector is compared with the key vectors of all earlier
    positions; the scores (after softmax) say how much of each earlier value vector to mix in.
    With several heads, the model can track several kinds of relationship at once.
    """

    def __init__(self, dim: int, heads: int, rng: np.random.Generator):
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.qkv = Linear(dim, 3 * dim, rng)
        self.proj = Linear(dim, dim, rng)

    def __call__(self, x: Tensor) -> Tensor:
        b, t, c = x.shape
        h, hd = self.heads, c // self.heads
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = (qkv[:, :, i * c:(i + 1) * c].reshape(b, t, h, hd).transpose(0, 2, 1, 3) for i in range(3))
        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(hd))  # (B, H, T, T)
        mask = np.triu(np.full((t, t), -1e9), k=1)  # -inf above the diagonal: no peeking ahead
        att = (scores + mask).softmax(axis=-1)
        out = (att @ v).transpose(0, 2, 1, 3).reshape(b, t, c)
        return self.proj(out)


class Block(Module):
    """Transformer block: x + attention(norm(x)), then x + mlp(norm(x)) (pre-norm residuals)."""

    def __init__(self, dim: int, heads: int, rng: np.random.Generator):
        self.ln1 = LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, rng)
        self.ln2 = LayerNorm(dim)
        self.fc1 = Linear(dim, 4 * dim, rng)
        self.fc2 = Linear(4 * dim, dim, rng)

    def __call__(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.fc2(self.fc1(self.ln2(x)).relu())


class TinyTransformer(Module):
    """A character-level GPT: predicts the next character from the previous `block_size` ones."""

    def __init__(self, vocab: int, dim: int = 48, heads: int = 4, layers: int = 2, block_size: int = 32,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.block_size = block_size
        self.tok = Embedding(vocab, dim, rng)
        self.pos = Embedding(block_size, dim, rng)
        self.blocks = [Block(dim, heads, rng) for _ in range(layers)]
        self.ln = LayerNorm(dim)
        self.head = Linear(dim, vocab, rng)

    def __call__(self, ids: np.ndarray) -> Tensor:
        b, t = ids.shape
        x = self.tok(ids) + self.pos(np.arange(t))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))  # logits (B, T, vocab)

    def loss(self, ids: np.ndarray, targets: np.ndarray) -> Tensor:
        logits = self(ids)
        b, t, v = logits.shape
        return logits.reshape(b * t, v).cross_entropy(targets.reshape(b * t))

    def generate(self, ids: list[int], n: int, rng: np.random.Generator, temperature: float = 0.8) -> list[int]:
        out = list(ids)
        for _ in range(n):
            ctx = np.array([out[-self.block_size:]])
            logits = self(ctx).data[0, -1] / temperature
            p = np.exp(logits - logits.max())
            out.append(int(rng.choice(len(p), p=p / p.sum())))
        return out
