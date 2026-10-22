"""Train a small neural network to separate a 2-D spiral. Run: python examples/train_mlp.py"""
import numpy as np

from jagents.nn.layers import MLP
from jagents.nn.optim import Adam


def spiral(n_per_class=100, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(classes):
        r = np.linspace(0.0, 1, n_per_class)
        t = np.linspace(c * 4, (c + 1) * 4, n_per_class) + rng.normal(0, 0.2, n_per_class)
        xs.append(np.c_[r * np.sin(t), r * np.cos(t)])
        ys.append(np.full(n_per_class, c))
    return np.vstack(xs), np.concatenate(ys)


def main(steps=400, verbose=True):
    x, y = spiral()
    model = MLP([2, 64, 64, 3], np.random.default_rng(0))
    opt = Adam(model.parameters(), lr=0.01)
    from jagents.nn.tensor import Tensor
    for step in range(steps):
        model.zero_grad()
        logits = model(Tensor(x))
        loss = logits.cross_entropy(y)
        loss.backward()
        opt.step()
        if verbose and step % 100 == 0:
            acc = (logits.data.argmax(1) == y).mean()
            print(f"step {step:4d}  loss {loss.data:.3f}  accuracy {acc:.2%}")
    acc = (model(Tensor(x)).data.argmax(1) == y).mean()
    if verbose:
        print(f"final accuracy {acc:.2%} with {model.num_parameters()} parameters")
    return acc


if __name__ == "__main__":
    main()
