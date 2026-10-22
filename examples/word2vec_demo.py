"""Learn word vectors from the corpus and show nearest neighbours. Run: python examples/word2vec_demo.py"""
from pathlib import Path

from jagents.text.word2vec import Word2Vec

texts = [p.read_text() for p in sorted((Path(__file__).parent.parent / "data" / "corpus").glob("*.md"))]
model = Word2Vec(dim=32, window=3, min_count=2)
losses = model.fit(texts, epochs=30)
print("loss per epoch:", " ".join(f"{l:.2f}" for l in losses[::5]))
for w in ["coffee", "tea", "database", "milk"]:
    print(w, "->", ", ".join(f"{n} ({s:.2f})" for n, s in model.most_similar(w, 5)))
print("(A corpus this small gives noisy neighbours; word2vec needs millions of words to shine.)")
