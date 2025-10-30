import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE

corpus = [
    "The quick brown fox jumps over the lazy dog".lower().split(),
    "A dog is a loyal pet and a friend".lower().split(),
    "The fox hunted the rabbit in the snowy field".lower().split(),
    "Quick movements are necessary for survival".lower().split(),
    "The brown leather jacket was quite fashionable".lower().split(),
    "Pet food comes in various shapes and sizes".lower().split(),
    "I love running in the field with my dog".lower().split()
]

print(f"Corpus size (sentences): {len(corpus)}")


model = Word2Vec(
    sentences=corpus,
    vector_size=50,  # 50 dimensions
    window=5,
    min_count=1,
    workers=4,
    sg=1
)

model.train(corpus, total_examples=model.corpus_count, epochs=100)
print(f"Model trained. Vocabulary size: {len(model.wv.index_to_key)}")


similarity = model.wv.similarity('dog', 'pet')
print(f"\nSimilarity between 'dog' and 'pet': {similarity:.4f}")

# We'll extract all unique words and their corresponding vectors
words = model.wv.index_to_key
vectors = model.wv[words]

print(f"Extracted {len(vectors)} vectors of dimension {vectors.shape[1]}")


tsne_model = TSNE(
    n_components=2,           # Target dimension: 2D
    perplexity=5,             # Controls local/global balance
    max_iter=2500,            # Updated parameter name
    random_state=42,          # For reproducibility
    learning_rate='auto',     # Auto-tune learning rate
    init='pca'                # Faster, stable initialization
)


vectors_2d = tsne_model.fit_transform(vectors)

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 12))

# Scatter plot the 2D points
ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='purple', alpha=0.8, s=50)

for i, word in enumerate(words):
    x = vectors_2d[i, 0]
    y = vectors_2d[i, 1]

    if word in ['dog', 'fox', 'rabbit', 'pet']:
        color = 'red'
        fontweight = 'bold'
    elif word in ['quick', 'brown', 'lazy']:
        color = 'blue'
        fontweight = 'normal'
    else:
        color = 'black'
        fontweight = 'normal'

    ax.annotate(
        word,
        xy=(x, y),
        xytext=(5, 2),
        textcoords='offset points',
        ha='right',
        va='bottom',
        fontsize=10,
        color=color,
        fontweight=fontweight
    )

ax.set_title('Word2Vec Embeddings Visualization using t-SNE', fontsize=16)
ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
plt.grid(True)
plt.show()
