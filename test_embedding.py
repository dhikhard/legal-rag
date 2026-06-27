from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the embedding model (downloads ~90MB on first run)
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Some legal sentences to embed
sentences = [
    "Section 420 of IPC deals with cheating and dishonestly inducing delivery of property.",
    "The accused was granted bail by the sessions court.",
    "A contract requires offer, acceptance, and consideration to be valid.",
    "The defendant filed an appeal in the High Court.",
]

print("Embedding sentences...")
embeddings = model.encode(sentences)

print(f"\nEach sentence is now a vector of {embeddings.shape[1]} numbers.\n")

sim_matrix = cosine_similarity(embeddings)

print("Similarity scores (1.0 = identical, 0.0 = unrelated):")
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        score = sim_matrix[i][j]
        print(f"  [{i+1}] vs [{j+1}]: {score:.3f}")

print("\nSentences:")
for i, s in enumerate(sentences):
    print(f"  [{i+1}] {s}")

print("\nPhase 1 complete! Embeddings are working.")