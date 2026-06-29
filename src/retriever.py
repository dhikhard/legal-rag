import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DB_PATH = "db"
COLLECTION_NAME = "legal_docs"

print("Loading retriever...")
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

def retrieve(query, top_k=5):
    print(f"Searching for: {query}")
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    print(f"Found {len(chunks)} relevant chunks.")
    return chunks, metadatas

if __name__ == "__main__":
    query = "What is the punishment for murder?"
    chunks, metadatas = retrieve(query)
    print(f"\nTop results for: '{query}'\n")
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
        print(f"--- Result {i+1} ---")
        print(f"Source: {meta['source']} | Chunk: {meta['chunk_index']}")
        print(chunk[:300])
        print()
