import os
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── Settings ──────────────────────────────────────────────
PDF_FOLDER = "data/pdfs"
CHROMA_DB_PATH = "db"
COLLECTION_NAME = "legal_docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ── Load embedding model ───────────────────────────────────
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ── Connect to ChromaDB ────────────────────────────────────
print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    print(f"Extracting text from {pdf_path}...")
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            full_text += f"\n[Page {page_num + 1}]\n{text}"
    return full_text

def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def ingest_pdf(pdf_path):
    """Full pipeline: extract → chunk → embed → store."""
    filename = os.path.basename(pdf_path)

    # Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print(f"No text found in {filename}, skipping.")
        return

    # Split into chunks
    chunks = split_into_chunks(text)
    print(f"Split into {len(chunks)} chunks.")

    # Embed and store in ChromaDB
    print("Embedding and storing chunks...")
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        collection.add(
            ids=[f"{filename}_chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": filename, "chunk_index": i}]
        )
        if (i + 1) % 50 == 0:
            print(f"  Stored {i + 1}/{len(chunks)} chunks...")

    print(f"Done! {len(chunks)} chunks stored for {filename}.")

# ── Run ingestion ──────────────────────────────────────────
if __name__ == "__main__":
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDFs found in data/pdfs/")
    else:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(PDF_FOLDER, pdf_file)
            ingest_pdf(pdf_path)

    print(f"\nTotal chunks in database: {collection.count()}")
    print("Ingestion complete! ChromaDB is ready.")