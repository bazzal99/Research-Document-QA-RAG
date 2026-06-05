"""
Step 3 & 4 — PDF Ingestion + Chunking + Embedding + FAISS Storage
Run this once (or whenever you add new PDFs to docs/)
"""

import os
import sys
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# ── Load environment ───────────────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
DOCS_DIR        = os.path.join(os.path.dirname(__file__), "..", "docs")
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")

BATCH_SIZE  = 10   # chunks per batch
DELAY       = 7    # seconds between batches (~10 batches/min = safe)


def load_pdfs(docs_dir: str):
    docs_dir = os.path.abspath(docs_dir)
    if not os.path.exists(docs_dir):
        print(f"❌ docs/ folder not found at: {docs_dir}"); sys.exit(1)
    pdf_files = [f for f in os.listdir(docs_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"❌ No PDF files found in: {docs_dir}"); sys.exit(1)

    print(f"📂 Loading PDFs from: {docs_dir}")
    loader = PyPDFDirectoryLoader(docs_dir)
    documents = loader.load()
    sources = set(os.path.basename(d.metadata["source"]) for d in documents)
    print(f"   ✅ Loaded {len(documents)} pages from {len(sources)} PDF(s)")
    for s in sorted(sources):
        count = sum(1 for d in documents if os.path.basename(d.metadata["source"]) == s)
        print(f"      - {s}: {count} page(s)")
    return documents


def chunk_documents(documents):
    print("\n✂️  Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    avg = sum(len(c.page_content) for c in chunks) // len(chunks)
    print(f"   ✅ Created {len(chunks)} chunks (avg {avg} chars each)")
    return chunks


def embed_and_store(chunks, vectorstore_dir: str):
    total   = len(chunks)
    batches = [chunks[i:i+BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]
    eta_min = round((len(batches) * DELAY) / 60, 1)

    print(f"\n🔢 Embedding {total} chunks in batches of {BATCH_SIZE}...")
    print(f"   Estimated time: ~{eta_min} min\n")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )

    vectorstore = None
    for i, batch in enumerate(batches):
        pct = round((i / len(batches)) * 100)
        print(f"   [{pct:3d}%] Batch {i+1}/{len(batches)} — {len(batch)} chunks...", end=" ", flush=True)

        success = False
        retries = 0
        while not success:
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    vectorstore.merge_from(FAISS.from_documents(batch, embeddings))
                success = True
                print("✅")
            except Exception as e:
                retries += 1
                wait = 30 * retries
                print(f"\n      ⚠️  Rate limit hit. Waiting {wait}s... (retry {retries})")
                time.sleep(wait)

        if i < len(batches) - 1:
            time.sleep(DELAY)

    os.makedirs(vectorstore_dir, exist_ok=True)
    vectorstore.save_local(vectorstore_dir)
    print(f"\n   ✅ FAISS index saved to: {os.path.abspath(vectorstore_dir)}")
    return vectorstore


if __name__ == "__main__":
    print("=" * 50)
    print("  RAG Ingestion Pipeline")
    print("=" * 50)

    if not GOOGLE_API_KEY:
        print("❌ GOOGLE_API_KEY not found."); sys.exit(1)

    docs   = load_pdfs(DOCS_DIR)
    chunks = chunk_documents(docs)
    embed_and_store(chunks, VECTORSTORE_DIR)

    print("\n🎉 Done! You can now run: python src/qa.py")