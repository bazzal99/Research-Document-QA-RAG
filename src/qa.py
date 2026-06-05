"""
Step 5 — Retrieval + Q&A Chain
Run this after ingest.py has built the FAISS index.
"""

import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── Load environment ───────────────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")

# ── Prompt ─────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a research assistant helping answer questions about academic documents.
Use ONLY the context below to answer. If the answer is not in the context, say "I don't have enough information in the provided documents to answer this."
Always mention which document your answer comes from.

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


def load_vectorstore():
    vdir = os.path.abspath(VECTORSTORE_DIR)
    if not os.path.exists(vdir):
        print(f"❌ No vectorstore found at: {vdir}")
        print("   Please run: python src/ingest.py first.")
        sys.exit(1)

    print("📦 Loading FAISS index...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )
    vectorstore = FAISS.load_local(
        vdir,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("   ✅ Vectorstore loaded")
    return vectorstore


def build_qa_chain(vectorstore):
    print("🤖 Loading Gemini Flash...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {os.path.basename(d.metadata['source'])}, page {d.metadata.get('page','?')}]\n{d.page_content}"
            for d in docs
        )

    chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    print("   ✅ Q&A chain ready\n")
    return chain, retriever


def ask(chain, retriever, question: str):
    print(f"\n❓ {question}")
    print("-" * 50)

    answer = chain.invoke(question)
    print(f"💡 {answer}")

    docs = retriever.invoke(question)
    print("\n📚 Sources:")
    seen = set()
    for doc in docs:
        src  = os.path.basename(doc.metadata["source"])
        page = doc.metadata.get("page", "?")
        key  = f"{src}:p{page}"
        if key not in seen:
            print(f"   - {src}, page {page}")
            seen.add(key)
    print("=" * 50)


if __name__ == "__main__":
    print("=" * 50)
    print("  Research Document Q&A System")
    print("=" * 50)

    if not GOOGLE_API_KEY:
        print("❌ GOOGLE_API_KEY not found."); sys.exit(1)

    vectorstore      = load_vectorstore()
    chain, retriever = build_qa_chain(vectorstore)

    print("Type your question and press Enter. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋"); break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye! 👋"); break
        ask(chain, retriever, question)