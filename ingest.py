import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = "data/"
DB_PATH = "chroma_db/"

def build_vector_database():
    # 1. Verify that the data folder contains at least one PDF
    has_pdf = os.path.isdir(DATA_PATH) and any(
        f.lower().endswith(".pdf") for f in os.listdir(DATA_PATH)
    )
    if not has_pdf:
        print(f"❌ Error: Put at least one PDF inside the '{DATA_PATH}' folder before running.")
        return

    print("⚡ [1/4] Reading PDF documents...")
    raw_docs = PyPDFDirectoryLoader(DATA_PATH).load()
    print(f"📖 Loaded {len(raw_docs)} pages.")

    if not any(doc.page_content.strip() for doc in raw_docs):
        print("❌ Error: No extractable text found. If your PDF is a scan or image, use a text-based PDF.")
        return

    print("✂️ [2/4] Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(raw_docs)
    print(f"🧩 Created {len(chunks)} chunks.")

    if not chunks:
        print("❌ Error: Splitting produced no chunks. Check your PDF content.")
        return

    print("🧠 [3/4] Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Clear old vectors by deleting the collection, not the whole folder.
    # This avoids the Windows file-lock error.
    old_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    try:
        old_db.delete_collection()
        print("🗑️ Cleared old vectors.")
    except Exception:
        pass  # No existing collection, so there's nothing to clear

    print("💾 [4/4] Writing vectors to chroma_db/...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH,
    )

    total = len(vector_db.get()["ids"])
    print(f"✨ SUCCESS: {total} chunks stored.")

    # Quick sanity check: confirm the database can be searched
    test = vector_db.similarity_search("work experience", k=1)
    if test:
        print(f"🔍 Sample retrieved chunk: {test[0].page_content[:150]}...")

if __name__ == "__main__":
    build_vector_database()