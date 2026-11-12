# Smart Portfolio AI Assistant

A RAG chatbot that answers questions about a resume using Google Gemini, LangChain, ChromaDB, and Streamlit.

## Setup
1. Clone the repo and install dependencies: `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and add your Google API key.
3. Create a `data/` folder in the project root and put your resume PDF inside it.
4. Build the vector database: `python ingest.py`
5. Start the app: `streamlit run app.py`

## Tech Stack
Python, Streamlit, LangChain, Google Gemini, ChromaDB, HuggingFace sentence-transformers