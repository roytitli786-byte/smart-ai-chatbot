# Smart Portfolio AI Assistant

A RAG chatbot that answers questions about a resume using Google Gemini, LangChain, ChromaDB, and Streamlit.

## Setup
1. Clone the repo and install dependencies: `pip install -r requirements.txt`
2. Create a `.env` file in the project root with:
        GOOGLE_API_KEY=your-google-api-key-here
        GEMINI_MODEL=your-gemini-model-here
3. Create a `data/` folder in the project root and put your resume PDF inside it.
4. Build the vector database: `python ingest.py`
5. Start the app: `streamlit run app.py`

## Tech Stack
Python, Streamlit, LangChain, Google Gemini, ChromaDB, HuggingFace sentence-transformers