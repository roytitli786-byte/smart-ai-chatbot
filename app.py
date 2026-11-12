import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- ENVIRONMENT ---
load_dotenv()  # reads GOOGLE_API_KEY from .env

st.set_page_config(page_title="Executive AI Portfolio Agent", page_icon="⚙️", layout="centered")
st.title("⚙️ Smart Portfolio AI Assistant")
st.caption("Advanced RAG Agent built with deep local text vectors and historical context chains.")

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY is not set. Add it to your .env file.")
    st.stop()

# --- CONTEXT ENGINE ---
@st.cache_resource
def load_context_engine():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory="chroma_db/", embedding_function=embeddings)
    if len(db.get()["ids"]) == 0:
        raise ValueError("chroma_db is empty. Run ingest.py first.")
    return db.as_retriever(search_kwargs={"k": 4})

try:
    retriever = load_context_engine()
    llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    temperature=0.2,
)
except Exception as e:
    st.error(f"Setup Error: {e}")
    st.stop()

# --- HISTORY-AWARE RETRIEVAL ---
context_system_prompt = (
    "Given a chat history and the latest user question which might reference context in the chat history, "
    "formulate a standalone question which can be understood without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
context_prompt = ChatPromptTemplate.from_messages([
    ("system", context_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)

# --- ANSWER GENERATION ---
executive_prompt_template = (
    "You are a professional executive talent manager representing the candidate based on the provided technical criteria.\n"
    "Use the background documents context matrix provided below to answer the user's question clearly, thoroughly, and confidently.\n\n"
    "Formatting Rules:\n"
    "- Always separate details into bold section headers or structured bullet points.\n"
    "- If the required answer details cannot be found inside the Context Matrix below, state professionally "
    "that you don't possess that specific record in your source data files, and decline to make things up.\n\n"
    "Context Matrix:\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", executive_prompt_template),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

# --- CHAT UI ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

if user_query := st.chat_input("Ask a question about experience, technical stacks, or achievements..."):
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing vectors & conversational history..."):
            try:
                response = rag_chain.invoke({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history,
                })
                answer = response["answer"]
            except Exception as e:
                answer = f"Sorry, something went wrong while generating a response: {e}"
            st.markdown(answer)

    st.session_state.chat_history.extend([
        HumanMessage(content=user_query),
        AIMessage(content=answer),
    ])