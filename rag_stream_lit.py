import os
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

# ───── Configuration ──────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
assert OPENROUTER_API_KEY, "Set OPENROUTER_API_KEY environment variable"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openrouter/auto"  # You can replace with a specific model like "mistralai/mistral-7b"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_TOKENS = 500  # Limit the output token count from the LLM

base_dir = os.path.dirname(os.path.abspath(__file__))
RESUME_DIR = os.path.join(base_dir, "examples")
VECTOR_STORE_DIR = os.path.join(base_dir, "resume_vector_db")

# ───── Validate Folders ────────────────────────────────────────────
if not os.path.exists(RESUME_DIR):
    raise FileNotFoundError(f"Resume directory '{RESUME_DIR}' does not exist.")
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# ───── Step 1: Create or Load Vectorstore ──────────────────────────
def get_vectorstore():
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    index_path = os.path.join(VECTOR_STORE_DIR, "index.faiss")

    if os.path.exists(index_path):
        return FAISS.load_local(VECTOR_STORE_DIR, embedder, allow_dangerous_deserialization=True)

    # Load and chunk documents
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for file in os.listdir(RESUME_DIR):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(RESUME_DIR, file))
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            for chunk in chunks:
                chunk.metadata = {"source": file}
            all_chunks.extend(chunks)

    # Build and persist FAISS
    vectorstore = FAISS.from_documents(all_chunks, embedder)
    vectorstore.save_local(VECTOR_STORE_DIR)
    return vectorstore

# ───── Step 2: Create RAG Chain ────────────────────────────────────
def get_rag_chain():
    vectorstore = get_vectorstore()
    llm = ChatOpenAI(
        openai_api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        model=LLM_MODEL,
        temperature=0.2,
        max_tokens=MAX_TOKENS,  # Apply token limit here
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

# ───── Streamlit UI ────────────────────────────────────────────────
st.set_page_config(page_title="Resume RAG Q&A", layout="wide")
st.title("📄 Resume Q&A RAG - Single Prompt")

query = st.text_input("🔍 Ask a question about the candidate resumes:")

if query:
    with st.spinner("Analyzing resumes..."):
        qa_chain = get_rag_chain()
        result = qa_chain(query)

        st.subheader("✅ Answer")
        st.markdown(result["result"])

        