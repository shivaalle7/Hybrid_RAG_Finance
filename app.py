
import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# =========================================================
# 1. CONFIGURATION
# =========================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "FINAL ANNUAL REPORT ENGLISH with cover (2).pdf"
CHROMA_DIR = BASE_DIR / "chromadbfinance_data_2026"
COLLECTION_NAME = "finance_report"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RETRIEVAL_K = 8

st.set_page_config(
    page_title="Hybrid RAG Finance Assistant",
    page_icon="📊",
    layout="wide",
)

# =========================================================
# 2. PAGE HEADER
# =========================================================

st.title("📊 Hybrid RAG Finance Assistant")

st.write(
    "Ask questions about your financial report using "
    "BM25 keyword search, semantic search, or hybrid retrieval."
)

# =========================================================
# 3. SIDEBAR
# =========================================================

st.sidebar.header("Settings")

groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    value=os.getenv("GROQ_API_KEY", ""),
    type="password",
)

retrieval_method = st.sidebar.selectbox(
    "Retrieval Method",
    [
        "Hybrid Search",
        "BM25 Keyword Search",
        "Semantic Search",
    ],
)

st.sidebar.caption(f"LLM: {GROQ_MODEL}")
st.sidebar.caption(f"Embedding: {EMBEDDING_MODEL}")

if st.sidebar.button("Rebuild Document Index"):
    st.cache_resource.clear()

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    st.rerun()

# =========================================================
# 4. LOAD DOCUMENTS AND CREATE RETRIEVERS
# =========================================================

@st.cache_resource(show_spinner="Loading PDF and building retrieval index...")
def load_rag_components(pdf_path, chroma_path):

    pdf_file = Path(pdf_path)
    db_path = Path(chroma_path)

    if not pdf_file.exists():
        raise FileNotFoundError(
            f"PDF not found:\n{pdf_file}\n\n"
            "Place the annual report PDF in the same folder as app.py."
        )

    # Load PDF pages
    loader = PyPDFLoader(str(pdf_file))
    pages = loader.load()

    if not pages:
        raise ValueError("The PDF contains no extractable text.")

    # Split extracted text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    chunks = splitter.split_documents(pages)

    if not chunks:
        raise ValueError(
            "No text chunks were created. "
            "The PDF may contain scanned pages without extractable text."
        )

    # Create embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # Open or create persistent Chroma database
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(db_path),
        embedding_function=embeddings,
    )

    # Index the document if the collection is empty
    existing_count = vector_store._collection.count()

    if existing_count == 0:
        vector_store.add_documents(chunks)

    # Semantic retrieval
    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K},
    )

    # BM25 keyword retrieval
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = RETRIEVAL_K

    # Hybrid retrieval
    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            vector_retriever,
            bm25_retriever,
        ],
        weights=[0.5, 0.5],
    )

    return {
        "vector_retriever": vector_retriever,
        "bm25_retriever": bm25_retriever,
        "hybrid_retriever": hybrid_retriever,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "indexed_count": vector_store._collection.count(),
    }


try:
    components = load_rag_components(
        str(PDF_PATH),
        str(CHROMA_DIR),
    )

except Exception as exc:
    st.error("Failed to load the document or initialize retrieval.")
    st.exception(exc)
    st.stop()

# =========================================================
# 5. DOCUMENT INFORMATION
# =========================================================

col1, col2, col3 = st.columns(3)

col1.metric("PDF Pages", components["page_count"])
col2.metric("Text Chunks", components["chunk_count"])
col3.metric("Indexed Chunks", components["indexed_count"])

st.success("Document retrieval system is ready.")

# =========================================================
# 6. QUESTION INPUT
# =========================================================

st.subheader("Ask a question about the financial report")

question = st.text_area(
    "Your question",
    placeholder=(
        "Example: What was the total revenue reported "
        "for FY 2023-24?"
    ),
    height=100,
)

ask_button = st.button(
    "Get Answer",
    type="primary",
    use_container_width=True,
)

# =========================================================
# 7. RETRIEVE DOCUMENT CHUNKS
# =========================================================

if ask_button:

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    if not groq_api_key.strip():
        st.warning(
            "Enter your Groq API key in the sidebar "
            "or configure GROQ_API_KEY in your .env file."
        )
        st.stop()

    try:
        # Select retrieval strategy
        if retrieval_method == "BM25 Keyword Search":
            retriever = components["bm25_retriever"]

        elif retrieval_method == "Semantic Search":
            retriever = components["vector_retriever"]

        else:
            retriever = components["hybrid_retriever"]

        with st.spinner("Retrieving relevant financial information..."):
            retrieved_docs = retriever.invoke(question)

        if not retrieved_docs:
            st.warning("No relevant document chunks were retrieved.")
            st.stop()

        # =================================================
        # 8. BUILD CONTEXT WITH SOURCE PAGES
        # =================================================

        context_sections = []

        for i, doc in enumerate(retrieved_docs, start=1):
            page = doc.metadata.get("page")

            if isinstance(page, int):
                page_label = str(page + 1)
            else:
                page_label = "Unknown"

            context_sections.append(
                f"[Source {i}; PDF page {page_label}]\n"
                f"{doc.page_content}"
            )

        context = "\n\n".join(context_sections)

        # =================================================
        # 9. INITIALIZE GROQ
        # =================================================

        llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=groq_api_key.strip(),
            temperature=0,
        )

        # =================================================
        # 10. FINANCIAL RAG PROMPT
        # =================================================

        system_prompt = """
You are a careful financial-report question-answering assistant.

Use only the financial report excerpts provided in the context.
Do not use outside knowledge to fill missing financial figures.

For every question:

1. Identify the exact financial metric requested.
2. Match the requested financial year precisely.
3. Distinguish carefully between:
   - Total revenue
   - Revenue receipts
   - Revenue expenditure
   - Capital expenditure
   - Total expenditure
   - Net profit or loss
   - Total assets and total liabilities
   - Earnings per share
   - Cash flow
4. Do not treat revenue expenditure as total revenue.
5. Do not treat total outgo as revenue.
6. Do not confuse assessment year (AY) with financial year (FY).
7. Use a number only if the supplied excerpts explicitly support it.
8. Do not calculate a missing number from unrelated figures.
9. If the answer is not present in the supplied context,
   state that it was not found in the retrieved excerpts.
10. Cite the PDF page number shown beside the relevant source.
11. If sources contain conflicting figures, explain the conflict.
12. Keep the answer clear and concise.

IMPORTANT:
A mention of a financial year does not prove that a figure
for that year answers the question.
"""

        human_prompt = f"""
USER QUESTION:
{question}

RETRIEVED FINANCIAL REPORT EXCERPTS:
{context}

Answer the question using the supplied excerpts.
Identify the correct metric and financial year.
If the required figure is missing, say so explicitly.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        # =================================================
        # 11. GENERATE ANSWER
        # =================================================

        with st.spinner("Generating answer with Groq..."):
            response = llm.invoke(messages)

        # =================================================
        # 12. DISPLAY ANSWER
        # =================================================

        st.subheader("📝 Answer")
        st.markdown(response.content)

        st.caption(
            f"Retrieval method: {retrieval_method} | "
            f"Model: {GROQ_MODEL}"
        )

        # =================================================
        # 13. SHOW RETRIEVED SOURCES
        # =================================================

        with st.expander(
            f"View retrieved document chunks ({len(retrieved_docs)})"
        ):
            for i, doc in enumerate(retrieved_docs, start=1):

                page = doc.metadata.get("page")

                if isinstance(page, int):
                    page_label = str(page + 1)
                else:
                    page_label = "Unknown"

                st.markdown(
                    f"**Source {i} — PDF page {page_label}**"
                )
                st.write(doc.page_content)
                st.divider()

    except Exception as exc:
        st.error("An error occurred while answering your question.")
        st.exception(exc)

        st.info(
            "Check your Groq API key, model access, PDF extraction, "
            "and installed package versions."
        )
