import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RETRIEVAL_K = 8

GROQ_API_KEY = st.secrets.get(
    "GROQ_API_KEY",
    os.getenv("GROQ_API_KEY", "")
)

st.set_page_config(
    page_title="Document AI Assistant",
    page_icon="📄",
    layout="wide"
)


# --------------------------------------------------
# INITIALIZE MODELS
# --------------------------------------------------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


@st.cache_resource
def load_llm():
    if not GROQ_API_KEY:
        raise ValueError(
            "Configure GROQ_API_KEY in Streamlit secrets."
        )

    return ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0
    )


# --------------------------------------------------
# DOCUMENT PROCESSING
# --------------------------------------------------

def process_uploaded_pdfs(uploaded_files):

    all_documents = []

    for uploaded_file in uploaded_files:

        if uploaded_file.size == 0:
            continue

        if not uploaded_file.name.lower().endswith(".pdf"):
            continue

        temp_path = None

        try:
            # Create a unique temporary file for this upload.
            with tempfile.NamedTemporaryFile(
                suffix=".pdf",
                delete=False
            ) as temp_file:

                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            loader = PyPDFLoader(temp_path)
            pages = loader.load()

            # Add the source filename to every page.
            for page in pages:
                page.metadata["source"] = uploaded_file.name

            all_documents.extend(pages)

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    if not all_documents:
        raise ValueError(
            "No readable PDF content was found."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(all_documents)

    if not chunks:
        raise ValueError(
            "The uploaded PDFs did not produce any text chunks."
        )

    return chunks


# --------------------------------------------------
# BUILD SESSION-SPECIFIC RETRIEVER
# --------------------------------------------------

def build_retriever(chunks):

    embeddings = load_embeddings()

    # In-memory vector index: not shared across users.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="user_documents"
    )

    semantic_retriever = vectorstore.as_retriever(
        search_kwargs={"k": RETRIEVAL_K}
    )

    keyword_retriever = BM25Retriever.from_documents(
        chunks
    )

    keyword_retriever.k = RETRIEVAL_K

    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            keyword_retriever,
            semantic_retriever
        ],
        weights=[0.5, 0.5]
    )

    return hybrid_retriever


# --------------------------------------------------
# ANSWER QUESTIONS
# --------------------------------------------------

def answer_question(question, retriever):

    relevant_documents = retriever.invoke(question)

    if not relevant_documents:
        return "I couldn't find relevant information in your documents."

    context_parts = []

    for document in relevant_documents:

        page_number = document.metadata.get("page", 0) + 1
        filename = document.metadata.get("source", "Unknown")

        context_parts.append(
            f"Source: {filename}\n"
            f"Page: {page_number}\n"
            f"Content: {document.page_content}"
        )

    context = "\n\n".join(context_parts)

    llm = load_llm()

    messages = [
        SystemMessage(
            content=(
                "You are a document question-answering assistant. "
                "Answer using only the supplied document context. "
                "Do not invent facts or figures. "
                "If the answer is not present in the context, "
                "clearly say that you could not find it in the "
                "provided documents. "
                "Cite the source filename and page number when possible."
            )
        ),
        HumanMessage(
            content=(
                f"DOCUMENT CONTEXT:\n{context}\n\n"
                f"USER QUESTION:\n{question}"
            )
        )
    ]

    response = llm.invoke(messages)

    return response.content


# --------------------------------------------------
# STREAMLIT USER INTERFACE
# --------------------------------------------------

st.title("📄 Document AI Assistant")
st.write(
    "Upload your PDF documents and ask questions "
    "to get answers based on their contents."
)

uploaded_files = st.file_uploader(
    "Upload your PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Process Documents", type="primary"):

    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        try:
            with st.spinner(
                "Extracting text and building your document index..."
            ):
                chunks = process_uploaded_pdfs(uploaded_files)
                retriever = build_retriever(chunks)

                st.session_state.retriever = retriever
                st.session_state.chat_history = []

                st.success(
                    f"Successfully indexed {len(chunks)} text chunks."
                )

        except Exception as exc:
            st.error(f"Document processing failed: {exc}")


if st.session_state.retriever is not None:

    st.subheader("💬 Ask Questions")

    question = st.chat_input(
        "Ask something about your uploaded documents..."
    )

    if question:

        with st.spinner("Finding information and generating an answer..."):

            try:
                answer = answer_question(
                    question,
                    st.session_state.retriever
                )

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": answer
                    }
                )

            except Exception as exc:
                st.error(f"Unable to generate an answer: {exc}")

    for message in st.session_state.chat_history:

        with st.chat_message("user"):
            st.write(message["question"])

        with st.chat_message("assistant"):
            st.write(message["answer"])
