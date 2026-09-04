import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_classic.retrievers import (
    EnsembleRetriever,
    BM25Retriever
)

from langchain_groq import ChatGroq

from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

st.set_page_config(
    page_title="Finance PDF RAG Chatbot",
    layout="wide"
)

st.title("Finance PDF RAG Chatbot")
st.write("Ask questions from the Finance Annual Report PDF")


groq_key = st.sidebar.text_input(
    "Enter Groq API Key",
    type="password"
)


@st.cache_resource
def load_rag_pipeline():



    loader = PyPDFLoader(
        file_path=r"FINAL ANNUAL REPORT ENGLISH with cover (2).pdf"
    )

    pages = loader.load()



    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(pages)



    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="chromadbfinance_data_2026",
        collection_name="langchain"
    )


    vector_retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )


    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 3


    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            vector_retriever,
            bm25_retriever
        ],
        weights=[0.7, 0.3]
    )

    return hybrid_retriever



with st.spinner("Loading Finance PDF..."):
    hybrid_retriever = load_rag_pipeline()

st.success("Finance PDF Loaded Successfully")



question = st.text_input(
    "Ask Question from Finance PDF"
)



if question:

    if not groq_key:
        st.error("Please Enter Groq API Key")

    else:


        retrieved_chunks = hybrid_retriever.invoke(question)



        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=groq_key
        )



        combined_input = f"""
        Based on the following documents, answer the question.

        Question:
        {question}

        Documents:
        {chr(10).join([f"- {doc.page_content}" for doc in retrieved_chunks])}

        Instructions:
        - Answer only from the provided documents.
        - If answer is not available, say:
          "I don't have enough information based on the provided documents."
        - Give detailed and clear explanation.
        """

        messages = [
            SystemMessage(
                content="You are a helpful finance assistant."
            ),
            HumanMessage(
                content=combined_input
            )
        ]



        with st.spinner("Generating Answer..."):
            result = llm.invoke(messages)



        st.subheader("Answer")
        st.write(result.content)



        st.subheader("Retrieved Chunks")

        for i, doc in enumerate(retrieved_chunks, 1):

            with st.expander(f"Chunk {i}"):

                st.write(doc.page_content)