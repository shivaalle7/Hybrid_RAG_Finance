# Finance PDF RAG Chatbot

A Hybrid Retrieval-Augmented Generation (RAG) application that allows users to ask questions about financial annual reports and receive document-grounded answers using Large Language Models.

## Project Overview

The Finance PDF RAG Chatbot combines semantic search and keyword-based retrieval to find relevant information from a financial report. The retrieved passages are passed to a Groq-hosted LLM to generate clear, context-based responses.

## Features

- PDF document loading and text extraction
- Text chunking with overlapping segments
- Semantic search using Hugging Face embeddings
- Vector storage and retrieval with ChromaDB
- Keyword search using BM25
- Hybrid retrieval using EnsembleRetriever
- AI-generated responses using Groq
- Retrieved text passages displayed for transparency
- Interactive web interface built with Streamlit

## Tech Stack

- **Language:** Python
- **Frontend:** Streamlit
- **LLM:** Groq — Llama 3.3 70B Versatile
- **Embeddings:** Sentence Transformers — all-MiniLM-L6-v2
- **Vector Database:** ChromaDB
- **Retrieval:** Semantic Search, BM25, EnsembleRetriever
- **Framework:** LangChain

## Architecture

1. Load the financial annual report PDF.
2. Extract document text using PyPDFLoader.
3. Split the extracted text into chunks with overlap.
4. Generate embeddings using Hugging Face.
5. Store document chunks in ChromaDB.
6. Retrieve relevant passages using vector search and BM25.
7. Combine retrieval results using weighted ensemble retrieval.
8. Send the question and retrieved passages to the Groq LLM.
9. Display the answer and retrieved passages in Streamlit.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/finance-pdf-rag-chatbot.git
cd finance-pdf-rag-chatbot
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the PDF

Place your authorized financial annual report PDF in the project root directory.

The current application expects this exact filename:

`FINAL ANNUAL REPORT ENGLISH with cover (2).pdf`

Alternatively, modify the application to support PDF uploads.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, enter your Groq API key in the sidebar, and start asking questions about the report.

## Example Questions

- What was the total revenue reported for FY 2023–24?
- What is the company's net profit for the reporting period?
- How well did the company perform financially during the year?
- Is the company financially stable based on its reported results?
- How did revenue, expenses, and net profit affect the company's overall financial performance?
- What do the company's assets, liabilities, and cash flow indicate about its financial stability?

The chatbot can answer only to the extent that the relevant information is present in the retrieved document passages.

## Security and Limitations

- Never commit API keys or credentials.
- Use financial documents you are authorized to process and share.
- The quality of answers depends on document extraction, retrieval quality, and the language model.
- This application is a document question-answering tool, not a substitute for professional financial advice.

## Future Improvements

- Allow PDF uploads through the interface.
- Support multiple financial documents.
- Display page numbers and source citations.
- Add conversation history.
- Deploy the application using Streamlit Community Cloud or another hosting platform.

## Author

Developed as a Hybrid RAG-Based Document Question Answering project using Python, LangChain, ChromaDB, Hugging Face Embeddings, BM25, and Groq.
