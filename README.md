# 📄 Hybrid RAG-Based Document Question Answering System

An AI-powered document question-answering application that allows users to upload PDF documents and ask questions about their contents using **Retrieval-Augmented Generation (RAG)**.

The application combines **BM25 keyword retrieval, semantic vector search, and Large Language Models (LLMs)** to retrieve relevant information and generate context-aware answers grounded in uploaded documents.

Built with Python, Streamlit, LangChain, ChromaDB, Hugging Face Embeddings, and Groq, this project demonstrates how hybrid information retrieval can improve document-based question answering.

## 🚀 Live Application

**Try the application:** Add your Streamlit Community Cloud URL here after deployment.

**GitHub Repository:** https://github.com/shivaalle7/Hybrid_RAG_Finance

---

## ✨ Features

- **📤 PDF Upload:** Upload one or multiple PDF documents directly through the web interface.
- **🔍 Hybrid Retrieval:** Combines BM25 keyword search and semantic vector search.
- **🧠 Semantic Search:** Uses Hugging Face sentence embeddings to retrieve contextually relevant passages.
- **📚 Vector Database:** Uses ChromaDB to index document chunks and support similarity search.
- **🤖 AI-Powered Answers:** Uses Groq-hosted LLMs to generate answers based on retrieved document content.
- **💬 Interactive Chat:** Ask questions and follow up on previously asked questions within the active session.
- **📑 Source Attribution:** Preserves document filenames and page metadata to help identify supporting information.
- **🛡️ Grounded Responses:** Instructs the LLM to avoid unsupported claims and acknowledge when information cannot be found.
- **☁️ Cloud Deployment:** Designed for deployment on Streamlit Community Cloud.
- **🔐 API Key Management:** Supports environment variables and Streamlit secrets for API key configuration.

## 🏗️ System Architecture

The application follows a Retrieval-Augmented Generation pipeline.

```text
                 USER
                   |
                   v
          Upload PDF Documents
                   |
                   v
          PDF Text Extraction
              PyPDFLoader
                   |
                   v
          Text Chunking
    RecursiveCharacterTextSplitter
                   |
                   v
          Document Processing
                   |
          +--------+---------+
          |                  |
          v                  v
   BM25 Keyword Search   Hugging Face
                         Embeddings
                              |
                              v
                         ChromaDB
                     Vector Similarity
          |                  |
          +--------+---------+
                   |
                   v
             Hybrid Retrieval
           EnsembleRetriever
                   |
                   v
          Relevant Document Chunks
                   |
                   v
             Groq-hosted LLM
                   |
                   v
          Context-grounded Answer
                   |
                   v
            Streamlit Chat UI
```

### How It Works

**1. Document Upload**

Users upload PDF documents through the Streamlit interface.

**2. Text Extraction**

PyPDFLoader extracts text and page metadata from the uploaded documents.

**3. Text Chunking**

RecursiveCharacterTextSplitter divides extracted text into smaller chunks to make retrieval more effective.

**4. Embedding Generation**

The `sentence-transformers/all-MiniLM-L6-v2` model converts document chunks into vector representations.

**5. Indexing**

ChromaDB stores the embeddings and supports semantic similarity search. BM25 builds a keyword-based retrieval index over the document chunks.

**6. Hybrid Retrieval**

EnsembleRetriever combines the results of BM25 and semantic retrieval to identify relevant passages for a user's question.

**7. Answer Generation**

A Groq-hosted LLM receives the question and retrieved context to generate a natural-language response.

**8. Response Delivery**

The application displays the answer through Streamlit. Document metadata can be used to identify the source and page associated with retrieved passages.

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Programming Language | Python | Application development |
| Web Framework | Streamlit | Interactive web interface |
| RAG Framework | LangChain | Document processing and retrieval orchestration |
| PDF Processing | PyPDFLoader / pypdf | PDF text extraction |
| Text Chunking | RecursiveCharacterTextSplitter | Document segmentation |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` | Semantic embeddings |
| Vector Database | ChromaDB | Vector storage and similarity search |
| Keyword Retrieval | BM25 | Lexical information retrieval |
| Hybrid Retrieval | EnsembleRetriever | Combines keyword and semantic retrieval |
| LLM Provider | Groq | Context-based answer generation |
| LLM Model | `openai/gpt-oss-120b` | Language generation |
| Configuration | python-dotenv / Streamlit Secrets | API key configuration |
| Deployment | Streamlit Community Cloud | Cloud hosting |

---

## 📁 Project Structure

```text
Hybrid_RAG_Finance/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── .streamlit/
    └── secrets.toml
```

**File descriptions**

- `app.py` — Main application, document processing, retrieval pipeline, and chat interface.
- `requirements.txt` — Python dependencies.
- `README.md` — Project documentation.
- `.gitignore` — Excludes sensitive and unnecessary files from version control.
- `.streamlit/secrets.toml` — Local configuration for secrets; should never be committed to GitHub.

The `.streamlit/secrets.toml` file is optional when using environment variables instead.

---

## ⚙️ Installation and Setup

### Prerequisites

Ensure the following are installed:

- Python 3.11 or another Python version compatible with your dependencies.
- Git.
- A Groq API key.
- Internet access for API requests and initial model downloads.

### Step 1: Clone the Repository

```bash
git clone https://github.com/shivaalle7/Hybrid_RAG_Finance.git
```

Navigate into the project directory:

```bash
cd Hybrid_RAG_Finance
```

### Step 2: Create a Virtual Environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure the Groq API Key

Create a `.streamlit` directory in the project root if it does not already exist.

Create `.streamlit/secrets.toml` and add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Replace the placeholder with your actual API key.

Alternatively, create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

If you use `.env`, ensure your application calls `load_dotenv()` before reading environment variables.

**Security:** Never commit your `.env` file, API keys, or `.streamlit/secrets.toml` to GitHub.

### Step 5: Run the Application

```bash
streamlit run app.py --server.fileWatcherType none
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

### Step 6: Upload and Query Documents

1. Open the application.
2. Upload one or more PDF documents.
3. Click **Process Documents**.
4. Wait for the indexing process to complete.
5. Enter a question about the uploaded documents.
6. Review the generated answer and verify it against the source PDF.
7. Ask additional questions about the indexed documents.

---

## ☁️ Deployment on Streamlit Community Cloud

The application can be deployed using Streamlit Community Cloud.

### Deployment Steps

1. Push the project source code to your GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in with GitHub.
4. Create a new application.
5. Select the `Hybrid_RAG_Finance` repository.
6. Select the `main` branch.
7. Set the main file path to `app.py`.
8. Add your Groq API key under the application's **Secrets** settings.

Configure the secret as follows:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

9. Deploy the application.

Once deployment completes, Streamlit will provide a public application URL.

**Important deployment considerations:**

- Do not upload API keys or secret files to GitHub.
- The embedding model may need to download during the first launch.
- Large PDFs can increase processing time and memory consumption.
- Local filesystem data may not persist indefinitely in the cloud environment.
- Uploaded document indexes should not be assumed to survive application restarts.
- Test document isolation and concurrent sessions before inviting multiple users.

Official deployment documentation: [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

---

## 🧪 Evaluation Strategy

The application should be evaluated using a verified dataset of questions, reference answers, and supporting document passages.

### 1. Retrieval Evaluation

Compare BM25, semantic search, and hybrid retrieval using the same test questions.

Recommended metrics:

- **Precision@K:** Proportion of retrieved passages that are relevant.
- **Recall@K:** Proportion of relevant passages successfully retrieved.
- **Mean Reciprocal Rank (MRR):** Ranking quality of the first relevant passage.

### 2. Answer Evaluation

Evaluate generated answers against verified reference answers and supporting document evidence.

Recommended metrics:

- Answer correctness.
- Faithfulness to retrieved context.
- Citation accuracy.
- Appropriate abstention when the answer is unavailable.

RAGAS can be considered for systematic evaluation of retrieval and generation quality.

### 3. Performance Evaluation

Measure:

- Document indexing time.
- Retrieval latency.
- End-to-end response latency.
- Memory usage during document processing.
- Behavior with multiple PDFs and concurrent sessions.

**Note:** No benchmark scores are claimed here. Quantitative results should be added only after running the evaluation and recording the measurements.

---

## ⚠️ Limitations

- PDF extraction quality depends on document structure and text accessibility.
- Scanned PDFs may require OCR.
- Complex tables, charts, and multi-column layouts may not be extracted reliably.
- Retrieval quality depends on chunk size, overlap, embedding quality, and retrieval configuration.
- LLM-generated answers may contain unsupported claims despite grounding instructions.
- Numerical calculations should be independently verified when accuracy is critical.
- Uploaded document indexes and chat history may be lost when a session resets.
- The current architecture is a prototype and does not provide a complete production security or multi-tenant storage system.

## 🔮 Future Enhancements

- Add a formal retrieval and answer evaluation pipeline.
- Implement a verified benchmark dataset.
- Add cross-encoder reranking to improve retrieval relevance.
- Introduce table-aware PDF extraction and OCR support.
- Improve source citations with clickable page references and retrieved passage previews.
- Add document deletion and index-reset controls.
- Support additional document formats.
- Optimize indexing and retrieval for large documents.
- Improve session management and error handling.
- Add authentication and stronger multi-user isolation.
- Introduce structured numerical verification for financial questions.
- Add automated regression testing to track quality improvements.

---

## 🎯 Use Cases

- Financial report analysis.
- Annual report question answering.
- Research paper exploration.
- Academic document assistance.
- Business report analysis.
- Policy and regulatory document search.
- General PDF-based question answering.

The reliability of answers depends on the quality of the uploaded documents, extraction process, retrieved evidence, and language model.

## 👨‍💻 Author

**Shiva Alle**

GitHub: [shivaalle7](https://github.com/shivaalle7)

---

## 📜 Disclaimer

This project is intended for educational, research, and document-analysis purposes. It does not replace professional financial, legal, or other expert advice.

Always verify important information against the original source documents.
