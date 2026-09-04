# Hybrid_RAG_Finance_documents_retrieval
An AI-powered question-answering system that combines vector search and BM25 retrieval to provide accurate, context-aware answers from financial documents using LangChain, embeddings, ChromaDB, and LLMs

## ▶️ Execution

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Hybrid-RAG-Financial-QA.git
cd Hybrid-RAG-Financial-QA
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

### 5. Open in Browser

After execution, Streamlit will provide a local URL such as:

```text
http://localhost:8501
```

Open the URL in your browser.

### 6. Ask Questions

Upload or use the financial document and enter questions such as:

```text
What was the company's total revenue?

What was the net income?

What are the company's major financial risks?

How did the company's revenue change?
```

The system retrieves relevant information using **Vector Search + BM25**, passes the retrieved context to the LLM, and generates the final answer.

