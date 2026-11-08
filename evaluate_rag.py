import argparse
import json
import os
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 8

# page numbers are PDF page numbers (1-based), not printed report page labels.
TEST_CASES = [
    {
        "id": "Q01",
        "question": "What was India's estimated real GDP growth for FY 2025-26?",
        "reference_answer": "7.4%",
        "evidence_terms": ["real GDP", "7.4", "2025-26"],
        "expected_pdf_pages": [9],
    },
    {
        "id": "Q02",
        "question": "What was the estimated real GVA growth for FY 2025-26?",
        "reference_answer": "7.3%",
        "evidence_terms": ["real GVA", "7.3", "2025-26"],
        "expected_pdf_pages": [9],
    },
    {
        "id": "Q03",
        "question": "What was India's estimated nominal GDP growth for FY 2025-26?",
        "reference_answer": "8.0%",
        "evidence_terms": ["nominal GDP", "8.0", "2025-26"],
        "expected_pdf_pages": [9],
    },
    {
        "id": "Q04",
        "question": "What was the estimated growth rate of the services sector in FY 2025-26?",
        "reference_answer": "9.1%",
        "evidence_terms": ["services", "9.1", "2025-26"],
        "expected_pdf_pages": [10],
    },
    {
        "id": "Q05",
        "question": "What was India's foodgrain production in 2024-25?",
        "reference_answer": "3,577.32 lakh metric tonnes (LMT)",
        "evidence_terms": ["foodgrain production", "3577.32", "2024-25"],
        "expected_pdf_pages": [10],
    },
    {
        "id": "Q06",
        "question": "What was retail inflation in FY 2025-26 from April to December?",
        "reference_answer": "1.7%",
        "evidence_terms": ["retail inflation", "1.7", "April-Dec"],
        "expected_pdf_pages": [14],
    },
    {
        "id": "Q07",
        "question": "What was the GNPA ratio of Scheduled Commercial Banks at the end of September 2025?",
        "reference_answer": "2.2%",
        "evidence_terms": ["GNPA", "2.2", "September 2025"],
        "expected_pdf_pages": [15],
    },
    {
        "id": "Q08",
        "question": "How much was mobilised from India's primary markets during FY 2025-26 through December 2025?",
        "reference_answer": "₹10.7 lakh crore",
        "evidence_terms": ["primary markets", "10.7 lakh crore", "December 2025"],
        "expected_pdf_pages": [16],
    },
    {
        "id": "Q09",
        "question": "What was India's merchandise trade deficit during April-November FY 2025-26?",
        "reference_answer": "USD 223.3 billion",
        "evidence_terms": ["trade deficit", "223.3", "April-Nov"],
        "expected_pdf_pages": [17],
    },
    {
        "id": "Q10",
        "question": "How did the current account deficit change between April-September FY 2024-25 and FY 2025-26?",
        "reference_answer": "It declined from USD 25.3 billion to USD 15.0 billion.",
        "evidence_terms": ["current account deficit", "25.3", "15.0", "FY26"],
        "expected_pdf_pages": [17],
    },
    {
        "id": "Q11",
        "question": "Which sector had the highest estimated real growth in FY 2025-26 among agriculture, industry, and services?",
        "reference_answer": "Services, estimated to grow by 9.1%.",
        "evidence_terms": ["Agriculture", "Industry", "Services", "9.1", "3.1", "6.2"],
        "expected_pdf_pages": [10],
    },
    {
        "id": "Q12",
        "question": "What was India's GDP growth in FY 2030-31 according to this report?",
        "reference_answer": "The report does not provide this figure in the evaluated passage; the system should say it cannot find the answer if no supporting passage is retrieved.",
        "evidence_terms": ["FY 2030-31", "GDP growth"],
        "expected_pdf_pages": [],
        "unanswerable": True,
    },
]

def normalized(text):
    return re.sub(r"\s+", " ", (text or "").lower()).replace("–", "-").replace("—", "-")

def evidence_match(doc_text, terms):
    text = normalized(doc_text)
    # A chunk counts as evidence only when it contains the core identifying term
    # and at least one additional evidence term; this remains a heuristic.
    normalized_terms = [normalized(t) for t in terms]
    if not normalized_terms:
        return False
    core_present = normalized_terms[0] in text
    other_hits = sum(1 for t in normalized_terms[1:] if t in text)
    return core_present and other_hits >= min(1, len(normalized_terms) - 1)

def page_number(doc):
    try:
        return int(doc.metadata.get("page", -1)) + 1
    except (TypeError, ValueError):
        return -1

def build_retrievers(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # Use a unique collection name for this run.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="rag_eval_" + str(int(time.time() * 1000)),
    )
    semantic = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = TOP_K
    hybrid = EnsembleRetriever(retrievers=[bm25, semantic], weights=[0.5, 0.5])
    return {"BM25": bm25, "Semantic": semantic, "Hybrid": hybrid}, vectorstore

def generate_answer(question, docs, api_key, model):
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage

    context = "\n\n".join(
        f"Source: {d.metadata.get('source', 'Unknown')}\n"
        f"Page: {page_number(d)}\nContent: {d.page_content}"
        for d in docs
    )
    llm = ChatGroq(model=model, groq_api_key=api_key, temperature=0)
    response = llm.invoke([
        SystemMessage(content=(
            "Answer only from the supplied document context. Do not invent facts. "
            "If the answer is not present, clearly say you could not find it. "
            "Cite the source filename and PDF page number."
        )),
        HumanMessage(content=f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION:\n{question}")
    ])
    return response.content

def main():
    global TOP_K
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to the PDF report")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--generate-answers", action="store_true")
    parser.add_argument("--groq-model", default="openai/gpt-oss-120b")
    parser.add_argument("--output", default="rag_evaluation_results.json")
    args = parser.parse_args()
    TOP_K = args.top_k

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    started = time.perf_counter()
    pages = PyPDFLoader(str(pdf_path)).load()
    for page in pages:
        page.metadata["source"] = pdf_path.name
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)
    indexing_seconds = time.perf_counter() - started
    print(f"Loaded {len(pages)} pages; created {len(chunks)} chunks.")
    print(f"PDF extraction + chunking: {indexing_seconds:.2f}s")

    build_started = time.perf_counter()
    retrievers, vectorstore = build_retrievers(chunks)
    indexing_seconds += time.perf_counter() - build_started
    print(f"Index creation completed. Total preprocessing/indexing: {indexing_seconds:.2f}s")

    api_key = os.getenv("GROQ_API_KEY", "")
    results = {
        "pdf": pdf_path.name,
        "pages": len(pages),
        "chunks": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "indexing_seconds": round(indexing_seconds, 3),
        "metric_note": (
            "Evidence/page metrics are heuristic proxies, not gold-labelled "
            "Precision@K or Recall@K."
        ),
        "retrievers": {},
    }

    for name, retriever in retrievers.items():
        per_question = []
        latencies = []
        evidence_hits = 0
        expected_page_hits = 0
        first_evidence_ranks = []
        answer_outputs = []

        for case in TEST_CASES:
            t0 = time.perf_counter()
            docs = retriever.invoke(case["question"])
            latency = time.perf_counter() - t0
            latencies.append(latency)

            matches = [
                i + 1 for i, doc in enumerate(docs)
                if evidence_match(doc.page_content, case["evidence_terms"])
            ]
            expected_pages = set(case["expected_pdf_pages"])
            retrieved_pages = [page_number(d) for d in docs]
            page_hit = bool(expected_pages.intersection(retrieved_pages))
            evidence_hit = bool(matches)

            if evidence_hit:
                evidence_hits += 1
                first_evidence_ranks.append(matches[0])
            if page_hit:
                expected_page_hits += 1

            row = {
                "id": case["id"],
                "question": case["question"],
                "reference_answer": case["reference_answer"],
                "unanswerable": case.get("unanswerable", False),
                "retrieval_latency_seconds": round(latency, 4),
                "retrieved_pages": retrieved_pages,
                "evidence_hit_heuristic": evidence_hit,
                "expected_page_hit": page_hit if expected_pages else None,
                "first_evidence_rank": matches[0] if matches else None,
                "retrieved_chunks": [
                    {
                        "rank": i + 1,
                        "page": page_number(doc),
                        "source": doc.metadata.get("source", "Unknown"),
                        "content_preview": doc.page_content[:350],
                    }
                    for i, doc in enumerate(docs)
                ],
            }

            if args.generate_answers:
                if not api_key:
                    raise SystemExit(
                        "--generate-answers requires the GROQ_API_KEY environment variable."
                    )
                row["generated_answer"] = generate_answer(
                    case["question"], docs, api_key, args.groq_model
                )
            per_question.append(row)

        results["retrievers"][name] = {
            "questions": len(TEST_CASES),
            "evidence_hit_rate_heuristic": round(evidence_hits / len(TEST_CASES), 4),
            "expected_page_hit_rate_answerable_questions": round(
                expected_page_hits / sum(bool(c["expected_pdf_pages"]) for c in TEST_CASES), 4
            ),
            "mean_first_evidence_rank": (
                round(statistics.mean(first_evidence_ranks), 3)
                if first_evidence_ranks else None
            ),
            "mean_retrieval_latency_seconds": round(statistics.mean(latencies), 4),
            "per_question": per_question,
        }
        print(
            f"{name}: evidence-hit proxy={evidence_hits}/{len(TEST_CASES)}, "
            f"page-hit={expected_page_hits}/"
            f"{sum(bool(c['expected_pdf_pages']) for c in TEST_CASES)}, "
            f"mean retrieval={statistics.mean(latencies):.4f}s"
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved detailed results to: {args.output}")
    print("Review the retrieved chunk previews before interpreting proxy scores.")

    # Best-effort cleanup of the temporary Chroma collection.
    try:
        vectorstore.delete_collection()
    except Exception:
        pass

if __name__ == "__main__":
    main()
