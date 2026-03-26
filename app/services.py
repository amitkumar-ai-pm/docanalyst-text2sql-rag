import re
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores.upstash import UpstashVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

ANALYTICS_PATTERNS = [
    r"\bmax\b",
    r"\bmaximum\b",
    r"\bmin\b",
    r"\bminimum\b",
    r"\bavg\b",
    r"\baverage\b",
    r"\bmean\b",
    r"\bsum\b",
    r"\bcount\b",
    r"\btop\b",
    r"\blowest\b",
    r"\bhighest\b",
    r"\bgroup by\b",
    r"\bhow many\b",
    r"\bunchanged\b",
    r"\bnot changed\b",
    r"\bstable\b",
    r"\bconstant\b",
]


def create_store(namespace: str | None = None) -> UpstashVectorStore:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return UpstashVectorStore(embedding=embeddings, namespace=namespace)


def create_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0)


def is_analytics_query(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in ANALYTICS_PATTERNS)


def answer_analytics(question: str, csv_path: str) -> str:
    q = question.lower()
    if (
        "unchanged" in q
        or "not changed" in q
        or "stable" in q
        or "constant" in q
    ):
        df = pd.read_csv(csv_path, usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"])

        # Price stability is evaluated at (store_id, item_id) level across weeks.
        grouped = (
            df.groupby(["store_id", "item_id"])
            .agg(
                weeks=("wm_yr_wk", "count"),
                unique_prices=("sell_price", "nunique"),
                min_price=("sell_price", "min"),
                max_price=("sell_price", "max"),
            )
            .reset_index()
        )

        stable = grouped[grouped["unique_prices"] == 1].sort_values("weeks", ascending=False)
        top = stable.head(20)

        if top.empty:
            return "No fully unchanged-price items were found in the evaluated data."

        preview = top.to_string(index=False)
        return (
            f"Found {len(stable)} (store_id, item_id) combinations with unchanged prices.\n"
            f"Top 20 by number of observed weeks:\n{preview}"
        )

    if "highest" in q or "max" in q or "maximum" in q:
        df = pd.read_csv(csv_path, usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"])
        max_price = df["sell_price"].max()
        rows = df[df["sell_price"] == max_price].sort_values(["item_id", "store_id", "wm_yr_wk"])
        preview = rows.head(10).to_string(index=False)
        return (
            f"Global max sell_price is {max_price}.\n"
            f"Rows with that max (first 10):\n{preview}\n"
            f"Total rows with max: {len(rows)}"
        )
    return "Analytics query detected, but this operation is not yet implemented."


def answer_rag(question: str, store: UpstashVectorStore, llm: ChatOpenAI, k: int = 4) -> dict[str, Any]:
    hits = store.similarity_search_with_score(question, k=k)
    context = "\n\n".join([f"[{i+1}] {doc.page_content}" for i, (doc, _) in enumerate(hits)])

    prompt = f"""
You are a document assistant.
Answer ONLY from the provided context.
If the answer is not in context, say: "I don't know based on the indexed documents."
When possible, reference citation numbers like [1], [2].

Context:
{context}

Question:
{question}
"""
    answer = llm.invoke(prompt).content
    citations = []
    for i, (doc, score) in enumerate(hits, 1):
        md = doc.metadata or {}
        citations.append(
            {
                "rank": i,
                "score": float(score),
                "source": md.get("source"),
                "chunk_index": md.get("chunk_index"),
                "start_row_estimate": md.get("start_row_estimate"),
                "rows_in_chunk": md.get("rows_in_chunk"),
                "snippet": doc.page_content[:180].replace("\n", " "),
            }
        )
    return {"answer": answer, "citations": citations}


def build_rag_debug(question: str, citations: list[dict[str, Any]], k: int = 5) -> dict[str, Any]:
    """Optional deep RAG debug payload with embeddings and cosine similarity."""
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    question_embedding = emb.embed_query(question)

    top = citations[:k]
    snippets = [c.get("snippet", "") for c in top]
    if snippets:
        snippet_embeddings = emb.embed_documents(snippets)
    else:
        snippet_embeddings = []

    q = np.array(question_embedding, dtype=float)
    qn = np.linalg.norm(q) + 1e-12
    matches = []
    for c, e in zip(top, snippet_embeddings):
        v = np.array(e, dtype=float)
        vn = np.linalg.norm(v) + 1e-12
        cosine = float(np.dot(q, v) / (qn * vn))
        matches.append(
            {
                "rank": c.get("rank"),
                "score": c.get("score"),
                "source": c.get("source"),
                "chunk_index": c.get("chunk_index"),
                "snippet": c.get("snippet"),
                "snippet_embedding": e,
                "cosine_with_question": cosine,
            }
        )

    return {
        "question_embedding": question_embedding,
        "top_k": len(matches),
        "matches": matches,
    }


def ask(question: str, csv_path: str, store: UpstashVectorStore, llm: ChatOpenAI) -> dict[str, Any]:
    if is_analytics_query(question):
        return {
            "route": "analytics",
            "answer": answer_analytics(question, csv_path),
            "citations": [],
        }
    rag = answer_rag(question, store, llm)
    return {
        "route": "rag",
        "answer": rag["answer"],
        "citations": rag["citations"],
    }


def ingest_csv(
    csv_path: str,
    store: UpstashVectorStore,
    *,
    max_source_rows: int = 100_000,
    chunk_read_size: int = 5_000,
    target_chars: int = 4_000,
    max_rows_per_doc: int = 100,
    batch_size: int = 200,
) -> dict[str, Any]:
    docs: list[Document] = []
    processed_rows = 0
    doc_index = 0

    buffer: list[str] = []
    buffer_rows = 0
    buffer_chars = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunk_read_size):
        chunk = chunk.fillna("")
        for _, row in chunk.iterrows():
            if processed_rows >= max_source_rows:
                break

            row_text = " | ".join([f"{col}: {row[col]}" for col in chunk.columns])
            should_flush = (
                (buffer_chars + len(row_text) > target_chars)
                or (buffer_rows >= max_rows_per_doc)
            )

            if should_flush and buffer:
                docs.append(
                    Document(
                        page_content="\n".join(buffer),
                        metadata={
                            "source": str(csv_path),
                            "file_type": "csv",
                            "chunk_index": doc_index,
                            "rows_in_chunk": buffer_rows,
                            "start_row_estimate": processed_rows - buffer_rows,
                        },
                    )
                )
                doc_index += 1
                buffer = []
                buffer_rows = 0
                buffer_chars = 0

            buffer.append(row_text)
            buffer_rows += 1
            buffer_chars += len(row_text)
            processed_rows += 1

        if processed_rows >= max_source_rows:
            break

    if buffer:
        docs.append(
            Document(
                page_content="\n".join(buffer),
                metadata={
                    "source": str(csv_path),
                    "file_type": "csv",
                    "chunk_index": doc_index,
                    "rows_in_chunk": buffer_rows,
                    "start_row_estimate": processed_rows - buffer_rows,
                },
            )
        )

    inserted = 0
    for start in range(0, len(docs), batch_size):
        end = min(start + batch_size, len(docs))
        batch = docs[start:end]
        ids = store.add_documents(batch)
        inserted += len(ids)

    return {
        "status": "ok",
        "processed_rows": processed_rows,
        "documents_built": len(docs),
        "vectors_inserted": inserted,
    }

