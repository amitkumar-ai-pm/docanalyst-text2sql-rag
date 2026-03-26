# RAG-based Document Q&A - Product and Technical Design

## 1) Product Overview

This project builds a **Document Q&A assistant** that answers user questions from uploaded documents.

Current focus:
- Local experimentation in `docs_assistant.ipynb`
- Data source: large CSV (M5 `sell_prices.csv`)
- Vector storage: **Upstash Vector**
- LLM + embeddings: **OpenAI** via LangChain

Target capability:
- Upload document(s)
- Ask questions
- Receive grounded answers with citations
- Handle both semantic Q&A and exact numeric questions

## Technical Architecture Diagram

![docanalyst_text2sql+RAG architecture](docanalyst_architecture.png)

---

## 2) Problem and Key Insight

For very large tabular files (for example, 6.8M rows), pure vector retrieval is not enough for exact analytical questions.

Example:
- Vector-only answer for "highest sell price" can be incorrect (semantic nearest chunks, not global aggregation).
- Exact analytics with pandas gave the true answer: `107.32`.

Therefore, the solution uses a **hybrid strategy**:
- **RAG path** for semantic/document-style questions.
- **Analytics path** for exact aggregations (max/min/avg/sum/count/group-by).

---

## 3) Current Stack

### Core
- Python 3.14 (`.venv`)
- Jupyter Notebook (`docs_assistant.ipynb`)
- LangChain ecosystem:
  - `langchain-openai`
  - `langchain-community`
  - `langchain-core`
  - `langchain-text-splitters`

### Models and Stores
- LLM: `ChatOpenAI(model="gpt-4.1-mini", temperature=0)`
- Embeddings: `OpenAIEmbeddings(model="text-embedding-3-small")`
- Vector DB: `UpstashVectorStore`

### Data Libraries
- `pandas` for exact analytics and chunked CSV reading

---

## 4) Environment Variables

The project expects these in `.env`:

```env
OPENAI_API_KEY=...
UPSTASH_VECTOR_REST_URL=...
UPSTASH_VECTOR_REST_TOKEN=...
```

Notebook startup loads `.env` and validates keys are present.

---

## 5) Upstash Vector Configuration

Recommended and currently used:
- Type: `Dense`
- Dimensions: `1536`
- Metric: `COSINE`
- Embedding model pairing: `text-embedding-3-small`

This dimension/model pairing must match, otherwise vector inserts/queries fail or degrade.

---

## 6) Data Ingestion Approach (Large CSV)

The dataset is large (~194MB, ~6.84M rows), so ingestion is memory-safe and cost-aware:

1. Read CSV in chunks (`pd.read_csv(..., chunksize=...)`)
2. Convert rows into normalized text (`col: value | col: value ...`)
3. Buffer multiple rows into a single LangChain `Document`
4. Add metadata:
   - `source`
   - `file_type`
   - `chunk_index`
   - `rows_in_chunk`
   - `start_row_estimate`
5. Upsert to Upstash in **batches** with progress logs

Important tuning knobs:
- `max_source_rows`
- `chunk_read_size`
- `target_chars`
- `max_rows_per_doc`
- `batch_size` for upsert

---

## 7) Retrieval and Answering Flow

### RAG Flow
1. User asks question.
2. Retrieve top-k chunks with `similarity_search_with_score`.
3. Build context from retrieved chunks.
4. Prompt LLM to answer only from context.
5. Return answer + citation metadata (scores and chunk info).

### Citation Format
Current citation includes:
- Similarity score
- Source file path
- Chunk index
- Approximate start row
- Rows in chunk

This supports auditability and trust.

---

## 8) Hybrid Routing Strategy

### Route 1: Analytics (Exact)
Used when question includes terms like:
- max / maximum
- min / minimum
- average / mean / avg
- sum
- count / how many
- top / group by

Execution:
- pandas computation on CSV (exact)
- return deterministic numeric result

### Route 2: RAG (Semantic)
Used for questions like:
- trends, patterns, interpretation
- item behavior over time
- explanatory or context-heavy questions

Execution:
- vector retrieval + LLM answer with sources

### Unified Entry Point
`ask(question, csv_path, store, llm)`

Returns:
- `[route=analytics] ...` for exact math
- `[route=rag] ...` for semantic retrieval answers

---

## 9) What Is Working Today

- Upstash index connection established
- Smoke test insertion and retrieval validated
- Large CSV chunked document ingestion implemented
- Batched vector upsert implemented
- RAG answers generated with citations
- Exact analytics path verified (global max sell_price)
- Hybrid router implemented and tested

---

## 10) Known Constraints

- Vector search is not a replacement for full-table aggregation.
- Current analytics handler is strongest for max/highest pattern; more operations should be added.
- Notebook-first implementation is excellent for learning and prototyping, but API/service layer is needed for productization.

---

## 11) Recommended Next Steps

### A) Expand analytics handlers
Add exact implementations for:
- min / lowest
- average
- sum
- count
- group by store/item/week
- top N items by metric

### B) Improve RAG quality
- Add namespace strategy per dataset/file
- Add metadata filters during retrieval
- Add short source snippet in citation output
- Enforce strict citation policy in prompt

### C) Move to FastAPI
Create service endpoints:
- `POST /ingest/csv` (background ingestion)
- `POST /ask` (hybrid router)
- `GET /health`

Return structured JSON:
- `route`
- `answer`
- `citations`
- `debug` (optional)

### D) Production hardening
- Add retries and structured logging
- Add request IDs and observability
- Add auth and multi-user dataset isolation
- Add background worker for long ingestion jobs

---

## 12) Suggested Project Structure (Next Phase)

```text
ch01/
  docs_assistant.ipynb
  .env
  app/
    main.py
    config.py
    router.py
    rag.py
    analytics.py
    ingestion.py
    schemas.py
  data/
  tests/
  requirements.txt
  README.md
```

---

## 13) Core Design Principle

Use the right engine for the right question:
- **Exact numeric question -> exact computation**
- **Semantic/context question -> RAG retrieval + LLM**

This principle is the foundation of reliable Document Q&A at scale.

---

## 14) Technical Architecture Diagram (Mermaid)

```mermaid
flowchart TD
    U[User in Streamlit UI] --> Q[Question Input]
    U --> IC[Ingestion Controls]
    U --> EV[Evaluation Trigger]

    subgraph UI["Streamlit App (ui.py)"]
      Q --> T{Use LangGraph hybrid agent?}
      T -->|ON| A1[/POST /ask/agent/]
      T -->|OFF| A2[/POST /ask/]
      IC --> I1[/POST /ingest/csv/]
      EV --> E1[/POST /evaluate/run/]
    end

    subgraph API["FastAPI Backend (app/main.py)"]
      A1 --> G[LangGraph Agent App]
      A2 --> S[Simple Router ask()]
      I1 --> ING[ingest_csv()]
      E1 --> ESUB[Run evaluation.py subprocess]
    end

    subgraph Agent["LangGraph Agent (app/agent_graph.py)"]
      G --> N1[Node 1: Decide Tool]
      N1 -->|sql| N2A[Node 2A: SQL Node]
      N1 -->|rag| N2B[Node 2B: RAG Node]
      N1 -->|both| N2A
      N2A --> N2B
      N2A --> N3[Node 3: Compose Answer]
      N2B --> N3
    end

    subgraph SQLPath["Exact Analytics Path"]
      N2A --> VSQL[SQL Guardrails Validator<br/>read-only, allowlist, CTE-safe]
      VSQL --> DDB[DuckDB in-memory view: data<br/>read_csv_auto(csv_path)]
      DDB --> SQLR[SQL rows preview + sql_query + sql_error]
    end

    subgraph RAGPath["Semantic RAG Path"]
      N2B --> VS[Upstash Vector Store]
      VS --> RET[Top-k retrieval + similarity scores]
      RET --> LLM[ChatOpenAI]
      LLM --> RAGR[Answer + citations]
      RAGR --> DBG[RAG debug optional:<br/>question embedding + top-5 snippet embeddings + cosine]
    end

    subgraph Ingest["Ingestion Path"]
      ING --> CSV[Read CSV in chunks]
      CSV --> CH[Build text chunks/docs]
      CH --> EMB[OpenAIEmbeddings]
      EMB --> UP[Batch upsert vectors]
      UP --> VS
    end

    subgraph Eval["Evaluation + Regression Tracking"]
      ESUB --> EQ[eval_questions.json]
      ESUB --> CALL[/POST /ask/agent per test/]
      CALL --> COMP[Compare agent vs reference_sql]
      COMP --> ECSV[eval_results_*.csv<br/>correct/partial/incorrect + outputs]
    end

    subgraph Obs["Observability"]
      API --> LOG[log_event()]
      LOG --> DB[(observability/events.db<br/>latest 100 events)]
      LOG --> J[(observability/events.jsonl<br/>raw archive)]
      ECSV --> OPage[Observability Streamlit page]
      DB --> OPage
      J --> OPage
    end

    N3 --> RESP[Final response:<br/>route, reason, answer, sql info, citations, rag similarity]
    S --> RESP
    RESP --> UIOut[Rendered in Streamlit chat]
```

