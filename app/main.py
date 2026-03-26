from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agent_graph import build_agent_graph
from app.config import load_env, validate_env
from app.observability import log_event
from app.services import ask, build_rag_debug, create_llm, create_store, ingest_csv
from time import perf_counter
import subprocess
import sys
from pathlib import Path

app = FastAPI(title="Hybrid RAG Document Q&A API")

_env_path = load_env()
_missing = validate_env()
if _missing:
    raise RuntimeError(f"Missing environment variables: {', '.join(_missing)}")

store = create_store()
llm = create_llm()
agent_app = build_agent_graph(llm, store)


def _rag_similarity_summary(citations: list[dict]) -> dict | None:
    if not citations:
        return None
    scores = [float(c.get("score", 0.0)) for c in citations if c.get("score") is not None]
    if not scores:
        return None
    return {
        "top_score": max(scores),
        "avg_score": sum(scores) / len(scores),
        "min_score": min(scores),
        "count": len(scores),
    }


class AskRequest(BaseModel):
    question: str
    csv_path: str
    namespace: str | None = None
    debug: bool = False


class IngestRequest(BaseModel):
    csv_path: str
    namespace: str | None = None
    max_source_rows: int = 100_000
    chunk_read_size: int = 5_000
    target_chars: int = 4_000
    max_rows_per_doc: int = 100
    batch_size: int = 200


class EvaluateRequest(BaseModel):
    api: str = "http://127.0.0.1:8000"
    csv_path: str
    namespace: str = "csv_sell_prices"
    questions_path: str = "eval_questions.json"
    debug: bool = True


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "env_loaded_from": str(_env_path)}


@app.post("/ask")
def ask_endpoint(payload: AskRequest) -> dict:
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    t0 = perf_counter()
    try:
        active_store = create_store(payload.namespace) if payload.namespace else store
        result = ask(payload.question, payload.csv_path, active_store, llm)
        result["rag_similarity"] = _rag_similarity_summary(result.get("citations", []))
        if payload.debug and result.get("route") == "rag":
            result["rag_debug"] = build_rag_debug(payload.question, result.get("citations", []), k=5)
        log_event(
            {
                "endpoint": "/ask",
                "question": payload.question,
                "csv_path": payload.csv_path,
                "namespace": payload.namespace,
                "route": result.get("route"),
                "answer": result.get("answer"),
                "rag_similarity": result.get("rag_similarity"),
                "sql_query": result.get("sql_query"),
                "sql_error": result.get("sql_error"),
                "citations_count": len(result.get("citations", [])),
                "latency_ms": round((perf_counter() - t0) * 1000, 2),
                "success": True,
                "debug_payload": result.get("rag_debug"),
            }
        )
        return result
    except FileNotFoundError:
        log_event(
            {
                "endpoint": "/ask",
                "question": payload.question,
                "csv_path": payload.csv_path,
                "namespace": payload.namespace,
                "latency_ms": round((perf_counter() - t0) * 1000, 2),
                "success": False,
                "error_message": f"CSV file not found: {payload.csv_path}",
            }
        )
        raise HTTPException(status_code=404, detail=f"CSV file not found: {payload.csv_path}")
    except Exception as e:
        log_event(
            {
                "endpoint": "/ask",
                "question": payload.question,
                "csv_path": payload.csv_path,
                "namespace": payload.namespace,
                "latency_ms": round((perf_counter() - t0) * 1000, 2),
                "success": False,
                "error_message": str(e),
            }
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/agent")
def ask_agent_endpoint(payload: AskRequest) -> dict:
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    t0 = perf_counter()
    try:
        # For namespace-specific RAG retrieval, build a temporary graph with the namespace store.
        if payload.namespace:
            ns_store = create_store(payload.namespace)
            ns_agent = build_agent_graph(llm, ns_store)
            result = ns_agent.invoke({"question": payload.question, "csv_path": payload.csv_path})
        else:
            result = agent_app.invoke({"question": payload.question, "csv_path": payload.csv_path})

        output = {
            "route": result.get("route", "unknown"),
            "route_reason": result.get("route_reason", ""),
            "answer": result.get("final_answer", ""),
            "sql_query": result.get("sql_query"),
            "sql_rows_preview": (result.get("sql_rows") or [])[:20],
            "citations": result.get("citations", []),
            "rag_similarity": _rag_similarity_summary(result.get("citations", [])),
            "sql_error": result.get("sql_error"),
        }
        if payload.debug and output.get("route") == "rag":
            output["rag_debug"] = build_rag_debug(payload.question, output.get("citations", []), k=5)
        log_event(
            {
                "endpoint": "/ask/agent",
                "question": payload.question,
                "csv_path": payload.csv_path,
                "namespace": payload.namespace,
                "route": output.get("route"),
                "route_reason": output.get("route_reason"),
                "answer": output.get("answer"),
                "rag_similarity": output.get("rag_similarity"),
                "sql_query": output.get("sql_query"),
                "sql_error": output.get("sql_error"),
                "citations_count": len(output.get("citations", [])),
                "latency_ms": round((perf_counter() - t0) * 1000, 2),
                "success": True,
                "debug_payload": output.get("rag_debug"),
            }
        )
        return output
    except FileNotFoundError:
        log_event(
            {
                "endpoint": "/ask/agent",
                "question": payload.question,
                "csv_path": payload.csv_path,
                "namespace": payload.namespace,
                "latency_ms": round((perf_counter() - t0) * 1000, 2),
                "success": False,
                "error_message": f"CSV file not found: {payload.csv_path}",
            }
        )
        raise HTTPException(status_code=404, detail=f"CSV file not found: {payload.csv_path}")
    except Exception as e:
        log_event(
            {
                "endpoint": "/ask/agent",
                "question": payload.question,
                "csv_path": payload.csv_path,
                "namespace": payload.namespace,
                "latency_ms": round((perf_counter() - t0) * 1000, 2),
                "success": False,
                "error_message": str(e),
            }
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/csv")
def ingest_csv_endpoint(payload: IngestRequest) -> dict:
    t0 = perf_counter()
    try:
        active_store = create_store(payload.namespace) if payload.namespace else store
        result = ingest_csv(
            payload.csv_path,
            active_store,
            max_source_rows=payload.max_source_rows,
            chunk_read_size=payload.chunk_read_size,
            target_chars=payload.target_chars,
            max_rows_per_doc=payload.max_rows_per_doc,
            batch_size=payload.batch_size,
        )
        log_event(
            {
                "endpoint": "/ingest/csv",
                "csv_path": payload.csv_path,
                "namespace": payload.namespace,
                "answer": str(result),
                "latency_ms": round((perf_counter() - t0) * 1000, 2),
                "success": True,
            }
        )
        return result
    except FileNotFoundError:
        log_event(
            {
                "endpoint": "/ingest/csv",
                "csv_path": payload.csv_path,
                "namespace": payload.namespace,
                "latency_ms": round((perf_counter() - t0) * 1000, 2),
                "success": False,
                "error_message": f"CSV file not found: {payload.csv_path}",
            }
        )
        raise HTTPException(status_code=404, detail=f"CSV file not found: {payload.csv_path}")
    except Exception as e:
        log_event(
            {
                "endpoint": "/ingest/csv",
                "csv_path": payload.csv_path,
                "namespace": payload.namespace,
                "latency_ms": round((perf_counter() - t0) * 1000, 2),
                "success": False,
                "error_message": str(e),
            }
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/run")
def evaluate_run_endpoint(payload: EvaluateRequest) -> dict:
    t0 = perf_counter()
    try:
        project_root = Path(__file__).resolve().parents[1]
        cmd = [
            sys.executable,
            "evaluation.py",
            "--api",
            payload.api,
            "--csv",
            payload.csv_path,
            "--questions",
            payload.questions_path,
            "--namespace",
            payload.namespace,
        ]
        if payload.debug:
            cmd.append("--debug")

        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
        )
        success = result.returncode == 0
        log_event(
            {
                "endpoint": "/evaluate/run",
                "csv_path": payload.csv_path,
                "namespace": payload.namespace,
                "answer": result.stdout[-3000:] if result.stdout else "",
                "error_message": result.stderr[-3000:] if result.stderr else None,
                "latency_ms": round((perf_counter() - t0) * 1000, 2),
                "success": success,
            }
        )
        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Evaluation failed. stderr: {result.stderr[-1000:]}",
            )
        return {"status": "ok", "stdout": result.stdout, "stderr": result.stderr}
    except HTTPException:
        raise
    except Exception as e:
        log_event(
            {
                "endpoint": "/evaluate/run",
                "csv_path": payload.csv_path,
                "namespace": payload.namespace,
                "latency_ms": round((perf_counter() - t0) * 1000, 2),
                "success": False,
                "error_message": str(e),
            }
        )
        raise HTTPException(status_code=500, detail=str(e))

