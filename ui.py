import os

import requests
import streamlit as st

st.set_page_config(page_title="Hybrid RAG Doc Assistant", layout="wide")
st.title("Hybrid RAG Document Q&A")
st.caption("Analytics route for exact numeric queries, RAG route for semantic queries.")

api_base = st.text_input("FastAPI base URL", value="http://127.0.0.1:8000")
default_csv_path = os.getenv("DEFAULT_CSV_PATH", "data/sell_prices.csv")
csv_path = st.text_input("CSV path on backend machine", value=default_csv_path)
namespace = st.text_input("Upstash namespace (optional)", value="csv_sell_prices")
use_langgraph_agent = st.toggle("Use LangGraph hybrid agent", value=True)

with st.expander("Ingestion Controls"):
    max_source_rows = st.number_input("Max source rows", min_value=1_000, value=100_000, step=1_000)
    chunk_read_size = st.number_input("Chunk read size", min_value=500, value=5_000, step=500)
    target_chars = st.number_input("Target chars per document", min_value=1_000, value=4_000, step=500)
    max_rows_per_doc = st.number_input("Max rows per document", min_value=20, value=100, step=10)
    batch_size = st.number_input("Batch size", min_value=20, value=200, step=20)
    if st.button("Ingest CSV to Vector DB"):
        try:
            with st.spinner("Ingesting... this can take a while for large files."):
                ingest_resp = requests.post(
                    f"{api_base}/ingest/csv",
                    json={
                        "csv_path": csv_path,
                        "namespace": namespace or None,
                        "max_source_rows": int(max_source_rows),
                        "chunk_read_size": int(chunk_read_size),
                        "target_chars": int(target_chars),
                        "max_rows_per_doc": int(max_rows_per_doc),
                        "batch_size": int(batch_size),
                    },
                    timeout=1800,
                )
            ingest_resp.raise_for_status()
            st.success("Ingestion complete")
            st.json(ingest_resp.json())
        except Exception as e:
            st.error(f"Ingestion failed: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                resp = requests.post(
                    f"{api_base}/ask/agent" if use_langgraph_agent else f"{api_base}/ask",
                    json={"question": question, "csv_path": csv_path, "namespace": namespace or None},
                    timeout=180,
                )
            resp.raise_for_status()
            data = resp.json()
            answer_text = f"**Route:** `{data['route']}`\n\n{data['answer']}"
            if data.get("route_reason"):
                answer_text = f"**Route:** `{data['route']}`  \n**Reason:** {data['route_reason']}\n\n{data['answer']}"
            if data.get("rag_similarity"):
                s = data["rag_similarity"]
                answer_text = (
                    answer_text
                    + "\n\n"
                    + f"**RAG similarity:** top={s['top_score']:.4f}, avg={s['avg_score']:.4f}, "
                    + f"min={s['min_score']:.4f} (k={s['count']})"
                )
            st.markdown(answer_text)

            if data.get("sql_query"):
                with st.expander("SQL used"):
                    st.code(data["sql_query"], language="sql")
                    if data.get("sql_rows_preview"):
                        st.write(data["sql_rows_preview"])

            if data.get("sql_error"):
                st.warning(f"SQL guardrail/execution warning: {data['sql_error']}")

            if data.get("citations"):
                with st.expander("Citations"):
                    for c in data["citations"]:
                        st.write(
                            f"[{c['rank']}] score={c['score']:.4f} | "
                            f"source={c.get('source')} | "
                            f"chunk_index={c.get('chunk_index')} | "
                            f"start_row_estimate={c.get('start_row_estimate')} | "
                            f"rows_in_chunk={c.get('rows_in_chunk')}"
                        )
                        st.caption(c.get("snippet", ""))

            st.session_state.messages.append({"role": "assistant", "content": answer_text})
        except Exception as e:
            err = f"Request failed: {e}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})

