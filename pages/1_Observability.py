from __future__ import annotations

import glob
import json
import sqlite3
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Observability", layout="wide")
st.title("Observability, Evaluation, and Q&A Archive")
api_base = st.text_input("FastAPI base URL", value="http://127.0.0.1:8000")

OBS_DIR = Path("observability")
DB_PATH = OBS_DIR / "events.db"


def load_events() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            """
            SELECT
                id, ts, endpoint, question, csv_path, namespace, route, route_reason,
                answer, rag_similarity_json, sql_query, sql_error, citations_count,
                latency_ms, success, error_message
            FROM events
            ORDER BY id DESC
            """,
            con,
        )
    finally:
        con.close()

    if "rag_similarity_json" in df.columns:
        def _parse_similarity(val: str):
            if not val:
                return None
            try:
                return json.loads(val)
            except Exception:
                return None

        sims = df["rag_similarity_json"].apply(_parse_similarity)
        df["rag_top_score"] = sims.apply(lambda x: x.get("top_score") if isinstance(x, dict) else None)
        df["rag_avg_score"] = sims.apply(lambda x: x.get("avg_score") if isinstance(x, dict) else None)

    return df


def load_eval_runs() -> pd.DataFrame:
    files = sorted(glob.glob(str(OBS_DIR / "eval_results_*.csv")))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["run_file"] = Path(f).name
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


events_df = load_events()
eval_df = load_eval_runs()

col1, col2, col3, col4 = st.columns(4)
if events_df.empty:
    col1.metric("Total events", "0")
    col2.metric("Success rate", "N/A")
    col3.metric("Avg latency (ms)", "N/A")
    col4.metric("RAG avg top score", "N/A")
else:
    total_events = len(events_df)
    success_rate = (events_df["success"].mean() * 100.0) if "success" in events_df else 0.0
    avg_latency = events_df["latency_ms"].dropna().mean()
    rag_avg_top = events_df["rag_top_score"].dropna().mean() if "rag_top_score" in events_df else None

    col1.metric("Total events", f"{total_events}")
    col2.metric("Success rate", f"{success_rate:.1f}%")
    col3.metric("Avg latency (ms)", f"{avg_latency:.1f}" if pd.notna(avg_latency) else "N/A")
    col4.metric("RAG avg top score", f"{rag_avg_top:.4f}" if pd.notna(rag_avg_top) else "N/A")

st.divider()

st.subheader("Q&A Archive")
if events_df.empty:
    st.info("No runtime events logged yet. Ask a few questions in the chat page first.")
else:
    endpoint_filter = st.multiselect(
        "Endpoint filter",
        options=sorted(events_df["endpoint"].dropna().unique().tolist()),
        default=sorted(events_df["endpoint"].dropna().unique().tolist()),
    )
    route_filter = st.multiselect(
        "Route filter",
        options=sorted(events_df["route"].dropna().unique().tolist()),
        default=sorted(events_df["route"].dropna().unique().tolist()),
    )
    success_filter = st.selectbox("Success filter", options=["all", "success only", "failed only"], index=0)

    filtered = events_df.copy()
    if endpoint_filter:
        filtered = filtered[filtered["endpoint"].isin(endpoint_filter)]
    if route_filter:
        filtered = filtered[filtered["route"].isin(route_filter)]
    if success_filter == "success only":
        filtered = filtered[filtered["success"] == 1]
    elif success_filter == "failed only":
        filtered = filtered[filtered["success"] == 0]

    display_cols = [
        "id",
        "ts",
        "endpoint",
        "route",
        "question",
        "success",
        "latency_ms",
        "citations_count",
        "rag_top_score",
        "sql_error",
    ]
    show_df = filtered[display_cols].head(300)
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    st.markdown("### Inspect Event Detail")
    selected_id = st.number_input(
        "Enter event id to inspect",
        min_value=int(events_df["id"].min()),
        max_value=int(events_df["id"].max()),
        value=int(events_df["id"].iloc[0]),
        step=1,
    )
    row = events_df[events_df["id"] == selected_id]
    if not row.empty:
        r = row.iloc[0]
        st.write(f"**Question:** {r.get('question')}")
        st.write(f"**Route:** {r.get('route')} | **Success:** {bool(r.get('success'))} | **Latency(ms):** {r.get('latency_ms')}")
        if pd.notna(r.get("route_reason")):
            st.write(f"**Route reason:** {r.get('route_reason')}")
        if pd.notna(r.get("answer")):
            st.markdown("**Answer**")
            st.code(str(r.get("answer"))[:4000])
        if pd.notna(r.get("sql_query")):
            st.markdown("**SQL query**")
            st.code(str(r.get("sql_query")), language="sql")
        if pd.notna(r.get("sql_error")):
            st.warning(f"SQL error: {r.get('sql_error')}")
        if pd.notna(r.get("error_message")):
            st.error(f"Error message: {r.get('error_message')}")

st.divider()

st.subheader("Evaluation History")
if eval_df.empty:
    st.info("No evaluation runs found. Run `evaluation.py` to generate reports.")
else:
    with st.expander("Run evaluation now"):
        default_csv = r"C:\Users\1036506\Downloads\data_M5\sell_prices.csv"
        run_csv_path = st.text_input("CSV path for eval run", value=default_csv)
        run_namespace = st.text_input("Namespace for eval run", value="csv_sell_prices")
        run_questions = st.text_input("Questions file path", value="eval_questions.json")
        run_debug = st.checkbox("Enable debug payload (embeddings, top matches)", value=True)
        if st.button("Run Evaluation"):
            try:
                with st.spinner("Running evaluation..."):
                    resp = requests.post(
                        f"{api_base}/evaluate/run",
                        json={
                            "api": api_base,
                            "csv_path": run_csv_path,
                            "namespace": run_namespace,
                            "questions_path": run_questions,
                            "debug": run_debug,
                        },
                        timeout=1800,
                    )
                resp.raise_for_status()
                st.success("Evaluation run finished. Reloading page data is recommended.")
                st.code(resp.json().get("stdout", "")[-2000:])
            except Exception as e:
                st.error(f"Failed to run evaluation: {e}")

    summary = (
        eval_df.groupby("run_file")["grade"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("run_file", ascending=False)
    )
    for col in ["correct", "partially_correct", "incorrect"]:
        if col not in summary.columns:
            summary[col] = 0
    summary["total"] = summary["correct"] + summary["partially_correct"] + summary["incorrect"]
    summary["correct_rate_pct"] = (summary["correct"] / summary["total"] * 100).round(1)

    # Parse timestamp from file name pattern: eval_results_YYYYMMDD_HHMMSS.csv
    summary["run_ts"] = pd.to_datetime(
        summary["run_file"].str.extract(r"eval_results_(\d{8}_\d{6})")[0],
        format="%Y%m%d_%H%M%S",
        errors="coerce",
    )
    trend = summary.sort_values("run_ts").copy()
    if not trend["run_ts"].isna().all():
        trend = trend.dropna(subset=["run_ts"])
        if not trend.empty:
            st.markdown("### Evaluation Trend")
            trend_chart = trend.set_index("run_ts")[["correct_rate_pct"]]
            st.line_chart(trend_chart, height=220)
            st.caption("Correct-rate trend across evaluation runs.")

    st.dataframe(
        summary[["run_file", "total", "correct", "partially_correct", "incorrect", "correct_rate_pct"]],
        use_container_width=True,
        hide_index=True,
    )

    run_choice = st.selectbox("Inspect evaluation run", options=summary["run_file"].tolist())
    run_df = eval_df[eval_df["run_file"] == run_choice]
    st.download_button(
        label="Download selected evaluation CSV",
        data=run_df.to_csv(index=False).encode("utf-8"),
        file_name=run_choice,
        mime="text/csv",
    )
    st.dataframe(run_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Raw JSONL Event Archive")
jsonl_path = OBS_DIR / "events.jsonl"
if jsonl_path.exists():
    st.write(f"Path: `{jsonl_path}`")
    text = jsonl_path.read_text(encoding="utf-8")
    st.download_button(
        label="Download raw events JSONL",
        data=text.encode("utf-8"),
        file_name="events.jsonl",
        mime="application/json",
    )
    with st.expander("Preview last 20 raw events"):
        lines = text.strip().splitlines()
        preview = "\n".join(lines[-20:]) if lines else "(empty)"
        st.code(preview)
else:
    st.info("No events.jsonl found yet. Ask questions to generate raw event logs.")

