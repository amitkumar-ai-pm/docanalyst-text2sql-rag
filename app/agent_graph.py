from __future__ import annotations

import re
from typing import Any, Literal, TypedDict

import duckdb
from langchain_community.vectorstores.upstash import UpstashVectorStore
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.services import answer_rag


class ToolDecision(BaseModel):
    route: Literal["sql", "rag", "both"] = Field(
        description="Choose sql for exact tabular math, rag for semantic retrieval, both if both are needed."
    )
    reason: str = Field(description="Short reason for the route.")


class SQLQuery(BaseModel):
    query: str = Field(description="A single read-only SQL query over table `data`.")


class AgentState(TypedDict, total=False):
    question: str
    csv_path: str
    route: str
    route_reason: str
    sql_query: str
    sql_rows: list[dict[str, Any]]
    sql_error: str
    rag_answer: str
    citations: list[dict[str, Any]]
    final_answer: str


def _rows_to_compact_table(rows: list[dict[str, Any]], max_rows: int = 10) -> str:
    if not rows:
        return "(no rows)"
    cols = list(rows[0].keys())
    lines = [" | ".join(cols)]
    lines.append("-" * len(lines[0]))
    for r in rows[:max_rows]:
        lines.append(" | ".join(str(r.get(c, "")) for c in cols))
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows)")
    return "\n".join(lines)


def _validate_sql_read_only(query: str) -> tuple[bool, str]:
    q = query.strip()
    # Allow a single trailing semicolon and normalize case for checks.
    if q.endswith(";"):
        q = q[:-1].strip()
    ql = q.lower()

    # Allow read-only SELECT statements, including CTE form: WITH ... SELECT ...
    if not (ql.startswith("select") or ql.startswith("with")):
        return False, "Only read-only SELECT/CTE queries are allowed."

    # Block multi-statement queries.
    if ";" in ql:
        return False, "Semicolons/multiple statements are not allowed."

    forbidden = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "truncate",
        "attach",
        "detach",
        "copy",
        "pragma",
        "call",
    ]
    hit = next((kw for kw in forbidden if re.search(rf"\b{kw}\b", ql)), None)
    if hit:
        return False, f"Disallowed keyword detected: {hit}"

    # Collect CTE names from WITH clauses so references to them are allowed.
    cte_names = set(
        name for name in re.findall(r"\bwith\s+([a-zA-Z_][\w]*)\s+as\b", ql)
    )
    cte_names.update(
        re.findall(r",\s*([a-zA-Z_][\w]*)\s+as\b", ql)
    )

    # Enforce table allowlist. We allow base table `data` and any CTE names.
    # Accept forms: FROM <name>, JOIN <name>
    table_refs = re.findall(r"\b(from|join)\s+([a-zA-Z_][\w]*)", ql)
    if not table_refs:
        return False, "No FROM/JOIN clause found."
    # SQL keywords that may appear after FROM/JOIN in valid read-only queries.
    keyword_aliases = {"lateral"}
    disallowed_tables = [
        tbl
        for _, tbl in table_refs
        if tbl != "data" and tbl not in cte_names and tbl not in keyword_aliases
    ]
    if disallowed_tables:
        return False, f"Disallowed table(s): {', '.join(sorted(set(disallowed_tables)))}. Only `data` is allowed."

    return True, ""


def _run_sql(csv_path: str, query: str) -> list[dict[str, Any]]:
    con = duckdb.connect(database=":memory:")
    safe_path = csv_path.replace("'", "''")
    # Avoid prepared-parameter placeholders here; DuckDB can reject them in CREATE VIEW.
    con.execute(f"CREATE VIEW data AS SELECT * FROM read_csv_auto('{safe_path}', HEADER=TRUE)")
    # Guardrail: cap output size to keep responses fast and readable.
    wrapped = f"SELECT * FROM ({query}) t LIMIT 200"
    rows = con.execute(wrapped).fetchdf()
    return rows.to_dict(orient="records")


def _get_table_schema(csv_path: str) -> list[dict[str, str]]:
    con = duckdb.connect(database=":memory:")
    safe_path = csv_path.replace("'", "''")
    con.execute(f"CREATE VIEW data AS SELECT * FROM read_csv_auto('{safe_path}', HEADER=TRUE)")
    df = con.execute("DESCRIBE data").fetchdf()
    return df.to_dict(orient="records")


def build_agent_graph(llm: ChatOpenAI, store: UpstashVectorStore):
    def decide_tool(state: AgentState) -> AgentState:
        model = llm.with_structured_output(ToolDecision)
        decision = model.invoke(
            f"""
You are a routing controller for a hybrid data assistant.
Question: {state['question']}

Rules:
- Use "sql" for exact numeric/table operations (max/min/avg/sum/count/group by/top/trend windows).
- Use "rag" for semantic, descriptive, or context-based questions.
- Use "both" only when query explicitly needs both exact math and semantic explanation.
"""
        )
        return {"route": decision.route, "route_reason": decision.reason}

    def sql_node(state: AgentState) -> AgentState:
        schema = _get_table_schema(state["csv_path"])
        schema_text = ", ".join([f"{c['column_name']} ({c['column_type']})" for c in schema])

        sql_model = llm.with_structured_output(SQLQuery)
        sql_spec = sql_model.invoke(
            f"""
Generate one read-only DuckDB SQL query for table `data` to answer:
{state['question']}

Constraints:
- Use only SELECT.
- Table name must be `data`.
- No DDL/DML.
- Use ONLY these columns: {schema_text}
- Do not invent column names.
- Return a concise result (aggregate or top rows) suitable for user display.
- For text filters like hobbies/foods/household, use case-insensitive match:
  UPPER(item_id) LIKE 'HOBBIES%' (or FOODS%, HOUSEHOLD%).
"""
        )
        query = sql_spec.query.strip().rstrip(";")
        ok, reason = _validate_sql_read_only(query)
        if not ok:
            return {"sql_query": query, "sql_error": reason, "sql_rows": []}
        try:
            rows = _run_sql(state["csv_path"], query)
            return {"sql_query": query, "sql_rows": rows}
        except Exception as e:
            # One automatic retry: repair SQL using actual DB error + allowlisted schema.
            repair_model = llm.with_structured_output(SQLQuery)
            repair = repair_model.invoke(
                f"""
The previous SQL query failed. Fix it.

Question:
{state['question']}

Original query:
{query}

Database error:
{str(e)}

Allowed table:
- data

Allowed columns only:
{schema_text}

Rules:
- Return one read-only SELECT query only.
- Do not invent column names.
- Keep query concise and valid DuckDB SQL.
"""
            )
            repaired_query = repair.query.strip().rstrip(";")
            ok2, reason2 = _validate_sql_read_only(repaired_query)
            if not ok2:
                return {
                    "sql_query": repaired_query,
                    "sql_error": f"repair_failed_guardrail: {reason2}",
                    "sql_rows": [],
                }
            try:
                rows2 = _run_sql(state["csv_path"], repaired_query)
                return {"sql_query": repaired_query, "sql_rows": rows2}
            except Exception as e2:
                return {
                    "sql_query": repaired_query,
                    "sql_error": f"repair_failed_execution: {str(e2)}",
                    "sql_rows": [],
                }

    def rag_node(state: AgentState) -> AgentState:
        rag = answer_rag(state["question"], store, llm, k=4)
        return {"rag_answer": rag["answer"], "citations": rag["citations"]}

    def compose_node(state: AgentState) -> AgentState:
        route = state.get("route", "rag")

        if route == "sql":
            if state.get("sql_error"):
                final = f"SQL route failed guardrail/execution: {state['sql_error']}"
            else:
                rows = state.get("sql_rows", [])
                if not rows:
                    final = (
                        "No rows were returned by the SQL query.\n"
                        "Try broadening filters (for example, case-insensitive item category match)."
                    )
                else:
                    table = _rows_to_compact_table(rows, max_rows=10)
                    final = (
                        f"SQL answer based on exact computation.\n\n"
                        f"Query used:\n{state.get('sql_query', '')}\n\n"
                        f"Rows preview:\n{table}"
                    )
            return {"final_answer": final}

        if route == "rag":
            ans = state.get("rag_answer", "No RAG answer.")
            # Citation guardrail: ensure citations are shown even if model forgets markers.
            cits = state.get("citations", [])
            return {"final_answer": ans, "citations": cits}

        # both: combine exact SQL output + RAG explanation/citations
        sql_part = ""
        if state.get("sql_error"):
            sql_part = f"SQL error: {state['sql_error']}"
        else:
            sql_part = (
                f"SQL query:\n{state.get('sql_query', '')}\n"
                f"Rows preview: {state.get('sql_rows', [])[:10]}"
            )
        final = (
            "Combined answer (exact analytics + semantic context):\n\n"
            f"{sql_part}\n\n"
            f"RAG answer:\n{state.get('rag_answer', '')}"
        )
        return {"final_answer": final, "citations": state.get("citations", [])}

    def route_after_decide(state: AgentState) -> str:
        return "sql_node" if state.get("route") in ("sql", "both") else "rag_node"

    def route_after_sql(state: AgentState) -> str:
        return "rag_node" if state.get("route") == "both" else "compose_node"

    builder = StateGraph(AgentState)
    builder.add_node("decide_tool", decide_tool)
    builder.add_node("sql_node", sql_node)
    builder.add_node("rag_node", rag_node)
    builder.add_node("compose_node", compose_node)

    builder.set_entry_point("decide_tool")
    builder.add_conditional_edges(
        "decide_tool",
        route_after_decide,
        {"sql_node": "sql_node", "rag_node": "rag_node"},
    )
    builder.add_conditional_edges(
        "sql_node",
        route_after_sql,
        {"rag_node": "rag_node", "compose_node": "compose_node"},
    )
    builder.add_edge("rag_node", "compose_node")
    builder.add_edge("compose_node", END)

    return builder.compile()

