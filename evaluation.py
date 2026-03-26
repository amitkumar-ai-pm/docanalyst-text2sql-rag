"""
Simple evaluation runner for /ask/agent.

Usage:
  .\.venv\Scripts\python.exe evaluation.py --api http://127.0.0.1:8000 --csv "C:\path\to\sell_prices.csv" --questions eval_questions.json
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import requests

MAX_EVAL_FILES = 3


def contains_all(text: str, tokens: list[str]) -> bool:
    t = text.lower()
    return all(tok.lower() in t for tok in tokens)


def _run_reference_sql(csv_path: str, query: str) -> list[dict[str, Any]]:
    con = duckdb.connect(database=":memory:")
    safe_path = csv_path.replace("'", "''")
    con.execute(f"CREATE VIEW data AS SELECT * FROM read_csv_auto('{safe_path}', HEADER=TRUE)")
    df = con.execute(query).fetchdf()
    return df.to_dict(orient="records")


def _extract_agent_rows(pred: dict[str, Any]) -> list[dict[str, Any]]:
    rows = pred.get("sql_rows_preview") or []
    return rows if isinstance(rows, list) else []


def _single_value_str(row_json: str) -> str:
    """If a row has one key, return its value as a string for easy Excel comparison."""
    if not row_json:
        return ""
    try:
        obj = json.loads(row_json)
        if isinstance(obj, dict) and len(obj) == 1:
            return str(next(iter(obj.values())))
    except Exception:
        return ""
    return ""


def _compare_reference(
    case: dict[str, Any], pred: dict[str, Any], csv_path: str
) -> tuple[bool | None, str, str, str]:
    ref_sql = case.get("reference_sql")
    if not ref_sql:
        return None, "no_reference", "", ""
    try:
        ref_rows = _run_reference_sql(csv_path, ref_sql)
    except Exception as e:
        return False, f"reference_sql_error: {e}", "", ""

    agent_rows = _extract_agent_rows(pred)
    ref_first = ref_rows[0] if ref_rows else {}
    agent_first = agent_rows[0] if agent_rows else {}
    ref_first_json = json.dumps(ref_first, ensure_ascii=False)
    agent_first_json = json.dumps(agent_first, ensure_ascii=False)

    if not ref_rows and not agent_rows:
        return True, "both_empty", ref_first_json, agent_first_json
    if not ref_rows or not agent_rows:
        return False, "one_empty", ref_first_json, agent_first_json

    # Compare first row + overlapping keys as lightweight exactness check.
    r0 = ref_rows[0]
    a0 = agent_rows[0]
    keys = sorted(set(r0.keys()).intersection(a0.keys()))
    if not keys:
        return False, "no_overlapping_columns", ref_first_json, agent_first_json
    same = all(str(r0[k]) == str(a0[k]) for k in keys)
    return same, f"compare_keys={keys}", ref_first_json, agent_first_json


def score_case(case: dict[str, Any], pred: dict[str, Any], csv_path: str) -> dict[str, Any]:
    expected_route = case.get("expected_route")
    route_ok = expected_route is None or pred.get("route") == expected_route

    answer = (pred.get("answer") or "").lower()
    must_contain = case.get("must_contain", [])
    must_not_contain = case.get("must_not_contain", [])

    contains_ok = contains_all(answer, must_contain) if must_contain else True
    forbidden_ok = all(tok.lower() not in answer for tok in must_not_contain)

    ref_ok, ref_note, ref_first_json, agent_first_json = _compare_reference(case, pred, csv_path)
    reference_ok = True if ref_ok is None else bool(ref_ok)

    # Three-level grading (reference checks strengthen SQL-style assertions).
    if route_ok and contains_ok and forbidden_ok and reference_ok:
        grade = "correct"
    elif route_ok and (contains_ok or forbidden_ok):
        grade = "partially_correct"
    else:
        grade = "incorrect"

    return {
        "grade": grade,
        "route_ok": route_ok,
        "contains_ok": contains_ok,
        "forbidden_ok": forbidden_ok,
        "reference_ok": reference_ok,
        "reference_note": ref_note,
        "reference_first_row_json": ref_first_json,
        "agent_first_row_json": agent_first_json,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--namespace", default="csv_sell_prices")
    parser.add_argument("--questions", default="eval_questions.json")
    parser.add_argument("--debug", action="store_true", help="Request RAG debug payload (embeddings + top matches).")
    args = parser.parse_args()

    questions_path = Path(args.questions)
    cases = json.loads(questions_path.read_text(encoding="utf-8"))

    out_dir = Path("observability")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"eval_results_{ts}.csv"

    rows = []
    for i, case in enumerate(cases, start=1):
        payload = {
            "question": case["question"],
            "csv_path": args.csv,
            "namespace": args.namespace,
            "debug": bool(args.debug),
        }
        try:
            resp = requests.post(f"{args.api}/ask/agent", json=payload, timeout=300)
            resp.raise_for_status()
            pred = resp.json()
            score = score_case(case, pred, args.csv)
            rows.append(
                {
                    "id": i,
                    "question": case["question"],
                    "expected_route": case.get("expected_route"),
                    "pred_route": pred.get("route"),
                    "grade": score["grade"],
                    "route_ok": score["route_ok"],
                    "contains_ok": score["contains_ok"],
                    "forbidden_ok": score["forbidden_ok"],
                    "reference_ok": score["reference_ok"],
                    "reference_note": score["reference_note"],
                    "reference_first_row_json": score["reference_first_row_json"],
                    "agent_first_row_json": score["agent_first_row_json"],
                    "reference_value": _single_value_str(score["reference_first_row_json"]),
                    "agent_value": _single_value_str(score["agent_first_row_json"]),
                    "rag_top_score": (pred.get("rag_similarity") or {}).get("top_score"),
                    "sql_error": pred.get("sql_error"),
                    "sql_query": pred.get("sql_query"),
                    "sql_rows_preview_json": json.dumps(pred.get("sql_rows_preview"), ensure_ascii=False),
                    "agent_answer": (pred.get("answer") or "").replace("\n", " "),
                    "rag_debug_json": json.dumps(pred.get("rag_debug"), ensure_ascii=False),
                    "answer_preview": (pred.get("answer") or "")[:300].replace("\n", " "),
                }
            )
        except Exception as e:
            rows.append(
                {
                    "id": i,
                    "question": case["question"],
                    "expected_route": case.get("expected_route"),
                    "pred_route": "error",
                    "grade": "incorrect",
                    "route_ok": False,
                    "contains_ok": False,
                    "forbidden_ok": False,
                    "reference_ok": False,
                    "reference_note": "request_error",
                    "reference_first_row_json": "",
                    "agent_first_row_json": "",
                    "reference_value": "",
                    "agent_value": "",
                    "rag_top_score": "",
                    "sql_error": str(e),
                    "sql_query": "",
                    "sql_rows_preview_json": "",
                    "agent_answer": "",
                    "rag_debug_json": "",
                    "answer_preview": "",
                }
            )

    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "question",
                "expected_route",
                "pred_route",
                "grade",
                "route_ok",
                "contains_ok",
                "forbidden_ok",
                "reference_ok",
                "reference_note",
                "reference_first_row_json",
                "agent_first_row_json",
                "reference_value",
                "agent_value",
                "rag_top_score",
                "sql_error",
                "sql_query",
                "sql_rows_preview_json",
                "agent_answer",
                "rag_debug_json",
                "answer_preview",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Retention policy: keep only latest N evaluation files.
    eval_files = sorted(out_dir.glob("eval_results_*.csv"))
    if len(eval_files) > MAX_EVAL_FILES:
        for old in eval_files[: len(eval_files) - MAX_EVAL_FILES]:
            try:
                old.unlink()
            except Exception:
                pass

    total = len(rows)
    correct = sum(1 for r in rows if r["grade"] == "correct")
    partial = sum(1 for r in rows if r["grade"] == "partially_correct")
    incorrect = sum(1 for r in rows if r["grade"] == "incorrect")
    print(f"Wrote: {out_file}")
    print(f"Total={total} | Correct={correct} | Partial={partial} | Incorrect={incorrect}")


if __name__ == "__main__":
    main()

