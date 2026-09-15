"""Write answers.xlsx (value, tier, citation) without openpyxl."""

from __future__ import annotations

from pathlib import Path


def write_answers_xlsx(
    path: str | Path,
    rows: list[dict],
    *,
    question_col: str = "question",
) -> Path:
    """rows: question, value, tier, doc, page, quote."""
    from rnsr.eval.datasets.office_gen import write_xlsx

    path = Path(path)
    header = [question_col, "value", "tier", "doc", "page", "quote"]
    grid = [header]
    for row in rows:
        grid.append([
            row.get(question_col) or row.get("question") or "",
            row.get("value") or row.get("model_answer") or "",
            row.get("tier") or "",
            row.get("doc") or "",
            row.get("page") or "",
            row.get("quote") or "",
        ])
    write_xlsx(path, {"answers": grid})
    return path
