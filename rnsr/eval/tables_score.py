"""Score extracted tables against a labelled set (synthetic or real)."""

from __future__ import annotations

import json
from pathlib import Path

from rnsr.config import Settings
from rnsr.db.artifact import CorpusDB
from rnsr.eval.datasets.messy_tables import load_labelled_tables
from rnsr.ingest.dispatch import is_ingestable
from rnsr.ingest.pipeline import ingest


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("$", "").replace("(", "").replace(")", "").split())


def _headers_match(got: list[str], expected: list[str]) -> bool:
    if len(got) != len(expected):
        return False
    return all(_norm(a) == _norm(b) or _norm(a).startswith(_norm(b).split()[0])
               for a, b in zip(got, expected, strict=True))


def score_labelled_tables(directory: str | Path, *,
                          out_db: str | Path | None = None,
                          config: Settings | None = None) -> dict:
    """Ingest every document in ``directory`` and score against labels.json."""
    directory = Path(directory)
    spec = load_labelled_tables(directory)
    files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and is_ingestable(p)
    )
    if not files:
        raise FileNotFoundError(f"no ingestable files in {directory}")
    out_db = Path(out_db) if out_db else directory / "corpus.db"
    if out_db.exists():
        out_db.unlink()
    ingest(files, out_db, config=config or Settings())

    with CorpusDB(out_db) as corpus:
        tables = corpus.manifest_dict().get("tables") or []
        results = []
        for label in spec.get("tables", []):
            doc = Path(label["doc"]).stem
            matches = [t for t in tables
                       if doc in (t.get("doc_id") or "")
                       or doc.replace("-", "_") in (t.get("doc_id") or "")]
            if label.get("table_idx") is not None and matches:
                # stable order by table_name
                matches = sorted(matches, key=lambda t: t.get("table_name") or "")
                idx = int(label["table_idx"])
                matches = matches[idx:idx + 1] if idx < len(matches) else []
            exp = label.get("expected") or {}
            got = matches[0] if matches else None
            checks: dict[str, bool] = {}
            if got is None:
                checks = {"extracted": False}
            else:
                headers = [c["name"] for c in got.get("schema") or []
                           if not str(c.get("name", "")).endswith("__raw")]
                n_total = int(got.get("n_total_rows") or 0)
                n_data = int(got.get("n_data_rows") or 0)
                if not n_data:
                    n_data = int(got.get("n_rows") or 0) - n_total
                checks["extracted"] = True
                checks["headers"] = _headers_match(
                    headers, [str(h) for h in exp.get("headers") or []]) if exp.get("headers") else True
                if exp.get("n_total_rows") is not None:
                    checks["n_total_rows"] = n_total == int(exp["n_total_rows"])
                if exp.get("n_data_rows") is not None:
                    checks["n_data_rows"] = n_data == int(exp["n_data_rows"])
                if exp.get("numeric_columns"):
                    numeric = {c["name"] for c in got.get("schema") or []
                               if c.get("type") != "TEXT"}
                    checks["numeric_columns"] = all(
                        any(want in name for name in numeric)
                        for want in exp["numeric_columns"]
                    )
            passed = all(checks.values()) if checks else False
            results.append({
                "doc": label.get("doc"),
                "must_pass": bool(label.get("must_pass", True)),
                "passed": passed,
                "checks": checks,
                "table_name": None if got is None else got.get("table_name"),
            })

    required = [r for r in results if r["must_pass"]]
    n_req = len(required)
    n_ok = sum(1 for r in required if r["passed"])
    report = {
        "directory": str(directory),
        "corpus_db": str(out_db),
        "n_labels": len(results),
        "n_required": n_req,
        "n_required_passed": n_ok,
        "pass_rate": round(n_ok / n_req, 4) if n_req else 1.0,
        "results": results,
    }
    (directory / "table_score.json").write_text(json.dumps(report, indent=2))
    return report
