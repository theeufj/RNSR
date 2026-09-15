"""Field-trial audit packet: evidence without full trajectories."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rnsr.harness.evidence import from_records
from rnsr.harness.trajectory import read_trajectory


def _sql_from_cells(records: list[dict]) -> list[str]:
    out: list[str] = []
    for rec in records:
        if rec.get("kind") != "cell":
            continue
        code = rec.get("code") or ""
        for line in code.splitlines():
            stripped = line.strip()
            low = stripped.lower()
            if low.startswith(("select ", "with ")) or (
                    "execute(" in stripped and any(k in low for k in ("select", "with "))):
                out.append(stripped)
    return out[:12]


def _pages(records: list[dict], verification: dict | None) -> list[int]:
    pages: set[int] = set()
    if verification:
        for q in verification.get("quotes") or []:
            page = q.get("page")
            if page is not None:
                pages.add(int(page))
    for rec in records:
        if rec.get("kind") == "search_rung":
            for hit in rec.get("hits") or []:
                prov = hit.get("provenance") or hit
                page = prov.get("page") or prov.get("_page")
                if page is not None:
                    pages.add(int(page))
    return sorted(pages)


def extract_evidence(records: list[dict], *, qid: str,
                     health: dict | None = None,
                     status_row: dict | None = None) -> dict:
    """Pull the publishable evidence slice from one trajectory."""
    start = next((r for r in records if r.get("kind") == "start"), {})
    final = next((r for r in reversed(records) if r.get("kind") == "final"), {})
    end = next((r for r in reversed(records) if r.get("kind") == "end"), {})
    verification = final.get("verification") or {}
    quotes = []
    for q in verification.get("quotes") or []:
        quotes.append({
            "quote": q.get("quote"),
            "matched": q.get("matched"),
            "doc_id": q.get("doc_id"),
            "page": q.get("page"),
        })
    answer = final.get("value")
    if isinstance(answer, dict) and qid in answer:
        answer = answer[qid]
    ev = from_records(records, status=(status_row or {}).get("status") or end.get("status") or "final",
                      health_grade=(health or {}).get("grade"), qid=qid)
    return {
        "qid": qid,
        "question": start.get("question"),
        "answer": answer,
        "status": (status_row or {}).get("status") or end.get("status"),
        "verified_quotes": quotes,
        "sql": _sql_from_cells(records),
        "pages": _pages(records, verification),
        "health": health,
        "tier": (status_row or {}).get("tier") or ev.tier,
        "evidence": ev.to_dict(),
    }


def export_audit(work_dir: str | Path, out_dir: str | Path, *,
                 key: str = "") -> dict:
    """Write per-question evidence.json plus review.csv. No trajectory copies."""
    work = Path(work_dir)
    out = Path(out_dir)
    ev_dir = out / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)

    health = None
    for candidate in (work / "run_report.json",
                      work.parent / "run_report.json",
                      out / "run_report.json"):
        if candidate.exists():
            health = json.loads(candidate.read_text()).get("health")
            break

    status_by_qid: dict[str, dict] = {}
    for candidate in (work / "answers_status.csv",
                      work.parent / "answers_status.csv"):
        if not candidate.exists():
            continue
        with open(candidate, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                qid = row.get("query_id") or row.get("qid") or ""
                if qid:
                    status_by_qid[qid] = row
        break

    traj_dir = work / "trajectories"
    if not traj_dir.is_dir():
        traj_dir = work
    paths = sorted(traj_dir.glob("*.jsonl")) + sorted(traj_dir.glob("*.jsonl.enc"))
    review_rows: list[dict] = []
    written = 0
    for path in paths:
        qid = path.name.split(".", 1)[0]
        records = read_trajectory(path, key=key)
        evidence = extract_evidence(
            records, qid=qid, health=health,
            status_row=status_by_qid.get(qid),
        )
        (ev_dir / f"{qid}.json").write_text(
            json.dumps(evidence, indent=2, default=str))
        written += 1
        review_rows.append({
            "qid": qid,
            "answer": evidence.get("answer") or "",
            "tier": evidence.get("tier") or "",
            "reviewer_mark": "",
            "note": "",
        })

    _rank = {"low": 0, "medium": 1, "high": 2}
    review_rows.sort(key=lambda r: _rank.get(r.get("tier") or "", 3))
    review_path = out / "review.csv"
    with open(review_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["qid", "answer", "tier",
                                           "reviewer_mark", "note"])
        w.writeheader()
        w.writerows(review_rows)
    return {"n": written, "evidence_dir": str(ev_dir), "review": str(review_path)}


_CORRECT = frozenset({"correct", "ok", "pass", "yes", "true", "1"})
_WRONG = frozenset({"wrong", "miss", "fail", "no", "false", "0", "incorrect"})


def score_review(path: str | Path) -> dict:
    """Score a filled-in review.csv into a miss report."""
    path = Path(path)
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            mark = (row.get("reviewer_mark") or "").strip().lower()
            if mark in _CORRECT:
                verdict = "ok"
            elif mark in _WRONG:
                verdict = "miss"
            else:
                verdict = "unmarked"
            rows.append({
                "qid": row.get("qid") or "",
                "answer": row.get("answer") or "",
                "reviewer_mark": row.get("reviewer_mark") or "",
                "note": row.get("note") or "",
                "verdict": verdict,
            })
    n_marked = sum(1 for r in rows if r["verdict"] != "unmarked")
    n_ok = sum(1 for r in rows if r["verdict"] == "ok")
    misses = [r for r in rows if r["verdict"] == "miss"]
    return {
        "total": len(rows),
        "marked": n_marked,
        "correct": n_ok,
        "misses": len(misses),
        "accuracy": (n_ok / n_marked) if n_marked else 0.0,
        "miss_list": misses,
        "unmarked": sum(1 for r in rows if r["verdict"] == "unmarked"),
    }
