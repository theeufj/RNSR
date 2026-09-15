"""Miss-cause classifier: join gold + trajectory + corpus health.

Every miss is assigned exactly one cause so engineering effort lands
where accuracy is actually lost:

  gold       reviewer marked the gold itself as wrong
  budget     the loop never produced a `final` (exhausted / recovered / error)
  format     substance matches after normalisation; the string form does not
  ingest     the gold's document, page, or table never made it into the artifact
  retrieval  the gold evidence was in the artifact but never surfaced to the loop
  reasoning  the gold evidence was seen; the answer is still wrong
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rnsr.eval.metrics import EvalResult, _as_number, _normalize, score_answer
from rnsr.harness.trajectory import read_trajectory

CAUSES = ("gold", "budget", "format", "ingest", "retrieval", "reasoning")

_LABEL_PREFIX = re.compile(
    r"^(?:answer|label|the answer is|final answer)\s*[:\-]\s*", re.I)
_SPLIT_PARTS = re.compile(r"\s*(?:,|;|/|\band\b)\s*")
_DOC_ID = re.compile(r"\b(?:doc(?:_id)?|document)\s*[=:]\s*['\"]?([A-Za-z0-9._-]+)")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._%-]{1,}")
_STOP = frozenset({
    "the", "and", "for", "was", "were", "with", "what", "which", "how",
    "many", "much", "does", "did", "from", "this", "that", "not", "found",
    "yes", "no",
})
_GOLD_ERROR = frozenset({
    "gold", "gold-error", "gold_error", "bad gold", "wrong gold",
})


@dataclass
class AutopsyItem:
    qid: str
    task_class: str
    predicted: str | None
    gold: str
    status: str
    cause: str
    reason: str
    exposure: str | None = None
    gold_doc: str | None = None
    seen_docs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _strip_label(text: str) -> str:
    return _LABEL_PREFIX.sub("", (text or "").strip()).strip(" .")


def format_only_miss(predicted: str | None, gold: str) -> bool:
    """True when the miss is a string-form mismatch, not a substance miss.

    Tie-sets (gold lists several labels, predicted is one of them) and
    `Answer:` / `Label:` prefixes are the live cases this exists for.
    """
    if predicted is None:
        return False
    p, g = _normalize(_strip_label(str(predicted))), _normalize(_strip_label(gold))
    if not p or not g:
        return False
    if p == g:
        return True
    if score_answer(p, g):
        return True
    gold_parts = [x for x in (_normalize(part) for part in _SPLIT_PARTS.split(g)) if x]
    pred_parts = [x for x in (_normalize(part) for part in _SPLIT_PARTS.split(p)) if x]
    if len(gold_parts) > 1 and p in gold_parts:
        return True
    if gold_parts and pred_parts and (
            set(pred_parts) <= set(gold_parts) or set(gold_parts) <= set(pred_parts)):
        return True
    gn, pn = _as_number(g), _as_number(p)
    return gn is not None and pn is not None and gn == pn


def _seen_docs(records: list[dict]) -> set[str]:
    docs: set[str] = set()
    for rec in records:
        kind = rec.get("kind")
        if kind == "final":
            verification = rec.get("verification") or {}
            quotes = verification.get("quotes") or []
            if isinstance(quotes, dict):
                quotes = [q for qs in quotes.values()
                          for q in (qs.get("quotes") or [qs])]
            for q in quotes:
                if isinstance(q, dict) and q.get("doc_id"):
                    docs.add(str(q["doc_id"]))
        blob = " ".join(str(rec.get(k) or "") for k in ("stdout", "code", "query"))
        docs.update(_DOC_ID.findall(blob))
        hits = rec.get("hits")
        if isinstance(hits, list):
            for hit in hits:
                if not isinstance(hit, dict):
                    continue
                doc_id = hit.get("doc_id") or (hit.get("provenance") or {}).get("doc_id")
                if doc_id:
                    docs.add(str(doc_id))
    return docs


def _seen_pages(records: list[dict]) -> set[int]:
    pages: set[int] = set()
    for rec in records:
        if rec.get("kind") == "final":
            for q in (rec.get("verification") or {}).get("quotes") or []:
                if isinstance(q, dict) and q.get("page") is not None:
                    pages.add(int(q["page"]))
        hits = rec.get("hits")
        if isinstance(hits, list):
            for hit in hits:
                if not isinstance(hit, dict):
                    continue
                page = hit.get("page") or (hit.get("provenance") or {}).get("page")
                if page is not None:
                    pages.add(int(page))
    return pages


def _trajectory_blob(records: list[dict]) -> str:
    parts: list[str] = []
    for rec in records:
        for key in ("stdout", "code", "query", "value"):
            val = rec.get(key)
            if val is not None:
                parts.append(str(val))
        for q in (rec.get("verification") or {}).get("quotes") or []:
            if isinstance(q, dict) and q.get("quote"):
                parts.append(str(q["quote"]))
    return _normalize(" ".join(parts))


def _gold_tokens(gold: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(gold) if t.lower() not in _STOP]


def _gold_text_seen(gold: str, records: list[dict]) -> bool:
    tokens = [t for t in _gold_tokens(gold) if not t.isdigit() or len(t) >= 3]
    if not tokens:
        tokens = _gold_tokens(gold)
    if not tokens:
        return False
    blob = _trajectory_blob(records)
    if not blob:
        return False
    hits = sum(1 for t in tokens if t in blob)
    return hits >= max(1, (len(tokens) + 1) // 2)


def _health_blocks_gold(meta: dict, health: dict | None,
                        manifest: dict | None) -> str | None:
    """Return an ingest reason if the gold's source never entered the artifact."""
    health = health or {}
    meta = meta or {}
    gold_doc = (meta.get("gold_doc") or meta.get("gold_doc_id") or "") or ""
    gold_page = meta.get("gold_page")
    gold_table = meta.get("gold_table") or ""
    exposure = meta.get("exposure") or ""

    failed = health.get("parse_failed") or []
    failed_ids = {str(x.get("doc_id") or x.get("source") or x) for x in failed} \
        if failed and isinstance(failed[0], dict) else {str(x) for x in failed}
    if gold_doc and any(gold_doc in x or x in gold_doc for x in failed_ids):
        return f"gold doc {gold_doc!r} is in parse_failed"

    untranscribed = health.get("scanned_pages_untranscribed") or []
    if exposure == "scanned_page" and untranscribed:
        return "gold depends on a scanned page that was not transcribed"
    if gold_doc and gold_page is not None:
        for entry in untranscribed:
            if isinstance(entry, dict):
                doc = str(entry.get("doc_id") or entry.get("source") or "")
                pages = entry.get("pages") or []
                if ((gold_doc in doc or doc in gold_doc)
                        and (not pages or int(gold_page) in {int(p) for p in pages})):
                    return f"gold page {gold_page} of {gold_doc!r} is untranscribed"

    tables = (manifest or {}).get("tables") or []
    if gold_table:
        for table in tables:
            name = table.get("table_name") or table.get("name") or ""
            if gold_table in name or name == gold_table:
                if table.get("status") == "untrusted":
                    return f"cited table {name} is untrusted"
                break

    if exposure == "attachment":
        docs = {(d.get("doc_id") if isinstance(d, dict) else str(d))
                for d in (manifest or {}).get("documents") or []}
        parent = meta.get("parent_doc") or gold_doc
        child = meta.get("child_doc") or meta.get("attachment_doc")
        if child and child not in docs:
            return f"email attachment {child!r} was not ingested"
        if parent and parent in docs and not child:
            return "email attachment was named but not parsed into a child document"

    if exposure == "sheet_identity":
        titles = [t.get("title") for t in tables if t.get("title")]
        wanted = meta.get("sheet_name")
        if wanted and wanted not in titles:
            return f"spreadsheet sheet {wanted!r} lost its identity at ingest"

    return None


def classify_miss(
    result: EvalResult,
    records: list[dict] | None = None,
    *,
    meta: dict | None = None,
    health: dict | None = None,
    manifest: dict | None = None,
    reviewer_mark: str | None = None,
) -> AutopsyItem:
    """Assign one cause to a miss (or to a correct result: cause stays empty)."""
    meta = dict(meta or {})
    records = records or []
    gold_doc = meta.get("gold_doc") or meta.get("gold_doc_id")
    exposure = meta.get("exposure")
    seen = sorted(_seen_docs(records))
    reason = ""
    cause = "reasoning"

    if result.correct:
        cause, reason = "ok", "answer agrees with gold"
    elif reviewer_mark and reviewer_mark.strip().lower() in _GOLD_ERROR:
        cause, reason = "gold", "reviewer marked the gold as wrong"
    elif result.status not in ("final",):
        cause, reason = "budget", f"loop status is {result.status!r}, not final"
    elif format_only_miss(result.predicted, result.gold):
        cause, reason = "format", "normalised / tie-set match; string form differs"
    else:
        ingest_reason = _health_blocks_gold(meta, health, manifest)
        if ingest_reason:
            cause, reason = "ingest", ingest_reason
        elif gold_doc and records and gold_doc not in seen and not any(
                gold_doc in d or d in gold_doc for d in seen):
            cause, reason = "retrieval", f"gold doc {gold_doc!r} never appeared in hits or quotes"
        elif meta.get("gold_page") is not None and records:
            wanted = int(meta["gold_page"])
            if wanted not in _seen_pages(records) and not _gold_text_seen(result.gold, records):
                cause, reason = "retrieval", f"gold page {wanted} never appeared in hits or quotes"
        elif records and not _gold_text_seen(result.gold, records):
            searched = any(r.get("kind") in ("search_rung", "cell") for r in records)
            if searched:
                cause, reason = "retrieval", "gold text never appeared in search hits or cell output"
            else:
                cause, reason = "retrieval", "loop never searched or opened a document"
        else:
            cause, reason = "reasoning", "gold evidence was visible; the answer is still wrong"

    return AutopsyItem(
        qid=result.qid,
        task_class=result.task_class,
        predicted=result.predicted,
        gold=result.gold,
        status=result.status,
        cause=cause,
        reason=reason,
        exposure=exposure,
        gold_doc=gold_doc,
        seen_docs=seen,
    )


def attach_causes(results: list[EvalResult], items: list[AutopsyItem]) -> list[EvalResult]:
    """Stamp ``cause`` onto each EvalResult (None / empty for correct)."""
    by_qid = {i.qid: i for i in items}
    for r in results:
        item = by_qid.get(r.qid)
        if item is None:
            continue
        r.cause = None if item.cause == "ok" else item.cause
        if item.gold_doc:
            r.retrieval_hit = item.gold_doc in item.seen_docs or any(
                item.gold_doc in d or d in item.gold_doc for d in item.seen_docs)
    return results


def _load_results(path: Path) -> list[EvalResult]:
    results: list[EvalResult] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        data.setdefault("expect", "value")
        data.setdefault("scored_by", "string")
        data.setdefault("trajectory_path", None)
        data.setdefault("cause", None)
        data.setdefault("retrieval_hit", None)
        known = set(EvalResult.__dataclass_fields__)
        results.append(EvalResult(**{k: v for k, v in data.items() if k in known}))
    return results


def _load_review_marks(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    import csv

    marks: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            qid = row.get("qid") or row.get("query_id") or ""
            if qid:
                marks[qid] = row.get("reviewer_mark") or row.get("note") or ""
    return marks


def _find_trajectory(run_dir: Path, qid: str, result: EvalResult) -> Path | None:
    if result.trajectory_path:
        p = Path(result.trajectory_path)
        if p.exists():
            return p
        alt = run_dir / result.trajectory_path
        if alt.exists():
            return alt
    for folder in (run_dir / "trajectories", run_dir):
        for suffix in (".jsonl", ".jsonl.enc"):
            candidate = folder / f"{qid}{suffix}"
            if candidate.exists():
                return candidate
    return None


def autopsy_run(
    run_dir: str | Path,
    *,
    golden: str | Path | None = None,
    review: str | Path | None = None,
    key: str = "",
    items_meta: dict[str, dict] | None = None,
) -> dict:
    """Classify every miss in an eval / answer-csv run directory.

    Reads ``results.jsonl`` when present (eval harness). Otherwise walks
    trajectories and pairs them with ``answers_status.csv`` / a golden JSON.
    """
    run_dir = Path(run_dir)
    health = None
    manifest = None
    for candidate in (run_dir / "run_report.json",
                      run_dir / "summary.json",
                      run_dir.parent / "run_report.json"):
        if candidate.exists():
            payload = json.loads(candidate.read_text())
            health = payload.get("health") or health
            manifest = payload.get("manifest") or manifest
            break

    results: list[EvalResult] = []
    results_path = run_dir / "results.jsonl"
    if results_path.exists():
        results = _load_results(results_path)
    else:
        from rnsr.eval.regression import load_field_answers, load_golden

        answers: dict[str, str] = {}
        for candidate in (run_dir / "answers_chunk1.csv",
                          run_dir / "answers.csv",
                          run_dir.parent / "answers_chunk1.csv"):
            if candidate.exists():
                answers = load_field_answers(candidate)
                break
        gold: dict[str, list[str]] = {}
        if golden:
            gold = load_golden(golden)
        for qid, answer in answers.items():
            g = "; ".join(gold.get(qid, [""]))
            results.append(EvalResult(
                qid=qid, task_class="default", predicted=answer, gold=g,
                correct=bool(g) and _normalize(answer) == _normalize(g),
                status="final", latency_s=0.0, cost_usd=0.0,
                sub_calls=0, iterations=0,
            ))

    marks = _load_review_marks(Path(review) if review else run_dir / "review.csv")
    items_meta = items_meta or {}
    classified: list[AutopsyItem] = []
    for result in results:
        path = _find_trajectory(run_dir, result.qid, result)
        records = read_trajectory(path, key=key) if path else []
        classified.append(classify_miss(
            result, records,
            meta=items_meta.get(result.qid, {}),
            health=health, manifest=manifest,
            reviewer_mark=marks.get(result.qid),
        ))
    attach_causes(results, classified)
    return build_ledger(results, classified)


def build_ledger(results: list[EvalResult],
                 items: list[AutopsyItem] | None = None) -> dict:
    """Aggregate autopsy items into the loss-ledger shape."""
    items = items or []
    misses = [i for i in items if i.cause != "ok"]
    by_cause: dict[str, int] = {c: 0 for c in CAUSES}
    by_cause.update(Counter(i.cause for i in misses))
    by_class: dict[str, dict[str, int]] = {}
    for i in misses:
        bucket = by_class.setdefault(i.task_class, {c: 0 for c in CAUSES})
        bucket[i.cause] = bucket.get(i.cause, 0) + 1
    n = len(results)
    n_miss = len(misses)
    return {
        "n": n,
        "n_correct": n - n_miss,
        "n_miss": n_miss,
        "accuracy": ((n - n_miss) / n) if n else 0.0,
        "cause_counts": by_cause,
        "cause_x_class": by_class,
        "items": [i.to_dict() for i in items],
        "retrieval_recall": (
            (sum(1 for r in results if r.retrieval_hit)
             / max(1, sum(1 for r in results if r.retrieval_hit is not None)))
            if any(r.retrieval_hit is not None for r in results) else None
        ),
    }


def render_ledger_md(ledger: dict, *, title: str = "Loss ledger") -> str:
    lines = [
        f"# {title}",
        "",
        f"n={ledger['n']}  correct={ledger['n_correct']}  "
        f"miss={ledger['n_miss']}  accuracy={ledger['accuracy']:.1%}",
        "",
        "## Misses by cause",
        "",
        "| cause | n | share of misses |",
        "|---|---:|---:|",
    ]
    n_miss = ledger["n_miss"] or 1
    for cause in CAUSES:
        n = ledger["cause_counts"].get(cause, 0)
        lines.append(f"| {cause} | {n} | {n / n_miss:.0%} |")
    if ledger.get("cause_x_class"):
        lines += ["", "## Cause × task class", "",
                  "| class | " + " | ".join(CAUSES) + " |",
                  "|---|" + "---:|" * len(CAUSES)]
        for cls, counts in sorted(ledger["cause_x_class"].items()):
            cells = " | ".join(str(counts.get(c, 0)) for c in CAUSES)
            lines.append(f"| {cls} | {cells} |")
    misses = [i for i in ledger.get("items", []) if i.get("cause") not in (None, "ok")]
    if misses:
        lines += ["", "## Miss list", ""]
        for i in misses:
            pred = (i.get("predicted") or "")[:80]
            gold = (i.get("gold") or "")[:80]
            lines.append(
                f"- `{i['qid']}` [{i.get('task_class')}] **{i['cause']}**: "
                f"{i.get('reason')} — predicted={pred!r} gold={gold!r}"
            )
    return "\n".join(lines) + "\n"


def write_ledger(ledger: dict, out_dir: str | Path, *,
                 title: str = "Loss ledger") -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "autopsy.json"
    md_path = out / "loss-ledger.md"
    json_path.write_text(json.dumps(ledger, indent=2, default=str))
    md_path.write_text(render_ledger_md(ledger, title=title))
    return {"json": str(json_path), "md": str(md_path)}
