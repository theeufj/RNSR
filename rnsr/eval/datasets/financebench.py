"""FinanceBench loader (§8) — numeric needles in real filings; the headline
demo for the SQL path. Questions from PatronusAI/financebench; PDFs
downloaded once into a local cache keyed by URL hash."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from rnsr.eval.datasets.base import EvalItem

DATASET_ID = "PatronusAI/financebench"
DEFAULT_CACHE = Path.home() / ".cache" / "rnsr" / "financebench"

_NUMERIC = re.compile(r"\d")


def _download(url: str, cache: Path) -> Path | None:
    import httpx

    out = cache / (hashlib.md5(url.encode()).hexdigest()[:8] + "_" + url.split("/")[-1])
    if out.exists():
        return out
    cache.mkdir(parents=True, exist_ok=True)
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=60,
                         headers={"User-Agent": "rnsr-eval/1.0"})
        resp.raise_for_status()
        out.write_bytes(resp.content)
        return out
    except Exception:
        return None


def load_financebench(limit: int | None = None,
                      cache: Path = DEFAULT_CACHE) -> list[EvalItem]:
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split="train")
    items: list[EvalItem] = []
    for row in ds:
        if limit and len(items) >= limit:
            break
        pdf = _download(row["doc_link"], cache)
        if pdf is None:
            continue  # unreachable filing; skip rather than fail the run
        gold = str(row["answer"])
        items.append(EvalItem(
            qid=row["financebench_id"],
            question=row["question"],
            gold=gold,
            task_class="numeric" if _NUMERIC.search(gold) else "textual",
            sources=[pdf],
            meta={"doc_name": row["doc_name"], "evidence": row.get("evidence")},
        ))
    return items
