"""Offline scoring audit of saved predictions; never a corrected live score.

Usage: python -m rnsr.eval.rescore --results OLD/results.jsonl --out AUDIT

Ambiguous historical positives require fresh independent adjudication.
This module does not import a provider client or modify the original run.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

from rnsr.eval.metrics import SCORER_VERSION, normalize_answer, score_answer


def audit_stored_predictions(results_path: str | Path, out_dir: str | Path) -> dict:
    """Write deterministic agreement and unresolved counts without model calls."""
    source = Path(results_path).resolve()
    out = Path(out_dir).resolve()
    rows_path = out / "scoring_audit.jsonl"
    summary_path = out / "scoring_audit_summary.json"
    if source in {rows_path, summary_path}:
        raise ValueError("Audit output must not replace its historical input file.")
    raw = source.read_bytes()
    rows = []
    for index, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            historical = json.loads(line)
            predicted, gold = historical["predicted"], historical["gold"]
            if not isinstance(gold, str):
                raise ValueError("gold must be a string")
            if not isinstance(historical.get("correct"), bool):
                raise ValueError("correct must be a boolean")
        except (ValueError, KeyError, TypeError):
            raise ValueError(f"Invalid stored evaluation result on line {index}.") from None
        deterministic = score_answer(predicted, gold)
        candidate_exists = predicted is not None and bool(str(predicted).strip())
        adjudication = candidate_exists and not deterministic
        normalized_candidate = normalize_answer(str(predicted)) if predicted is not None else ""
        normalized_gold = normalize_answer(gold)
        rows.append({
            "row_number": index,
            "qid": historical.get("qid"),
            "historical_correct": historical["correct"],
            "historical_scored_by": historical.get("scored_by", "unknown"),
            "exact_text_agreement": bool(normalized_candidate and normalized_gold)
                and normalized_candidate == normalized_gold,
            "deterministic_agreement": deterministic,
            "requires_adjudication": adjudication,
            "status": ("deterministic_agreement" if deterministic else
                       "requires_adjudication" if adjudication else "no_prediction"),
        })
    summary = {
        "schema_version": 1,
        "scorer_version": SCORER_VERSION,
        "kind": "offline_scoring_audit",
        "source_results_sha256": sha256(raw).hexdigest(),
        "n": len(rows),
        "historical_correct_count": sum(r["historical_correct"] for r in rows),
        "deterministic_agreement_count": sum(r["deterministic_agreement"] for r in rows),
        "requires_adjudication_count": sum(r["requires_adjudication"] for r in rows),
        "historical_positives_requiring_adjudication": sum(
            r["historical_correct"] and r["requires_adjudication"] for r in rows),
        "no_prediction_count": sum(r["status"] == "no_prediction" for r in rows),
        "provider_calls": 0,
        "accuracy_established": False,
        "limitation": ("Deterministic agreement is not a corrected benchmark accuracy. "
                       "Unresolved predictions need independent adjudication; no live "
                       "queries or model judges were run. Historical files are unchanged."),
    }
    out.mkdir(parents=True, exist_ok=True)
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    summary = audit_stored_predictions(arguments.results, arguments.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
