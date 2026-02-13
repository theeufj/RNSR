"""
Contradiction Detection Benchmark for RNSR

Two-tier benchmark:

Tier 1 -- Synthetic Ground Truth (self-contained, zero external deps)
    Uses test PDFs from ``scripts/generate_test_pdfs.py`` with known
    embedded contradictions.  Validates:
    - Single-document: Greenfield Annual Report (revenue, profit, headcount)
    - Cross-document: Expert Reports A/B + Incident Report (diagnosis,
      speed, admission days, treatment, work fitness)

Tier 2 -- ContractNLI (stretch goal, not yet implemented)
    607 NDAs with 17 hypotheses, labeled Entailment/Contradiction/NotMentioned.

Metrics:
    - contradiction_recall: fraction of known contradictions detected
    - precision: fraction of detected results that match a known pair
    - f1: harmonic mean of recall and precision
    - type_accuracy: fraction of detected contradictions with correct type

Usage:
    from rnsr.benchmarks.contradiction_bench import run_contradiction_benchmark
    results = run_contradiction_benchmark()
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import structlog

from rnsr.benchmarks.standard_benchmarks import BenchmarkDataset, BenchmarkQuestion

logger = structlog.get_logger(__name__)

TEST_DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "test-documents"


# ============================================================================
# Ground-truth contradictions
# ============================================================================

@dataclass
class ExpectedContradiction:
    """A known contradiction we expect the detector to find."""
    id: str
    description: str
    source_1_keywords: list[str]  # keywords from source 1 section/claim
    source_2_keywords: list[str]  # keywords from source 2 section/claim
    expected_type: str  # "numeric", "direct", "semantic"
    # At least one keyword from each side must appear in the detected claim
    claim_1_keywords: list[str]
    claim_2_keywords: list[str]


# ── Single-doc: Greenfield Annual Report ──

GREENFIELD_CONTRADICTIONS: list[ExpectedContradiction] = [
    ExpectedContradiction(
        id="gf_revenue",
        description="Revenue: $892M (Exec Summary) vs $887M (CEO Letter)",
        source_1_keywords=["executive", "summary"],
        source_2_keywords=["ceo", "letter", "shareholders"],
        expected_type="numeric",
        claim_1_keywords=["892"],
        claim_2_keywords=["887"],
    ),
    ExpectedContradiction(
        id="gf_profit",
        description="Net profit: $134M (Exec Summary) vs $127.4M (Financial Highlights)",
        source_1_keywords=["executive", "summary"],
        source_2_keywords=["profitability", "financial"],
        expected_type="numeric",
        claim_1_keywords=["134"],
        claim_2_keywords=["127"],
    ),
    ExpectedContradiction(
        id="gf_headcount",
        description="Employees: 3,200 (Exec Summary) vs 3,450 (CEO Letter)",
        source_1_keywords=["executive", "summary"],
        source_2_keywords=["ceo", "letter", "shareholders"],
        expected_type="numeric",
        claim_1_keywords=["3,200", "3200"],
        claim_2_keywords=["3,450", "3450"],
    ),
    ExpectedContradiction(
        id="gf_offices",
        description="Offices: 14 (Exec Summary) vs 12 (HR section)",
        source_1_keywords=["executive", "summary"],
        source_2_keywords=["human", "resources"],
        expected_type="numeric",
        claim_1_keywords=["14"],
        claim_2_keywords=["12"],
    ),
    ExpectedContradiction(
        id="gf_cardioven",
        description="Cardioven sales: $312M (CEO Letter) vs $298M (Product Performance)",
        source_1_keywords=["ceo", "letter", "shareholders"],
        source_2_keywords=["product", "performance"],
        expected_type="numeric",
        claim_1_keywords=["312"],
        claim_2_keywords=["298"],
    ),
]


# ── Cross-doc: Expert Reports + Incident Report ──

CROSSDOC_CONTRADICTIONS: list[ExpectedContradiction] = [
    ExpectedContradiction(
        id="cd_diagnosis",
        description="PTSD (Hartley) vs Adjustment Disorder (Webb)",
        source_1_keywords=["hartley", "diagnosis"],
        source_2_keywords=["webb", "diagnosis"],
        expected_type="direct",
        claim_1_keywords=["ptsd", "post-traumatic"],
        claim_2_keywords=["does not meet", "adjustment disorder"],
    ),
    ExpectedContradiction(
        id="cd_speed",
        description="Forklift speed: 15 km/h (Hartley) vs 8 km/h (Incident) vs 5 km/h (Webb)",
        source_1_keywords=["hartley", "history"],
        source_2_keywords=["webb", "incident", "description"],
        expected_type="numeric",
        claim_1_keywords=["15"],
        claim_2_keywords=["5", "8"],
    ),
    ExpectedContradiction(
        id="cd_admission",
        description="Hospital admission: 6 days (Hartley) vs 7 days (Incident) vs 3 days (Webb)",
        source_1_keywords=["hartley", "history"],
        source_2_keywords=["webb", "injuries", "incident"],
        expected_type="numeric",
        claim_1_keywords=["6"],
        claim_2_keywords=["3", "7"],
    ),
    ExpectedContradiction(
        id="cd_gaf",
        description="GAF score: 45 (Hartley) vs 62 (Webb)",
        source_1_keywords=["hartley", "diagnosis"],
        source_2_keywords=["webb", "diagnosis"],
        expected_type="numeric",
        claim_1_keywords=["45"],
        claim_2_keywords=["62"],
    ),
    ExpectedContradiction(
        id="cd_treatment_sessions",
        description="Treatment: 20 sessions TF-CBT (Hartley) vs 6-8 sessions CBT (Webb)",
        source_1_keywords=["hartley", "treatment"],
        source_2_keywords=["webb", "treatment"],
        expected_type="numeric",
        claim_1_keywords=["20"],
        claim_2_keywords=["6", "8"],
    ),
    ExpectedContradiction(
        id="cd_fitness",
        description="Unfit 12 months (Hartley) vs fit for graduated return (Webb)",
        source_1_keywords=["hartley", "capacity", "work"],
        source_2_keywords=["webb", "capacity", "work"],
        expected_type="direct",
        claim_1_keywords=["unfit", "12 months"],
        claim_2_keywords=["fit", "graduated", "return"],
    ),
]


# ============================================================================
# Scoring
# ============================================================================

@dataclass
class ContradictionScore:
    """Scores for a contradiction detection run."""
    scenario: str
    total_expected: int
    total_detected: int
    true_positives: int
    contradiction_recall: float
    precision: float
    f1: float
    matched_details: list[dict[str, Any]] = field(default_factory=list)
    unmatched_expected: list[str] = field(default_factory=list)


def _contradiction_matches(
    detected: Any,  # FactContradiction
    expected: ExpectedContradiction,
) -> bool:
    """Check if a detected contradiction matches an expected one.

    Matching is fuzzy: at least one keyword from each expected claim side
    must appear somewhere in the detected contradiction's claims, sources,
    or explanation.
    """
    # Build a combined searchable string from the detected contradiction
    parts = [
        getattr(detected, "claim_1", ""),
        getattr(detected, "claim_2", ""),
        getattr(detected, "source_1", ""),
        getattr(detected, "source_2", ""),
        getattr(detected, "explanation", ""),
    ]
    combined = " ".join(parts).lower()

    has_claim1 = any(kw.lower() in combined for kw in expected.claim_1_keywords)
    has_claim2 = any(kw.lower() in combined for kw in expected.claim_2_keywords)

    return has_claim1 and has_claim2


def evaluate_contradictions(
    detected: list[Any],  # list of FactContradiction
    expected: list[ExpectedContradiction],
) -> ContradictionScore:
    """Score detected contradictions against ground truth.

    Args:
        detected: Output from a contradiction detection function.
        expected: Known ground-truth contradictions.

    Returns:
        ContradictionScore with recall, precision, and F1.
    """
    matched_expected: set[str] = set()
    matched_detected: set[int] = set()
    details: list[dict[str, Any]] = []

    for exp in expected:
        for i, det in enumerate(detected):
            if i in matched_detected:
                continue
            if _contradiction_matches(det, exp):
                matched_expected.add(exp.id)
                matched_detected.add(i)
                details.append({
                    "expected_id": exp.id,
                    "expected_desc": exp.description,
                    "detected_type": getattr(det, "type", ""),
                    "expected_type": exp.expected_type,
                    "confidence": getattr(det, "confidence", 0),
                })
                break

    tp = len(matched_expected)
    recall = tp / len(expected) if expected else 0.0
    precision = tp / len(detected) if detected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    unmatched = [
        exp.description for exp in expected
        if exp.id not in matched_expected
    ]

    return ContradictionScore(
        scenario="",
        total_expected=len(expected),
        total_detected=len(detected),
        true_positives=tp,
        contradiction_recall=recall,
        precision=precision,
        f1=f1,
        matched_details=details,
        unmatched_expected=unmatched,
    )


# ============================================================================
# Tier 1: Synthetic PDF evaluation
# ============================================================================


def _ingest_and_build_kg(pdf_path: str, doc_id: str) -> tuple[dict, Any, Any]:
    """Ingest a PDF and return (skeleton, kv_store, kg)."""
    from rnsr import ingest_document, build_skeleton_index
    from rnsr.indexing.knowledge_graph import InMemoryKnowledgeGraph
    from rnsr.extraction.rlm_unified_extractor import extract_entities_and_relationships

    result = ingest_document(pdf_path)
    skeleton, kv_store = build_skeleton_index(result.tree)

    kg = InMemoryKnowledgeGraph()
    for node_id, node in skeleton.items():
        content = kv_store.get(node_id) or ""
        if len(content.strip()) < 20:
            continue
        try:
            ext_result = extract_entities_and_relationships(
                node_id=node_id,
                doc_id=doc_id,
                header=node.header,
                content=content,
            )
            for entity in ext_result.entities:
                kg.add_entity(entity)
            for rel in ext_result.relationships:
                kg.add_relationship(rel)
        except Exception as exc:
            logger.debug("extraction_error", node=node_id, error=str(exc))

    return skeleton, kv_store, kg


def run_single_doc_benchmark() -> ContradictionScore | None:
    """Run single-document contradiction benchmark on Greenfield report."""
    from rnsr.analysis.contradiction_detector import detect_document_contradictions

    pdf_path = TEST_DOCS_DIR / "Contradictions - Greenfield Annual Report.pdf"
    if not pdf_path.exists():
        logger.warning("contradiction_bench_pdf_missing", path=str(pdf_path))
        return None

    logger.info("contradiction_bench_single_doc", doc=pdf_path.name)

    skeleton, kv_store, kg = _ingest_and_build_kg(
        str(pdf_path), "greenfield"
    )

    # Detect with LLM if available
    try:
        from rnsr.llm import get_llm
        _llm = get_llm()
        llm_fn = lambda prompt: str(_llm.complete(prompt))
    except Exception:
        llm_fn = None

    detected = detect_document_contradictions(
        kg=kg,
        skeleton=skeleton,
        kv_store=kv_store,
        doc_id="greenfield",
        llm_fn=llm_fn,
    )

    score = evaluate_contradictions(detected, GREENFIELD_CONTRADICTIONS)
    score.scenario = "single_doc_greenfield"

    logger.info(
        "contradiction_bench_single_result",
        recall=f"{score.contradiction_recall:.0%}",
        precision=f"{score.precision:.0%}",
        f1=f"{score.f1:.0%}",
        tp=score.true_positives,
        expected=score.total_expected,
        detected=score.total_detected,
    )

    return score


def run_cross_doc_benchmark() -> ContradictionScore | None:
    """Run cross-document contradiction benchmark on expert reports."""
    from rnsr.analysis.contradiction_detector import detect_cross_document_contradictions
    from rnsr.indexing.knowledge_graph import InMemoryKnowledgeGraph

    pdf_files = [
        ("CrossDoc - Expert Report A (Dr Hartley).pdf", "hartley"),
        ("CrossDoc - Expert Report B (Dr Webb).pdf", "webb"),
        ("CrossDoc - Employer Incident Report.pdf", "incident"),
    ]

    documents: list[tuple[str, dict, Any]] = []
    workspace_kg = InMemoryKnowledgeGraph()

    for pdf_name, doc_id in pdf_files:
        pdf_path = TEST_DOCS_DIR / pdf_name
        if not pdf_path.exists():
            logger.warning("contradiction_bench_pdf_missing", path=str(pdf_path))
            return None

        skeleton, kv_store, kg = _ingest_and_build_kg(str(pdf_path), doc_id)
        documents.append((doc_id, skeleton, kv_store))

        # Merge entities and relationships into workspace KG
        for entity in kg.find_entities_in_document(doc_id):
            workspace_kg.add_entity(entity)
        for entity in kg.find_entities_in_document(doc_id):
            for rel in kg.get_entity_relationships(entity.id):
                workspace_kg.add_relationship(rel)

    logger.info("contradiction_bench_cross_doc", num_docs=len(documents))

    # Get LLM if available
    try:
        from rnsr.llm import get_llm
        _llm = get_llm()
        llm_fn = lambda prompt: str(_llm.complete(prompt))
    except Exception:
        llm_fn = None

    detected = detect_cross_document_contradictions(
        kg=workspace_kg,
        documents=documents,
        llm_fn=llm_fn,
    )

    score = evaluate_contradictions(detected, CROSSDOC_CONTRADICTIONS)
    score.scenario = "cross_doc_expert_reports"

    logger.info(
        "contradiction_bench_cross_result",
        recall=f"{score.contradiction_recall:.0%}",
        precision=f"{score.precision:.0%}",
        f1=f"{score.f1:.0%}",
        tp=score.true_positives,
        expected=score.total_expected,
        detected=score.total_detected,
        unmatched=score.unmatched_expected,
    )

    return score


# ============================================================================
# Loader (for integration with run_all_benchmarks.py)
# ============================================================================


class ContradictionBenchLoader:
    """Loader for the contradiction detection benchmark.

    Returns a ``BenchmarkDataset`` describing the expected contradictions
    as BenchmarkQuestion instances.  The actual evaluation is done via
    ``run_contradiction_benchmark()``.
    """

    @staticmethod
    def load(
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> BenchmarkDataset:
        """Load contradiction ground truth as a BenchmarkDataset."""
        questions: list[BenchmarkQuestion] = []

        all_expected = GREENFIELD_CONTRADICTIONS + CROSSDOC_CONTRADICTIONS
        if max_samples:
            all_expected = all_expected[:max_samples]

        for exp in all_expected:
            questions.append(
                BenchmarkQuestion(
                    id=exp.id,
                    question=exp.description,
                    answer="contradiction",
                    context=exp.claim_1_keywords + exp.claim_2_keywords,
                    reasoning_type="contradiction",
                    metadata={
                        "expected_type": exp.expected_type,
                    },
                )
            )

        return BenchmarkDataset(
            name="contradiction",
            description=(
                "Contradiction detection benchmark. "
                "Single-doc: Greenfield Annual Report (5 known contradictions). "
                "Cross-doc: Expert Reports A/B + Incident Report (6 known contradictions)."
            ),
            questions=questions,
            metrics=["recall", "precision", "f1"],
            source_url="",
        )


# ============================================================================
# Combined runner
# ============================================================================


def run_contradiction_benchmark() -> dict[str, Any]:
    """Run the full contradiction benchmark (single-doc + cross-doc).

    Returns:
        Dict with ``single_doc`` and ``cross_doc`` score dicts.
    """
    results: dict[str, Any] = {"single_doc": None, "cross_doc": None}

    single = run_single_doc_benchmark()
    if single:
        results["single_doc"] = {
            "scenario": single.scenario,
            "recall": single.contradiction_recall,
            "precision": single.precision,
            "f1": single.f1,
            "true_positives": single.true_positives,
            "total_expected": single.total_expected,
            "total_detected": single.total_detected,
            "matched": single.matched_details,
            "unmatched": single.unmatched_expected,
        }

    cross = run_cross_doc_benchmark()
    if cross:
        results["cross_doc"] = {
            "scenario": cross.scenario,
            "recall": cross.contradiction_recall,
            "precision": cross.precision,
            "f1": cross.f1,
            "true_positives": cross.true_positives,
            "total_expected": cross.total_expected,
            "total_detected": cross.total_detected,
            "matched": cross.matched_details,
            "unmatched": cross.unmatched_expected,
        }

    return results
