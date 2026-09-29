"""Question-derived evidence requirements, never inferred from reference answers.

These distinctions constrain what a reviewer must establish. They do not decide
whether a particular source satisfies a category or manufacture a replacement
answer. Caller definitions and the exact question remain the task authority.
"""
from __future__ import annotations

import re

VERSION = "task-requirements-v2"

_PRICE = {
    "requirement_id": "actual_price_constraint",
    "requirement": (
        "For a claimed price restriction, identify a binding constraint on the actual price a party may "
        "charge, raise or reduce (such as a floor, ceiling, price approval or mandated pricing rule). "
        "A royalty/fee valuation basis for an already-permitted discounted sale is not by itself a "
        "restriction on the sale price. Distinguish the price charged to the buyer from the amount "
        "used to calculate payments to another party. Financial consequences alone do not establish "
        "a prohibition on raising or lowering prices."
    ),
    "support_example": "The retailer shall not sell the product below the agreed minimum resale price.",
    "insufficient_example": "If a product is sold below list price, royalties are still calculated using list price.",
}
_EXCLUSIVE = {
    "requirement_id": "exclusive_counterparty_commitment",
    "requirement": (
        "For a claimed exclusive dealing commitment, identify the exclusive counterparty/relationship "
        "and the actual duty to obtain requirements from that party or refrain from dealing with "
        "third parties in the requested scope. Approval of a pool of suppliers is a qualification "
        "rule, not sole-source procurement from the counterparty. A non-compete or non-solicitation "
        "restriction limited to competitors/customer diversion does not alone establish exclusivity "
        "with the counterparty. Do not expand a competitor restriction to all other parties; inspect "
        "the operative obligation, covered transactions and beneficiaries against the question."
    ),
    "support_example": "Buyer shall purchase all of its requirements for these components exclusively from Seller.",
    "insufficient_example": "Buyer may use any approved supplier and must not divert customers to competitors.",
}
_UNLIMITED = {
    "requirement_id": "affirmative_unlimited_usage_grant",
    "requirement": (
        "For a claimed enterprise/all-you-can-eat/unlimited usage licence, identify affirmative "
        "language granting the relevant unlimited or enterprise-wide usage scope. No additional "
        "cost, royalty-free status, or the absence of an express cap in a short excerpt does not "
        "establish unlimited usage. Do not infer an affirmative unlimited right from silence. "
        "A genuine unlimited grant may still have a fixed term, territory or permitted purpose."
    ),
    "support_example": "Licensee may deploy unlimited copies to any number of users throughout its enterprise during the term.",
    "insufficient_example": "Licensee may use the data at no additional cost solely to operate the licensed business.",
}
_DIRECTION = {
    "requirement_id": "requested_direction",
    "requirement": (
        "The requested result is direction (increase/decrease or higher/lower), not a precise amount "
        "or reconstruction of a separate financial line item. Relevant narrative evidence can "
        "establish direction without exact amounts. For an expense as a share of sales, expense "
        "deleverage means its share increased and expense leverage means its share decreased, "
        "when that is the stated source context. Do not substitute the direction of a combined "
        "expense total for the requested component. An answer claiming NOT_FOUND solely because "
        "exact amounts are absent does not answer a directional question when the relevant "
        "direction is stated. A missing period/scope still requires retrieval rather than guessing."
    ),
    "support_example": "Payroll deleverage increased the payroll share of sales; no separate payroll dollar amount is needed to state that direction.",
    "insufficient_example": "The direction cannot be answered because exact payroll dollars are not separately reported.",
}


def requires_classification_proof(question: str, category_definitions: str | None = None) -> bool:
    """Identify semantic classification aggregates, not ordinary factual counts.

    Caller labels are task vocabulary, never expected counts. Requiring the
    parent proof prevents a reviewer from replacing strict recorded row labels
    with an informal recount from quoted examples.
    """
    text = " ".join(question.casefold().split())
    aggregate = bool(re.search(
        r"\b(?:how many|count(?:s|ed|ing)?|number of|frequency|frequencies|distribution|"
        r"breakdown|aggregate|aggregating|aggregation|most common|least common|"
        r"more common|less common|most frequent|least frequent|more frequent|less frequent|"
        r"highest frequency|lowest frequency)\b", text))
    if not aggregate:
        return False
    explicit = bool(re.search(
        r"\b(?:classify|classifying|categorize|categorizing|categorise|categorising|"
        r"labeling|labelling)\b", text))
    population = bool(re.search(r"\b(?:instances?|rows?|records?|examples?|items?|entries|questions?|comments?|utterances?)\b", text))
    categories = bool(re.search(r"\b(?:categor(?:y|ies)|labels?|classes|classification)\b", text))
    category_comparison = categories and bool(re.search(
        r"\b(?:frequency|frequencies|(?:most|least|more|less) (?:common|frequent))\b", text))
    if explicit or (population and categories) or category_comparison:
        return True
    definitions = category_definitions or ""
    if not definitions.strip():
        return False
    # Common caller rubric formats: "Positive: ...; Negative: ..." or one
    # definition per line/sentence. A category word in a count question signals
    # label aggregation; unrelated definitions must not turn a factual employee
    # count or contract duration into a semantic classification task.
    labels = re.findall(r"(?:^|[.;\n])\s*([^:;.\n]{1,80}):", definitions)
    return bool(categories or any(re.search(r"(?<!\w)" + re.escape(label.strip().casefold()) +
                                           r"(?!\w)", text) for label in labels if label.strip()))


def extract_requirements(question: str, category_definitions: str | None = None) -> dict:
    """Produce bounded review criteria solely from the caller's task wording."""
    text = " ".join(question.casefold().split())
    amount = bool(re.search(r"\b(?:by how much|how much|what (?:was|is|were|are) the (?:amount|percentage|ratio)|calculate|compute|quantify)\b", text))
    direction = bool(re.search(
        r"\b(?:increas(?:e|ed|ing) or decreas(?:e|ed|ing)|decreas(?:e|ed) or increas(?:e|ed)|"
        r"higher or lower|rose or fell|risen or fallen|increase versus decrease|direction)\b", text))
    checks = []
    if re.search(r"\bprice restrictions?\b|\brestriction\w* on\b.{0,90}\b(?:raise|reduce|change)\b.{0,35}\bprices?\b", text):
        checks.append(dict(_PRICE))
    if re.search(r"\bexclusivity\b|\bexclusive dealing\b|\bexclusive counterparty\b|\bsole[- ]source\b", text):
        checks.append(dict(_EXCLUSIVE))
    if (re.search(r"\bunlimited\b|\ball[- /]you[- ]can[- ]eat\b|\benterprise\b", text)
            and re.search(r"\bli[cs]en[cs]e\b|\blicen[cs]ing\b", text)):
        checks.append(dict(_UNLIMITED))
    if direction and not amount:
        checks.append(dict(_DIRECTION))
    years = list(dict.fromkeys(re.findall(r"\b(?:19|20)\d{2}\b|(?<=fy)(?:19|20)\d{2}\b", text)))
    quarter = bool(re.search(r"\bquarter\b|\bq[1-4]\b", text))
    fiscal = bool(re.search(r"\bfy\s*\d|\bfiscal\b", text))
    if checks and years:
        checks.append({
            "requirement_id": "requested_period_scope",
            "requirement": (
                "Establish the requested period from the evidence's actual headings/qualifications. "
                "A publication date is not automatically the fiscal period. A quarter cannot stand "
                "in for a full fiscal year, and a heading for one period cannot be attached to a "
                "value or narrative from another. If fiscal/calendar labels differ, require an "
                "explicit source-supported mapping; do not invent one."
            ),
        })
    return {
        "version": VERSION,
        "answer_form": "direction" if direction and not amount else "quantity_or_calculation" if amount else "as_asked",
        "requested_years": years,
        "requested_period_scope": "quarter" if quarter else "fiscal_year" if fiscal else "as_asked",
        "required_parent_proofs": (["classification_aggregate"]
                                   if requires_classification_proof(question, category_definitions) else []),
        "checks": checks,
        "definition_policy": (
            "Apply the exact caller question and supplied category definitions. These generic "
            "distinctions do not broaden a category. For negative/absence answers, establish "
            "coverage of the requested source scope; a missing qualifying excerpt alone is insufficient."
        ),
    }
