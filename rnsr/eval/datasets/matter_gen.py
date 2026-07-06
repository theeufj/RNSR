"""Synthetic legal matter-file benchmark: the multi-document regime.

Generates a realistic commercial-dispute matter as real PDFs — an MSA,
amendments that override its terms, an SOW with rate tables, numbered
invoices, breach/demand correspondence, an optional personal guarantee,
and bulk file notes — totalling ~1M characters across dozens of documents
(beyond a single context window). Every gold is exact by construction, and
the operative facts carry internal redundancy (the demand letter restates
the invoice arithmetic) in the §3.3 spirit.

Question classes:
  single-doc    a fact in one document among many
  cross-doc     the CURRENT term after amendments (requires date ordering)
  aggregation   totals/counts across the invoice set
  timeline      computed deadlines (breach date + cure period)
  absent        guarantees/clauses that may not exist at all
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

from rnsr.eval.datasets.base import EvalItem

_ROLES = ["Senior Consultant", "Project Manager", "Data Engineer", "Analyst"]
_TOPICS = [
    "office relocation logistics", "quarterly staff townhall", "IT system migration",
    "catering arrangements for the client seminar", "carpark licence renewal",
    "annual insurance review", "records management audit", "travel policy update",
]
_FILLER_SENTENCES = [
    "The team discussed {topic} at length and resolved to revisit the item next month.",
    "Counsel noted that no action arises from {topic} at this stage.",
    "An estimate of ${amt:,} was tabled in relation to {topic}, subject to approval.",
    "It was agreed that {topic} raises no contractual implications for current engagements.",
    "The file was updated to reflect correspondence received regarding {topic}.",
    "A follow-up meeting on {topic} was scheduled for the {day}th of the month.",
    "Attendees confirmed the figures circulated previously on {topic} remain indicative only.",
]


def _fmt(d: date) -> str:
    return d.strftime("%-d %B %Y")


class MatterFacts:
    """All operative facts, derived deterministically from the seed."""

    def __init__(self, seed: int):
        rng = random.Random(seed)
        self.client = rng.choice(["Meridian Logistics Pty Ltd", "Harbourline Freight Pty Ltd"])
        self.vendor = rng.choice(["Corvid Systems Pty Ltd", "Atlas Digital Pty Ltd"])
        self.vendor_signatory = rng.choice(["Marcus Ellery", "Diane Okafor"])
        self.msa_date = date(2022, rng.randint(2, 6), rng.randint(1, 28))
        self.payment_terms_0 = rng.choice([30, 45])
        self.indemnity_cap_0 = rng.randint(15, 40) * 100_000
        self.a1_date = self.msa_date + timedelta(days=rng.randint(200, 320))
        self.payment_terms_1 = rng.choice([14, 21])          # amendment 1 changes terms
        self.a2_date = self.a1_date + timedelta(days=rng.randint(120, 240))
        self.indemnity_cap_2 = self.indemnity_cap_0 + rng.randint(5, 15) * 100_000
        self.rates = {r: rng.randint(140, 320) * 5 for r in _ROLES}
        n_inv = rng.randint(9, 14)
        self.invoices = []
        inv_date = self.a1_date + timedelta(days=30)
        for i in range(1, n_inv + 1):
            amount = rng.randint(180, 950) * 100
            paid = rng.random() < 0.6
            self.invoices.append({"no": f"INV-{2300 + i}", "date": inv_date,
                                  "amount": amount, "paid": paid})
            inv_date += timedelta(days=rng.randint(18, 40))
        if not any(not i["paid"] for i in self.invoices):
            self.invoices[-1]["paid"] = False
        self.breach_date = inv_date + timedelta(days=rng.randint(10, 30))
        self.cure_days = rng.choice([14, 21, 28])
        self.guarantee_exists = rng.random() < 0.5
        self.guarantor = "Peter Halloway"

    @property
    def total_invoiced(self) -> int:
        return sum(i["amount"] for i in self.invoices)

    @property
    def outstanding(self) -> list[dict]:
        return [i for i in self.invoices if not i["paid"]]

    @property
    def cure_deadline(self) -> date:
        return self.breach_date + timedelta(days=self.cure_days)


def _pdf(path: Path, title: str, blocks: list, tables: list | None = None) -> None:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 10)]
    for b in blocks:
        story.append(Paragraph(b, styles["Heading2"] if b.isupper() else styles["BodyText"]))
        story.append(Spacer(1, 6))
    for grid in tables or []:
        story.append(Table(grid, style=TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, "black"),
            ("BACKGROUND", (0, 0), (-1, 0), "#dddddd"),
        ])))
        story.append(Spacer(1, 10))
    SimpleDocTemplate(str(path), pagesize=LETTER).build(story)


def _write_operative_docs(out: Path, f: MatterFacts) -> None:
    _pdf(out / "01_master_services_agreement.pdf",
         f"Master Services Agreement — {f.client} and {f.vendor}",
         [f"This Master Services Agreement is made on {_fmt(f.msa_date)} between "
          f"{f.client} (the Client) and {f.vendor} (the Vendor).",
          "PAYMENT TERMS",
          f"The Client shall pay each correctly rendered invoice within "
          f"{f.payment_terms_0} days of receipt (net {f.payment_terms_0} days).",
          "LIMITATION AND INDEMNITY",
          f"The Vendor's aggregate liability under the indemnity in this Agreement "
          f"is capped at ${f.indemnity_cap_0:,}.",
          "DISPUTES",
          "Any dispute arising under this Agreement shall be determined exclusively "
          "by the courts of New South Wales. For the avoidance of doubt the parties "
          "do not agree to refer disputes to arbitration.",
          "EXECUTION",
          f"Signed for the Vendor by {f.vendor_signatory}, Director."])
    _pdf(out / "02_amendment_1.pdf",
         "Amendment No. 1 to Master Services Agreement",
         [f"This Amendment No. 1 is made on {_fmt(f.a1_date)}.",
          "VARIATION",
          f"Clause 'Payment Terms' of the Agreement is deleted and replaced such "
          f"that invoices are payable within {f.payment_terms_1} days of receipt "
          f"(net {f.payment_terms_1} days). All other terms are unchanged."])
    _pdf(out / "03_amendment_2.pdf",
         "Amendment No. 2 to Master Services Agreement",
         [f"This Amendment No. 2 is made on {_fmt(f.a2_date)}.",
          "VARIATION",
          f"The indemnity cap in the clause 'Limitation and Indemnity' is increased "
          f"to ${f.indemnity_cap_2:,} with effect from the date of this Amendment. "
          "All other terms are unchanged."])
    _pdf(out / "04_statement_of_work.pdf",
         "Statement of Work No. 1 — Professional Services Rates",
         ["The following hourly rates apply to services performed under the MSA."],
         tables=[[["Role", "Hourly Rate ($)"]] + [[r, f"{v:,}"] for r, v in f.rates.items()]])
    for inv in f.invoices:
        _pdf(out / f"inv_{inv['no']}.pdf",
             f"Tax Invoice {inv['no']} — {f.vendor}",
             [f"Date: {_fmt(inv['date'])}", f"To: {f.client}",
              "Professional services rendered under MSA / SOW No. 1."],
             tables=[[["Description", "Amount ($)"],
                      ["Professional services", f"{inv['amount']:,}"],
                      ["TOTAL DUE", f"{inv['amount']:,}"]]])
    paid = [i["no"] for i in f.invoices if i["paid"]]
    _pdf(out / "20_remittance_letter.pdf",
         f"Letter — Remittance advice from {f.client}",
         [f"We confirm payment in full of the following invoices: {', '.join(paid)}. "
          "No other invoices have been paid to date."])
    out_total = sum(i["amount"] for i in f.outstanding)
    _pdf(out / "21_breach_notice.pdf",
         f"Notice of Breach — {f.vendor} to {f.client}",
         [f"Date: {_fmt(f.breach_date)}",
          f"You are in breach of the Agreement by reason of non-payment. You must "
          f"remedy the breach within {f.cure_days} days of the date of this notice, "
          "failing which we reserve all rights."])
    _pdf(out / "22_demand_letter.pdf",
         f"Letter of Demand — {f.vendor} to {f.client}",
         [f"As at the date of this letter the total amount outstanding across unpaid "
          f"invoices is ${out_total:,}. Payment is demanded within 7 days."])
    if f.guarantee_exists:
        _pdf(out / "23_personal_guarantee.pdf",
             "Deed of Guarantee and Indemnity",
             [f"{f.guarantor} unconditionally guarantees to {f.vendor} the due and "
              f"punctual payment of all amounts owed by {f.client} under the MSA."])


def _write_filler(out: Path, seed: int, n_filler: int, chars_each: int) -> None:
    rng = random.Random(seed + 777)
    for k in range(n_filler):
        topic_pool = rng.sample(_TOPICS, 3)
        blocks = []
        text_len = 0
        while text_len < chars_each:
            s = rng.choice(_FILLER_SENTENCES).format(
                topic=rng.choice(topic_pool), amt=rng.randint(50, 900) * 10,
                day=rng.randint(1, 28))
            blocks.append(s + " " + rng.choice(_FILLER_SENTENCES).format(
                topic=rng.choice(topic_pool), amt=rng.randint(50, 900) * 10,
                day=rng.randint(1, 28)))
            text_len += len(blocks[-1])
        _pdf(out / f"note_{k:02d}.pdf", f"File Note {k + 1} — internal memorandum", blocks)


def generate_matter(out_dir: str | Path, *, n_filler: int = 32,
                    filler_chars: int = 26_000, seed: int = 5) -> list[EvalItem]:
    """Write the matter file (PDFs) and return its questions with exact golds."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = MatterFacts(seed)
    # Idempotent: reportlab stamps creation dates into the bytes, so
    # regeneration would change content hashes and defeat corpus caching.
    if not (out / "01_master_services_agreement.pdf").exists():
        _write_operative_docs(out, f)
        _write_filler(out, seed, n_filler, filler_chars)
    sources = sorted(out.glob("*.pdf"))

    out_total = sum(i["amount"] for i in f.outstanding)
    qs = [
        ("single-doc",
         "What was the indemnity cap under the Master Services Agreement as "
         "originally executed?", f"${f.indemnity_cap_0:,}"),
        ("cross-doc",
         "What is the indemnity cap currently in effect, taking into account all "
         "amendments?", f"${f.indemnity_cap_2:,}"),
        ("cross-doc",
         "What payment terms currently apply to invoices, taking into account all "
         "amendments?", f"net {f.payment_terms_1} days"),
        ("aggregation",
         "How many tax invoices has the Vendor issued in this matter?",
         str(len(f.invoices))),
        ("aggregation",
         "What is the total amount across ALL tax invoices issued (paid and "
         "unpaid)?", f"${f.total_invoiced:,}"),
        ("aggregation",
         "What is the total amount outstanding across unpaid invoices?",
         f"${out_total:,}"),
        ("timeline",
         "By what date must the Client remedy the breach identified in the notice "
         "of breach?", _fmt(f.cure_deadline)),
        ("timeline",
         "On what date was Amendment No. 2 made?", _fmt(f.a2_date)),
        ("single-doc",
         "What is the hourly rate for a Project Manager under Statement of Work "
         "No. 1?", f"${f.rates['Project Manager']:,}"),
        ("single-doc",
         "Who signed the Master Services Agreement for the Vendor?",
         f.vendor_signatory),
        ("absent",
         f"Is there a personal guarantee from {f.guarantor} anywhere in the "
         "matter file?", "Yes" if f.guarantee_exists else "No"),
        ("absent",
         "Do the parties agree to arbitration of disputes under the MSA?",
         "No"),
    ]
    return [
        EvalItem(qid=f"matter-{seed}-{i:02d}", question=q, gold=gold,
                 task_class=cls, sources=list(sources),
                 meta={"n_docs": len(sources)})
        for i, (cls, q, gold) in enumerate(qs)
    ]
