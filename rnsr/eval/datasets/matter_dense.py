"""Dense synthetic matter: long rather than numerous, with aggregation golds.

A 36-month professional-services engagement between a client and a vendor
across three projects, written as PDFs and text files with exact golds by
construction. Where ``matter_gen`` is ~250k tokens with 40 one-line invoices,
this writes ~2.8M characters (~750k tokens): hundreds of multi-line tax
invoices with GST and TOTAL rows, credit notes and reissues, monthly
remittance advices, pro-forma quotes as distractors, a mid-stream rate
amendment with two invoices that wrongly keep the old rate, and multi-page
status reports whose "indicative" hours disagree with what was invoiced.

Question classes (``question_type`` in answer_key.json):
  aggregation   sums/counts over invoice lines by project, FY, role, person
  set_logic     outstanding = invoices - credit notes - remittances, at a date
  join          facts that need two tables (rate table x invoice lines)
  cross_doc     the current term after an amendment
  value_lookup  one fact in one document among hundreds
  timeline      computed or located dates
  existence     a yes/no the documents settle explicitly
  unanswerable  things the matter does not contain

Layout (same as testMatter/matters/<name>/ plus scorer inputs):
  documents/            the corpus
  qa.json               [{id, question, answer, unanswerable}]
  answer_key.json       [{id, question, answer, question_type, sources}]
  questions.csv         ground_truth_question,field_id   (answer-csv input)
  golden.json           {items:[{id, question, golden, note, field_context}]}
  map.json              standalone items, needs_value false (rnsr regress --map)

    python -m rnsr.eval.datasets.matter_dense --out testMatter/matters/dense_corvid_meridian --seed 11
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

# --------------------------------------------------------------------------- facts

ROLES = ["Senior Consultant", "Consultant", "Project Manager", "Data Engineer", "Analyst"]
PEOPLE = [("Priya Natarajan", "Senior Consultant"), ("Tom Abernethy", "Senior Consultant"),
          ("Ling Zhao", "Consultant"), ("Marcus Oyelaran", "Consultant"), ("Sofia Reinholt", "Consultant"),
          ("Declan Murray", "Project Manager"), ("Hannah Kowalczyk", "Project Manager"),
          ("Arjun Mehta", "Data Engineer"), ("Beatriz Camargo", "Data Engineer"), ("Noah Fitzgerald", "Data Engineer"),
          ("Imogen Blake", "Analyst"), ("Yusuf Demir", "Analyst")]
PROJECTS = {"PRJ-ALPHA": "Warehouse Management System Replacement",
            "PRJ-BRAVO": "Fleet Telematics Integration",
            "PRJ-CHARLIE": "Data Platform Migration"}
CENT = Decimal("0.01")


def money(x: Decimal) -> str:
    return f"${x.quantize(CENT, ROUND_HALF_UP):,.2f}"


def fmt(d: date) -> str:
    return d.strftime("%-d %B %Y")


def fy(d: date) -> int:
    """Australian financial year label: FY2025 = 1 Jul 2024 .. 30 Jun 2025."""
    return d.year + 1 if d.month >= 7 else d.year


def month_starts(first: date, n: int):
    y, m = first.year, first.month
    for _ in range(n):
        yield date(y, m, 1)
        m += 1
        if m > 12:
            y, m = y + 1, 1


class Facts:
    def __init__(self, seed: int, n_months: int = 36, invoices_per_month: int = 10):
        rng = random.Random(seed)
        self.seed = seed
        self.client = "Meridian Logistics Pty Ltd"
        self.vendor = "Corvid Systems Pty Ltd"
        self.vendor_signatory = rng.choice(["Marcus Ellery", "Diane Okafor"])
        self.client_signatory = rng.choice(["Helen Tran", "Rowan Castellano"])
        self.start = date(2023, 7, 1)
        self.months = list(month_starts(self.start, n_months))
        self.as_at = self.months[-1].replace(day=28) + timedelta(days=4)  # last day of final month
        self.msa_date = self.start - timedelta(days=rng.randint(20, 45))
        self.payment_terms_0 = 30
        self.payment_terms_1 = rng.choice([14, 21])
        self.a1_date = date(2024, rng.randint(2, 5), rng.randint(1, 28))
        self.indemnity_cap_0 = rng.randint(15, 30) * 100_000
        self.indemnity_cap_2 = self.indemnity_cap_0 + rng.randint(5, 12) * 100_000
        self.a2_date = date(2025, rng.randint(2, 9), rng.randint(1, 28))
        self.rates_0 = {"Senior Consultant": 285, "Consultant": 210, "Project Manager": 260,
                        "Data Engineer": 240, "Analyst": 165}
        self.sow_amend_date = date(2025, 1, 1)
        self.rates_1 = {**self.rates_0, "Senior Consultant": 310, "Data Engineer": 265}
        self.guarantee_exists = rng.random() < 0.5
        self.guarantor = "Peter Halloway"
        self.people = PEOPLE
        self.pm = {"PRJ-ALPHA": "Declan Murray", "PRJ-BRAVO": "Hannah Kowalczyk", "PRJ-CHARLIE": "Declan Murray"}
        # ---- invoices -------------------------------------------------------
        self.invoices: list[dict] = []
        no = 2400
        project_cycle = list(PROJECTS)
        for mi, m in enumerate(self.months):
            for k in range(invoices_per_month):
                no += 1
                project = project_cycle[(mi + k) % 3] if k < 6 else rng.choice(project_cycle)
                inv_date = m + timedelta(days=rng.randint(1, 27))
                rates = self.rates_1 if inv_date >= self.sow_amend_date else self.rates_0
                n_lines = rng.randint(3, 7)
                lines = []
                for p, role in rng.sample(self.people, n_lines):
                    hours = Decimal(rng.randint(4, 36) * 5) / 2  # 10.0 .. 90.0 in 2.5 steps
                    rate = Decimal(rates[role])
                    lines.append({"person": p, "role": role, "hours": hours, "rate": rate,
                                  "amount": (hours * rate).quantize(CENT)})
                self.invoices.append({"no": f"INV-{no}", "date": inv_date, "project": project,
                                      "po": f"PO-{rng.randint(70000, 79999)}",
                                      "period": m.strftime("%B %Y"), "lines": lines,
                                      "status": "issued", "replaces": None, "rate_error": False})
        # Two invoices after the SOW amendment wrongly bill Senior Consultant at the old rate.
        after = [i for i in self.invoices if i["date"] >= self.sow_amend_date + timedelta(days=20)
                 and any(ln["role"] == "Senior Consultant" for ln in i["lines"])]
        for inv in rng.sample(after, min(2, len(after))):
            inv["rate_error"] = True
            for ln in inv["lines"]:
                if ln["role"] == "Senior Consultant":
                    ln["rate"] = Decimal(self.rates_0["Senior Consultant"])
                    ln["amount"] = (ln["hours"] * ln["rate"]).quantize(CENT)
        # Credit notes: cancel some invoices in full and reissue with a corrected PO/period.
        self.credit_notes: list[dict] = []
        cancel = rng.sample([i for i in self.invoices if not i["rate_error"]], max(12, len(self.invoices) // 24))
        for inv in cancel:
            inv["status"] = "cancelled"
            cn_date = inv["date"] + timedelta(days=rng.randint(7, 25))
            no += 1
            self.credit_notes.append({"no": f"CN-{len(self.credit_notes) + 101}", "date": cn_date,
                                      "against": inv["no"], "amount": self.total(inv),
                                      "reason": rng.choice(["incorrect purchase order reference",
                                                            "service period misstated",
                                                            "duplicate of an earlier invoice"])})
            reissue = {**inv, "no": f"INV-{no}", "date": cn_date + timedelta(days=rng.randint(1, 6)),
                       "po": f"PO-{rng.randint(70000, 79999)}", "status": "issued",
                       "replaces": inv["no"], "lines": [dict(ln) for ln in inv["lines"]]}
            self.invoices.append(reissue)
        self.invoices.sort(key=lambda i: (i["date"], i["no"]))
        # Disputed invoices: never paid.
        live = [i for i in self.invoices if i["status"] == "issued" and i["replaces"] is None]
        pool = [i for i in live if i["date"] < self.months[-min(6, len(self.months) - 1)]]
        self.disputed = sorted(rng.sample(pool, min(5, len(pool))), key=lambda i: i["date"])
        for inv in self.disputed:
            inv["status"] = "disputed"
        # Remittances: monthly, pay issued invoices dated >= 45 days earlier, none in last 3 months.
        self.remittances: list[dict] = []
        paid: set[str] = set()
        for m in self.months[2:]:
            rem_date = m + timedelta(days=rng.randint(10, 24))
            due = [i for i in self.invoices if i["status"] == "issued" and i["no"] not in paid
                   and i["date"] <= rem_date - timedelta(days=45)]
            if not due:
                continue
            self.remittances.append({"no": f"REM-{m:%Y%m}", "date": rem_date,
                                     "invoices": [i["no"] for i in due],
                                     "amount": sum(self.total(i) for i in due)})
            paid.update(i["no"] for i in due)
        self.paid = paid
        # Pro-forma quotes: invoice-shaped distractors that are not tax invoices.
        self.quotes = []
        for q in range(8):
            m = rng.choice(self.months)
            self.quotes.append({"no": f"Q-{2400 + q * 7}", "date": m + timedelta(days=rng.randint(1, 27)),
                                "project": rng.choice(project_cycle),
                                "amount": Decimal(rng.randint(180, 950) * 100)})
        # Breach / demand after the as-at date is irrelevant; set within the last months.
        self.breach_date = self.months[-2] + timedelta(days=rng.randint(3, 20))
        self.cure_days = rng.choice([14, 21, 28])
        self.demand_as_at = self.months[-4] + timedelta(days=15)   # stale figure in the demand letter

    # --- derived -------------------------------------------------------------
    @staticmethod
    def subtotal(inv: dict) -> Decimal:
        return sum((ln["amount"] for ln in inv["lines"]), Decimal(0))

    @classmethod
    def gst(cls, inv: dict) -> Decimal:
        return (cls.subtotal(inv) * Decimal("0.10")).quantize(CENT, ROUND_HALF_UP)

    @classmethod
    def total(cls, inv: dict) -> Decimal:
        return cls.subtotal(inv) + cls.gst(inv)

    def live(self) -> list[dict]:
        return [i for i in self.invoices if i["status"] in ("issued", "disputed")]

    def outstanding_at(self, d: date) -> Decimal:
        paid = {no for r in self.remittances if r["date"] <= d for no in r["invoices"]}
        return sum((self.total(i) for i in self.live() if i["date"] <= d and i["no"] not in paid), Decimal(0))

    @property
    def cure_deadline(self) -> date:
        return self.breach_date + timedelta(days=self.cure_days)


# --------------------------------------------------------------------------- prose

_WORKSTREAMS = ["data migration", "interface build", "user acceptance testing", "cut-over planning",
                "device provisioning", "reporting layer", "security hardening", "change management",
                "vendor onboarding", "performance tuning", "defect triage", "training delivery",
                "environment refresh", "reconciliation", "release packaging"]
_SYSTEMS = ["the staging environment", "the SAP interface", "the telematics gateway", "the Snowflake tenant",
            "the WMS pilot site", "the identity provider", "the nightly batch", "the dashboard suite",
            "the depot handhelds", "the archive store", "the message bus", "the test harness"]
_ISSUES = ["intermittent timeouts", "a schema drift", "duplicate records", "a certificate expiry",
           "scope creep on reporting", "late sign-off from operations", "a licensing shortfall",
           "unplanned leave", "a regression in the picking workflow", "firmware incompatibility",
           "a data quality finding", "ambiguity in the acceptance criteria"]
_ACTIONS = ["agreed to re-baseline the plan", "raised a change request", "escalated to the steering committee",
            "deferred the item to the next release", "scheduled a workshop", "accepted the risk",
            "requested a vendor fix", "added two test cycles", "reallocated effort from {ws}",
            "confirmed no contractual implication arises", "asked for a written position", "closed the action"]
_OPEN = ["During {month} the team focused on {ws} across {sys}.",
         "{ws_c} progressed against the plan, with {sys} the main area of attention.",
         "The period was dominated by {ws} and the knock-on effects for {sys}.",
         "Effort in {month} concentrated on {ws}; {sys} remained stable throughout."]
_MID = ["{issue_c} was identified in {sys} and the team {action}.",
        "Operations reported {issue}; after review the parties {action}.",
        "A recurring theme this month was {issue}, which affected {ws}.",
        "The steering committee noted {issue} and {action}.",
        "{person} led {ws} and reported that {sys} would be ready after {ws2}.",
        "Testing of {sys} surfaced {issue}; {person} {action}.",
        "No further issues were recorded against {sys} once {ws} completed.",
        "{person} and {person2} coordinated {ws} with the client's {ws2} lead."]
_CLOSE = ["Next month's priorities are {ws} and {ws2}.", "Risks are reviewed in the table below.",
          "Hours shown are indicative management figures and are not the invoiced hours.",
          "The figures in this report are compiled from timesheets before invoicing and may differ from the tax invoices."]


class Prose:
    def __init__(self, rng: random.Random, facts: Facts):
        self.rng, self.f = rng, facts

    def slots(self, month: str) -> dict:
        r = self.rng
        ws, ws2 = r.sample(_WORKSTREAMS, 2)
        issue = r.choice(_ISSUES)
        return {"month": month, "ws": ws, "ws2": ws2, "ws_c": ws[0].upper() + ws[1:], "sys": r.choice(_SYSTEMS),
                "issue": issue, "issue_c": issue[0].upper() + issue[1:],
                "action": r.choice(_ACTIONS).format(ws=r.choice(_WORKSTREAMS)),
                "person": r.choice(self.f.people)[0], "person2": r.choice(self.f.people)[0]}

    def paragraph(self, month: str, n: int = 5) -> str:
        parts = [self.rng.choice(_OPEN).format(**self.slots(month))]
        parts += [self.rng.choice(_MID).format(**self.slots(month)) for _ in range(n - 1)]
        return " ".join(parts)

    def report(self, month: str, chars: int) -> list[str]:
        blocks = []
        total = 0
        while total < chars:
            heading = self.rng.choice(["PROGRESS", "ISSUES AND DECISIONS", "RISKS", "STAKEHOLDERS",
                                       "SCHEDULE", "QUALITY", "DEPENDENCIES", "NEXT PERIOD"])
            para = self.paragraph(month, self.rng.randint(4, 8))
            blocks += [heading, para]
            total += len(para)
        blocks.append(self.rng.choice(_CLOSE).format(**self.slots(month)))
        return blocks


# --------------------------------------------------------------------------- writers

def _pdf(path: Path, title: str, blocks: list[str], tables: list[list[list[str]]] | None = None,
         table_titles: list[str] | None = None) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 10)]
    for b in blocks:
        story.append(Paragraph(b, styles["Heading2"] if b.isupper() else styles["BodyText"]))
        story.append(Spacer(1, 6))
    for ti, grid in enumerate(tables or []):
        if table_titles and ti < len(table_titles):
            story.append(Paragraph(table_titles[ti], styles["Heading3"]))
        story.append(Table(grid, style=TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, "black"),
            ("BACKGROUND", (0, 0), (-1, 0), "#dddddd"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]), repeatRows=1))
        story.append(Spacer(1, 10))
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


def _invoice_pdf(out: Path, f: Facts, inv: dict) -> str:
    name = f"{inv['date']:%Y-%m-%d}_Tax_Invoice_{inv['no']}_{inv['project']}.pdf"
    blocks = [f"Tax Invoice {inv['no']}", f"Date: {fmt(inv['date'])}", f"From: {f.vendor} (ABN 51 824 753 556)",
              f"To: {f.client}", f"Project: {inv['project']} — {PROJECTS[inv['project']]}",
              f"Purchase order: {inv['po']}", f"Service period: {inv['period']}",
              "Professional services rendered under the Master Services Agreement and Statement of Work No. 1."]
    if inv["replaces"]:
        blocks.append(f"This invoice replaces {inv['replaces']}, which was cancelled by credit note.")
    grid = [["Consultant", "Role", "Hours", "Rate ($/hr)", "Amount ($)"]]
    for ln in inv["lines"]:
        grid.append([ln["person"], ln["role"], f"{ln['hours']:.1f}", f"{ln['rate']:,.2f}", f"{ln['amount']:,.2f}"])
    grid += [["", "", "", "Subtotal (ex GST)", f"{f.subtotal(inv):,.2f}"],
             ["", "", "", "GST (10%)", f"{f.gst(inv):,.2f}"],
             ["", "", "", "TOTAL DUE (inc GST)", f"{f.total(inv):,.2f}"]]
    terms = f.payment_terms_1 if inv["date"] >= f.a1_date else f.payment_terms_0
    blocks.append(f"Payment terms: net {terms} days from receipt. Please quote the invoice number on remittance.")
    _pdf(out / name, f"TAX INVOICE {inv['no']}", blocks, [grid])
    return name


def _write_corpus(out: Path, f: Facts, report_chars: int) -> dict[str, list[str]]:
    """Write all documents; return {doc_kind: [file names]} for source keys."""
    rng = random.Random(f.seed + 101)
    prose = Prose(rng, f)
    docs: dict[str, list[str]] = {}

    def add(kind: str, name: str) -> None:
        docs.setdefault(kind, []).append(name)

    # Operative documents
    n = f"{f.msa_date:%Y-%m-%d}_Master_Services_Agreement.pdf"
    _pdf(out / n, f"Master Services Agreement — {f.client} and {f.vendor}", [
        f"This Master Services Agreement is made on {fmt(f.msa_date)} between {f.client} (the Client) and {f.vendor} (the Vendor).",
        "PAYMENT TERMS", f"The Client shall pay each correctly rendered tax invoice within {f.payment_terms_0} days of receipt (net {f.payment_terms_0} days).",
        "LIMITATION AND INDEMNITY", f"The Vendor's aggregate liability under the indemnity in this Agreement is capped at ${f.indemnity_cap_0:,}.",
        "DISPUTES", "Any dispute arising under this Agreement shall be determined exclusively by the courts of Victoria. The parties do not agree to refer disputes to arbitration.",
        "EXECUTION", f"Signed for the Vendor by {f.vendor_signatory}, Director. Signed for the Client by {f.client_signatory}, Chief Operating Officer."])
    add("msa", n)
    n = f"{f.msa_date - timedelta(days=40):%Y-%m-%d}_MSA_Draft_v3_SUPERSEDED.pdf"
    _pdf(out / n, "DRAFT v3 — Master Services Agreement — NOT EXECUTED", [
        "DRAFT FOR NEGOTIATION ONLY. SUPERSEDED BY THE EXECUTED AGREEMENT.",
        "PAYMENT TERMS (DRAFT)", f"The Client shall pay each invoice within {f.payment_terms_0 + 30} days of receipt.",
        "LIMITATION AND INDEMNITY (DRAFT)", f"The Vendor's aggregate liability is capped at ${f.indemnity_cap_0 + 750_000:,}.",
        "STATUS", "This version was not executed."])
    add("draft", n)
    n = f"{f.a1_date:%Y-%m-%d}_Amendment_No_1_Payment_Terms.pdf"
    _pdf(out / n, "Amendment No. 1 to Master Services Agreement", [
        f"This Amendment No. 1 is made on {fmt(f.a1_date)}.", "VARIATION",
        f"The clause 'Payment Terms' is deleted and replaced such that tax invoices are payable within {f.payment_terms_1} days of receipt (net {f.payment_terms_1} days). All other terms are unchanged."])
    add("amend1", n)
    n = f"{f.a2_date:%Y-%m-%d}_Amendment_No_2_Indemnity_Cap.pdf"
    _pdf(out / n, "Amendment No. 2 to Master Services Agreement", [
        f"This Amendment No. 2 is made on {fmt(f.a2_date)}.", "VARIATION",
        f"The indemnity cap in the clause 'Limitation and Indemnity' is increased to ${f.indemnity_cap_2:,} with effect from the date of this Amendment. All other terms are unchanged."])
    add("amend2", n)
    n = f"{f.msa_date:%Y-%m-%d}_Statement_of_Work_No_1_Rates.pdf"
    _pdf(out / n, "Statement of Work No. 1 — Professional Services Rates",
         ["The following hourly rates (ex GST) apply to services performed under the MSA from the Commencement Date.",
          "Projects in scope: " + "; ".join(f"{k} ({v})" for k, v in PROJECTS.items()) + "."],
         [[["Role", "Hourly rate ($, ex GST)"]] + [[r, f"{v:,}"] for r, v in f.rates_0.items()]])
    add("sow", n)
    n = f"{f.sow_amend_date:%Y-%m-%d}_SOW_No_1_Rate_Variation.pdf"
    changed = [r for r in ROLES if f.rates_1[r] != f.rates_0[r]]
    _pdf(out / n, "Statement of Work No. 1 — Rate Variation",
         [f"With effect from {fmt(f.sow_amend_date)} the hourly rates for the following roles are varied as set out below. Rates for all other roles are unchanged. Invoices for services performed on or after that date must apply the varied rates."],
         [[["Role", "Previous rate ($/hr)", "Varied rate ($/hr)"]] + [[r, f"{f.rates_0[r]:,}", f"{f.rates_1[r]:,}"] for r in changed]])
    add("sow_var", n)
    if f.guarantee_exists:
        n = f"{f.msa_date + timedelta(days=3):%Y-%m-%d}_Deed_of_Guarantee_Halloway.pdf"
        _pdf(out / n, "Deed of Guarantee and Indemnity",
             [f"{f.guarantor} unconditionally guarantees to {f.vendor} the due and punctual payment of all amounts owed by {f.client} under the MSA."])
        add("guarantee", n)

    # Invoices, credit notes, quotes, remittances
    for inv in f.invoices:
        add("invoice", _invoice_pdf(out, f, inv))
    for cn in f.credit_notes:
        n = f"{cn['date']:%Y-%m-%d}_Credit_Note_{cn['no']}_against_{cn['against']}.pdf"
        _pdf(out / n, f"CREDIT NOTE {cn['no']}", [
            f"Date: {fmt(cn['date'])}", f"From: {f.vendor}", f"To: {f.client}",
            f"This credit note cancels tax invoice {cn['against']} in full. Reason: {cn['reason']}. A replacement invoice follows under separate cover.",
            f"Amount credited (inc GST): {money(cn['amount'])}"])
        add("credit_note", n)
    for q in f.quotes:
        n = f"{q['date']:%Y-%m-%d}_Pro_Forma_Quote_{q['no']}_{q['project']}.pdf"
        _pdf(out / n, f"PRO FORMA QUOTE {q['no']} — NOT A TAX INVOICE", [
            f"Date: {fmt(q['date'])}", f"Project: {q['project']}",
            "This document is an indicative quotation for budgeting purposes only. It is not a tax invoice and no amount is payable on it."],
            [[["Description", "Indicative amount ($)"], ["Proposed additional services", f"{q['amount']:,.2f}"],
              ["TOTAL (indicative)", f"{q['amount']:,.2f}"]]])
        add("quote", n)
    for r in f.remittances:
        n = f"{r['date']:%Y-%m-%d}_Remittance_Advice_{r['no']}.pdf"
        _pdf(out / n, f"Remittance Advice {r['no']} — {f.client}", [
            f"Date: {fmt(r['date'])}", f"To: {f.vendor}",
            "We advise payment by electronic funds transfer of the following tax invoices in full:",
            ", ".join(r["invoices"]) + ".", f"Total remitted: {money(r['amount'])}."])
        add("remittance", n)

    # Disputes, breach, demand (stale figure), file notes
    for inv in f.disputed:
        d = inv["date"] + timedelta(days=rng.randint(10, 30))
        n = f"{d:%Y-%m-%d}_Letter_Dispute_{inv['no']}.pdf"
        _pdf(out / n, f"Letter — {f.client} to {f.vendor} — Invoice {inv['no']}", [
            f"Date: {fmt(d)}", f"We dispute tax invoice {inv['no']} dated {fmt(inv['date'])} ({money(f.total(inv))} inc GST) on the basis that the hours recorded for {inv['lines'][0]['person']} exceed the approved allocation for {inv['project']}. Payment is withheld pending resolution. This letter is without prejudice to our rights under the Agreement."])
        add("dispute", n)
    n = f"{f.breach_date:%Y-%m-%d}_Notice_of_Breach.pdf"
    _pdf(out / n, f"Notice of Breach — {f.vendor} to {f.client}", [
        f"Date: {fmt(f.breach_date)}",
        f"You are in breach of the Agreement by reason of non-payment of tax invoices. You must remedy the breach within {f.cure_days} days of the date of this notice, failing which we reserve all rights."])
    add("breach", n)
    stale = f.outstanding_at(f.demand_as_at)
    n = f"{f.demand_as_at + timedelta(days=2):%Y-%m-%d}_Letter_of_Demand.pdf"
    _pdf(out / n, f"Letter of Demand — {f.vendor} to {f.client}", [
        f"Date: {fmt(f.demand_as_at + timedelta(days=2))}",
        f"As at {fmt(f.demand_as_at)} the total amount outstanding across unpaid tax invoices was {money(stale)}. Payment is demanded within 7 days. Interest accrues under the Agreement."])
    add("demand", n)

    # Monthly status reports per project (multi-page) and steering minutes
    for m in f.months:
        label = m.strftime("%B %Y")
        for code, title in PROJECTS.items():
            lines = [ln for i in f.invoices if i["project"] == code and i["period"] == label and i["status"] != "cancelled"
                     for ln in i["lines"]]
            by_role: dict[str, Decimal] = {}
            for ln in lines:
                by_role[ln["role"]] = by_role.get(ln["role"], Decimal(0)) + ln["hours"]
            # Indicative figures: deliberately off from the invoiced hours.
            grid = [["Role", "Indicative hours (timesheets)"]] + [
                [r, f"{(h * Decimal(rng.choice(['0.9', '0.95', '1.05', '1.1']))).quantize(Decimal('0.5')):.1f}"]
                for r, h in sorted(by_role.items())]
            n = f"{m:%Y-%m}_Status_Report_{code}.pdf"
            blocks = [f"Project: {code} — {title}", f"Reporting period: {label}", f"Project manager: {f.pm[code]}",
                      f"Prepared for: {f.client} steering committee"] + prose.report(label, report_chars)
            _pdf(out / n, f"Monthly Status Report — {code} — {label}", blocks, [grid],
                 ["Effort summary (indicative, pre-invoicing)"])
            add("status_report", n)
        if m.month in (1, 4, 7, 10):
            n = f"{m:%Y-%m-%d}_Steering_Committee_Minutes.txt"
            body = [f"STEERING COMMITTEE — {f.client} / {f.vendor}", f"Date: {fmt(m + timedelta(days=rng.randint(5, 20)))}",
                    f"Attendees: {f.client_signatory}, {f.vendor_signatory}, " + ", ".join(sorted(set(f.pm.values()))), ""]
            for code in PROJECTS:
                body += [f"{code}", prose.paragraph(label, 6), ""]
            body.append("ACTIONS: " + prose.paragraph(label, 3))
            (out / n).write_text("\n".join(body), encoding="utf-8")
            add("minutes", n)
    return docs


# --------------------------------------------------------------------------- questions

def _questions(f: Facts, docs: dict[str, list[str]]) -> list[dict]:
    live = f.live()
    inv_names = {i["no"]: n for i, n in zip(f.invoices, docs["invoice"], strict=True)}

    def names(invs):
        return [inv_names[i["no"]] for i in invs]

    def by(pred):
        return [i for i in live if pred(i)]

    fy25 = lambda i: fy(i["date"]) == 2025  # noqa: E731
    fy26 = lambda i: fy(i["date"]) == 2026  # noqa: E731
    sub = Facts.subtotal
    tot = Facts.total
    qs: list[dict] = []

    def q(qid, qtype, question, answer, sources, note="", unanswerable=False):
        qs.append({"id": qid, "question_type": qtype, "question": question, "answer": answer,
                   "sources": sorted(set(sources))[:400], "note": note, "unanswerable": unanswerable})

    # ---- aggregation
    alpha25 = by(lambda i: i["project"] == "PRJ-ALPHA" and fy25(i))
    q("agg_alpha_fy25_exgst", "aggregation",
      "What was the total amount invoiced (ex GST) for project PRJ-ALPHA in FY2025 (1 July 2024 to 30 June 2025), excluding invoices that were cancelled by credit note?",
      money(sum((sub(i) for i in alpha25), Decimal(0))), names(alpha25))
    bravo = by(lambda i: i["project"] == "PRJ-BRAVO")
    q("agg_bravo_count", "aggregation",
      "How many tax invoices were issued for project PRJ-BRAVO over the whole engagement, counting a replacement invoice but not the cancelled invoice it replaced?",
      str(len(bravo)), names(bravo))
    charlie26 = by(lambda i: i["project"] == "PRJ-CHARLIE" and fy26(i))
    q("agg_charlie_fy26_incgst", "aggregation",
      "What was the total invoiced (inc GST) for project PRJ-CHARLIE in FY2026?",
      money(sum((tot(i) for i in charlie26), Decimal(0))), names(charlie26))
    de_hours = sum((ln["hours"] for i in live for ln in i["lines"] if ln["role"] == "Data Engineer"), Decimal(0))
    q("agg_de_hours", "aggregation",
      "How many hours in total were invoiced for the Data Engineer role across all projects over the whole engagement (excluding cancelled invoices)?",
      f"{de_hours:,.1f} hours", names(live))
    gst25 = sum((Facts.gst(i) for i in by(lambda i: i["date"].year == 2025)), Decimal(0))
    q("agg_gst_cal2025", "aggregation",
      "How much GST was charged on tax invoices dated in calendar year 2025 (excluding cancelled invoices)?",
      money(gst25), names(by(lambda i: i["date"].year == 2025)))
    big = by(lambda i: tot(i) > Decimal(80000))
    q("agg_count_over_80k", "aggregation",
      "How many live tax invoices (not cancelled) have a total due including GST greater than $80,000?",
      str(len(big)), names(big))
    priya = [i for i in live if any(ln["person"] == "Priya Natarajan" for ln in i["lines"])]
    q("agg_priya_total", "aggregation",
      "What is the total amount (ex GST) invoiced for work by Priya Natarajan across the engagement, excluding cancelled invoices?",
      money(sum((ln["amount"] for i in priya for ln in i["lines"] if ln["person"] == "Priya Natarajan"), Decimal(0))), names(priya))
    totals = {p: sum((sub(i) for i in live if i["project"] == p), Decimal(0)) for p in PROJECTS}
    top = max(totals, key=totals.get)
    q("agg_top_project", "aggregation",
      "Which project has the highest total invoiced amount (ex GST) over the engagement, excluding cancelled invoices?",
      f"{top} — {PROJECTS[top]}", names(live), note=f"totals: {', '.join(f'{k} {money(v)}' for k, v in totals.items())}")
    months = {}
    for i in live:
        months[i["period"]] = months.get(i["period"], Decimal(0)) + tot(i)
    top_month = max(months, key=months.get)
    q("agg_top_month", "aggregation",
      "Which service period (month) had the highest total invoiced amount including GST, excluding cancelled invoices?",
      top_month, names(by(lambda i: i["period"] == top_month)), note=f"{money(months[top_month])}")
    people_c = {ln["person"] for i in live if i["project"] == "PRJ-CHARLIE" for ln in i["lines"]}
    q("agg_charlie_people", "aggregation",
      "How many distinct consultants billed time to project PRJ-CHARLIE over the engagement?",
      str(len(people_c)), names(by(lambda i: i["project"] == "PRJ-CHARLIE")))
    q("agg_credit_notes_total", "aggregation",
      "What is the total value (inc GST) of all credit notes issued by the vendor?",
      money(sum((c["amount"] for c in f.credit_notes), Decimal(0))), docs["credit_note"])
    q("agg_credit_notes_count", "aggregation", "How many credit notes did the vendor issue?",
      str(len(f.credit_notes)), docs["credit_note"])
    q("agg_remit_fy25", "aggregation",
      "What was the total amount remitted by the client under remittance advices dated in FY2025?",
      money(sum((r["amount"] for r in f.remittances if fy(r["date"]) == 2025), Decimal(0))),
      [n for r, n in zip(f.remittances, docs["remittance"], strict=True) if fy(r["date"]) == 2025])
    avg26 = (sum((tot(i) for i in by(fy26)), Decimal(0)) / len(by(fy26))).quantize(CENT, ROUND_HALF_UP)
    q("agg_avg_fy26", "aggregation",
      "What was the average total due (inc GST) per live tax invoice dated in FY2026, to the nearest cent?",
      money(avg26), names(by(fy26)))
    q("agg_disputed_total", "aggregation",
      "What is the total value (inc GST) of the tax invoices the client formally disputed by letter?",
      money(sum((tot(i) for i in f.disputed), Decimal(0))), docs["dispute"] + names(f.disputed))
    # ---- set logic
    q("set_outstanding_asat", "set_logic",
      f"What was the total amount outstanding (inc GST) on tax invoices as at {fmt(f.as_at)}, after all credit notes and all remittance advices dated on or before that day?",
      money(f.outstanding_at(f.as_at)), names(live) + docs["remittance"] + docs["credit_note"],
      note="the letter of demand states an earlier, stale figure")
    unpaid_fy25 = [i for i in by(fy25) if i["no"] not in f.paid]
    q("set_unpaid_fy25_count", "set_logic",
      "How many live tax invoices dated in FY2025 had not been paid under any remittance advice by the end of the engagement?",
      str(len(unpaid_fy25)), names(unpaid_fy25) + docs["remittance"])
    q("set_reissued_count", "set_logic",
      "How many tax invoices are replacements for an earlier invoice that was cancelled by credit note?",
      str(sum(1 for i in f.invoices if i["replaces"])), names([i for i in f.invoices if i["replaces"]]))
    # ---- join
    err = [i for i in f.invoices if i["rate_error"]]
    q("join_rate_error_count", "join",
      f"After the SOW rate variation took effect on {fmt(f.sow_amend_date)}, how many tax invoices still billed the Senior Consultant role at the superseded hourly rate of ${f.rates_0['Senior Consultant']}?",
      str(len(err)), names(err) + docs["sow_var"])
    q("join_rate_error_invoices", "join",
      "Which invoice numbers billed the Senior Consultant role at the superseded rate after the rate variation took effect?",
      ", ".join(i["no"] for i in err), names(err) + docs["sow_var"])
    under = sum(((Decimal(f.rates_1["Senior Consultant"]) - Decimal(f.rates_0["Senior Consultant"])) * ln["hours"]
                 for i in err for ln in i["lines"] if ln["role"] == "Senior Consultant"), Decimal(0))
    q("join_rate_error_shortfall", "join",
      "By how much (ex GST) were those rate-error invoices under-billed in total, compared with the varied Senior Consultant rate?",
      money(under), names(err) + docs["sow_var"])
    # ---- cross-doc
    q("cross_de_rate_now", "cross_doc", "What is the current hourly rate for a Data Engineer under Statement of Work No. 1?",
      f"${f.rates_1['Data Engineer']}", docs["sow"] + docs["sow_var"])
    q("cross_payment_terms_now", "cross_doc",
      "If the client receives a tax invoice from the vendor today, within how many days must it be paid?",
      f"{f.payment_terms_1} days", docs["msa"] + docs["amend1"])
    q("cross_indemnity_now", "cross_doc",
      "What is the indemnity cap currently in effect under the MSA, taking into account all amendments?",
      f"${f.indemnity_cap_2:,}", docs["msa"] + docs["amend2"])
    q("cross_indemnity_original", "cross_doc",
      "What was the indemnity cap under the Master Services Agreement as originally executed?",
      f"${f.indemnity_cap_0:,}", docs["msa"])
    # ---- value lookup
    sample = live[len(live) // 3]
    q("lookup_po", "value_lookup", f"What purchase order number is quoted on tax invoice {sample['no']}?",
      sample["po"], [inv_names[sample["no"]]])
    q("lookup_signatory", "value_lookup", "Who signed the Master Services Agreement for the Vendor?",
      f.vendor_signatory, docs["msa"])
    q("lookup_pm_bravo", "value_lookup", "Who is the project manager for PRJ-BRAVO according to the monthly status reports?",
      f.pm["PRJ-BRAVO"], [n for n in docs["status_report"] if "PRJ-BRAVO" in n][:3])
    sample2 = live[2 * len(live) // 3]
    q("lookup_inv_total", "value_lookup", f"What is the total due including GST on tax invoice {sample2['no']}?",
      money(tot(sample2)), [inv_names[sample2["no"]]])
    q("lookup_stale_demand", "value_lookup",
      "What outstanding amount does the vendor's letter of demand state?", money(f.outstanding_at(f.demand_as_at)), docs["demand"])
    # ---- timeline
    q("time_cure_deadline", "timeline",
      "By what date must the client remedy the breach identified in the vendor's notice of breach?", fmt(f.cure_deadline), docs["breach"])
    q("time_last_remittance", "timeline", "What is the date of the most recent remittance advice in the matter?",
      fmt(f.remittances[-1]["date"]), docs["remittance"][-1:])
    first_c = min(by(lambda i: i["project"] == "PRJ-CHARLIE"), key=lambda i: i["date"])
    q("time_first_charlie", "timeline", "What is the date of the earliest live tax invoice for project PRJ-CHARLIE?",
      fmt(first_c["date"]), [inv_names[first_c["no"]]])
    q("time_amend2", "timeline", "On what date was Amendment No. 2 to the MSA made?", fmt(f.a2_date), docs["amend2"])
    # ---- unanswerable / absent
    q("abs_guarantee", "unanswerable" if not f.guarantee_exists else "value_lookup",
      f"Is there a personal guarantee from {f.guarantor} anywhere in the matter file?",
      "Yes" if f.guarantee_exists else "No", docs.get("guarantee", docs["msa"]), unanswerable=not f.guarantee_exists)
    q("abs_arbitration", "existence", "Do the parties agree to arbitration of disputes under the MSA?", "No", docs["msa"])
    q("abs_delta", "unanswerable", "How many tax invoices were issued for project PRJ-DELTA?", "None — there is no project PRJ-DELTA in the matter",
      docs["sow"], unanswerable=True)
    q("abs_quote_invoice", "unanswerable",
      f"What tax invoice was issued for pro forma quote {f.quotes[0]['no']}?", "None — pro forma quotes are not tax invoices and no invoice references that quote",
      docs["quote"][:1], unanswerable=True)
    return qs


# --------------------------------------------------------------------------- entry

FIELD_CONTEXT = ("Benchmark question over one commercial matter's document corpus (hundreds of tax invoices, "
                 "credit notes, remittances, reports). The reference is an exact gold computed from the documents. "
                 "Money must match to the cent and counts exactly; a candidate that states the reference value and "
                 "adds correct supporting detail (document names, components of the sum) is equivalent. A different "
                 "number is not equivalent. For unanswerable items the candidate must decline or state the premise is "
                 "wrong; a confident value is wrong.")


def generate(out_dir: Path, seed: int, n_months: int, invoices_per_month: int, report_chars: int) -> dict:
    out = Path(out_dir)
    docs_dir = out / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    f = Facts(seed, n_months=n_months, invoices_per_month=invoices_per_month)
    if not any(docs_dir.glob("*Master_Services_Agreement.pdf")):
        docs = _write_corpus(docs_dir, f, report_chars)
        (out / "doc_index.json").write_text(json.dumps(docs, indent=1))
    else:
        docs = json.loads((out / "doc_index.json").read_text())
    qs = _questions(f, docs)
    (out / "qa.json").write_text(json.dumps([{"id": q["id"], "question": q["question"], "answer": q["answer"],
                                              "unanswerable": q["unanswerable"]} for q in qs], indent=1, ensure_ascii=False))
    (out / "answer_key.json").write_text(json.dumps(qs, indent=1, ensure_ascii=False))
    with open(out / "questions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ground_truth_question", "field_id"])
        for q in qs:
            w.writerow([q["question"], q["id"]])
    (out / "golden.json").write_text(json.dumps({"items": [
        {"id": q["id"], "question": q["question"], "golden": "" if q["unanswerable"] else q["answer"],
         "note": (q["answer"] if q["unanswerable"] else q["note"]), "field_context": FIELD_CONTEXT + f" Question type: {q['question_type']}."}
        for q in qs]}, indent=1, ensure_ascii=False))
    (out / "map.json").write_text(json.dumps({"items": [
        {"kind": "standalone", "field_id": q["id"], "question": q["question"], "needs_value": False} for q in qs]}, indent=1))
    n_files = sum(len(v) for v in docs.values())
    return {"documents": n_files, "invoices": len(f.invoices), "live_invoices": len(f.live()),
            "credit_notes": len(f.credit_notes), "remittances": len(f.remittances), "questions": len(qs),
            "by_type": {t: sum(1 for q in qs if q["question_type"] == t) for t in sorted({q["question_type"] for q in qs})}}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--months", type=int, default=36)
    ap.add_argument("--invoices-per-month", type=int, default=10)
    ap.add_argument("--report-chars", type=int, default=16_000, help="prose per monthly status report")
    args = ap.parse_args(argv)
    print(json.dumps(generate(args.out, args.seed, args.months, args.invoices_per_month, args.report_chars), indent=1))


if __name__ == "__main__":
    main()
