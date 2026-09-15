"""Synthetic general-office corpus: mixed formats, known failure exposures.

A workplace dump a knowledge worker would actually have — email with an
attached invoice, a Word memo, a multi-sheet Excel workbook whose sheets
share a header, a scanned receipt (no text layer), and a policy that was
superseded. Every gold is exact by construction; each item records the
ingest/retrieval exposure it depends on so the autopsy classifier can be
checked against a known cause.
"""

from __future__ import annotations

import random
import zipfile
from email.message import EmailMessage
from email.utils import formatdate
from pathlib import Path

from rnsr.eval.datasets.base import EvalItem

# Minimal SpreadsheetML / WordprocessingML — generation must not depend on
# anydoc/openpyxl/python-docx so the benchmark can be built in CI.


def _xlsx_sheet_xml(name: str, rows: list[list[object]]) -> str:
    cells = []
    for r_i, row in enumerate(rows, start=1):
        parts = []
        for c_i, value in enumerate(row):
            col = chr(ord("A") + c_i)
            ref = f"{col}{r_i}"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                parts.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                escaped = (str(value).replace("&", "&amp;")
                           .replace("<", "&lt;").replace(">", "&gt;"))
                parts.append(f'<c r="{ref}" t="inlineStr"><is><t>{escaped}</t></is></c>')
        cells.append(f'<row r="{r_i}">{"".join(parts)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(cells)}</sheetData></worksheet>'
    )


def write_xlsx(path: Path, sheets: dict[str, list[list[object]]]) -> None:
    """Write a multi-sheet .xlsx (shared-string-free, inline strings)."""
    sheet_names = list(sheets)
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets>"
        + "".join(
            f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>'
            for i, name in enumerate(sheet_names, start=1)
        )
        + "</sheets></workbook>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(
            f'<Relationship Id="rId{i}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{i}.xml"/>'
            for i in range(1, len(sheet_names) + 1)
        )
        + "</Relationships>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        + "".join(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for i in range(1, len(sheet_names) + 1)
        )
        + "</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", rels)
        for i, name in enumerate(sheet_names, start=1):
            zf.writestr(f"xl/worksheets/sheet{i}.xml",
                        _xlsx_sheet_xml(name, sheets[name]))


def write_docx(path: Path, title: str, paragraphs: list[str]) -> None:
    """Write a single-section .docx with a heading and body paragraphs."""
    def p_xml(text: str, style: str = "Normal") -> str:
        escaped = (text.replace("&", "&amp;").replace("<", "&lt;")
                   .replace(">", "&gt;"))
        return (f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>'
                f'<w:r><w:t xml:space="preserve">{escaped}</w:t></w:r></w:p>')

    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        + p_xml(title, "Heading1")
        + "".join(p_xml(p) for p in paragraphs)
        + "</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/></Relationships>'
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document)


def write_scanned_pdf(path: Path, caption: str) -> None:
    """Near-blank PDF: extractable text is well under the scanned-page threshold.

    Ingest flags pages with < 50 extracted characters as scanned. The gold
    lives in a sidecar ``.gold.txt`` so a transcriber (or a human) can
    recover it; the PDF itself has no usable text layer.
    """
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen.canvas import Canvas

    c = Canvas(str(path), pagesize=LETTER)
    c.setFillColorRGB(0.96, 0.96, 0.94)
    c.rect(72, 600, 400, 120, fill=1, stroke=0)
    c.save()
    _ = caption


def write_text_pdf(path: Path, title: str, blocks: list[str]) -> None:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 10)]
    for b in blocks:
        story.append(Paragraph(b, styles["BodyText"]))
        story.append(Spacer(1, 6))
    SimpleDocTemplate(str(path), pagesize=LETTER).build(story)


class OfficeFacts:
    """Deterministic workplace facts, derived from the seed."""

    def __init__(self, seed: int):
        rng = random.Random(seed)
        self.company = rng.choice(["Northwind Trading", "Contoso Logistics"])
        self.author = rng.choice(["Priya Shah", "Marcus Chen"])
        self.q3_revenue = rng.choice([1_240_000, 1_375_000, 980_000])
        self.quarters = {
            "Q1": {"Revenue": rng.randint(80, 140) * 1000,
                   "Cost": rng.randint(40, 90) * 1000},
            "Q2": {"Revenue": rng.randint(80, 140) * 1000,
                   "Cost": rng.randint(40, 90) * 1000},
            "Q3": {"Revenue": self.q3_revenue,
                   "Cost": rng.randint(40, 90) * 1000},
            "Q4": {"Revenue": rng.randint(80, 140) * 1000,
                   "Cost": rng.randint(40, 90) * 1000},
        }
        for q in self.quarters.values():
            q["Profit"] = q["Revenue"] - q["Cost"]
        self.widget_price = rng.choice([47, 53, 61])
        self.limit_v1 = rng.choice([500, 400])
        self.limit_v2 = self.limit_v1 + rng.choice([250, 350])
        self.receipt_total = rng.choice([312, 287, 445])
        self.invoice_no = f"INV-{rng.randint(9000, 9999)}"


def generate_office(out_dir: str | Path, *, seed: int = 7) -> list[EvalItem]:
    """Write the office dump and return questions with exact golds + exposures."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = OfficeFacts(seed)
    marker = out / ".office_gen.json"
    if not marker.exists():
        _write_docs(out, f)
        marker.write_text(f'{{"seed": {seed}}}\n')
    sources = sorted(
        p for p in out.iterdir()
        if p.is_file() and p.suffix.lower() in {
            ".pdf", ".docx", ".xlsx", ".eml", ".txt", ".md",
        }
    )
    return _questions(f, sources, seed)


def _write_docs(out: Path, f: OfficeFacts) -> None:
    write_docx(out / "memo_q3_review.docx",
               f"{f.company} — Q3 operating review",
               [f"Prepared by {f.author}.",
                f"Q3 revenue closed at ${f.q3_revenue:,}.",
                "Headcount was unchanged. No cryptocurrency positions are held."])

    sheets = {
        q: [["Revenue", "Cost", "Profit"],
            [vals["Revenue"], vals["Cost"], vals["Profit"]]]
        for q, vals in f.quarters.items()
    }
    write_xlsx(out / "budget.xlsx", sheets)

    invoice_path = out / "_invoice_widget.pdf"
    write_text_pdf(invoice_path,
                   f"Tax Invoice {f.invoice_no} — {f.company}",
                   [f"Widget unit price: ${f.widget_price}.",
                    "Quantity: 100. Payment due net 14."])
    invoice_bytes = invoice_path.read_bytes()
    invoice_path.unlink()  # lives only as an .eml attachment

    msg = EmailMessage()
    msg["From"] = f"{f.author.replace(' ', '.').lower()}@example.test"
    msg["To"] = "ceo@example.test"
    msg["Subject"] = f"{f.company} Q3 invoice"
    msg["Date"] = formatdate()
    msg.set_content(
        "Please see the attached invoice for the widget unit price. "
        "Do not use the superseded draft policy for expense limits."
    )
    msg.add_attachment(invoice_bytes, maintype="application", subtype="pdf",
                       filename="invoice_widget.pdf")
    (out / "cfo_q3_invoice.eml").write_bytes(msg.as_bytes())

    write_text_pdf(out / "policy_v1_superseded.pdf",
                   f"{f.company} — Expense policy (DRAFT, SUPERSEDED)",
                   ["DRAFT FOR REVIEW ONLY — SUPERSEDED BY THE CURRENT POLICY.",
                    f"The single-transaction expense limit is ${f.limit_v1}.",
                    "This version was not approved."])
    write_text_pdf(out / "policy_v2_current.pdf",
                   f"{f.company} — Expense policy (current)",
                   ["This policy supersedes all earlier drafts.",
                    f"The single-transaction expense limit is ${f.limit_v2}.",
                    "Approved and in force."])

    write_scanned_pdf(out / "receipt_scan.pdf",
                      f"Receipt total ${f.receipt_total}")
    # The gold for the scan lives next to the image so a transcriber (or a
    # human) can recover it; the PDF itself has no text layer.
    (out / "receipt_scan.gold.txt").write_text(
        f"Receipt total ${f.receipt_total}\n")

    (out / "notes.txt").write_text(
        f"{f.company} weekly notes.\n"
        "Parking permits renewed. Catering for the offsite is booked.\n"
        "No change to the expense policy this week.\n"
    )
    (out / "agenda.md").write_text(
        f"# {f.company} staff meeting\n\n"
        "- Review Q3 close\n"
        "- Confirm current expense policy is v2\n"
    )


def _questions(f: OfficeFacts, sources: list[Path], seed: int) -> list[EvalItem]:
    q2 = f.quarters["Q2"]
    total_rev = sum(q["Revenue"] for q in f.quarters.values())
    specs = [
        ("lookup",
         f"What was Q3 revenue as stated in the {f.company} operating review memo?",
         f"${f.q3_revenue:,}",
         {"gold_doc": "memo_q3_review", "exposure": "docx",
          "gold_text": f"${f.q3_revenue:,}"}),
        ("sheet-specific",
         "What was Q2 Profit in the budget workbook?",
         f"{q2['Profit']:,}",
         {"gold_doc": "budget", "exposure": "sheet_identity",
          "sheet_name": "Q2", "gold_text": str(q2["Profit"])}),
        ("aggregation",
         "What is the sum of Revenue across all four quarters in the budget workbook?",
         f"{total_rev:,}",
         {"gold_doc": "budget", "exposure": "sheet_identity",
          "gold_text": str(total_rev)}),
        ("cross-doc",
         "What is the widget unit price on the invoice attached to the CFO email?",
         f"${f.widget_price}",
         {"gold_doc": "cfo_q3_invoice", "exposure": "attachment",
          "parent_doc": "cfo_q3_invoice",
          "child_doc": "invoice_widget",
          "gold_text": f"${f.widget_price}"}),
        ("supersession",
         "What is the current single-transaction expense limit, taking the "
         "superseding policy into account?",
         f"${f.limit_v2}",
         {"gold_doc": "policy_v2_current", "exposure": "supersession",
          "gold_text": f"${f.limit_v2}"}),
        ("absent",
         "Does the operating review memo say the company holds cryptocurrency positions?",
         "No",
         {"gold_doc": "memo_q3_review", "exposure": None, "gold_text": "No"}),
        ("lookup",
         "What was the receipt total on the scanned receipt?",
         f"${f.receipt_total}",
         {"gold_doc": "receipt_scan", "exposure": "scanned_page",
          "gold_page": 1, "gold_text": f"${f.receipt_total}"}),
    ]
    return [
        EvalItem(
            qid=f"office-{seed}-{i:02d}",
            question=q, gold=gold, task_class=cls,
            sources=list(sources),
            expect="absent" if cls == "absent" else "value",
            meta={"n_docs": len(sources), **meta},
        )
        for i, (cls, q, gold, meta) in enumerate(specs)
    ]
