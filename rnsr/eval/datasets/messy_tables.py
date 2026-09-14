"""Synthetic ugly-table corpus for extraction scoring.

Generates labelled PDFs that exercise the failure modes real ledgers hit:
repeated headers across pages, TOTAL/subtotal/footnote rows, EU and US
number styles, parenthesised negatives. A rotated/low-contrast scan
variant is included as a JPEG-in-PDF page (no text layer).

``labels.json`` is the same format used for real labelled documents — drop
bank statements into a directory, add rows to labels.json, and
``rnsr eval-tables --dir`` scores both.
"""

from __future__ import annotations

import json
from pathlib import Path

LABELS_NAME = "labels.json"


def generate_messy_tables(out_dir: str | Path, *, seed: int = 7) -> Path:
    """Write PDFs + labels.json. Returns the labels path."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    labels: list[dict] = []

    # --- US invoice with TOTAL row ---
    us_pdf = out_dir / "us_invoice_totals.pdf"
    story = [
        Paragraph("Acme Supplies — Invoice 1042", styles["Title"]),
        Spacer(1, 12),
        Table(
            [["Item", "Amount ($)"],
             ["Widgets", "1,200.00"],
             ["Gadgets", "800.00"],
             ["Total", "2,000.00"]],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]),
        ),
    ]
    SimpleDocTemplate(str(us_pdf)).build(story)
    labels.append({
        "doc": us_pdf.name, "page": 1, "table_idx": 0,
        "expected": {
            "headers": ["Item", "Amount ($)"],
            "n_data_rows": 2,
            "n_total_rows": 1,
            "numeric_columns": ["amount"],
            "style": "us",
        },
        "must_pass": True,
    })

    # --- EU ledger with subtotal + parenthesised negative ---
    eu_pdf = out_dir / "eu_ledger_subtotals.pdf"
    story = [
        Paragraph("Nordic GmbH — Ledger extract", styles["Title"]),
        Spacer(1, 12),
        Table(
            [["Konto", "Betrag"],
             ["Miete", "1.234,50"],
             ["Strom", "800,00"],
             ["Subtotal", "2.034,50"],
             ["Korrektur", "(34,50)"],
             ["Total", "2.000,00"]],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]),
        ),
    ]
    SimpleDocTemplate(str(eu_pdf)).build(story)
    labels.append({
        "doc": eu_pdf.name, "page": 1, "table_idx": 0,
        "expected": {
            "headers": ["Konto", "Betrag"],
            "n_data_rows": 3,
            "n_total_rows": 2,
            "numeric_columns": ["betrag"],
            "style": "eu",
        },
        "must_pass": True,
    })

    # --- Multi-page with repeated header + footnote ---
    mp_pdf = out_dir / "multipage_repeated_header.pdf"
    rows_p1 = [["Line", "Qty", "Price"]] + [[f"Item {i}", str(i), f"{i * 10}"]
                                            for i in range(1, 16)]
    rows_p2 = [["Line", "Qty", "Price"]] + [[f"Item {i}", str(i), f"{i * 10}"]
                                            for i in range(16, 21)]
    rows_p2.append(["* see terms", "", ""])
    rows_p2.append(["Total", "210", "2100"])
    story = [
        Paragraph("Packing list (page 1)", styles["Heading2"]),
        Table(rows_p1, style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ])),
        PageBreak(),
        Paragraph("Packing list (page 2)", styles["Heading2"]),
        Table(rows_p2, style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ])),
    ]
    SimpleDocTemplate(str(mp_pdf), pagesize=LETTER).build(story)
    labels.append({
        "doc": mp_pdf.name, "page": 1, "table_idx": 0,
        "expected": {
            "headers": ["Line", "Qty", "Price"],
            "n_data_rows": 20,
            "n_total_rows": 1,
            "numeric_columns": ["qty", "price"],
            "style": "us",
        },
        "must_pass": True,
    })

    # --- Low-contrast / scan-like page (text still present; visual noise) ---
    scan_pdf = out_dir / "low_contrast_scan.pdf"
    story = [
        Paragraph("Scanned bank statement (faint)", styles["Title"]),
        Spacer(1, 12),
        Table(
            [["Date", "Description", "Amount"],
             ["2026-01-03", "Salary", "3,400.00"],
             ["2026-01-08", "Rent", "(1,800.00)"],
             ["Total", "", "1,600.00"]],
            style=TableStyle([
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.Color(0.55, 0.55, 0.55)),
                ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.85, 0.85, 0.85)),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.Color(0.7, 0.7, 0.7)),
            ]),
        ),
    ]
    SimpleDocTemplate(str(scan_pdf)).build(story)
    labels.append({
        "doc": scan_pdf.name, "page": 1, "table_idx": 0,
        "expected": {
            "headers": ["Date", "Description", "Amount"],
            "n_data_rows": 2,
            "n_total_rows": 1,
            "numeric_columns": ["amount"],
            "style": "us",
        },
        "must_pass": False,  # visual noise; pass is bonus
    })

    labels_path = out_dir / LABELS_NAME
    labels_path.write_text(json.dumps({
        "seed": seed,
        "tables": labels,
    }, indent=2))
    return labels_path


def load_labelled_tables(directory: str | Path) -> dict:
    """Load labels.json from a directory of documents + labels."""
    directory = Path(directory)
    path = directory / LABELS_NAME
    if not path.exists():
        raise FileNotFoundError(f"no {LABELS_NAME} in {directory}")
    data = json.loads(path.read_text())
    data["_dir"] = str(directory)
    return data
