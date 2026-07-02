"""Custom numeric-needle generator (§8): synthesize small filings as real
PDFs with planted figures, so the *entire* pipeline (Docling parse → typed
tables → SQL) is exercised and every gold answer is known exactly.

Each document gets several tables of random figures plus prose that never
mentions the needle values — the needle is only reachable through the
table (or exhaustive search), which is precisely the class vector search
misses.
"""

from __future__ import annotations

import random
from pathlib import Path

from rnsr.eval.datasets.base import EvalItem

_SEGMENTS = ["Widgets", "Gadgets", "Fasteners", "Adhesives", "Optics", "Sensors"]
_METRICS = [("Revenue ($M)", 100, 9_999), ("Operating Income ($M)", 10, 999),
            ("Headcount", 50, 20_000)]
_YEARS = [2021, 2022, 2023]

_PROSE = (
    "The company delivered another year of disciplined execution across its "
    "portfolio. Management remains focused on operational excellence, margin "
    "discipline, and long-term shareholder value. Segment performance varied "
    "by end market, with detailed figures presented in the tables herein. "
)


def generate_needle_set(
    out_dir: str | Path,
    *,
    n_docs: int = 3,
    tables_per_doc: int = 6,
    questions_per_doc: int = 3,
    prose_blocks: int = 25,   # bulk per table section; the gate regime is LARGE documents
    seed: int = 11,
) -> list[EvalItem]:
    """Write PDFs to out_dir and return needle questions with exact golds."""
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

    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    items: list[EvalItem] = []

    for d in range(n_docs):
        company = f"Company{chr(65 + d)}"
        pdf_path = out_dir / f"{company.lower()}_filing.pdf"
        story = [Paragraph(f"{company} Corp — Annual Report", styles["Title"])]
        needles: list[tuple[str, str, int, int, str]] = []  # metric, seg, year, value, caption

        # unique (metric, year) per table so every question has exactly one answer
        combos = [(m, lo, hi, y) for (m, lo, hi) in _METRICS for y in _YEARS]
        rng.shuffle(combos)
        if tables_per_doc > len(combos):
            raise ValueError(f"tables_per_doc must be <= {len(combos)}")

        for t in range(tables_per_doc):
            metric, lo, hi, year = combos[t]
            caption = f"{metric} by segment, fiscal {year}"
            segments = rng.sample(_SEGMENTS, 4)
            values = [rng.randint(lo, hi) for _ in segments]
            grid = [["Segment", metric]]
            grid += [[s, f"{v:,}"] for s, v in zip(segments, values, strict=True)]
            grid.append(["Total", f"{sum(values):,}"])

            story += [
                Spacer(1, 10),
                Paragraph(caption, styles["Heading2"]),
                *[Paragraph(_PROSE, styles["BodyText"]) for _ in range(prose_blocks)],
                Table(grid, style=TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.5, "black"),
                    ("BACKGROUND", (0, 0), (-1, 0), "#dddddd"),
                ])),
            ]
            for s, v in zip(segments, values, strict=True):
                needles.append((metric, s, year, v, caption))
            if t % 2 == 1:
                story.append(PageBreak())

        SimpleDocTemplate(str(pdf_path), pagesize=LETTER).build(story)

        for q in range(questions_per_doc):
            metric, seg, year, value, caption = rng.choice(needles)
            items.append(EvalItem(
                qid=f"needle-{company.lower()}-{q}",
                question=(f"According to {company} Corp's filing, what was the "
                          f"{metric} for the {seg} segment in fiscal {year}?"),
                gold=str(value),
                task_class="numeric",
                sources=[pdf_path],
                meta={"caption": caption},
            ))
    return items
