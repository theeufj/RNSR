"""Shared fixtures: isolated configuration and synthetic PDFs."""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_offline_configuration(request, monkeypatch):
    """Offline tests must not consume developer credentials or local settings."""
    if request.node.get_closest_marker("live"):
        return
    from rnsr import config

    for name in tuple(os.environ):
        if name.startswith("RNSR_") or name in {
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "LLM_PROVIDER",
        }:
            monkeypatch.delenv(name)
    repository_env = Path(__file__).resolve().parents[1] / ".env"
    real_load_dotenv = config.load_dotenv

    def load_test_dotenv(dotenv_path=None, *args, **kwargs):
        candidate = Path(dotenv_path or ".env")
        if candidate == Path(".env") or candidate.resolve() == repository_env:
            return False
        return real_load_dotenv(dotenv_path, *args, **kwargs)

    monkeypatch.setattr(config, "load_dotenv", load_test_dotenv)


@pytest.fixture(scope="session")
def fixture_pdf(tmp_path_factory):
    """A small financial-report-like PDF: headings, prose, and one table
    whose line items sum to the stated total (checksum-friendly)."""
    reportlab = pytest.importorskip("reportlab")  # noqa: F841
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    path = tmp_path_factory.mktemp("pdfs") / "acme_2023_report.pdf"
    styles = getSampleStyleSheet()
    story = [
        Paragraph("ACME Corporation Annual Report 2023", styles["Title"]),
        Spacer(1, 12),
        Paragraph("Item 7. Management Discussion", styles["Heading1"]),
        Paragraph(
            "Net revenue for fiscal 2023 was $3,234 million, driven by strong "
            "Widgets performance. Gadgets revenue was $2,000 million.",
            styles["BodyText"],
        ),
        Spacer(1, 12),
        Paragraph("Revenue by Segment", styles["Heading2"]),
        Table(
            [
                ["Segment", "Revenue ($M)", "Margin %"],
                ["Widgets", "$1,234", "45%"],
                ["Gadgets", "$2,000", "30%"],
                ["Total", "$3,234", "75%"],
            ],
            style=TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, "black"),
                ("BACKGROUND", (0, 0), (-1, 0), "#dddddd"),
            ]),
        ),
        Spacer(1, 12),
        Paragraph(
            "The company expects continued growth in fiscal 2024.", styles["BodyText"]
        ),
    ]
    SimpleDocTemplate(str(path), pagesize=LETTER).build(story)
    return path


@pytest.fixture(scope="session")
def scanned_pdf(fixture_pdf, tmp_path_factory):
    """An image-only PDF (the fixture rasterized) — no text layer at all."""
    pytest.importorskip("pypdfium2")
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(fixture_pdf))
    try:
        images = [page.render(scale=2.0).to_pil() for page in pdf]
    finally:
        pdf.close()
    out = tmp_path_factory.mktemp("pdfs") / "acme_scanned.pdf"
    images[0].save(out, format="PDF", save_all=True, append_images=images[1:])
    return out
