"""Built-in parsers for text-like formats: .txt, .md, .eml.

Stdlib-only and deterministic — these formats need no parsing library.
Markdown keeps its heading structure (structure-aware chunking) and its
GFM tables (RawTable -> typed SQL path); email keeps headers, the best
body part (plain preferred, HTML stripped), and attachment names so
absence questions can see what was attached but not ingested. Everything
lands on page 1, like the office formats.
"""

from __future__ import annotations

import email
import email.policy
import re
from html.parser import HTMLParser
from pathlib import Path

from rnsr.ingest.model import Element, ParsedDocument, RawTable
from rnsr.ingest.parse import _sha256, make_doc_id, render_table_text

TEXT_PARSER_NAME = "text"
MARKDOWN_PARSER_NAME = "markdown"
EML_PARSER_NAME = "eml"


def _document(path: Path, doc_id: str | None, parser: str) -> ParsedDocument:
    return ParsedDocument(
        doc_id=doc_id or make_doc_id(path),
        source_path=str(path),
        sha256=_sha256(path),
        n_pages=1,
        parser=parser,
    )


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_text(path: str | Path, doc_id: str | None = None) -> ParsedDocument:
    """Plain text: one element per blank-line-separated paragraph."""
    path = Path(path)
    parsed = _document(path, doc_id, TEXT_PARSER_NAME)
    for para in re.split(r"\n\s*\n", _read(path)):
        if para.strip():
            parsed.elements.append(Element("text", para.strip(), 1))
    return parsed


# --- markdown ---------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _split_md_row(line: str) -> list[str]:
    row = line.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [c.strip() for c in row.split("|")]


def parse_markdown(path: str | Path, doc_id: str | None = None) -> ParsedDocument:
    """Markdown: ATX headings, GFM tables, fenced code, paragraphs.

    Line-based and intentionally minimal (no escaped-pipe or inline-HTML
    handling): headings drive chunking, tables reach the SQL path, and
    everything else is retained verbatim as paragraph text.
    """
    path = Path(path)
    parsed = _document(path, doc_id, MARKDOWN_PARSER_NAME)
    lines = _read(path).split("\n")
    para: list[str] = []

    def flush() -> None:
        if para:
            text = "\n".join(para).strip()
            if text:
                parsed.elements.append(Element("text", text, 1))
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith(("```", "~~~")):
            flush()
            fence = stripped[:3]
            block = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(fence):
                block.append(lines[i])
                i += 1
            if i < len(lines):
                block.append(lines[i])
                i += 1
            parsed.elements.append(Element("text", "\n".join(block), 1))
            continue

        m = _HEADING_RE.match(stripped)
        if m:
            flush()
            parsed.elements.append(
                Element("heading", m.group(2), 1, heading_level=len(m.group(1))))
            i += 1
            continue

        if (stripped.startswith("|") and i + 1 < len(lines)
                and _TABLE_SEP_RE.match(lines[i + 1])):
            flush()
            header = _split_md_row(stripped)
            i += 2
            rows: list[list[str | None]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(list(_split_md_row(lines[i])))
                i += 1
            parsed.elements.append(
                Element("table", render_table_text(header, rows), 1))
            if rows:
                parsed.tables.append(RawTable(
                    page=1, header=header, rows=rows,
                    extractor=MARKDOWN_PARSER_NAME))
            continue

        if not stripped:
            flush()
        else:
            para.append(line)
        i += 1
    flush()
    return parsed


# --- email ------------------------------------------------------------------


class _HTMLText(HTMLParser):
    """Collect visible text from an HTML body; block tags become newlines."""

    _SKIP = frozenset({"script", "style", "head"})
    _BLOCK = frozenset({"p", "div", "br", "li", "tr", "table", "h1", "h2",
                        "h3", "h4", "h5", "h6", "blockquote"})

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        collapsed = re.sub(r"[ \t]+", " ", "".join(self.parts))
        return re.sub(r"\n\s*\n+", "\n\n", collapsed).strip()


def _strip_html(html: str) -> str:
    p = _HTMLText()
    p.feed(html)
    return p.text()


def parse_eml(path: str | Path, doc_id: str | None = None) -> ParsedDocument:
    """RFC-822 email: subject as heading, headers, best body, attachment names.

    Attachments are named but not ingested — visible, never silent."""
    path = Path(path)
    parsed = _document(path, doc_id, EML_PARSER_NAME)
    msg = email.message_from_bytes(path.read_bytes(), policy=email.policy.default)

    subject = str(msg.get("Subject", "")).strip() or "(no subject)"
    parsed.elements.append(Element("heading", subject, 1, heading_level=1))
    for h in ("From", "To", "Cc", "Date"):
        if msg.get(h):
            parsed.elements.append(Element("text", f"{h}: {msg[h]}", 1))

    body = msg.get_body(preferencelist=("plain", "html"))
    if body is not None:
        try:
            content = body.get_content()
        except Exception:
            content = ""
        if body.get_content_subtype() == "html":
            content = _strip_html(content)
        for para in re.split(r"\n\s*\n", content):
            if para.strip():
                parsed.elements.append(Element("text", para.strip(), 1))

    for att in msg.iter_attachments():
        name = att.get_filename() or f"unnamed ({att.get_content_type()})"
        parsed.elements.append(Element("text", f"[attachment: {name}]", 1))
    return parsed
