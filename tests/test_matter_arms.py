"""Baseline arms: DCI sandbox namespace, prompt mode, evidence status."""
from __future__ import annotations

import os
import sqlite3

import pytest

from rnsr.env.sandbox_child import _dci_namespace
from rnsr.eval.matter_arms import flat_context, load_documents
from rnsr.harness.evidence import from_records
from rnsr.harness.prompts.base import render_system


@pytest.fixture
def ns(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    import tempfile

    tempfile.tempdir = None  # re-read TMPDIR
    try:
        yield _dci_namespace({"2026-01-01 Letter.docx": "Dear Sir\nThe rent is $500\nRegards",
                              "notes": "rent review\nRent is now $550"})
    finally:
        tempfile.tempdir = None


def test_dci_files_and_helpers(ns):
    assert ns["ls"]() == ["2026-01-01_Letter.docx.txt", "notes.txt"]
    assert os.path.isdir(ns["CORPUS_DIR"])
    hits = ns["grep"](r"rent is").splitlines()
    assert hits == ["2026-01-01_Letter.docx.txt:2: The rent is $500", "notes.txt:2: Rent is now $550"]
    assert ns["grep"]("rent", path="notes.txt", max_hits=1).splitlines()[-1].startswith("[grep] stopped")
    assert ns["grep"]("nothing-here") == "[grep] no matches"
    assert ns["read"]("notes.txt", 2) == "2: Rent is now $550"
    assert ns["grep"]("500", context=1) == ("2026-01-01_Letter.docx.txt:1: Dear Sir\n"
                                            "2026-01-01_Letter.docx.txt:2: The rent is $500\n"
                                            "2026-01-01_Letter.docx.txt:3: Regards\n--")


def test_dci_prompt_mode():
    text = render_system("dci")
    assert "CORPUS_DIR" in text and "grep(" in text and "no index" in text
    with pytest.raises(ValueError):
        render_system("shell")


def test_dci_review_not_applicable():
    ev = from_records([{"kind": "start", "mode": "dci"},
                       {"kind": "final", "value": "x", "verification": None}], status="final")
    assert ev.claim_review_status == "not_applicable"
    assert ev.claim_review_required is False


def test_load_documents_one_file_per_document(tmp_path):
    db = tmp_path / "corpus.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE documents(doc_id TEXT, source_path TEXT);
        CREATE TABLE doc_text(doc_id TEXT, page INTEGER, text TEXT);
        INSERT INTO documents VALUES ('b','x/B Letter.pdf'),('a','x/A.docx');
        INSERT INTO doc_text VALUES ('a',1,'one'),('a',2,'two'),('b',1,'bee');
    """)
    conn.commit()
    conn.close()
    docs = load_documents(db)
    assert list(docs) == ["A.txt", "B Letter.txt"]
    assert docs["A.txt"] == "one\n\ntwo"
    assert flat_context(docs).startswith("===== A.txt =====\none")
