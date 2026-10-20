# Okafor & Nguyen (family law, Victoria) - synthetic benchmark matter

All people, firms, addresses, numbers and events are fictional. The matter is a combined parenting and property dispute in the Federal Circuit and Family Court of Australia (Division 2), file MLC1184/2026. The mother, Ada Okafor (a nurse in Bendigo), wants to relocate her children Zara (b. 14 Feb 2016) and Lachlan (b. 2 Sep 2018) to Brisbane; the father, Trung Nguyen (a builder, sole director of Nguyen Constructions Pty Ltd and appointor of the Nguyen Family Trust), opposes. The pool also includes the Banksia Court home, super splitting, a small business and a trust. Interim orders were made on 26 Aug 2026 and consent orders (drafts v1 and v2) are being negotiated.

Cast: Ada Okafor (client, applicant); Trung Nguyen (respondent); Fiona Brennan-Hale (principal) and Sam Odhiambo (paralegal), Harcourt Dunne Family Lawyers; Callum Reyes (counsel); Declan Pryce, Castellan & Pryce Lawyers (other side); Marguerite Sallis (Independent Children's Lawyer); Dr Helena Voss (family report); Judge Ruth Ashdown.

Key dates: married 9 Mar 2013; separated 18 Jun 2025; s60I certificates 20 Nov 2025 (non-attendance, FDR-2025-0471) and 17 Feb 2026 (attended, FDR-2026-0093); application filed 9 Apr 2026; family report 12 Aug 2026; interim hearing 26 Aug 2026; draft v1 14 Sep 2026; draft v2 30 Sep 2026; final hearing listed 15 Mar 2027.

Deliberately superseded facts: Ada's address (Hargreaves St to 8 Larkspur Lane), work days, Trung's hours and income, the Coral Coast start date, and draft v1 versus v2 (relocation date, flights, sale split, super split).

Files: 37 in documents/ - 31 text documents (13 .docx, 13 .pdf, 5 .txt, including the payslip as a text PDF and two distractors: a utilities bill and a newsletter) and 6 images (4 .png, 2 .jpg: bank statement, handwritten note, school report, text-message screenshot, licence, signed receipt).

Load: upload every file in documents/ to a Matter AI matter and wait for processing (images need OCR/vision).

Benchmark: qa.json holds 85 questions (id, question, answer, optional unanswerable: true); answer_key.json gives the question type and source files per id, in the same order. Load qa.json in the benchmark page, run each question against the matter, and grade against the answer (unanswerable ones should be declined, not guessed).

Rebuild: from the repo root run build_docs.py, build_images.py, build_qa.py then verify.py in build/ with `uv run --with python-docx --with reportlab --with pillow --with pypdf python <script>`.

## The mixed-source set (`qa_mixed.json`, 2026-10-12)

116 questions whose answers live, by construction, in one of: the matter documents (20, sampled from `qa.json`), the
knowledge graph (the 21 of `qa_graph.json`), the firm's practice-management database (30 keyed lookups and
aggregations plus 8 conflicts where the database is authoritative over a document), the firm's policy corpus (15),
a court-list / company-register API (10); 6 need two sources (`multi`) and 6 are unanswerable (`none`).
`answer_key_mixed.json` gives `question_type`, `oracle_source` (`matter_docs | graph | practice_db | policy_docs |
court_api | none | multi`), `sources` and `conflict` per id, in the same order. Ids are prefixed by slice
(`doc_`, `graph_`, `db_`, `agg_`, `conf_`, `pol_`, `api_`, `multi_`, `none_`).

The data behind the non-document slices lives in `build/`: `practice_world.py` (the database world; writes
`build/practice/00_schema.sql` and `10_seed.sql`), `policy_content.py` + `build_policy_docs.py` (ten policy documents
into `policies/` and `build/practice/20_policy_chunks.sql`), `seed_policy_chunks.py` (bge-small embeddings into
`build/practice/30_policy_embeddings.sql`), `court_api_data.json` + `court_api.py` (the FastAPI mock), and
`verify_mixed.py`, which fails when any database, policy or API answer token appears in a document's text, in the
image text (`export_image_text.py` -> `build/image_text.json`) or in the graph's facts (bolt 7689), or when a `none`
answer appears anywhere; `--plant <id>` proves the check catches a leaked answer.

Rebuild, from the repo root: `uv run python build/practice_world.py`; `uv run --with python-docx --with reportlab
python build/build_policy_docs.py`; `uv run python build/seed_policy_chunks.py`; `uv run python
build/export_image_text.py`; `uv run python build/build_qa_mixed.py`; `uv run --with python-docx --with pypdf python
build/verify_mixed.py` (the graph check reads `CTXR_CONN_NEO4J_COMMUNITY_PASSWORD` from `.env`). Serve: `make
practice-db-up` (Docker, :5433) and `make court-api` (:8120); register and run: `make bench-mixed` (see the repo README).
