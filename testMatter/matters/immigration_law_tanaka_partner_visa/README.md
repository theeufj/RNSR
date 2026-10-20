# Tanaka Partner visa matter (fictional immigration law test data)

## Matter
Australian immigration matter (Migration Act 1958 (Cth), Department of Home Affairs, review by the Administrative Review Tribunal). Haruto Tanaka, a Japanese chef holding a Temporary Graduate (485) visa, lodged an onshore Partner visa (820/801) application with his Australian citizen partner Eleanor Marsh. A section 56 request, a processing delay and bridging visas A and B followed, the application was refused on a relationship-evidence ground, and an ART review (hearing listed, then re-listed) is pending. All people, IDs, addresses and phone numbers are invented.

## Cast and dates
- Cast: Haruto Tanaka (applicant), Eleanor Marsh (sponsor), Priya Raman (agent, MARN 9910042), Daniel Okafor and Mei-Ling Chen (Form 888 friends), Marcus Lindqvist (delegate), Hugh Sandberg (ART Member), Thomas Hargreaves JP.
- Key dates: lodged 14 October 2024 (App ID EGO0482915); s56 request 12 December 2024, response 8 January 2025; refusal 12 September 2025 (letter dated 11 September, corrected 22 September); ART application 6 October 2025 (case 2025/07731, deadline 10 October 2025); hearing 19 March 2026 (originally 12 March 2026).
- Deliberate superseded facts: lodgement date, decision date, passport number (TR9004127 correct vs TR9004172), review deadline, address (Banksia St to Wattle Grove Rd), hearing and filing dates, rent, move-in date.

## Files and use
- Files: 43 in documents/ (19 PDF, 11 DOCX, 5 TXT, 8 images: 7 PNG, 1 JPG). Images are generated scans/photos (passport, grant notice, lease, messages, bank statement, 2 captioned photos, Japanese police certificate). Distractors: a Penrith electricity bill and a community newsletter.
- Load: create a Matter AI matter and upload every file in documents/ (all types, including images).
- Benchmark: qa.json holds 90 questions (keys id, question, answer, optional unanswerable: true). Ask each question against the matter on the benchmark page and compare to the gold answer; for unanswerable items the correct behaviour is to say the matter does not contain the answer. answer_key.json (same order) gives question_type and the source files for scoring retrieval and citations.
- Rebuild and check: run the scripts in build/ (make_docs.py, make_images.py, make_qa.py), then build/verify.py, using `uv run --with python-docx --with reportlab --with pillow --with pypdf python <script>` from the repo root.
