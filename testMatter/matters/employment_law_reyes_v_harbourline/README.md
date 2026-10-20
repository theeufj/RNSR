# Reyes v Harbourline Freight (employment law, Australia) - benchmark matter

Fictional matter: Marisol Reyes, a senior logistics coordinator (Level 5), was dismissed by Harbourline Freight Pty Ltd (Port Botany, NSW) in May 2026 after a Bay 6 forklift safety complaint (12 March 2026) and a payroll query about overtime paid at single time (17 March 2026). Claims: unfair dismissal (s394, FWC matter U2026/4821), general protections (s340/s365, C2026/3390) and underpaid overtime and annual leave loading under the Harbourline Enterprise Agreement 2023 (AG2023/1982), totalling $8,388.40 per the calculation. All people, entities, numbers and addresses are invented.

Cast: Marisol Reyes (client); Declan Ashworth (Operations Manager, her manager from 1 July 2025; Helena Brandt before); Priya Venkataraman (General Manager); Lorraine Whitcombe (HR); Janelle Ng (payroll); Aisha Rahman (witness); Tomasz Kowalczyk (Dock Supervisor); Hannah Lindqvist, Owen Pretorius, Jun Takahashi (Mercer & Lindqvist Lawyers); Gareth Coyle (Barrett Coyle, for Harbourline); Ruth Fenwick (conciliator); Callum Drysdale (FWO inspector).

Key dates: employment from 14 Feb 2022; pay rate $44.00 to $46.20 on 1 July 2025; poor review 2 Apr 2026; show cause 20 Apr (amended 24 Apr); suspension 21 Apr; termination letter 5 May, corrected 11 May to a 12 May 2026 effective date; 21-day deadline 26 May (from 5 May) or 2 June (from 12 May); F2 lodged 22 May; F3 4 Jun; conciliation 30 Jun (unresolved); offers $18,500 (2 Jul) then $22,000 (6 Jul); hearing 24 Nov 2026.

Superseded facts (latest document wins): base rate, reporting line, termination date, show-cause allegation 1 and deadline, settlement offer, F2 deadline.

Files: 40 in documents/ (12 .docx, 15 .pdf, 5 .txt, 8 images: 6 .jpg and 2 .png scans/photos, including a payslip photo, handwritten timesheet, incident form, hazard tag, SMS screenshot, medical certificate, signed termination letter and staff ID card). Distractors: a staff newsletter and an electricity bill.

Load: create a Matter AI matter and upload every file in documents/ (images included).

Benchmark: qa.json is a list of {id, question, answer[, unanswerable]} (111 entries; unanswerable ones have a false premise or are not covered). answer_key.json has the same order with question_type and source file names for scoring retrieval/citations. Paste or import qa.json into the benchmark page and grade answers against it.

Rebuild: from the repo root run `uv run --with python-docx --with reportlab --with pillow --with pypdf python data/matters/employment_law_reyes_v_harbourline/build/<script>.py` for make_docs.py, make_images.py, make_qa.py, then verify.py.
