#!/usr/bin/env python3
"""Generate test PDFs for timeline and contradiction features.

Usage:
    python scripts/generate_test_pdfs.py

Creates PDFs in test-documents/ with hierarchical structure (varied font sizes)
so RNSR's font histogram algorithm can detect the document hierarchy.
"""

from pathlib import Path
from fpdf import FPDF

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "test-documents"


def _ascii(text: str) -> str:
    """Replace Unicode characters with ASCII equivalents for core fonts."""
    return (
        text.replace("\u2014", "--")   # em-dash
            .replace("\u2013", "-")    # en-dash
            .replace("\u2018", "'")    # left single quote
            .replace("\u2019", "'")    # right single quote
            .replace("\u201c", '"')    # left double quote
            .replace("\u201d", '"')    # right double quote
            .replace("\u2026", "...")  # ellipsis
    )


class StructuredPDF(FPDF):
    """PDF with helper methods for hierarchical document structure."""

    def h1(self, text: str):
        self.set_font("Helvetica", "B", 22)
        self.cell(0, 14, _ascii(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def h2(self, text: str):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 11, _ascii(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def h3(self, text: str):
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 9, _ascii(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body(self, text: str):
        self.set_font("Helvetica", "", 11)
        self.multi_cell(0, 6, _ascii(text))
        self.ln(3)

    def separator(self):
        self.ln(6)


# =============================================================================
# 1. Timeline Test: Project History
# =============================================================================
def create_timeline_project():
    pdf = StructuredPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pdf.h1("Meridian Infrastructure Project — Full History")

    pdf.h2("1. Project Inception")
    pdf.body(
        "On 15 January 2019, the Meridian City Council approved the initial feasibility "
        "study for the Northern Bypass Highway project. The council voted 7-2 in favour, "
        "with Councillor Sarah Chen and Councillor David Okonkwo casting dissenting votes. "
        "The feasibility study was awarded to Hargrove Engineering Pty Ltd on 3 March 2019 "
        "for a contract value of $2.4 million."
    )
    pdf.body(
        "Hargrove Engineering submitted the completed feasibility report on 28 August 2019. "
        "The report recommended a 12.5 km dual carriageway route with an estimated total "
        "construction cost of $340 million. The Environmental Impact Assessment (EIA) was "
        "lodged with the Department of Environment on 15 October 2019."
    )

    pdf.h2("2. Planning and Approvals")
    pdf.body(
        "The Department of Environment issued conditional approval for the EIA on "
        "22 February 2020. Key conditions included a fauna corridor at km 4.2, noise "
        "barriers along the Henderson Valley section, and quarterly water quality monitoring "
        "at Willow Creek."
    )
    pdf.body(
        "Public consultation commenced on 1 April 2020 and ran until 30 June 2020. A total "
        "of 1,247 submissions were received. The revised planning application was submitted "
        "to the State Planning Authority on 14 September 2020."
    )
    pdf.body(
        "Final planning approval was granted on 5 March 2021, subject to 23 conditions. "
        "Detailed design commenced on 1 May 2021 under a $15.8 million contract with "
        "BridgePoint Design Group."
    )

    pdf.h2("3. Procurement and Construction")
    pdf.body(
        "The main construction tender was issued on 10 November 2021 with a closing date "
        "of 28 February 2022. Four consortia submitted bids. Following evaluation, the "
        "contract was awarded to Pacific Alliance Construction JV on 18 May 2022 for "
        "$378 million."
    )
    pdf.body(
        "Construction officially commenced on 1 August 2022 with an expected completion "
        "date of 30 June 2025. Site clearing and earthworks began at the southern interchange "
        "on 15 August 2022."
    )
    pdf.body(
        "On 12 December 2022, unexpected contaminated soil was discovered at the former "
        "Dawson Industrial Estate (km 2.1–2.8). Remediation works added $14.2 million to "
        "the project cost and delayed the southern section by approximately four months."
    )

    pdf.h2("4. Milestones and Progress")
    pdf.h3("4.1 Year 1 — August 2022 to July 2023")
    pdf.body(
        "The Willow Creek Bridge (328 metres) reached structural completion on 20 June 2023. "
        "Road base was laid on 8.4 km of the 12.5 km route by 31 July 2023. The project was "
        "reported as 38% complete at the end of Year 1."
    )

    pdf.h3("4.2 Year 2 — August 2023 to July 2024")
    pdf.body(
        "Asphalt paving commenced on 3 September 2023. Noise barriers along Henderson Valley "
        "were installed between 15 November 2023 and 28 February 2024. The fauna corridor "
        "bridge at km 4.2 was completed on 10 April 2024. By 31 July 2024, the project was "
        "reported as 74% complete."
    )

    pdf.h3("4.3 Year 3 — August 2024 to Present")
    pdf.body(
        "Traffic signalling and intelligent transport systems (ITS) installation began on "
        "5 September 2024. Line marking and road furniture installation commenced on "
        "18 November 2024. Final safety audits are scheduled for 15 March 2025. The revised "
        "completion date is 30 September 2025 due to the earlier contamination delay."
    )

    pdf.h2("5. Financial Summary")
    pdf.body(
        "Original approved budget: $340 million (28 August 2019). "
        "Revised budget after contamination: $354.2 million (approved 14 March 2023). "
        "Current forecast at completion: $361.5 million (as of 31 December 2024). "
        "Total expenditure to date: $267.3 million."
    )

    pdf.h2("6. Key Personnel")
    pdf.body(
        "Project Director: Margaret Liu (appointed 1 June 2021). "
        "Chief Engineer: Dr Rajesh Patel (appointed 1 August 2022). "
        "Environmental Manager: James Whitfield (appointed 15 March 2020). "
        "Community Liaison Officer: Amanda Torres (appointed 1 April 2020). "
        "Councillor Sarah Chen remained the council's project champion throughout."
    )

    pdf.output(str(OUTPUT_DIR / "Timeline - Meridian Project History.pdf"))
    print("  Created: Timeline - Meridian Project History.pdf")


# =============================================================================
# 2. Timeline Test: Legal Case Chronology
# =============================================================================
def create_timeline_legal():
    pdf = StructuredPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pdf.h1("Chronology of Proceedings — Baxter v Thornton Industries")

    pdf.h2("Background")
    pdf.body(
        "Ms Rachel Baxter was employed by Thornton Industries Ltd as a Senior Process "
        "Engineer from 14 February 2017. Her annual salary at the time of termination "
        "was $142,500. She reported to Plant Manager Mr Gregory Thornton."
    )

    pdf.h2("Events Leading to Dispute")
    pdf.body(
        "On 3 July 2022, Ms Baxter submitted a formal complaint to the HR department "
        "alleging unsafe working conditions in Plant B. The complaint was acknowledged "
        "on 5 July 2022 by HR Manager Lisa Carmichael."
    )
    pdf.body(
        "An internal investigation commenced on 18 July 2022 and concluded on "
        "29 August 2022. The investigation report found that three of Ms Baxter's "
        "five allegations were substantiated."
    )
    pdf.body(
        "On 15 September 2022, Ms Baxter was issued a formal warning for alleged "
        "insubordination during a meeting on 8 September 2022. Ms Baxter denied the "
        "allegations and filed a grievance on 22 September 2022."
    )
    pdf.body(
        "Ms Baxter's employment was terminated on 30 November 2022. The termination "
        "letter cited performance concerns and the formal warning."
    )

    pdf.h2("Proceedings")
    pdf.body(
        "Ms Baxter filed an unfair dismissal application with the Fair Work Commission "
        "on 14 December 2022. A conciliation conference was held on 2 February 2023 "
        "but did not result in settlement."
    )
    pdf.body(
        "The matter proceeded to hearing on 15 May 2023, 16 May 2023, and 17 May 2023. "
        "Commissioner Angela Morrison presided. Evidence was given by Ms Baxter, "
        "Mr Thornton, Ms Carmichael, and two independent witnesses."
    )
    pdf.body(
        "The decision was handed down on 28 July 2023. Commissioner Morrison found "
        "that the dismissal was unfair and ordered reinstatement effective 1 September 2023 "
        "with back pay of $98,750 covering the period of dismissal."
    )

    pdf.h2("Post-Decision")
    pdf.body(
        "Thornton Industries filed an appeal on 25 August 2023. The appeal was heard "
        "on 12 November 2023 by a Full Bench comprising Vice President Roberts, "
        "Deputy President Singh, and Commissioner Lee."
    )
    pdf.body(
        "The Full Bench dismissed the appeal on 19 January 2024 and upheld the original "
        "decision in its entirety. Ms Baxter was reinstated on 5 February 2024."
    )

    pdf.output(str(OUTPUT_DIR / "Timeline - Baxter v Thornton.pdf"))
    print("  Created: Timeline - Baxter v Thornton.pdf")


# =============================================================================
# 3. Contradiction Test: Single Document with Internal Conflicts
# =============================================================================
def create_contradictions_single():
    pdf = StructuredPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pdf.h1("Annual Report 2024 — Greenfield Pharmaceuticals Ltd")

    pdf.h2("1. Executive Summary")
    pdf.body(
        "Greenfield Pharmaceuticals achieved record revenue of $892 million in FY2024, "
        "representing growth of 18% over the prior year. Net profit after tax was "
        "$134 million. The company employed 3,200 staff across 14 global offices."
    )

    pdf.h2("2. CEO Letter to Shareholders")
    pdf.body(
        "Dear Shareholders, I am pleased to report that FY2024 was a transformative year "
        "for Greenfield. Total revenue reached $887 million, up from $756 million in FY2023. "
        "We expanded our workforce to 3,450 employees and opened two new research facilities "
        "in Singapore and Munich."
    )
    pdf.body(
        "Our flagship product, Cardioven, generated $312 million in sales during the year, "
        "solidifying its position as the market leader in cardiovascular treatments."
    )

    pdf.h2("3. Financial Highlights")
    pdf.h3("3.1 Revenue")
    pdf.body(
        "Total consolidated revenue for the twelve months ended 30 June 2024 was "
        "$892.3 million (FY2023: $755.8 million), an increase of 18.0%. Revenue from "
        "the Americas region was $412 million. Revenue from EMEA was $298 million. "
        "Revenue from Asia-Pacific was $182.3 million."
    )

    pdf.h3("3.2 Profitability")
    pdf.body(
        "Net profit after tax was $127.4 million (FY2023: $98.2 million). The effective "
        "tax rate was 24.1%. Operating expenses increased by 12% to $623 million, primarily "
        "driven by R&D investment."
    )

    pdf.h3("3.3 Product Performance")
    pdf.body(
        "Cardioven, the company's leading cardiovascular drug, recorded annual sales of "
        "$298 million (FY2023: $265 million), representing 33% of total revenue. Neurolix, "
        "the neurological treatment launched in Q2, contributed $45 million in its first "
        "partial year."
    )

    pdf.h2("4. Human Resources")
    pdf.body(
        "As at 30 June 2024, Greenfield employed 3,200 full-time equivalent staff "
        "(FY2023: 2,850). The company operates from 12 offices globally, with the "
        "largest sites in Boston (headquarters, 680 staff) and London (420 staff)."
    )
    pdf.body(
        "Voluntary turnover was 8.2% for the year, down from 11.5% in FY2023. The "
        "company did not undertake any restructuring or redundancy programs during FY2024."
    )

    pdf.h2("5. Research and Development")
    pdf.body(
        "R&D expenditure totalled $189 million in FY2024 (FY2023: $156 million), "
        "representing 21.2% of revenue. The company has 8 compounds in clinical trials."
    )
    pdf.body(
        "Phase III trials for Pulmonex (chronic obstructive pulmonary disease) were "
        "completed in April 2024 with positive results. Regulatory submission to the "
        "FDA was filed on 15 August 2024. Approval is not expected until Q2 2025."
    )

    pdf.h2("6. Outlook")
    pdf.body(
        "Management expects FY2025 revenue of $980–$1,020 million, driven by continued "
        "growth of Cardioven and the full-year contribution of Neurolix. Pulmonex approval "
        "is anticipated in early 2025, with launch expected in Q3 2025."
    )

    pdf.output(str(OUTPUT_DIR / "Contradictions - Greenfield Annual Report.pdf"))
    print("  Created: Contradictions - Greenfield Annual Report.pdf")


# =============================================================================
# 4. Cross-doc Contradiction: Expert Report A
# =============================================================================
def create_crossdoc_expert_a():
    pdf = StructuredPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pdf.h1("Expert Medical Report — Dr Fiona Hartley")
    pdf.body("Prepared for: Workers Compensation Commission of NSW")
    pdf.body("Date: 14 March 2024")
    pdf.body("Re: Mr Benjamin Reeves (DOB: 22 April 1985)")
    pdf.separator()

    pdf.h2("1. Qualifications")
    pdf.body(
        "I am Dr Fiona Hartley, MBBS, FRANZCP, a consultant psychiatrist with 18 years "
        "of clinical experience. I have been asked to provide an independent assessment "
        "of Mr Reeves' psychological condition following a workplace incident."
    )

    pdf.h2("2. History of Injury")
    pdf.body(
        "Mr Reeves was involved in a workplace accident on 5 June 2023 at the Warwick "
        "Distribution Centre. He was struck by a forklift travelling at approximately "
        "15 km/h, resulting in a fractured left tibia and soft tissue injuries to the "
        "lower back. He was transported to Westmead Hospital by ambulance and admitted "
        "for 6 days."
    )

    pdf.h2("3. Current Symptoms")
    pdf.body(
        "At the time of my examination on 8 March 2024, Mr Reeves reported persistent "
        "anxiety, nightmares occurring 4-5 times per week, hypervigilance in workplace "
        "settings, and depressed mood. He described difficulty concentrating and stated "
        "he has been unable to return to work since the accident."
    )

    pdf.h2("4. Diagnosis")
    pdf.body(
        "In my opinion, Mr Reeves meets the diagnostic criteria for Post-Traumatic Stress "
        "Disorder (PTSD) under DSM-5. He also presents with a moderate Major Depressive "
        "Episode secondary to the PTSD and chronic pain. His GAF score is 45, indicating "
        "serious impairment in occupational and social functioning."
    )

    pdf.h2("5. Capacity for Work")
    pdf.body(
        "Mr Reeves is currently totally unfit for his pre-injury role as a warehouse "
        "supervisor. In my opinion, he is unlikely to be fit for any form of employment "
        "for at least the next 12 months. His prognosis is guarded given the severity "
        "of the PTSD and the comorbid depression."
    )

    pdf.h2("6. Treatment Recommendations")
    pdf.body(
        "I recommend weekly trauma-focused cognitive behavioural therapy (TF-CBT) for a "
        "minimum of 20 sessions, continuation of his current medication (sertraline 150mg "
        "daily), and review in 6 months. A graduated return-to-work program should not be "
        "attempted before March 2025 at the earliest."
    )

    pdf.output(str(OUTPUT_DIR / "CrossDoc - Expert Report A (Dr Hartley).pdf"))
    print("  Created: CrossDoc - Expert Report A (Dr Hartley).pdf")


# =============================================================================
# 5. Cross-doc Contradiction: Expert Report B (conflicting opinions)
# =============================================================================
def create_crossdoc_expert_b():
    pdf = StructuredPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pdf.h1("Independent Medical Examination — Dr Marcus Webb")
    pdf.body("Prepared for: Insurer — National Mutual Workers Insurance")
    pdf.body("Date: 22 April 2024")
    pdf.body("Re: Mr Benjamin Reeves (DOB: 22 April 1985)")
    pdf.separator()

    pdf.h2("1. Qualifications")
    pdf.body(
        "I am Dr Marcus Webb, MBBS, FRACP, FAFRM, a specialist in occupational and "
        "rehabilitation medicine with 22 years of experience. I was engaged by National "
        "Mutual Workers Insurance to conduct an independent assessment of Mr Reeves."
    )

    pdf.h2("2. History of Injury")
    pdf.body(
        "Mr Reeves sustained injuries in a workplace incident on 5 June 2023 at the "
        "Warwick Distribution Centre. According to the incident report, he was clipped by "
        "a slow-moving forklift travelling at approximately 5 km/h. He sustained a hairline "
        "fracture of the left tibia and minor bruising to the lumbar region. He was taken to "
        "Westmead Hospital where he was admitted for 3 days before discharge."
    )

    pdf.h2("3. Current Presentation")
    pdf.body(
        "I examined Mr Reeves on 18 April 2024. He reported ongoing anxiety related to "
        "the workplace, occasional sleep disturbance, and low mood. On examination, he "
        "was well-groomed, made good eye contact, and his affect was mildly restricted but "
        "not flat. His concentration appeared adequate during our 90-minute consultation."
    )

    pdf.h2("4. Diagnosis")
    pdf.body(
        "In my opinion, Mr Reeves does not meet the full diagnostic criteria for PTSD "
        "under DSM-5. His symptoms are more consistent with an Adjustment Disorder with "
        "mixed anxiety and depressed mood. His GAF score is 62, indicating mild to moderate "
        "impairment. The depressive symptoms are mild and do not warrant a separate diagnosis "
        "of Major Depressive Episode."
    )

    pdf.h2("5. Capacity for Work")
    pdf.body(
        "Mr Reeves is fit for suitable duties on a graduated return-to-work basis. He could "
        "commence with 3 days per week in a supervisory role that does not involve direct "
        "forklift operation. I would expect him to be capable of full pre-injury duties "
        "within 8 to 12 weeks of commencing a return-to-work program."
    )

    pdf.h2("6. Treatment Recommendations")
    pdf.body(
        "I recommend a short course of 6-8 sessions of CBT with a focus on anxiety "
        "management and workplace re-integration. His current medication (sertraline 150mg) "
        "should be reviewed with a view to tapering and cessation within 3 months. A "
        "graduated return-to-work program should commence within 4 weeks."
    )

    pdf.h2("7. Comment on Dr Hartley's Report")
    pdf.body(
        "I have reviewed the report of Dr Fiona Hartley dated 14 March 2024. I respectfully "
        "disagree with her diagnosis of PTSD and the assessment that Mr Reeves is unfit for "
        "12 months. The GAF score of 45 assigned by Dr Hartley is inconsistent with my "
        "clinical findings, which support a score of 62. The description of the incident "
        "as a high-speed impact is not supported by the employer's incident report, which "
        "records the forklift speed as approximately 5 km/h, not 15 km/h."
    )

    pdf.output(str(OUTPUT_DIR / "CrossDoc - Expert Report B (Dr Webb).pdf"))
    print("  Created: CrossDoc - Expert Report B (Dr Webb).pdf")


# =============================================================================
# 6. Cross-doc Contradiction: Employer's Incident Report
# =============================================================================
def create_crossdoc_incident():
    pdf = StructuredPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pdf.h1("Workplace Incident Report")
    pdf.body("Warwick Distribution Centre — Incident #WDC-2023-0047")
    pdf.body("Date of Incident: 5 June 2023")
    pdf.body("Date of Report: 7 June 2023")
    pdf.separator()

    pdf.h2("1. Injured Worker Details")
    pdf.body(
        "Name: Benjamin Reeves. Date of Birth: 22 April 1985. Position: Warehouse "
        "Supervisor (Level 3). Employment Start Date: 8 January 2018. "
        "Normal Hours: Monday to Friday, 7:00 AM to 3:30 PM."
    )

    pdf.h2("2. Incident Description")
    pdf.body(
        "At approximately 10:15 AM on 5 June 2023, Mr Reeves was walking through Aisle 7 "
        "of Warehouse B when he was struck by a counterbalance forklift operated by Mr Kyle "
        "Patterson (Forklift Operator, Level 2). The forklift was travelling in reverse at "
        "an estimated speed of 8 km/h."
    )
    pdf.body(
        "Mr Reeves was knocked to the ground and the rear wheel of the forklift made "
        "contact with his left lower leg. First aid was administered at the scene by "
        "Mr David Chen (First Aid Officer). An ambulance was called at 10:22 AM and "
        "Mr Reeves was transported to Westmead Hospital, arriving at approximately 10:55 AM."
    )

    pdf.h2("3. Injuries Sustained")
    pdf.body(
        "As per the hospital discharge summary (12 June 2023, 7 days admission), "
        "Mr Reeves sustained: a comminuted fracture of the left tibial shaft, "
        "contusion and soft tissue damage to the lumbar spine, and multiple abrasions "
        "to the left hip and thigh."
    )

    pdf.h2("4. Investigation Findings")
    pdf.body(
        "The investigation found that: (a) Mr Patterson did not sound the horn before "
        "reversing, in breach of Standard Operating Procedure WHS-014; (b) the convex "
        "mirror at the Aisle 7 junction had been damaged and not replaced; (c) Mr Reeves "
        "was wearing high-visibility clothing as required."
    )

    pdf.h2("5. Witness Statements")
    pdf.body(
        "Mr David Chen (First Aid Officer): \"I heard a shout and ran to Aisle 7. "
        "Ben was on the ground holding his leg. Kyle was very shaken. The forklift "
        "was still running. I applied a splint and called 000.\""
    )
    pdf.body(
        "Mr Kyle Patterson (Forklift Operator): \"I was reversing out of Aisle 7. "
        "I did not see Mr Reeves. I was travelling slowly, maybe 8 or 9 km/h. "
        "I heard a thud and stopped immediately.\""
    )

    pdf.h2("6. Corrective Actions")
    pdf.body(
        "All aisle junction mirrors were inspected and replaced where necessary (completed "
        "8 June 2023). Refresher training on SOP WHS-014 was conducted for all forklift "
        "operators on 12 June 2023. A proximity warning system trial was approved on "
        "20 June 2023."
    )

    pdf.output(str(OUTPUT_DIR / "CrossDoc - Employer Incident Report.pdf"))
    print("  Created: CrossDoc - Employer Incident Report.pdf")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating test PDFs in test-documents/...\n")

    create_timeline_project()
    create_timeline_legal()
    create_contradictions_single()
    create_crossdoc_expert_a()
    create_crossdoc_expert_b()
    create_crossdoc_incident()

    print(f"\nDone! {6} PDFs created in {OUTPUT_DIR}")
