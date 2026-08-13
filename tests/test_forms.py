"""Spec-driven question enrichment, fan-out, and regression scoring."""

import json

import pytest

from rnsr.eval.regression import (
    load_golden,
    score_run,
    string_agrees,
)
from rnsr.forms import build_questions
from rnsr.forms.fanout import fan_out
from rnsr.forms.spec import Convention, FormSpec, load_spec, parse_field

GROUP_NOTES = (
    'For Respondent 1 — Daniel Robert Mitchell. Respondent 1: Gender - M (male)'
    "\n\nFIELD TYPE: radio button. Respond with the exact option text shown on "
    'the form, or "No" if not applicable.'
    '\n\nMUTUALLY EXCLUSIVE GROUP "Respondent 1: gender" (What is {{person}}\'s '
    'gender?): this field and ["Respondent 1: Gender - F (female)"] are '
    'alternative answers to the same question — EXACTLY ONE of them may have a '
    "value."
)


def _gender_spec() -> FormSpec:
    # titles mirror the vendor's phrasing, which is what marks a field as
    # recording its selection with "yes" rather than the option's full text
    male = parse_field({
        "id": "r1_male",
        "title": '[Radio button] Is Daniel Robert Mitchell a male? If this is '
                 'correct, return exactly "yes"',
        "notes": GROUP_NOTES})
    female = parse_field({
        "id": "r1_female",
        "title": '[Radio button] Is Daniel Robert Mitchell a female? If this is '
                 'correct, return exactly "yes"',
        "notes": GROUP_NOTES.replace("Gender - M (male)", "Gender - F (female)")
                           .replace('["Respondent 1: Gender - F (female)"]',
                                    '["Respondent 1: Gender - M (male)"]')})
    return FormSpec(form="Initiating Application",
                    roles={"applicant_1": "Sarah Jane Mitchell",
                           "respondent_1": "Daniel Robert Mitchell"},
                    fields=[male, female])


class TestSpecParsing:
    def test_vendor_notes_yield_structure(self):
        f = parse_field({"id": "r1_male", "title": 'return exactly "yes"',
                         "notes": GROUP_NOTES})
        assert f.role == "Respondent 1"
        assert f.subject == "Daniel Robert Mitchell"
        assert f.group == "Respondent 1: gender"
        assert f.group_question == "What is {{person}}'s gender?"
        assert f.field_type == "radio button"
        assert f.needs_value is False
        assert f.wants_yes is True

    def test_date_fields_need_a_value(self):
        f = parse_field({"id": "dom", "title": "Date of marriage",
                         "notes": "FIELD TYPE: date."})
        assert f.needs_value is True

    def test_validate_catches_a_sibling_outside_its_group(self):
        spec = _gender_spec()
        spec.fields[1].group = "Some other group"
        problems = spec.validate()
        assert any("outside the group" in p for p in problems)

    def test_build_refuses_an_inconsistent_spec(self):
        spec = _gender_spec()
        spec.fields[1].group = "Some other group"
        with pytest.raises(ValueError, match="inconsistent"):
            build_questions(spec)

    def test_loads_a_vendor_golden_export_directly(self, tmp_path):
        path = tmp_path / "spec.json"
        path.write_text(json.dumps({
            "context": "Form: Initiating Application.\nRoles: Applicant 1 = "
                       "Sarah Jane Mitchell; respondent_1 = Daniel Mitchell.",
            "fields": [{"id": "r1_male", "title": "t", "notes": GROUP_NOTES,
                        "golden": ["yes"]}],
        }))
        spec = load_spec(path)
        assert spec.form == "Initiating Application"
        assert spec.roles["applicant_1"] == "Sarah Jane Mitchell"
        assert spec.fields[0].golden == ["yes"]


class TestEnrichment:
    def test_group_becomes_one_single_choice_question(self):
        items = build_questions(_gender_spec())
        assert len(items) == 1
        item = items[0]
        assert item.kind == "group" and item.mode == "options"
        assert [m["id"] for m in item.members] == ["r1_male", "r1_female"]
        assert "exactly one option may hold a value" in item.question
        assert "option_id: r1_male" in item.question

    def test_question_names_the_subject_and_the_wrong_party(self):
        question = build_questions(_gender_spec())[0].question
        assert "THIS QUESTION IS ABOUT: Respondent 1 - Daniel Robert Mitchell" in question
        assert "Do NOT use details belonging to Applicant 1" in question

    def test_conventions_attach_by_group_and_carry_no_matter_facts(self):
        spec = _gender_spec()
        spec.conventions = [Convention(
            text="FORM CONVENTION: a name is not evidence of gender.",
            groups=("Respondent 1: gender",))]
        question = build_questions(spec)[0].question
        assert "a name is not evidence of gender" in question

    def test_convention_scoped_elsewhere_is_not_attached(self):
        spec = _gender_spec()
        spec.conventions = [Convention(text="UNRELATED", groups=("55c",))]
        assert "UNRELATED" not in build_questions(spec)[0].question

    def test_value_group_asks_for_the_value_not_an_option_id(self):
        notes = ('FIELD TYPE: date.\n\nMUTUALLY EXCLUSIVE GROUP "Date of '
                 'marriage" (What is the date of marriage?): this field and '
                 '["Date of marriage - not applicable"] are alternative '
                 "answers to the same question")
        date_field = parse_field({"id": "dom", "title": "Date of marriage",
                                  "notes": notes})
        na = parse_field({
            "id": "dom_na", "title": "Not applicable",
            "notes": 'Date of marriage - not applicable\n\nFIELD TYPE: '
                     'checkbox.\n\nMUTUALLY EXCLUSIVE GROUP "Date of marriage" '
                     '(What is the date of marriage?): this field and ["Date '
                     'of marriage"] are alternative answers'})
        na.option_label = "Date of marriage - not applicable"
        date_field.option_label = "Date of marriage"
        spec = FormSpec(fields=[date_field, na])
        item = build_questions(spec)[0]
        assert item.mode == "value"
        assert "DD/MM/YYYY" in item.question
        assert "not_applicable" in item.question

    def test_large_corpus_note_steers_away_from_reading_everything(self):
        spec = _gender_spec()
        spec.corpus_note = "999 files across seven folders."
        question = build_questions(spec)[0].question
        assert "999 files" in question
        assert "do not try to read every document" in question.lower()

    def test_reproduces_the_hand_written_testmatter_structure(self):
        """The generic layer must match the script whose accuracy was measured;
        otherwise shipping it would ship a different system."""
        from pathlib import Path

        golden = Path("testMatter/golden.json")
        legacy = Path("testMatter/questions_enriched_map.json")
        if not (golden.exists() and legacy.exists()):
            pytest.skip("testMatter fixtures not present")
        items = build_questions(load_spec(golden))
        expected = json.loads(legacy.read_text())["items"]
        assert {i.item_id for i in items} == {i["item_id"] for i in expected}


class TestFanOut:
    def test_winner_takes_yes_and_siblings_take_no(self):
        items = build_questions(_gender_spec())
        fields, notes = fan_out(items, ["ANSWER: r1_male\nVALUE: n/a"])
        assert fields == {"r1_male": "yes", "r1_female": "No"}
        assert notes == []

    def test_unknown_leaves_every_option_negative(self):
        items = build_questions(_gender_spec())
        fields, _ = fan_out(items, ["ANSWER: unknown\nVALUE: n/a"])
        assert set(fields.values()) == {"No"}

    def test_choice_recovered_from_prose(self):
        items = build_questions(_gender_spec())
        fields, notes = fan_out(
            items, ["Having read the intake form, the answer is r1_male."])
        assert fields["r1_male"] == "yes"
        assert any("recovered choice from prose" in n for n in notes)

    def test_contradictory_siblings_are_impossible(self):
        # the failure this design removes: asked separately, both options
        # were answered "yes"
        items = build_questions(_gender_spec())
        fields, _ = fan_out(items, ["ANSWER: r1_female\nVALUE: n/a"])
        assert sum(v == "yes" for v in fields.values()) == 1

    def test_answer_count_mismatch_is_refused(self):
        items = build_questions(_gender_spec())
        with pytest.raises(ValueError, match="answers for"):
            fan_out(items, [])


class TestRegressionScoring:
    def test_blank_golden_is_satisfied_by_a_negative(self):
        assert string_agrees([], "No")
        assert string_agrees([], "Not found in matter corpus")
        assert not string_agrees([], "Yes")

    def test_option_text_and_yes_are_equivalent(self):
        assert string_agrees(["55e: No"], "no")
        assert string_agrees(["yes"], "Respondent 1: Gender - M (male) - yes")

    def test_wording_differences_still_need_the_judge(self):
        assert not string_agrees(["Party to a marriage, Parent"],
                                 "The applicant is a spouse and a parent")

    def test_scores_and_gates_on_accuracy(self):
        golden = {"a": ["yes"], "b": ["14/02/2014"], "c": []}
        answers = {"a": "yes", "b": "14/06/2014", "c": "No"}
        report = score_run(golden, answers, min_accuracy=0.9)
        assert report.correct == 2 and report.total == 3
        assert not report.passed
        assert report.substantive == (1, 2)
        assert report.summary()["disagreements"][0]["field_id"] == "b"

    def test_report_writes_comparison_and_summary(self, tmp_path):
        report = score_run({"a": ["yes"]}, {"a": "yes"}, min_accuracy=1.0)
        path = report.write(tmp_path)
        assert path.exists()
        assert (tmp_path / "comparison.csv").read_text().startswith("field_id")
        assert json.loads(path.read_text())["passed"] is True

    def test_load_golden_accepts_vendor_and_item_shapes(self, tmp_path):
        vendor = tmp_path / "v.json"
        vendor.write_text(json.dumps({"fields": [{"id": "a", "golden": ["x"]}]}))
        assert load_golden(vendor) == {"a": ["x"]}
        items = tmp_path / "i.json"
        items.write_text(json.dumps({"items": [{"qid": "b", "golden": "y"}]}))
        assert load_golden(items) == {"b": ["y"]}
