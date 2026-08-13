"""Form-field question enrichment (spec-driven).

The measured accuracy on both matter test sets depended on enriched
questions: role maps so the parties stay straight, mutually exclusive
groups collapsed into one decision, evidence rules, and per-field answer
shapes. That logic lived in per-test-set scripts under testMatter/, which
made it unshippable — a new form meant a new script.

This package is the same logic driven by a declarative FormSpec, so a new
form is a JSON file rather than Python. Conventions (how a particular form
wants its numbered parent slots or its de facto items answered) are data on
the spec, not branches in code.
"""

from rnsr.forms.enrich import build_questions, render_field_question
from rnsr.forms.fanout import fan_out
from rnsr.forms.spec import Convention, FormField, FormSpec, QuestionItem

__all__ = [
    "Convention",
    "FormField",
    "FormSpec",
    "QuestionItem",
    "build_questions",
    "fan_out",
    "render_field_question",
]
