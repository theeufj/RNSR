"""Single-field discovery nudges must retain person/property uncertainty."""
from rnsr.harness.negative_audit import audit_negatives
from rnsr.ingest.model import Element, ParsedDocument
from rnsr.ingest.pipeline import ingest


class Events:
    def __init__(self):
        self.events = []

    def event(self, name, **fields):
        self.events.append((name, fields))


def corpus(tmp_path, text):
    def parse(path):
        return ParsedDocument(doc_id='letter', source_path=str(path), sha256='e' * 64,
                              parser='test', n_pages=1, elements=[Element('text', text, 1)])
    out = tmp_path / 'source.db'
    ingest([tmp_path / 'letter.txt'], out, parse=parse)
    return str(out)


def questions():
    boilerplate = 'Do not guess gender from a name. Read retained evidence.'
    return [('a', boilerplate + '\nSubject: Morgan. Field: Gender'),
            ('b', boilerplate + '\nSubject: Morgan. Field: Citizenship')]


def test_gender_field_discovery_survives_shared_warning(tmp_path):
    db = corpus(tmp_path, 'For Morgan, our signed record expressly lists male.')
    events = Events()
    result = audit_negatives({'value': {'a': 'unknown', 'b': 'unknown'}}, questions(), db, events)
    assert result and 'expressly lists male' in result
    assert events.events == [('negative_audit', {'flagged': ['a']})]
    assert 'not established contradictions' in result
    assert 'exact person, period, and requested property' in result


def test_discovery_does_not_treat_other_person_as_contradiction(tmp_path):
    db = corpus(tmp_path, 'The patient Taylor has a recorded sex: female.')
    result = audit_negatives({'value': {'a': 'unknown'}}, questions(), db, Events())
    assert result and 'Taylor' in result
    assert 'retrieval candidates' in result
    assert 'otherwise resubmit the negative unchanged' in result
    assert 'are not interchangeable facts' in result


def test_shared_warning_alone_does_not_trigger_gender_probe(tmp_path):
    db = corpus(tmp_path, 'The patient Taylor has a recorded sex: female.')
    result = audit_negatives({'value': {'b': 'unknown'}}, questions(), db, Events())
    assert result is None


def test_honorifics_do_not_supply_gender_evidence(tmp_path):
    db = corpus(tmp_path, 'Mr Morgan signed this letter. No attributes are stated.')
    result = audit_negatives({'value': {'a': 'unknown'}}, questions(), db, Events())
    assert result is None


def test_verified_negative_does_not_get_unquoted_fallback(tmp_path):
    db = corpus(tmp_path, 'The patient Taylor has a recorded sex: female.')
    final = {'value': {'a': 'unknown'}, 'verification': {'a': {'passed': True, 'quotes': ['a quote']}}}
    assert audit_negatives(final, questions(), db, Events()) is None
