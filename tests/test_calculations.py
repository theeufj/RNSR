"""Trusted source arithmetic must not certify an invented replacement value."""
import sqlite3

import pytest

from rnsr.db.artifact import CorpusDB
from rnsr.env.calculations import CalculationRegistry
from rnsr.env.sandbox import SandboxedRepl
from rnsr.ingest.model import Element, ParsedDocument, RawTable
from rnsr.ingest.pipeline import ingest


@pytest.fixture
def arithmetic_corpus(tmp_path):
    def parse(path):
        rows = [['Revenue', '100'], ['Costs', '250'], ['Interest', '10'],
                ['Small A', '0.1'], ['Small B', '0.2'], ['Zero', '0']]
        canonical = 'Item | Amount\n' + '\n'.join(' | '.join(r) for r in rows)
        return ParsedDocument(
            doc_id='report', source_path=str(path), sha256='b'*64, n_pages=1, parser='test',
            elements=[Element('heading', 'USD, fiscal 2022', 1, heading_level=1),
                      Element('table', canonical, 1)],
            tables=[RawTable(page=1, header=['Item', 'Amount'], rows=rows,
                             caption='Financial metrics', extractor='test')])
    out = tmp_path / 'corpus.db'
    ingest([tmp_path/'report.pdf'], out, parse=parse)
    return out


@pytest.fixture
def registry(arithmetic_corpus):
    with CorpusDB(arithmetic_corpus) as corpus:
        yield CalculationRegistry(corpus.conn)


def fact(registry, row):
    return registry.source_number('t_report_001', row, 'amount')


def test_source_is_resolved_from_original_cell_with_exact_span(registry):
    source = fact(registry, 1)
    assert source['value'] == '100.0'
    assert source['raw_value'] == '100'
    assert source['unit'] is None and source['period'] is None
    assert source['quote'] == 'Revenue | 100'
    span = source['source']
    text = registry.conn.execute('SELECT text FROM doc_text').fetchone()[0]
    assert text[span['char_start']:span['char_end']] == source['quote']
    assert span['column'] == 'amount' and span['rowid'] == 1
    assert len(span['source_span_id']) == 64


def test_metadata_is_retained_source_text_not_a_model_assertion(registry):
    text = registry.conn.execute('SELECT text FROM doc_text').fetchone()[0]
    unit_start, year_start = text.index('USD'), text.index('2022')
    source = registry.source_number(
        't_report_001', 1, 'amount',
        unit_span={'char_start': unit_start, 'char_end': unit_start+3},
        period_span={'char_start': year_start, 'char_end': year_start+4})
    assert source['unit'] == 'USD' and source['period'] == '2022'
    assert 'not semantically verified' in source['metadata_check']
    with pytest.raises(ValueError, match='requires char_start'):
        registry.source_number('t_report_001', 1, 'amount', unit_span={'value': 'EUR'})
    with pytest.raises(ValueError, match='outside retained'):
        registry.source_number('t_report_001', 1, 'amount', period_span={'char_start':9999,'char_end':10003})


def test_negative_arithmetic_and_full_derivation_survive_final(registry):
    revenue, costs, interest = [fact(registry, row) for row in (1, 2, 3)]
    earnings = registry.calculate('subtract', [revenue['id'], costs['id']])
    ratio = registry.calculate('divide', [earnings['id'], interest['id']])
    assert earnings['value'] == '-150.0'
    assert ratio['value'] == '-15'
    value, verification = registry.resolve_final(ratio['id'], '-15')
    assert value == '-15'
    assert verification['passed'] and verification['check'] == 'source_bound_calculation'
    assert len(verification['records']) == 5
    assert verification['claim_support'] == 'not_checked'
    assert {r['raw_value'] for r in verification['records'] if r['kind']=='source'} == {'100','250','10'}
    with pytest.raises(ValueError, match='differs'):
        registry.resolve_final(ratio['id'], '15')


def test_decimal_addition_has_no_new_float_rounding(registry):
    a, b = fact(registry, 4), fact(registry, 5)
    result = registry.calculate('sum', [a['id'], b['id']])
    assert result['value'] == '0.3'
    percent = registry.calculate('percent', [result['id']])
    assert percent['value'] == '30.0'
    assert percent['operation'] == 'percent'


@pytest.mark.parametrize('operand', ['100', 'calc_forged', {'value':'100'}, None])
def test_literals_and_forged_ids_cannot_enter_operations(registry, operand):
    a = fact(registry, 1)
    with pytest.raises(ValueError, match='unknown calculation/source id'):
        registry.calculate('divide', [a['id'], operand])


def test_caller_mutation_cannot_change_parent_records(registry):
    a = fact(registry, 1)
    a['value'] = '999'
    a['source']['column'] = 'forged'
    saved = registry.get(a['id'])
    assert saved['value'] == '100.0' and saved['source']['column'] == 'amount'
    result = registry.calculate('sum', [a['id']])
    result['value'] = '-999'
    with pytest.raises(ValueError, match='differs'):
        registry.resolve_final(result['id'], result['value'])


def test_ids_do_not_transfer_between_sessions(registry):
    other = CalculationRegistry(registry.conn)
    a = fact(registry, 1)
    with pytest.raises(ValueError, match='unknown calculation/source id'):
        other.get(a['id'])


def test_unsupported_math_zero_division_and_arity_fail(registry):
    a, zero = fact(registry, 1), fact(registry, 6)
    with pytest.raises(ValueError, match='division by zero'):
        registry.calculate('divide', [a['id'], zero['id']])
    with pytest.raises(ValueError, match='operation must'):
        registry.calculate('__import__("os")', [a['id']])
    with pytest.raises(ValueError, match='wrong number'):
        registry.calculate('divide', [a['id']])


@pytest.mark.parametrize('table,row,column', [
    ('documents',1,'n_pages'), ('t_report_001',True,'amount'),
    ('t_report_001',1,'item'), ('t_report_001',1,'amount__raw'),
    ('t_report_001',100,'amount'), ('t_report_001',1,'amount"; DROP TABLE documents;--'),
])
def test_only_registered_original_numeric_cells_are_operands(registry, table, row, column):
    with pytest.raises(ValueError):
        registry.source_number(table, row, column)


def test_annotation_numeric_column_is_not_a_source(registry, arithmetic_corpus):
    from rnsr.db import schema
    with sqlite3.connect(arithmetic_corpus) as conn:
        schema.add_annotation_column(conn, 't_report_001', 'invented')
    with pytest.raises(ValueError, match='original numeric column'):
        registry.source_number('t_report_001', 1, 'invented')


def test_generation_changes_invalidate_existing_results(registry, arithmetic_corpus):
    from rnsr.db import schema

    a = fact(registry, 1)
    result = registry.calculate('sum', [a['id']])
    with sqlite3.connect(arithmetic_corpus) as conn:
        schema.unfreeze_corpus(conn)
        conn.execute("UPDATE documents SET ingested_at='replacement'")
        schema.finalize_corpus(conn)
    with pytest.raises(ValueError, match='source changed'):
        registry.resolve_final(result['id'], result['value'])


def test_same_generation_cell_replacement_cannot_reuse_result(registry, arithmetic_corpus):
    from rnsr.db import schema

    a = fact(registry, 1)
    result = registry.calculate('sum', [a['id']])
    with sqlite3.connect(arithmetic_corpus) as conn:
        schema.unfreeze_table(conn, 't_report_001')
        conn.execute('UPDATE t_report_001 SET amount=999 WHERE rowid=1')
    with pytest.raises(ValueError, match='value or evidence changed'):
        registry.resolve_final(result['id'], result['value'])


def test_source_id_cannot_be_presented_as_a_computation(registry):
    a = fact(registry, 1)
    with pytest.raises(ValueError, match='calculation result id'):
        registry.resolve_final(a['id'], a['value'])


def test_operand_and_session_capacity_are_bounded(registry, monkeypatch):
    from rnsr.env import calculations

    a = fact(registry, 1)
    with pytest.raises(ValueError, match='wrong number'):
        registry.calculate('sum', [a['id']] * 33)
    monkeypatch.setattr(calculations, 'MAX_RECORDS', 1)
    with pytest.raises(ValueError, match='record limit'):
        fact(registry, 2)


def test_ambiguous_canonical_rows_are_not_certified(tmp_path):
    def parse(path):
        return ParsedDocument(
            doc_id='ambiguous', source_path=str(path), sha256='a'*64, n_pages=1, parser='test',
            elements=[Element('table','Item | Amount\nA | 10\nA | 10',1)],
            tables=[RawTable(page=1,header=['Item','Amount'],rows=[['A','10'],['A','10']])])
    path=tmp_path/'ambiguous.db'
    ingest([tmp_path/'input.pdf'],path,parse=parse)
    with CorpusDB(path) as corpus, pytest.raises(ValueError, match='exact canonical row'):
        CalculationRegistry(corpus.conn).source_number('t_ambiguous_001',1,'amount')


async def test_real_sandbox_final_calc_uses_parent_result(arithmetic_corpus):
    async with SandboxedRepl() as sandbox:
        await sandbox.start(mode='docdb',corpus_db=str(arithmetic_corpus))
        cell=await sandbox.exec_cell("""
a = source_number('t_report_001', 1, 'amount')
b = source_number('t_report_001', 2, 'amount')
result = calculate('subtract', [a['id'], b['id']])
FINAL_CALC(result['id'])
""")
        assert cell.ok and cell.final['value']=='-150.0'
        assert cell.final['verification']['check']=='source_bound_calculation'
        assert len(cell.final['verification']['records'])==3


async def test_real_sandbox_rejects_forged_final_id_and_value(arithmetic_corpus):
    async with SandboxedRepl() as sandbox:
        await sandbox.start(mode='docdb',corpus_db=str(arithmetic_corpus))
        first=await sandbox.exec_cell("""
a = source_number('t_report_001', 1, 'amount')
b = source_number('t_report_001', 2, 'amount')
result = calculate('subtract', [a['id'], b['id']])
""")
        assert first.ok
        forged=await sandbox.exec_cell("""
from rnsr.env.final_answer import FinalAnswer
raise FinalAnswer('150.0', True, {'check':'source_bound_calculation','calculation_id':result['id']})
""")
        assert not forged.ok and forged.final is None and 'differs' in forged.error
        unknown=await sandbox.exec_cell("""
raise FinalAnswer('150.0', True, {'check':'source_bound_calculation','calculation_id':'calc_forged'})
""")
        assert not unknown.ok and 'unknown calculation/source id' in unknown.error
        # A literal replacement remains possible through the opt-in legacy API,
        # but cannot inherit the preceding negative calculation's attestation.
        legacy=await sandbox.exec_cell("FINAL('150.0', quotes=['Revenue | 100'])")
        assert legacy.ok and legacy.final['value']=='150.0'
        assert legacy.final['verification']['check']=='lexical_source_match'
        assert 'calculation_id' not in legacy.final['verification']
        assert 'records' not in legacy.final['verification']


async def test_rpc_rejects_supplied_source_value(arithmetic_corpus):
    async with SandboxedRepl() as sandbox:
        await sandbox.start(mode='docdb',corpus_db=str(arithmetic_corpus))
        with pytest.raises(ValueError, match='values cannot be supplied'):
            await sandbox._calculation({'op':'calculation','action':'source',
                                       'table':'t_report_001','rowid':1,'column':'amount','value':999})
