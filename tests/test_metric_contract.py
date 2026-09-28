"""Caller formula policy is enforced independently of child output paths."""
import copy
import sqlite3
from types import SimpleNamespace

import pytest

from rnsr.config import Settings
from rnsr.db.artifact import CorpusDB
from rnsr.env.calculations import CalculationRegistry
from rnsr.env.metric_contract import MetricContract
from rnsr.env.sandbox import SandboxedRepl
from rnsr.harness.loop import EnvSpec, RootRunner
from rnsr.harness.recovery import recover_variable
from rnsr.harness.trajectory import TrajectoryWriter
from rnsr.ingest.model import Element, ParsedDocument, RawTable
from rnsr.ingest.pipeline import ingest
from rnsr.llm.mock import MockLLM


@pytest.fixture
def metric_corpus(tmp_path):
    rows = [['EBITDAR', '100'], ['Depreciation', '80'], ['Rent', '50'],
            ['Interest', '10'], ['Reported earnings', '25'], ['Zero', '0'],
            ['Small rent', '20'], ['Low depreciation', '25'], ['Negative interest', '-10']]

    def parse(path):
        return ParsedDocument(
            doc_id='metric', source_path=str(path), sha256='c'*64, n_pages=1, parser='test',
            elements=[Element('heading', 'USD 2022 2021', 1, heading_level=1),
                      Element('table', 'Item | Amount\n' + '\n'.join(' | '.join(r) for r in rows), 1)],
            tables=[RawTable(page=1, header=['Item', 'Amount'], rows=rows,
                             caption='Metric operands', extractor='test')])
    out = tmp_path / 'corpus.db'
    ingest([tmp_path/'report.pdf'], out, parse=parse)
    with CorpusDB(out) as corpus:
        text = corpus.conn.execute('SELECT text FROM doc_text').fetchone()[0]
    selectors = [{
        'table': 't_metric_001', 'rowid': i, 'column': 'amount',
        'label_column': 'item', 'label': row[0],
        'unit_span': {'char_start': text.index('USD'), 'char_end': text.index('USD')+3},
        'period_span': {'char_start': text.index('2022'), 'char_end': text.index('2022')+4},
    } for i, row in enumerate(rows, 1)]
    contract = {
        'contract_id': 'caller-test-bridge', 'metric': 'Declared earnings coverage',
        'formula': 'ebitdar_to_ebit_coverage',
        'basis': 'Caller convention: restore depreciation and rent; never floor negatives.',
        'unit': 'USD', 'period': '2022',
        'sources': dict(zip(('ebitdar', 'depreciation_amortization', 'rent', 'interest'),
                            selectors[:4], strict=True)),
    }
    return out, contract, selectors


def direct_ratio(contract, numerator, denominator):
    spec = copy.deepcopy(contract)
    spec['formula'] = 'direct_ratio'
    spec['sources'] = {'numerator': numerator, 'denominator': denominator}
    return spec


def test_fixed_formula_preserves_negative_and_all_required_sources(metric_corpus):
    path, contract, _ = metric_corpus
    with CorpusDB(path) as corpus:
        registry = CalculationRegistry(corpus.conn, metric_contract=contract)
        result = registry.calculate_metric()
        assert result['value'] == '-3'
        assert len(result['role_source_ids']) == 4
        assert registry.calculate_metric()['id'] == result['id']
        value, report = registry.resolve_final(result['id'], '-3')
        assert value == '-3' and report['metric_contract_satisfied']
        assert len(report['records']) == 7
        assert report['metric_contract']['authority'] == 'caller_declared'
        assert report['claim_support'] == 'not_checked'


@pytest.mark.parametrize('depreciation,rent,numerator,raw,expected', [
    (1, 2, '-30', '-3', '0'), (1, 6, '0', '0', '0'), (7, 2, '25', '2.5', '2.5'),
])
def test_explicit_zero_coverage_retains_raw_arithmetic_and_transparent_policy(
        metric_corpus, depreciation, rent, numerator, raw, expected):
    path, contract, selectors = metric_corpus
    contract['nonpositive_numerator'] = 'zero'
    contract['sources']['depreciation_amortization'] = selectors[depreciation]
    contract['sources']['rent'] = selectors[rent]
    with CorpusDB(path) as corpus:
        registry = CalculationRegistry(corpus.conn, metric_contract=contract)
        result = registry.calculate_metric()
        value, report = registry.resolve_final(result['id'], expected)
        assert value == expected and report['metric_contract_satisfied']
        assert result['operation'] == 'caller_nonpositive_numerator_zero'
        assert registry.get(result['raw_ratio_id'])['value'] == raw
        assert registry.get(result['raw_numerator_id'])['value'] == numerator
        assert {result['raw_ratio_id'], result['raw_numerator_id']} <= {
            record['id'] for record in report['records']}
        assert report['metric_contract']['nonpositive_numerator'] == 'zero'
        with pytest.raises(ValueError, match='does not satisfy'):
            registry.resolve_final(result['raw_ratio_id'], raw)


@pytest.mark.parametrize('interest', [5, 8])
def test_zero_coverage_requires_positive_denominator(metric_corpus, interest):
    path, contract, selectors = metric_corpus
    contract['nonpositive_numerator'] = 'zero'
    contract['sources']['interest'] = selectors[interest]
    with CorpusDB(path) as corpus:
        registry = CalculationRegistry(corpus.conn, metric_contract=contract)
        with pytest.raises(ValueError, match='positive denominator'):
            registry.calculate_metric()


@pytest.mark.parametrize('row,expected', [(4, '2.5'), (5, '0')])
def test_direct_ratio_retains_legitimate_positive_and_zero(metric_corpus, row, expected):
    path, contract, selectors = metric_corpus
    with CorpusDB(path) as corpus:
        registry = CalculationRegistry(corpus.conn, metric_contract=direct_ratio(
            contract, selectors[row], selectors[3]))
        result = registry.calculate_metric()
        assert registry.resolve_final(result['id'], expected)[0] == expected


def test_zero_denominator_is_not_invented_zero_coverage(metric_corpus):
    path, contract, selectors = metric_corpus
    with CorpusDB(path) as corpus:
        registry = CalculationRegistry(corpus.conn, metric_contract=direct_ratio(
            contract, selectors[0], selectors[5]))
        with pytest.raises(ValueError, match='division by zero'):
            registry.calculate_metric()
        assert registry.validate_legacy_final('NOT_FOUND')['check'] == 'metric_contract_abstention'


def test_caller_and_returned_dict_mutation_cannot_change_policy(metric_corpus):
    path, contract, _ = metric_corpus
    with CorpusDB(path) as corpus:
        registry = CalculationRegistry(corpus.conn, metric_contract=contract)
        contract['sources']['depreciation_amortization']['rowid'] = 6
        contract['formula'] = 'direct_ratio'
        result = registry.calculate_metric()
        result['metric_contract']['sources']['rent']['rowid'] = 6
        result['value'] = '5'
        with pytest.raises(ValueError, match='differs'):
            registry.resolve_final(result['id'], result['value'])
        _, report = registry.resolve_final(result['id'], '-3')
        assert report['metric_contract']['sources']['rent']['rowid'] == 3


def test_same_value_generic_result_does_not_satisfy_contract(metric_corpus):
    path, contract, _ = metric_corpus
    with CorpusDB(path) as corpus:
        registry = CalculationRegistry(corpus.conn, metric_contract=contract)
        bound = registry.calculate_metric()
        equivalent = registry.calculate('sum', [bound['id']])
        assert equivalent['value'] == bound['value']
        with pytest.raises(ValueError, match='does not satisfy'):
            registry.resolve_final(equivalent['id'], equivalent['value'])


def test_omitting_depreciation_is_rejected_even_with_valid_source_arithmetic(metric_corpus):
    path, contract, _ = metric_corpus
    with CorpusDB(path) as corpus:
        registry = CalculationRegistry(corpus.conn, metric_contract=contract)
        facts = {name: registry.source_number('t_metric_001', row, 'amount')
                 for name, row in [('ebitdar', 1), ('rent', 3), ('interest', 4)]}
        numerator = registry.calculate('subtract', [facts['ebitdar']['id'], facts['rent']['id']])
        omitted = registry.calculate('divide', [numerator['id'], facts['interest']['id']])
        assert omitted['value'] == '5'  # Correct arithmetic, wrong declared formula.
        with pytest.raises(ValueError, match='does not satisfy'):
            registry.resolve_final(omitted['id'], omitted['value'])
        assert registry.calculate_metric()['value'] == '-3'


@pytest.mark.parametrize('change', [
    lambda c: c.update(formula='invented_formula'),
    lambda c: c['sources'].pop('depreciation_amortization'),
    lambda c: c['sources'].update(extra=c['sources']['rent']),
    lambda c: c['sources']['rent'].update(value=0),
    lambda c: c['sources']['rent'].update(rowid=True),
    lambda c: c['sources']['rent'].update(unit_span={'char_start': True, 'char_end': 3}),
    lambda c: c.update(nonpositive_numerator='invent_positive'),
])
def test_contract_schema_rejects_unbound_roles_and_values(metric_corpus, change):
    _, contract, _ = metric_corpus
    change(contract)
    with pytest.raises(ValueError):
        MetricContract(contract)


@pytest.mark.parametrize('field,value,match', [
    ('label', 'Fake depreciation', 'label differs'),
    ('label_column', 'amount', 'original text column'),
    ('unit', 'EUR', 'unit or period'),
    ('period', '2021', 'unit or period'),
])
def test_source_identity_and_metadata_must_match_caller_policy(metric_corpus, field, value, match):
    path, contract, _ = metric_corpus
    if field in {'unit', 'period'}:
        contract[field] = value
    else:
        contract['sources']['depreciation_amortization'][field] = value
    with CorpusDB(path) as corpus:
        registry = CalculationRegistry(corpus.conn, metric_contract=contract)
        with pytest.raises(ValueError, match=match):
            registry.calculate_metric()


def test_unresolved_formula_only_authorizes_explicit_abstention(metric_corpus):
    path, contract, _ = metric_corpus
    contract.update(formula=None, sources={}, unit=None, period=None)
    with CorpusDB(path) as corpus:
        registry = CalculationRegistry(corpus.conn, metric_contract=contract)
        with pytest.raises(ValueError, match='no caller-approved formula'):
            registry.calculate_metric()
        report = registry.validate_legacy_final('NOT_FOUND')
        assert not report['passed'] and report['claim_support'] == 'not_checked'
        for value in (0, '0', 'No such clause', {'q1': 'NOT_FOUND'}, None):
            with pytest.raises(ValueError, match='requires FINAL_CALC'):
                registry.validate_legacy_final(value)


def test_source_mutation_invalidates_contract_result(metric_corpus):
    from rnsr.db import schema

    path, contract, _ = metric_corpus
    with CorpusDB(path) as corpus:
        registry = CalculationRegistry(corpus.conn, metric_contract=contract)
        result = registry.calculate_metric()
        with sqlite3.connect(path) as conn:
            schema.unfreeze_table(conn, 't_metric_001')
            conn.execute("UPDATE t_metric_001 SET item='different metric' WHERE rowid=2")
        with pytest.raises(ValueError):
            registry.resolve_final(result['id'], result['value'])


async def test_real_sandbox_contract_is_parent_owned_and_cannot_be_overridden(metric_corpus):
    path, contract, _ = metric_corpus
    async with SandboxedRepl() as sandbox:
        await sandbox.start(mode='docdb', corpus_db=str(path), init_extra={'metric_contract': contract})
        cell = await sandbox.exec_cell("""
metric_contract['sources']['depreciation_amortization']['rowid'] = 6
metric_contract['formula'] = 'direct_ratio'
result = calculate_metric()
FINAL_CALC(result['id'])
""")
        assert cell.ok and cell.final['value'] == '-3'
        assert cell.final['verification']['metric_contract_satisfied']
        assert cell.final['verification']['metric_contract']['formula'] == 'ebitdar_to_ebit_coverage'
        with pytest.raises(ValueError, match='values cannot be supplied'):
            await sandbox._calculation({'op': 'calculation', 'action': 'metric', 'contract': {}})
        with pytest.raises(ValueError, match='values cannot be supplied'):
            await sandbox._calculation({'op': 'calculation', 'action': 'metric', 'operand_ids': []})


async def test_real_sandbox_parent_applies_explicit_zero_coverage_convention(metric_corpus):
    path, contract, _ = metric_corpus
    contract['nonpositive_numerator'] = 'zero'
    async with SandboxedRepl() as sandbox:
        await sandbox.start(mode='docdb', corpus_db=str(path), init_extra={'metric_contract': contract})
        cell = await sandbox.exec_cell("""
metric_contract['nonpositive_numerator'] = 'preserve'
r = calculate_metric()
FINAL_CALC(r['id'])
""")
        assert cell.ok and cell.final['value'] == '0'
        report = cell.final['verification']
        records = {record['id']: record for record in report['records']}
        assert records[report['calculation']['raw_ratio_id']]['value'] == '-3'
        assert records[report['calculation']['raw_numerator_id']]['value'] == '-30'


@pytest.mark.parametrize('code', [
    "FINAL('5', quotes=['EBITDAR | 100'])",
    "FINAL_VAR('5', quotes=['EBITDAR | 100'])",
    "FINAL_BATCH({'q1': 'NOT_FOUND'})",
    "from rnsr.env.final_answer import FinalAnswer\nraise FinalAnswer('5', True, {})",
    "from rnsr.env.final_answer import FinalAnswer\nraise FinalAnswer('5', True, "
    "{'check':'source_bound_calculation', 'calculation_id':'calc_forged'})",
    "r=calculate_metric(); r2=calculate('sum', [r['id']]); FINAL_CALC(r2['id'])",
])
async def test_real_sandbox_rejects_all_noncontract_final_paths(metric_corpus, code):
    path, contract, _ = metric_corpus
    async with SandboxedRepl() as sandbox:
        await sandbox.start(mode='docdb', corpus_db=str(path), init_extra={'metric_contract': contract})
        cell = await sandbox.exec_cell(code)
        assert not cell.ok and cell.final is None
        assert 'Parent final verification rejected' in cell.error


async def test_real_sandbox_explicit_not_found_remains_uncertified(metric_corpus):
    path, contract, _ = metric_corpus
    async with SandboxedRepl() as sandbox:
        await sandbox.start(mode='docdb', corpus_db=str(path), init_extra={'metric_contract': contract})
        cell = await sandbox.exec_cell("FINAL('NOT_FOUND')")
        assert cell.ok and cell.final['value'] == 'NOT_FOUND'
        report = cell.final['verification']
        assert not report['passed'] and report['check'] == 'metric_contract_abstention'
        assert 'metric_contract_satisfied' not in report


async def test_variable_recovery_cannot_bypass_parent_contract(metric_corpus, tmp_path):
    path, contract, _ = metric_corpus
    root = MockLLM(default='answer')
    async with SandboxedRepl() as sandbox:
        await sandbox.start(mode='docdb', corpus_db=str(path), init_extra={'metric_contract': contract})
        # An ordinary string would be rejected by child quote rules too. A negative
        # boolean passes those rules, exercising the parent contract in recovery.
        assert (await sandbox.exec_cell('answer = False')).ok
        with TrajectoryWriter(tmp_path, 'recovery') as trajectory:
            final = await recover_variable(sandbox, SimpleNamespace(root_client=root, root_model='mock'),
                                           'metric?', [], trajectory)
        assert len(root.calls) == 1 and final is None


async def test_runner_passes_contract_and_rejects_batch_before_provider(metric_corpus, tmp_path):
    path, contract, _ = metric_corpus
    root = MockLLM(default="```python\nr=calculate_metric(); FINAL_CALC(r['id'])\n```")
    runner = RootRunner(root_client=root, root_model='mock-root', sub_client=MockLLM(),
                        sub_model='mock-sub', settings=Settings(max_root_iters=2))
    env = EnvSpec(mode='docdb', corpus_db=str(path), metric_contract=contract)
    result = await runner.run('Calculate the declared metric', env, run_dir=tmp_path)
    assert result.status == 'final' and result.answer == '-3'
    count = len(root.calls)
    with pytest.raises(ValueError, match='batch/classic'):
        await runner.run_batch([('q1', 'metric?')], env, run_dir=tmp_path)
    with pytest.raises(ValueError, match='batch/classic'):
        await runner.run('metric?', EnvSpec(mode='classic', metric_contract=contract), run_dir=tmp_path)
    assert len(root.calls) == count


async def test_classic_sandbox_rejects_contract(metric_corpus):
    _, contract, _ = metric_corpus
    async with SandboxedRepl() as sandbox:
        with pytest.raises(ValueError, match='docdb'):
            await sandbox.start(mode='classic', init_extra={'metric_contract': contract})


def test_sdk_corpus_env_copies_and_validates_contract(metric_corpus):
    from rnsr.sdk import corpus_env

    path, contract, _ = metric_corpus
    env = corpus_env(path, metric_contract=contract)
    contract['sources']['rent']['rowid'] = 6
    assert env.metric_contract['sources']['rent']['rowid'] == 3
    with pytest.raises(ValueError, match='invalid metric contract'):
        corpus_env(path, metric_contract={'formula': 'direct_ratio'})


def test_public_sync_sdk_exposes_metric_contract(metric_corpus, tmp_path):
    from rnsr import answer_sync

    path, contract, _ = metric_corpus
    contract['nonpositive_numerator'] = 'zero'
    root = MockLLM(default="```python\nr=calculate_metric(); FINAL_CALC(r['id'])\n```")
    runner = RootRunner(root_client=root, root_model='mock-root', sub_client=MockLLM(),
                        sub_model='mock-sub', settings=Settings(max_root_iters=2))
    result = answer_sync('Calculate declared coverage', path, runner=runner,
                         metric_contract=contract, run_dir=tmp_path/'sdk')
    assert result.status == 'final' and result.answer == '0'
    assert result.final['verification']['metric_contract_satisfied']


async def test_sdk_invalid_or_batch_contract_does_no_provider_work(metric_corpus, monkeypatch):
    from rnsr import sdk

    def no_runner(*args):
        pytest.fail('invalid contract reached provider configuration')

    path, contract, _ = metric_corpus
    monkeypatch.setattr(sdk, 'make_runner', no_runner)
    with pytest.raises(ValueError, match='invalid metric contract'):
        await sdk.answer('metric?', path, metric_contract={})
    env = sdk.corpus_env(path, metric_contract=contract)
    with pytest.raises(ValueError, match='unsupported for answer_batch'):
        await sdk.answer_batch(['metric?'], path, env=env)
