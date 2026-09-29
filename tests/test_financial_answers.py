"""Financial definitions and final values stay bound to original evidence."""
from decimal import Decimal

import pytest

from rnsr.db.artifact import CorpusDB
from rnsr.env.calculations import CalculationRegistry
from rnsr.env.financial import render_calculations
from rnsr.env.sandbox import SandboxedRepl
from rnsr.ingest.model import Element, ParsedDocument, RawTable
from rnsr.ingest.pipeline import ingest


@pytest.fixture
def financial_corpus(tmp_path):
    rows = [['Current assets', '700'], ['Current liabilities', '400'],
            ['Operating current assets', '180'], ['Operating current liabilities', '220'],
            ['Cost of sales', '1000'], ['Beginning inventory', '80'], ['Ending inventory', '120'],
            ['Revenue', '900'], ['Gross profit', '90']]

    def parse(path):
        return ParsedDocument(
            doc_id='finance', source_path=str(path), sha256='f'*64, n_pages=1, parser='test',
            elements=[Element('heading', 'USD FY2025', 1, heading_level=1),
                      Element('text', 'Prior loss was (1,250). EUR FY2024. Payroll deleverage increased. Ratios 1.25 and -10; total 12,345.', 1),
                      Element('table', 'Metric | Amount\n' + '\n'.join(' | '.join(row) for row in rows), 1)],
            tables=[RawTable(page=1, header=['Metric', 'Amount'], rows=rows, extractor='test')])
    path = tmp_path / 'financial.db'
    ingest([tmp_path/'report.pdf'], path, parse=parse)
    return path


@pytest.fixture
def financial_registry(financial_corpus):
    with CorpusDB(financial_corpus) as db:
        yield CalculationRegistry(db.conn)


def spans(registry, unit='USD', period='FY2025'):
    text = registry.conn.execute('SELECT text FROM doc_text').fetchone()[0]
    return {key: {'char_start': text.index(value), 'char_end': text.index(value)+len(value)}
            for key, value in [('unit_span', unit), ('period_span', period)]}


def source(registry, row, **kwargs):
    return registry.source_number('t_finance_001', row, 'amount', **spans(registry, **kwargs))['id']


def test_total_and_operating_working_capital_are_distinct(financial_registry):
    registry = financial_registry
    total = registry.calculate_financial('working_capital', {
        'current_assets': source(registry, 1), 'current_liabilities': source(registry, 2)}, convention='total')
    operating = registry.calculate_financial('working_capital', {
        'operating_current_assets': source(registry, 3),
        'operating_current_liabilities': source(registry, 4)}, convention='operating')
    assert Decimal(total['value']) == 300 and Decimal(operating['value']) == -40
    answer, proof = registry.resolve_finals('Total: {total}; operating: {operating}.',
                                           {'total': total['id'], 'operating': operating['id']}, decimals=0)
    assert answer == 'Total: 300; operating: -40.'
    assert len(proof['calculation_answers']) == 2
    assert {r['financial_metric']['convention'] for r in proof['records'] if 'financial_metric' in r} == {
        'total', 'operating'}
    assert all(q['source_context']['page'] == 1 for q in proof['quotes'])


def test_inventory_alternatives_are_explicit_not_silently_selected(financial_registry):
    registry = financial_registry
    inputs = {'cost_of_sales': source(registry, 5), 'ending_inventory': source(registry, 7)}
    year_end = registry.calculate_financial('inventory_turnover', inputs, convention='year_end')
    average = registry.calculate_financial('inventory_turnover', {
        **inputs, 'beginning_inventory': source(registry, 6, period='FY2024')}, convention='average')
    assert float(year_end['value']) == pytest.approx(8.3333333333)
    assert average['value'] == '10'
    answer, proof = registry.resolve_finals('Year-end: {end}; average inventory: {average}.',
        {'end': year_end['id'], 'average': average['id']}, decimals=2)
    assert answer == 'Year-end: 8.33; average inventory: 10.00.'
    assert proof['calculation_answers']['end']['value'] == year_end['value']


def test_gross_margin_percent_keeps_inputs_and_formula(financial_registry):
    registry = financial_registry
    result = registry.calculate_financial('gross_margin', {
        'gross_profit': source(registry, 9), 'revenue': source(registry, 8)},
        convention='gross_profit_over_revenue')
    assert result['value'] == '10.0'
    _, proof = registry.resolve_final(result['id'], result['value'])
    assert {'divide', 'percent'} <= {r.get('operation') for r in proof['records']}


def test_financial_inputs_require_units_and_periods_and_explicit_roles(financial_registry):
    registry = financial_registry
    unbound = registry.source_number('t_finance_001', 1, 'amount')['id']
    liability = source(registry, 2)
    with pytest.raises(ValueError, match='unit and period'):
        registry.calculate_financial('working_capital', {
            'current_assets': unbound, 'current_liabilities': liability}, convention='total')
    with pytest.raises(ValueError, match='exactly these input roles'):
        registry.calculate_financial('working_capital', {'current_assets': source(registry, 1)},
                                     convention='total')
    with pytest.raises(ValueError, match='different units'):
        registry.calculate_financial('working_capital', {
            'current_assets': source(registry, 1, unit='EUR'),
            'current_liabilities': liability}, convention='total')
    with pytest.raises(ValueError, match='unknown financial metric'):
        registry.calculate_financial('working_capital', {}, convention='whatever_matches_the_reference')


def test_prose_number_is_bound_to_exact_original_span(financial_registry):
    registry = financial_registry
    text = registry.conn.execute('SELECT text FROM doc_text').fetchone()[0]
    start = text.index('(1,250)')
    record = registry.source_text_number('finance', start, start+7,
                                         **spans(registry, period='FY2024'))
    assert record['value'] == '-1250' and record['raw_value'] == '(1,250)'
    result = registry.calculate('sum', [record['id']])
    answer, proof = registry.resolve_final(result['id'], '-1250')
    assert answer == '-1250' and proof['quotes'][0]['quote'] == '(1,250)'
    assert 'Prior loss' in proof['quotes'][0]['source_context']['text']
    with pytest.raises(ValueError, match='exactly one numeric'):
        registry.source_text_number('finance', start-5, start+7)


@pytest.mark.parametrize('whole,part', [('(1,250)', '1,250'), ('1.25', '25'), ('1.25', '1'),
                                       ('-10', '10'), ('12,345', '12'), ('12,345', '345')])
def test_prose_offsets_cannot_strip_sign_or_magnitude(financial_registry, whole, part):
    registry = financial_registry
    text = registry.conn.execute('SELECT text FROM doc_text').fetchone()[0]
    start = text.index(whole) + whole.index(part)
    with pytest.raises(ValueError, match='complete numeric token'):
        registry.source_text_number('finance', start, start + len(part))


@pytest.mark.parametrize('template,values', [
    ('It is 999, not {value}.', {'value': '-40'}),
    ('It is {value.real}.', {'value': '-40'}),
    ('It is {value!r}.', {'value': '-40'}),
    ('It is {value:.2f}.', {'value': '-40'}),
    ('It is {unknown}.', {'value': '-40'}),
    ('It is {first}.', {'first': '10', 'second': '20'}),
    ('It is ² and {value}.', {'value': '-40'}),
])
def test_final_templates_cannot_introduce_replacement_numerals(template, values):
    with pytest.raises(ValueError):
        render_calculations(template, values)


def test_parent_rounding_never_changes_raw_result(financial_registry):
    registry = financial_registry
    result = registry.calculate('divide', [source(registry, 5), source(registry, 7)])
    answer, proof = registry.resolve_finals('Turnover: {ratio}.', {'ratio': result['id']}, decimals=1)
    assert answer == 'Turnover: 8.3.'
    assert proof['calculation_answers']['ratio']['value'] == result['value']


async def test_sandbox_financial_answer_and_forged_replacement(financial_corpus):
    async with SandboxedRepl() as sandbox:
        await sandbox.start(mode='docdb', corpus_db=str(financial_corpus))
        setup = await sandbox.exec_cell("""
text = doc['finance']
unit = {'char_start': text.index('USD'), 'char_end': text.index('USD')+3}
period = {'char_start': text.index('FY2025'), 'char_end': text.index('FY2025')+6}
a = source_number('t_finance_001', 3, 'amount', unit_span=unit, period_span=period)
b = source_number('t_finance_001', 4, 'amount', unit_span=unit, period_span=period)
r = calculate_financial('working_capital', {'operating_current_assets': a['id'],
    'operating_current_liabilities': b['id']}, convention='operating')
""")
        assert setup.ok, setup.error
        good = await sandbox.exec_cell("FINAL_CALCS('Operating working capital: {value}.', {'value': r['id']}, decimals=0)")
        assert good.ok and good.final['value'] == 'Operating working capital: -40.'
        bad = await sandbox.exec_cell("""
from rnsr.env.final_answer import FinalAnswer
raise FinalAnswer('Operating working capital: 40.', is_var=True,
    verification={'check': 'source_bound_calculations',
        'template': 'Operating working capital: {value}.', 'results': {'value': r['id']}, 'decimals': 0})
""")
        assert not bad.ok and bad.final is None
        assert 'differs from parent-rendered' in bad.error


def test_prompt_documents_enforced_proofs_and_preserves_negative_values():
    from rnsr.harness.prompts.base import _FINANCIAL_ADDON, render_system

    prompt = render_system('docdb')
    assert 'FINAL_CALCS' in prompt and '{prior}' in prompt
    assert 'semantic_classify' in prompt and 'expected_count' in prompt
    assert 'Preserve negative computed' in _FINANCIAL_ADDON
    assert 'do not answer NOT_FOUND solely' in _FINANCIAL_ADDON


@pytest.mark.parametrize('template,values', [
    ('-{x}', {'x': '2.5'}), ('+{x}', {'x': '2.5'}), ('−{x}', {'x': '2.5'}),
    ('- {x}', {'x': '2.5'}), ('--{x}', {'x': '-2.5'}), ('-${x}', {'x': '2.5'}),
    ('({x})', {'x': '2.5'}), ('( {x} )', {'x': '2.5'}), ('(${x})', {'x': '2.5'}),
    ('({x}%)', {'x': '2.5'}),
    ('{x}{y}', {'x': '2', 'y': '5'}), ('{x}{y}', {'x': '2', 'y': '-5'}),
    ('{x}.{y}', {'x': '2', 'y': '5'}), ('{x},{y}', {'x': '2', 'y': '500'}),
    ('{x}e{y}', {'x': '2', 'y': '5'}), ('{x}E-{y}', {'x': '2', 'y': '5'}),
    ('.{x}', {'x': '25'}),
])
def test_answer_templates_cannot_alter_or_join_bound_numeric_tokens(template, values):
    with pytest.raises(ValueError, match='numeric token'):
        render_calculations(template, values)


@pytest.mark.parametrize('template,values,expected', [
    ('Working capital: {x}.', {'x': '-2.5'}, 'Working capital: -2.5.'),
    ('Revenue ${x}; expense ${y}.', {'x': '250', 'y': '-5'}, 'Revenue $250; expense $-5.'),
    ('First {x}, second {y}.', {'x': '2', 'y': '500'}, 'First 2, second 500.'),
    ('{x}, {y}', {'x': '2', 'y': '500'}, '2, 500'),
    ('Margin: {x}%.', {'x': '2.5'}, 'Margin: 2.5%.'),
    ('(margin: {x}%)', {'x': '2.5'}, '(margin: 2.5%)'),
    ('- Ratio: {x}\n- Coverage: {y}', {'x': '2.5', 'y': '-3'}, '- Ratio: 2.5\n- Coverage: -3'),
    ('{x}; repeated {x}.', {'x': '-2.5'}, '-2.5; repeated -2.5.'),
    ('{{result}}: {x}.', {'x': '-2.5'}, '{result}: -2.5.'),
    ('{{{x}}}; {x}', {'x': '-2.5'}, '{-2.5}; -2.5'),
])
def test_template_token_guard_preserves_normal_prose_and_separators(template, values, expected):
    assert render_calculations(template, values) == expected


@pytest.mark.parametrize('template,two_values', [
    ('-{x}', False), ('({x})', False), ('{x}{y}', True), ('{x}e{y}', True),
])
def test_parent_final_resolution_rejects_numerical_reinterpretation(financial_registry, template, two_values):
    registry = financial_registry
    first = registry.calculate('sum', [source(registry, 9)])
    results = {'x': first['id']}
    if two_values:
        results['y'] = registry.calculate('sum', [source(registry, 7)])['id']
    with pytest.raises(ValueError, match='numeric token'):
        registry.resolve_finals(template, results)
    assert registry.get(first['id'])['value'] == '90'
    assert registry.resolve_finals('Gross profit: {x}.', {'x': first['id']})[0] == 'Gross profit: 90.'
