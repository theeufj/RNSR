"""Named calculation conventions and parent-rendered numerical answers.

These conventions describe the arithmetic, not which convention a question
intends. Source scope and economic meaning still require evidence review.
"""
from __future__ import annotations

import re
import string
from decimal import ROUND_HALF_EVEN, Decimal, localcontext

_NUMBER_BODY = r'(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)'
_SIGNED_NUMBER = (r'(?:[+\-−﹣＋－]\s*(?:[$€£¥]\s*)?)*' + _NUMBER_BODY
                  + r'(?:[eE]\s*[+\-−]?\s*\d+)?')
_NUMBER_TOKEN = re.compile(r'\(\s*[$€£¥]?\s*' + _SIGNED_NUMBER + r'\s*%?\s*\)|' + _SIGNED_NUMBER)


def financial_metric(registry, metric: str, inputs: dict[str, str], *, convention: str):
    specifications = {
        ('working_capital', 'total'): ('current_assets', 'current_liabilities'),
        ('working_capital', 'operating'): ('operating_current_assets', 'operating_current_liabilities'),
        ('quick_ratio', 'liquid_assets'): ('cash', 'short_term_investments', 'receivables', 'current_liabilities'),
        ('quick_ratio', 'cash_receivables'): ('cash', 'receivables', 'current_liabilities'),
        ('gross_margin', 'gross_profit_over_revenue'): ('gross_profit', 'revenue'),
        ('inventory_turnover', 'year_end'): ('cost_of_sales', 'ending_inventory'),
        ('inventory_turnover', 'average'): ('cost_of_sales', 'beginning_inventory', 'ending_inventory'),
        ('effective_tax_rate', 'tax_over_pretax_income'): ('income_tax_expense', 'pretax_income'),
    }
    if not isinstance(metric, str) or not isinstance(convention, str):
        raise ValueError('metric and convention must be explicit names')
    roles = specifications.get((metric, convention))
    if roles is None:
        raise ValueError('unknown financial metric/convention; choose an explicit supported basis')
    if not isinstance(inputs, dict) or set(inputs) != set(roles):
        raise ValueError(f'{metric}/{convention} requires exactly these input roles: {roles}')
    records = {role: registry.get(inputs[role]) for role in roles}
    facts = {}
    for record in records.values():
        for source_id in ([record['id']] if record['kind'] == 'source' else record['source_ids']):
            facts[source_id] = registry.get(source_id)
    if any(not fact.get('unit') or not fact.get('period') for fact in facts.values()):
        raise ValueError('financial inputs require source-bound unit and period spans')
    units = {' '.join(fact['unit'].casefold().split()) for fact in facts.values()}
    if len(units) != 1:
        raise ValueError('financial inputs have different units; resolve scaling before calculating')

    def calc(op, ids):
        return registry.calculate(op, ids)['id']

    if metric == 'working_capital':
        result_id = calc('subtract', [inputs[role] for role in roles])
    elif metric == 'quick_ratio':
        numerator = calc('sum', [inputs[role] for role in roles[:-1]])
        result_id = calc('divide', [numerator, inputs['current_liabilities']])
    elif metric == 'inventory_turnover':
        denominator = (inputs['ending_inventory'] if convention == 'year_end' else
                       calc('mean', [inputs['beginning_inventory'], inputs['ending_inventory']]))
        result_id = calc('divide', [inputs['cost_of_sales'], denominator])
    else:
        ratio = calc('divide', [inputs[role] for role in roles])
        result_id = calc('percent', [ratio])
    # This is parent-generated metadata over parent-owned inputs, never a
    # caller's asserted answer or a benchmark reference value.
    registry._records[result_id]['financial_metric'] = {
        'metric': metric, 'convention': convention, 'input_roles': dict(inputs),
        'scope': 'arithmetic verified; source scope and convention require claim review',
    }
    return registry.get(result_id)


def render_calculations(template: str, values: dict[str, str], *, decimals: int | None = None):
    """Only placeholder values introduce numerals; no eval/attribute formatting."""
    if not isinstance(template, str) or not template.strip() or len(template) > 8000:
        raise ValueError('calculation answer requires a nonempty bounded template')
    if not isinstance(values, dict) or not values or len(values) > 16:
        raise ValueError('calculation answer requires 1-16 named results')
    if any(not isinstance(key, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', key)
           for key in values):
        raise ValueError('calculation placeholder names must be identifiers')
    if decimals is not None and (type(decimals) is not int or not 0 <= decimals <= 12):
        raise ValueError('decimals must be an integer from 0 to 12 or None')
    pieces, used, value_spans = [], set(), []
    length = 0
    for literal, field, fmt, conversion in string.Formatter().parse(template):
        if any(char.isnumeric() for char in literal):
            raise ValueError('numeric answer literals are not allowed; use bound result placeholders')
        pieces.append(literal)
        length += len(literal)
        if field is None:
            continue
        if field not in values or fmt or conversion:
            raise ValueError('use only named result placeholders, without formatting or attribute access')
        value = values[field]
        if decimals is not None:
            with localcontext() as ctx:
                ctx.prec = max(64, len(value) + decimals + 4)
                value = format(Decimal(value).quantize(Decimal(1).scaleb(-decimals),
                                                     rounding=ROUND_HALF_EVEN), 'f')
        pieces.append(value)
        if value_spans and value_spans[-1][1] == length:
            raise ValueError('calculation placeholders require a separator; numeric token joins are not allowed')
        value_spans.append((length, length + len(value)))
        length += len(value)
        used.add(field)
    if used != set(values):
        raise ValueError('every supplied calculation must appear in the answer template')
    rendered = ''.join(pieces)
    # Source-bound numerals must stay complete numeric tokens. Merely forbidding
    # literal digits still permits -{x}, {x}{y}, {x}.{y}, {x}e{y}, or accounting
    # parentheses to change their value/sign after the arithmetic was verified.
    tokens = [(match.start(), match.end()) for match in _NUMBER_TOKEN.finditer(rendered)]
    if tokens != value_spans:
        raise ValueError('template changes a bound numeric token; signs, joins and accounting wrappers are not allowed')
    return rendered
