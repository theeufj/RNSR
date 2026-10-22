"""Opt-in, parent-owned arithmetic over immutable extracted source cells.

This proves which retained values entered an operation, not that the chosen
cells, formula, or financial convention answer the question. Unit/period
spans prove occurrence only; their interpretation remains a separate check.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import re
import secrets
import sqlite3
import time
from decimal import ROUND_HALF_EVEN, Decimal, DecimalException, localcontext

from rnsr.db.metadata import decode_table_schema
from rnsr.db.schema import quote_ident
from rnsr.env.evidence import SourceContext

# Session limits. Operands, sources and records bound the parent's work per
# untrusted request; 0 means unlimited (the research default, see Settings).
# A sum over N invoices is one certificate over N rows, not N round-trips, so
# the aggregate path does not need an operand cap at all.
MAX_RECORDS = 0
MAX_OPERANDS = 0
MAX_SOURCES = 0
MAX_ROW_CHARS = 20_000
PRECISION = 50
AGGREGATE_QUOTE_SAMPLE = 3        # quotes forwarded per aggregate; the certificate carries every row
MAX_FINAL_QUOTES = 32             # the reviewer's evidence limit for calculation proofs
AGGREGATE_OPERATIONS = ('sum', 'count', 'mean', 'min', 'max')


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _decimal(value) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("calculation source must be a finite numeric cell")
    if len(str(value)) > 128:
        raise ValueError("calculation value exceeds the numeric size limit")
    try:
        number = Decimal(str(value))
    except DecimalException as exc:
        raise ValueError("calculation source is not numeric") from exc
    if not number.is_finite() or (number and abs(number.adjusted()) > 1000):
        raise ValueError("calculation source must be finite and bounded")
    return number


def _render(value: Decimal) -> str:
    if not value.is_finite() or (value and abs(value.adjusted()) > 1000):
        raise ValueError("calculation result exceeds the numeric size limit")
    return format(value, 'f')


PRESENT_DECIMALS = 4


def present(value: str) -> str:
    """The answer form of a certified value: at most four decimal places,
    trailing zeros dropped. The record keeps the full-precision value; a
    50-digit quotient is evidence, not an answer (seen live on FinanceBench)."""
    number = Decimal(value)
    if number.as_tuple().exponent >= -PRESENT_DECIMALS:
        return value                      # already a presentable rendering
    with localcontext() as ctx:
        ctx.prec = PRECISION
        rounded = number.quantize(Decimal(1).scaleb(-PRESENT_DECIMALS), rounding=ROUND_HALF_EVEN)
    text = format(rounded, 'f')
    return text.rstrip('0').rstrip('.') if '.' in text else text


class CalculationRegistry:
    """Session-local facts/results. Never instantiate this in the child.

    Values come from typed numeric columns in the original artifact, not
    from RPC arguments. Decimal arithmetic avoids further binary-float
    rounding; precision lost during ingestion cannot be reconstructed here.
    """

    def __init__(self, conn: sqlite3.Connection, *, metric_contract: dict | None = None,
                 max_records: int = MAX_RECORDS, max_operands: int = MAX_OPERANDS,
                 max_sources: int = MAX_SOURCES):
        self.conn = conn
        self.max_records, self.max_operands, self.max_sources = max_records, max_operands, max_sources
        self._deadline: float | None = None
        self._context = SourceContext(conn, check=self._check_deadline)
        self._records: dict[str, dict] = {}
        self._generation = self._read_generation()
        self._generation_by_id = {row[0]: row for row in self._generation}
        from rnsr.env.metric_contract import MetricContract

        self._metric_contract = MetricContract(metric_contract) if metric_contract is not None else None
        self._metric_result_id: str | None = None

    def _check_deadline(self):
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise TimeoutError("source calculation exceeded cell wall-clock deadline")

    @contextlib.contextmanager
    def deadline(self, deadline: float | None):
        """Scope the parent-owned connection's SQL and Python work to one cell.

        The sandbox owns this connection and its progress handler. Restore the
        enclosing scope before its transaction is rolled back, including after
        SQLite interruption. A nested scope can never extend an outer deadline.
        """
        previous = self._deadline
        self._deadline = (deadline if previous is None else previous if deadline is None
                          else min(previous, deadline))

        def install():
            self.conn.set_progress_handler(
                (lambda: int(time.monotonic() >= self._deadline))
                if self._deadline is not None else None, 1000)

        install()
        try:
            self._check_deadline()
            yield
            self._check_deadline()
        except Exception:
            # Deadline-triggered SQLite interruption is reported as a timeout;
            # other failures retain their original type while time remains.
            self._check_deadline()
            raise
        finally:
            self._deadline = previous
            install()

    def _read_generation(self):
        rows = []
        for row in self.conn.execute(
                'SELECT doc_id,sha256,content_sha256,ingested_at FROM documents ORDER BY doc_id'):
            self._check_deadline()
            rows.append(tuple(row))
        self._check_deadline()
        return tuple(rows)

    def _check_generation(self):
        if self._read_generation() != self._generation:
            raise ValueError("calculation source changed; restart the sandbox session")

    def _lookup(self, record_id: str) -> dict:
        self._check_deadline()
        if not isinstance(record_id, str) or record_id not in self._records:
            raise ValueError("unknown calculation/source id for this session")
        return self._records[record_id]

    def _store(self, record: dict, prefix: str) -> dict:
        self._check_deadline()
        if self.max_records and len(self._records) >= self.max_records:
            raise ValueError("calculation session record limit exceeded")
        record['id'] = prefix + secrets.token_hex(16)
        self._records[record['id']] = copy.deepcopy(record)
        result = copy.deepcopy(record)
        self._check_deadline()
        return result

    def _read_cell(self, table, rowid, column) -> dict:
        self._check_deadline()
        if (not isinstance(table, str) or not isinstance(column, str)
                or isinstance(rowid, bool) or not isinstance(rowid, int) or rowid < 1):
            raise ValueError("source_number requires table, positive rowid and column")
        meta = self.conn.execute(
            'SELECT doc_id,title,schema_json FROM manifest_tables WHERE table_name=?',
            (table,)).fetchone()
        if meta is None:
            raise ValueError("calculation sources require a registered table")
        schema = decode_table_schema(meta[2])
        col = next((c for c in schema.columns if c.name == column), None)
        if col is None or col.annotation or col.type not in {'INTEGER', 'REAL', 'NUMERIC'}:
            raise ValueError("calculation source must be an original numeric column")
        cur = self.conn.execute(f'SELECT * FROM {quote_ident(table)} WHERE rowid=?', (rowid,))
        row = cur.fetchone()
        if row is None:
            raise ValueError("calculation source row does not exist")
        values = dict(zip((c[0] for c in cur.description), row, strict=True))
        number = _decimal(values[column])
        context = self._context(table=table, rowid=rowid)
        if context['row_location'] != 'exact':
            raise ValueError("calculation source needs one exact canonical row location")
        start, end = context['char_start'], context['char_end']
        if end - start > MAX_ROW_CHARS:
            raise ValueError("calculation source row exceeds the evidence limit")
        quote = self._span(meta[0], start, end)['text']
        generation = self._generation_by_id[meta[0]]
        self._check_deadline()
        return {'value': _render(number), 'raw_value': values.get(col.raw_col or column),
                'source': {'doc_id': meta[0], 'table': table, 'rowid': rowid,
                           'column': column, 'page': context['page'],
                           'char_start': start, 'char_end': end,
                           'source_span_id': _digest([generation, start, end]),
                           'generation_sha256': _digest(generation),
                           'table_title': meta[1], 'heading_paths': context['heading_paths'],
                           'extractor': values.get('_extractor'),
                           'coercion_rule': col.coercion_rule}, 'quote': quote}

    def _span(self, doc_id, start, end) -> dict:
        self._check_deadline()
        if (not isinstance(doc_id, str) or isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, int) or not isinstance(end, int)
                or not 0 <= start < end or end - start > MAX_ROW_CHARS):
            raise ValueError("invalid calculation source span")
        rows = []
        for row in self.conn.execute(
            'SELECT char_start,char_end,text FROM doc_text '
            'WHERE doc_id=? AND char_start<? AND char_end>? ORDER BY page',
                (doc_id, end, start)):
            self._check_deadline()
            rows.append(row)
        if not rows or rows[0][0] > start or rows[-1][1] < end:
            raise ValueError("calculation source span is outside retained text")
        text = ''.join(r[2] for r in rows)[start-rows[0][0]:end-rows[0][0]]
        if len(text) != end - start:
            raise ValueError("calculation source span crosses a retained-text gap")
        self._check_deadline()
        return {'doc_id': doc_id, 'char_start': start, 'char_end': end, 'text': text,
                'source_span_id': _digest([self._generation, doc_id, start, end, text])}

    def _metadata_span(self, span, doc_id):
        if span is None:
            return None
        if not isinstance(span, dict) or set(span) != {'char_start', 'char_end'}:
            raise ValueError("unit/period span requires char_start and char_end")
        if (not isinstance(span['char_start'], int) or not isinstance(span['char_end'], int)
                or span['char_end'] - span['char_start'] > 256):
            raise ValueError("unit/period span exceeds 256 characters")
        return self._span(doc_id, span['char_start'], span['char_end'])

    def source_number(self, table: str, rowid: int, column: str, *,
                      unit_span: dict | None = None, period_span: dict | None = None) -> dict:
        self._check_generation()
        cell = self._read_cell(table, rowid, column)
        doc_id = cell['source']['doc_id']
        unit = self._metadata_span(unit_span, doc_id)
        period = self._metadata_span(period_span, doc_id)
        return self._store({'kind': 'source', **cell,
                            'unit': unit['text'] if unit else None,
                            'period': period['text'] if period else None,
                            'unit_source': unit, 'period_source': period,
                            'metadata_check': 'source_text_only; scope not semantically verified'}, 'src_')

    def _validate_sources(self, source_ids):
        self._check_generation()
        for source_id in source_ids:
            fact = self._lookup(source_id)
            if fact['kind'] == 'aggregate':
                self._validate_rows(fact)
                continue
            source = fact['source']
            current = (self._read_text_number(source['doc_id'], source['char_start'], source['char_end'])
                       if source.get('kind') == 'text' else
                       self._read_cell(source['table'], source['rowid'], source['column']))
            if any(current[key] != fact[key] for key in ('value', 'raw_value', 'source', 'quote')):
                raise ValueError("calculation source value or evidence changed")
            for field in ('unit_source', 'period_source'):
                span = fact[field]
                if span and self._span(span['doc_id'], span['char_start'], span['char_end']) != span:
                    raise ValueError("calculation metadata source changed")
        self._check_deadline()

    def calculate(self, operation: str, operand_ids: list[str]) -> dict:
        self._check_deadline()
        many = self.max_operands or float('inf')
        arities = {'sum': (1, many), 'mean': (1, many), 'subtract': (2, 2),
                   'multiply': (2, 2), 'divide': (2, 2), 'percent': (1, 1)}
        if not isinstance(operation, str) or operation not in arities:
            raise ValueError("operation must be sum, mean, subtract, multiply, divide or percent")
        lo, hi = arities[operation]
        if not isinstance(operand_ids, list) or not lo <= len(operand_ids) <= hi:
            raise ValueError("wrong number of calculation operands")
        operands = [self._lookup(key) for key in operand_ids]
        sources = list(dict.fromkeys(source for record in operands
                                    for source in ([record['id']] if record['kind'] in ('source', 'aggregate')
                                                   else record['source_ids'])))
        if self.max_sources and len(sources) > self.max_sources:
            raise ValueError("calculation exceeds the source evidence limit")
        self._validate_sources(sources)
        values = [_decimal(r['value']) for r in operands]
        try:
            with localcontext() as ctx:
                ctx.prec = PRECISION
                ctx.rounding = ROUND_HALF_EVEN
                if operation == 'sum':
                    value = sum(values, Decimal(0))
                elif operation == 'mean':
                    value = sum(values, Decimal(0)) / len(values)
                elif operation == 'subtract':
                    value = values[0] - values[1]
                elif operation == 'multiply':
                    value = values[0] * values[1]
                elif operation == 'divide':
                    value = values[0] / values[1]
                else:
                    value = values[0] * Decimal(100)
                rendered = _render(value)
        except DecimalException as exc:
            raise ValueError("invalid decimal calculation (including division by zero)") from exc
        return self._store({'kind': 'calculation', 'value': rendered,
                            'operation': operation, 'operand_ids': list(operand_ids),
                            'source_ids': sources, 'precision': PRECISION,
                            'rounding': 'ROUND_HALF_EVEN',
                            'unit': 'percent' if operation == 'percent' else None,
                            'period': None,
                            'metadata_check': 'operand units/periods retained; compatibility not certified'}, 'calc_')

    # --- aggregate certificate ----------------------------------------------

    def _row_fact(self, row) -> dict:
        if isinstance(row, (list, tuple)) and len(row) == 3:
            row = {'table': row[0], 'rowid': row[1], 'column': row[2]}
        if not isinstance(row, dict) or set(row) != {'table', 'rowid', 'column'}:
            raise ValueError("aggregate rows are {table, rowid, column} triples")
        cell = self._read_cell(row['table'], row['rowid'], row['column'])
        src = cell['source']
        return {'table': src['table'], 'rowid': src['rowid'], 'column': src['column'],
                'doc_id': src['doc_id'], 'page': src['page'],
                'char_start': src['char_start'], 'char_end': src['char_end'],
                'source_span_id': src['source_span_id'], 'generation_sha256': src['generation_sha256'],
                'value': cell['value'], 'raw_value': cell['raw_value'], 'quote': cell['quote']}

    @staticmethod
    def _rows_digest(rows) -> str:
        return _digest([(r['table'], r['rowid'], r['column'], r['value']) for r in rows])

    def _validate_rows(self, record: dict) -> None:
        current = [self._row_fact({'table': r['table'], 'rowid': r['rowid'], 'column': r['column']})
                   for r in record['rows']]
        if self._rows_digest(current) != record['rows_digest'] or any(
                c['quote'] != r['quote'] or c['source_span_id'] != r['source_span_id']
                for c, r in zip(current, record['rows'], strict=True)):
            raise ValueError("aggregate source rows changed since the certificate was issued")

    def _select_rows(self, selection_sql: str, column: str | None) -> list[dict]:
        """Run the controller's selection SQL on the parent's read-only snapshot
        and return (table, rowid, column) triples. The predicate becomes part
        of the certificate: the reviewer judges the SQL, not a paraphrase."""
        if not isinstance(selection_sql, str) or len(selection_sql) > 8000:
            raise ValueError("selection_sql must be a string of at most 8000 characters")
        head = selection_sql.lstrip().lower()
        if not (head.startswith("select") or head.startswith("with")) or ";" in selection_sql:
            raise ValueError("selection_sql must be a single SELECT statement")
        rows = []
        cur = self.conn.execute(selection_sql)
        width = len(cur.description or ())
        if width not in (2, 3) or (width == 2 and not column):
            raise ValueError("selection_sql must return (table_name, rowid) with column=..., or (table_name, rowid, column)")
        for rec in cur:
            self._check_deadline()
            rows.append({'table': rec[0], 'rowid': rec[1], 'column': rec[2] if width == 3 else column})
        return rows

    def aggregate(self, operation: str, rows: list | None = None, *, selection: str | None = None,
                  selection_sql: str | None = None, column: str | None = None) -> dict:
        """One certificate for an operation over an exact set of source rows.

        Every row is re-read from the frozen tables with its canonical span, so
        the certificate binds (operation, rows, values) and can be revalidated
        at FINAL time. Rows come either from ``rows`` (triples chosen by the
        controller; ``selection`` is its own description, recorded verbatim and
        not verified) or from ``selection_sql``, which the parent executes on
        its read-only snapshot so that the predicate itself is certified.
        """
        self._check_deadline()
        self._check_generation()
        if not isinstance(operation, str) or operation not in AGGREGATE_OPERATIONS:
            raise ValueError("aggregate operation must be sum, count, mean, min or max")
        if selection_sql is not None:
            if rows:
                raise ValueError("pass either rows or selection_sql, not both")
            rows = self._select_rows(selection_sql, column)
            selection = selection_sql
        if not isinstance(rows, list) or not rows:
            raise ValueError("aggregate requires a nonempty list of rows (or a selection_sql that matches rows)")
        if selection is not None and (not isinstance(selection, str) or len(selection) > 8000):
            raise ValueError("selection must be a string of at most 8000 characters")
        seen = set()
        facts = []
        for row in rows:
            fact = self._row_fact(row)
            key = (fact['table'], fact['rowid'], fact['column'])
            if key in seen:
                raise ValueError(f"aggregate row repeated: {key}")
            seen.add(key)
            facts.append(fact)
        values = [_decimal(f['value']) for f in facts]
        with localcontext() as ctx:
            ctx.prec = PRECISION
            ctx.rounding = ROUND_HALF_EVEN
            value = {'sum': lambda: sum(values, Decimal(0)),
                     'count': lambda: Decimal(len(values)),
                     'mean': lambda: sum(values, Decimal(0)) / len(values),
                     'min': lambda: min(values), 'max': lambda: max(values)}[operation]()
            rendered = _render(value)
        tables = sorted({f['table'] for f in facts})
        documents = sorted({f['doc_id'] for f in facts})
        columns = sorted({f['column'] for f in facts})
        return self._store({
            'kind': 'aggregate', 'value': rendered, 'operation': operation,
            'n_rows': len(facts), 'rows': facts, 'rows_digest': self._rows_digest(facts),
            'tables': tables, 'documents': documents, 'columns': columns,
            'min': _render(min(values)), 'max': _render(max(values)),
            'rows_sample': [{k: f[k] for k in ('table', 'rowid', 'column', 'value')}
                            for f in facts[:8]],
            'selection': selection,
            'selection_executed_by': 'parent' if selection_sql is not None else 'controller',
            'operand_ids': [], 'source_ids': [],
            'precision': PRECISION, 'rounding': 'ROUND_HALF_EVEN', 'unit': None, 'period': None,
            'metadata_check': ('rows re-read from frozen tables and bound by digest; '
                               + ('selection SQL executed by the parent on its read-only snapshot'
                                  if selection_sql is not None else
                                  'selection is model-described and not verified'))}, 'agg_')

    def get(self, record_id: str) -> dict:
        record = self._lookup(record_id)
        sources = [record_id] if record['kind'] in ('source', 'aggregate') else record['source_ids']
        self._validate_sources(sources)
        if record.get('metric_contract') and self._metric_contract:
            self._metric_contract.validate_sources(self)
        result = copy.deepcopy(record)
        self._check_deadline()
        return result

    def _read_text_number(self, doc_id: str, char_start: int, char_end: int) -> dict:
        span = self._span(doc_id, char_start, char_end)
        raw = span['text']
        if len(raw) > 128 or not re.fullmatch(
                r'(?:[-+−]?\d+(?:,\d{3})*(?:\.\d+)?|\(\d+(?:,\d{3})*(?:\.\d+)?\))', raw):
            raise ValueError('source text span must contain exactly one numeric literal, without units')
        # Occurrence of "10" inside "-10", "(10)", "1.10" or "10,000"
        # is not evidence for positive ten. Resolve the whole surrounding
        # numeric token, retaining its sign/grouping before accepting offsets.
        source_end = self.conn.execute('SELECT max(char_end) FROM doc_text WHERE doc_id=?',
                                       (doc_id,)).fetchone()[0]
        window_start = max(0, char_start - 256)
        window = self._span(doc_id, window_start, min(source_end, char_end + 256))['text']
        tokens = re.finditer(
            r'(?<![\w.,])(?:\(\s*(?:[-+−]\s*)?\d+(?:,\d{3})*(?:\.\d+)?\s*\)'
            r'|(?:[-+−]\s*)?\d+(?:,\d{3})*(?:\.\d+)?)(?![\w]|\.\d|,\d)', window)
        if not any((window_start + token.start(), window_start + token.end()) == (char_start, char_end)
                   for token in tokens):
            raise ValueError('source text span must include the complete numeric token and its sign')
        normalized = raw.replace(',', '').replace('−', '-')
        if normalized.startswith('('):
            normalized = '-' + normalized[1:-1]
        generation = self._generation_by_id.get(doc_id)
        if generation is None:
            raise ValueError('source text document is not registered')
        context = self._context(doc_id=doc_id, char_start=char_start, char_end=char_end)
        return {'value': _render(_decimal(normalized)), 'raw_value': raw, 'quote': raw,
                'source': {'kind': 'text', 'doc_id': doc_id, 'char_start': char_start,
                           'char_end': char_end, 'page': context['page'],
                           'heading_paths': context['heading_paths'],
                           'source_span_id': span['source_span_id'],
                           'generation_sha256': _digest(generation)}}

    def source_text_number(self, doc_id: str, char_start: int, char_end: int, *,
                           unit_span: dict | None = None, period_span: dict | None = None) -> dict:
        """Bind one original numeric span when a source is prose, not a typed table."""
        self._check_generation()
        cell = self._read_text_number(doc_id, char_start, char_end)
        unit, period = (self._metadata_span(s, doc_id) for s in (unit_span, period_span))
        return self._store({'kind': 'source', **cell,
                            'unit': unit['text'] if unit else None,
                            'period': period['text'] if period else None,
                            'unit_source': unit, 'period_source': period,
                            'metadata_check': 'source_text_only; scope not semantically verified'}, 'src_')

    def calculate_financial(self, metric: str, inputs: dict[str, str], *, convention: str) -> dict:
        from rnsr.env.financial import financial_metric

        return financial_metric(self, metric, inputs, convention=convention)

    def resolve_finals(self, template: str, results: dict[str, str], *,
                       decimals: int | None = None) -> tuple[str, dict]:
        """Render numerical claims from immutable calculations, not model literals."""
        from rnsr.env.financial import render_calculations

        if not isinstance(results, dict) or not results or len(results) > 16:
            raise ValueError('calculation answer requires 1-16 named results')
        records, reports, values = {}, [], {}
        for name, result_id in results.items():
            result = self.get(result_id)
            _presented, report = self.resolve_final(result_id, result['value'])
            values[name] = result['value']     # templates round the raw value themselves
            reports.append(report)
            records.update({record['id']: record for record in report['records']})
        answer = render_calculations(template, values, decimals=decimals)
        quotes = []
        seen = set()
        for report in reports:
            for quote in report['quotes']:
                identity = (quote['doc_id'], quote['char_start'], quote['char_end'])
                if identity not in seen:
                    quotes.append(quote)
                    seen.add(identity)
        proof = {'passed': True, 'check': 'source_bound_calculations',
                        'answer': answer, 'quotes': quotes, 'records': list(records.values()),
                        'calculation_answers': {
                            name: {'value': value, 'calculation_id': results[name],
                                   'rendered_value': render_calculations('{value}', {'value': value},
                                                                        decimals=decimals)}
                            for name, value in values.items()},
                        'claim_support': 'not_checked'}
        if self._metric_contract is not None:
            # Each component was resolved against the same caller contract
            # above; retain its authority through narrative rendering too.
            proof['metric_contract'] = self._metric_contract.describe()
            proof['metric_contract_satisfied'] = True
        return answer, proof

    def calculate_metric(self) -> dict:
        """Execute the caller's fixed formula; model-supplied operands are not accepted."""
        self._check_deadline()
        if self._metric_contract is None:
            raise ValueError("no caller-issued metric contract is installed")
        if self._metric_result_id is None:
            result, role_source_ids, numerator_id = self._metric_contract.compute(self)
            if self._metric_contract.nonpositive_numerator == 'zero':
                numerator = self.get(numerator_id)
                # This is a caller-declared convention, not another source value.
                # Preserve both the raw numerator and ratio in the expression DAG.
                result = self._store({
                    'kind': 'calculation', 'value': ('0' if _decimal(numerator['value']) <= 0
                                                    else result['value']),
                    'operation': 'caller_nonpositive_numerator_zero',
                    'operand_ids': [result['id'], numerator_id],
                    'raw_ratio_id': result['id'], 'raw_numerator_id': numerator_id,
                    'source_ids': result['source_ids'], 'precision': PRECISION,
                    'rounding': 'ROUND_HALF_EVEN', 'unit': None, 'period': None,
                    'metadata_check': 'caller convention; not inferred from source or benchmark',
                }, 'calc_')
            record = self._records[result['id']]
            record['metric_contract'] = self._metric_contract.describe()
            record['role_source_ids'] = role_source_ids
            self._metric_result_id = result['id']
        return self.get(self._metric_result_id)

    def validate_legacy_final(self, value) -> dict | None:
        """Strict mode permits only explicit abstention outside its bound result."""
        self._check_deadline()
        if self._metric_contract is None:
            return None
        if not isinstance(value, str) or value != 'NOT_FOUND':
            raise ValueError("metric contract requires FINAL_CALC(calculate_metric()['id']) or NOT_FOUND")
        return {'passed': False, 'check': 'metric_contract_abstention', 'answer': value,
                'quotes': [], 'metric_contract': self._metric_contract.describe(),
                'claim_support': 'not_checked'}

    def resolve_final(self, result_id: str, submitted_value) -> tuple[str, dict]:
        result = self.get(result_id)
        if result['kind'] not in ('calculation', 'aggregate'):
            raise ValueError("FINAL_CALC requires a calculation or aggregate result id")
        if self._metric_contract is not None and (
                result_id != self._metric_result_id
                or result.get('metric_contract', {}).get('sha256') != self._metric_contract.digest):
            raise ValueError("FINAL_CALC result does not satisfy the caller-issued metric contract")
        if not isinstance(submitted_value, str) or submitted_value not in (result['value'], present(result['value'])):
            raise ValueError("FINAL_CALC value differs from the parent calculation result")
        facts = [self.get(source_id) for source_id in result['source_ids']]
        if result['kind'] == 'aggregate':
            facts = [result]
        quotes = []
        for fact in facts:
            if fact['kind'] == 'aggregate':
                # A bounded sample of row quotes; the certificate itself carries every row.
                for row in fact['rows'][:AGGREGATE_QUOTE_SAMPLE]:
                    quotes.append({'quote': row['quote'], 'matched': True,
                                   'source_context': self._context(
                                       doc_id=row['doc_id'], char_start=row['char_start'],
                                       char_end=row['char_end']),
                                   **{key: row[key] for key in ('doc_id', 'char_start', 'char_end')}})
                continue
            source = fact['source']
            quotes.append({'quote': fact['quote'], 'matched': True,
                           'source_context': self._context(
                               doc_id=source['doc_id'], char_start=source['char_start'],
                               char_end=source['char_end']),
                           **{key: source[key] for key in ('doc_id', 'char_start', 'char_end')}})
        quotes = quotes[:MAX_FINAL_QUOTES]
        # Include the entire bounded expression graph, not just the last step.
        graph = {}
        def visit(record_id):
            self._check_deadline()
            if record_id in graph:
                return
            record = self._lookup(record_id)
            graph[record_id] = copy.deepcopy(record)
            for key in record.get('operand_ids', []):
                visit(key)
        visit(result_id)
        report = {'passed': True, 'check': 'source_bound_calculation',
                                 'answer': present(result['value']), 'quotes': quotes,
                                 'calculation_id': result_id, 'calculation': result,
                                 'records': list(graph.values()),
                                 'claim_support': 'not_checked'}
        if self._metric_contract:
            report['metric_contract'] = self._metric_contract.describe()
            report['metric_contract_satisfied'] = True
        self._check_deadline()
        return present(result['value']), report
