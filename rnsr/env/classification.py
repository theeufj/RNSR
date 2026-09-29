"""Final aggregates derived from a revalidated classification contract."""
from __future__ import annotations

from rnsr.env.annotate import Annotator
from rnsr.env.evidence import SourceContext


def classification_final(conn, table: str, column: str, operation: str, labels=None,
                         *, cancelled=lambda: False):
    if not all(isinstance(value, str) and value for value in (table, column, operation)):
        raise ValueError('aggregate table, column and operation must be nonempty strings')
    if cancelled():
        raise TimeoutError('classification aggregate cancelled')
    # Keep scope, active labels, annotation metadata and quoted source on the
    # same snapshot even when this helper is used outside the sandbox.
    own_snapshot = not conn.in_transaction
    if own_snapshot:
        conn.execute('BEGIN')
    try:
        result = _classification_final(conn, table, column, operation, labels, cancelled=cancelled)
        if cancelled():
            raise TimeoutError('classification aggregate cancelled')
        return result
    finally:
        if own_snapshot:
            conn.rollback()


def _classification_final(conn, table: str, column: str, operation: str, labels,
                          *, cancelled):
    proof = Annotator(conn, None, cancelled=cancelled).classification_counts(table, column)
    counts = proof['counts']
    if labels is None:
        labels = []
    if (not isinstance(labels, list) or any(not isinstance(label, str) or label not in counts
                                          for label in labels)
            or len(labels) != len(set(labels))):
        raise ValueError('aggregate labels must be distinct members of the classification vocabulary')
    if operation == 'count' and len(labels) == 1:
        answer = f'Answer: {counts[labels[0]]}'
    elif operation == 'compare' and len(labels) == 2:
        left, right = (counts[label] for label in labels)
        comparison = ('more common than' if left > right else 'less common than' if left < right
                      else 'same frequency as')
        answer = f'Answer: {labels[0]} is {comparison} {labels[1]}'
    elif operation in {'least', 'most'} and not labels:
        extreme = (min if operation == 'least' else max)(counts.values())
        answer = 'Label: ' + ', '.join(label for label, count in counts.items() if count == extreme)
    else:
        raise ValueError('use count with one label, compare with two, or least/most without labels')
    record = conn.execute('SELECT doc_id FROM manifest_tables WHERE table_name=?', (table,)).fetchone()
    source = conn.execute('SELECT text,char_start FROM doc_text WHERE doc_id=? ORDER BY page LIMIT 1',
                          (record[0],)).fetchone()
    quotes = []
    if source and source[0]:
        text, start = source[0][:1200], source[1]
        quotes.append({'quote': text, 'matched': True, 'doc_id': record[0],
                       'char_start': start, 'char_end': start + len(text),
                       'source_context': SourceContext(conn)(doc_id=record[0], char_start=start,
                                                             char_end=start + len(text))})
    contract = conn.execute('SELECT where_clause,prompt FROM annotation_log WHERE id=?',
                            (proof['annotation_id'],)).fetchone()
    return answer, {'passed': True, 'check': 'classification_aggregate', 'answer': answer,
                    'quotes': quotes, 'classification': {**proof, 'table': table, 'column': column,
                                                        'where': contract[0], 'instruction': contract[1],
                                                        'operation': operation, 'labels': labels},
                    'claim_support': 'not_checked'}
