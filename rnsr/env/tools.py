"""Builds the preloaded docdb namespace inside the sandbox child (spec §4).

    db                 sqlite3 connection (read-only source artifact)
    doc                dict: doc_id -> full raw text
    manifest           dict form of the manifest
    semantic_annotate  batched sub-LM pass writing a real column (§4.1)
    search             the tiered ladder (§5)
    verify             exact quote matching (§6)
    source_context     retained section/page context for a span or table row
    schema_map         cross-table column-correspondence proposals (§9)

llm_query/llm_map/FINAL/FINAL_VAR are installed by the child itself.
"""

from __future__ import annotations

import json
import sqlite3
from numbers import Integral


def build_namespace(corpus_db: str, child, init_msg: dict) -> dict:
    from rnsr.db.artifact import CorpusDB
    from rnsr.env.search import Ladder
    from rnsr.env.verify import Verifier

    with CorpusDB(corpus_db, mode="ro") as ro:
        manifest = ro.manifest_dict()

    from pathlib import Path

    conn = sqlite3.connect(Path(corpus_db).resolve().as_uri() + "?mode=ro", uri=True)
    from rnsr.db.schema import apply_read_pragmas, quote_ident, validate_frozen

    apply_read_pragmas(conn)  # mmap-backed reads via the shared OS page cache
    validate_frozen(conn)

    def _authorizer(action, _arg1, _arg2, _dbname, _source):
        if action == sqlite3.SQLITE_ATTACH:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    conn.set_authorizer(_authorizer)
    from rnsr.env.lazydoc import LazyDoc

    doc = LazyDoc(conn)  # bounded memory at any corpus size

    verifier = Verifier(doc)
    ladder = Ladder(
        conn=conn, doc=doc, manifest=manifest, rpc=child.rpc,
        expansion_max_rounds=init_msg.get("expansion_max_rounds", 3),
        enable_embeddings=init_msg.get("enable_embeddings", True),
    )
    def semantic_annotate(table, new_col, prompt, *, where=None, batch_size=None,
                          model="sub", force=False, votes=1, allowed_labels=None):
        result = child.rpc({"op": "annotate", "table": table, "new_col": new_col,
                            "prompt": prompt, "where": where,
                            "batch_size": batch_size, "model": model,
                            "force": force, "votes": votes, "allowed_labels": allowed_labels})
        # The parent owns all writes; refresh metadata after a successful write.
        with CorpusDB(corpus_db, mode="ro") as refreshed:
            manifest.update(refreshed.manifest_dict())
        return result["result"]

    def semantic_classify(table, new_col, prompt, *, allowed_labels, where, expected_count,
                          model="sub", votes=3, batch_size=None, force=False):
        # DataFrame counts are commonly NumPy integers. Normalize before the
        # JSON RPC transport; accepting strings or floats would weaken scope checks.
        if (not isinstance(expected_count, Integral) or isinstance(expected_count, bool)
                or expected_count < 0):
            raise ValueError("expected_count must be a nonnegative integer")
        expected_count = int(expected_count)
        result = child.rpc({'op': 'classify', 'table': table, 'new_col': new_col,
                            'prompt': prompt, 'allowed_labels': allowed_labels, 'where': where,
                            'expected_count': expected_count, 'model': model, 'votes': votes,
                            'batch_size': batch_size, 'force': force})
        with CorpusDB(corpus_db, mode='ro') as refreshed:
            manifest.update(refreshed.manifest_dict())
        return result['result']

    def classification_counts(table, column):
        return child.rpc({'op': 'classification_counts', 'table': table, 'column': column})['result']

    def FINAL_CLASSIFICATION(table, column, operation, labels=None):  # noqa: N802
        from rnsr.env.final_answer import FinalAnswer

        arguments = {'table': table, 'column': column, 'operation': operation, 'labels': labels}
        result = child.rpc({'op': 'classification_final', **arguments})['result']
        raise FinalAnswer(result['answer'], is_var=True,
                          verification={'check': 'classification_aggregate', **arguments,
                                        'annotation_version': result['verification']['classification']['annotation_version']})

    def schema_map(table_a: str, table_b: str) -> list[dict]:
        """Sub-LM *proposals* for column correspondences — never auto-applied."""
        def describe(name: str) -> str:
            t = next((t for t in manifest.get("tables", [])
                      if t["table_name"] == name), None)
            cols = [c["name"] for c in (t or {}).get("schema", [])]
            rows = conn.execute(
                f"SELECT * FROM {quote_ident(name)} LIMIT 3"
            ).fetchall()
            return f"{name}: columns={cols} sample_rows={rows[:3]}"

        prompt = (
            "Two tables extracted from different documents may describe the "
            "same kind of data with drifted headers.\n"
            f"A) {describe(table_a)}\nB) {describe(table_b)}\n\n"
            'Propose column correspondences as JSON: [{"a": "<colA>", '
            '"b": "<colB>", "confidence": 0-1, "reason": "..."}]. '
            "Only include pairs you believe correspond. Return only JSON."
        )
        reply = child.rpc({"op": "llm_batch", "prompts": [prompt],
                           "model": "sub"})["results"][0]
        try:
            proposals = json.loads(reply[reply.find("["): reply.rfind("]") + 1])
            return proposals if isinstance(proposals, list) else []
        except (json.JSONDecodeError, ValueError):
            return []

    def FINAL(answer, quotes=None):  # noqa: N802
        from rnsr.env.final_answer import FinalAnswer
        from rnsr.env.finalize import validate_final

        report = validate_final(answer, quotes, verifier)
        raise FinalAnswer(answer, is_var=False, verification=report)

    def FINAL_VAR(value, quotes=None):  # noqa: N802
        # Variables carry the same evidence contract as literal values.
        from rnsr.env.final_answer import FinalAnswer
        from rnsr.env.finalize import validate_final

        report = validate_final(value, quotes, verifier)
        raise FinalAnswer(value, is_var=True, verification=report)

    def FINAL_BATCH(answers, quotes=None):  # noqa: N802
        from rnsr.env.final_answer import FinalAnswer
        from rnsr.env.finalize import validate_final

        try:
            reports = validate_final(answers, quotes, verifier, batch=True)
        except ValueError as exc:
            raise ValueError(f"FINAL_BATCH rejected: {exc}") from exc
        raise FinalAnswer(dict(answers), is_var=True, verification=reports)

    def source_number(table, rowid, column, *, unit_span=None, period_span=None):
        """Bind a numeric source cell; metadata spans are exact text, not inferred scope."""
        return child.rpc({'op': 'calculation', 'action': 'source', 'table': table,
                          'rowid': rowid, 'column': column, 'unit_span': unit_span,
                          'period_span': period_span})['result']

    def calculate(operation, operand_ids):
        """Decimal arithmetic over source/result IDs; no literal replacement operands."""
        return child.rpc({'op': 'calculation', 'action': 'compute',
                          'operation': operation, 'operand_ids': operand_ids})['result']

    def source_text_number(doc_id, char_start, char_end, *, unit_span=None, period_span=None):
        return child.rpc({'op': 'calculation', 'action': 'source_text', 'doc_id': doc_id,
                          'char_start': char_start, 'char_end': char_end,
                          'unit_span': unit_span, 'period_span': period_span})['result']

    def calculate_financial(metric, inputs, *, convention):
        return child.rpc({'op': 'calculation', 'action': 'financial', 'metric': metric,
                          'inputs': inputs, 'convention': convention})['result']

    def calculation(result_id):
        return child.rpc({'op': 'calculation', 'action': 'get',
                          'record_id': result_id})['result']

    def FINAL_CALC(result_id):  # noqa: N802
        """The parent resolves this result again before accepting the final answer."""
        from rnsr.env.final_answer import FinalAnswer

        result = calculation(result_id)
        raise FinalAnswer(result['value'], is_var=True,
                          verification={'check': 'source_bound_calculation',
                                        'calculation_id': result_id})

    def calculate_metric():
        """Execute the exact caller-issued metric contract, when configured."""
        return child.rpc({'op': 'calculation', 'action': 'metric'})['result']

    def FINAL_CALCS(template, results, *, decimals=None):  # noqa: N802
        from rnsr.env.final_answer import FinalAnswer

        arguments = {'template': template, 'results': results, 'decimals': decimals}
        result = child.rpc({'op': 'calculation', 'action': 'render', **arguments})['result']
        raise FinalAnswer(result[0], is_var=True,
                          verification={'check': 'source_bound_calculations', **arguments})

    return {
        "db": conn,
        "doc": doc,
        "manifest": manifest,
        "semantic_annotate": semantic_annotate,
        "semantic_classify": semantic_classify,
        "classification_counts": classification_counts,
        "FINAL_CLASSIFICATION": FINAL_CLASSIFICATION,
        "search": ladder.search,
        "verify": verifier.verify,
        "source_context": doc.context,
        "schema_map": schema_map,
        "FINAL_VAR": FINAL_VAR,
        "FINAL": FINAL,  # overrides the unverified classic-mode FINAL
        "FINAL_BATCH": FINAL_BATCH,  # ditto, with per-field quote checks
        "source_number": source_number,
        "source_text_number": source_text_number,
        "calculate": calculate,
        "calculate_financial": calculate_financial,
        "calculation": calculation,
        "FINAL_CALC": FINAL_CALC,
        "FINAL_CALCS": FINAL_CALCS,
        "calculate_metric": calculate_metric,
        "metric_contract": init_msg.get("metric_contract"),
    }
