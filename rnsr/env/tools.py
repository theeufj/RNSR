"""Builds the preloaded docdb namespace inside the sandbox child (spec §4).

    db                 sqlite3 connection (rw; source data trigger-frozen)
    doc                dict: doc_id -> full raw text
    manifest           dict form of the manifest
    semantic_annotate  batched sub-LM pass writing a real column (§4.1)
    search             the tiered ladder (§5)
    verify             exact quote matching (§6)
    schema_map         cross-table column-correspondence proposals (§9)

llm_query/llm_map/FINAL/FINAL_VAR are installed by the child itself.
"""

from __future__ import annotations

import json
import sqlite3


def build_namespace(corpus_db: str, child, init_msg: dict) -> dict:
    from rnsr.db.artifact import CorpusDB
    from rnsr.env.annotate import Annotator
    from rnsr.env.search import Ladder
    from rnsr.env.verify import Verifier

    with CorpusDB(corpus_db, mode="ro") as ro:
        manifest = ro.manifest_dict()

    conn = sqlite3.connect(corpus_db)  # rw for annotations; triggers guard sources
    from rnsr.env.lazydoc import LazyDoc

    doc = LazyDoc(conn)  # bounded memory at any corpus size

    verifier = Verifier(doc)
    ladder = Ladder(
        conn=conn, doc=doc, manifest=manifest, rpc=child.rpc,
        expansion_max_rounds=init_msg.get("expansion_max_rounds", 3),
    )
    annotator = Annotator(
        conn, child.rpc,
        char_budget=init_msg.get("sub_call_char_budget", 200_000),
        default_batch_size=init_msg.get("annotate_batch_size", 40),
    )

    def semantic_annotate(table, new_col, prompt, *, where=None, batch_size=None,
                          model="sub", force=False, votes=1):
        return annotator.annotate(table, new_col, prompt, where=where,
                                  batch_size=batch_size, model=model, force=force,
                                  votes=votes)

    def schema_map(table_a: str, table_b: str) -> list[dict]:
        """Sub-LM *proposals* for column correspondences — never auto-applied."""
        def describe(name: str) -> str:
            t = next((t for t in manifest.get("tables", [])
                      if t["table_name"] == name), None)
            cols = [c["name"] for c in (t or {}).get("schema", [])]
            rows = conn.execute(
                f'SELECT * FROM "{name}" LIMIT 3'
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

    rejections = {"n": 0}

    def FINAL(answer, quotes=None):  # noqa: N802 — spec-mandated name
        """Docdb FINAL enforces §6: answers carry quotes, verified by code.

        A FINAL whose quotes fail verification raises — the failure report
        becomes the loop observation instead of an accepted answer. To
        prevent rejection death-spirals (seen live: 20 iterations of quote
        retries), the third attempt is accepted with the failed verification
        recorded rather than blocked again.
        """
        from rnsr.env.sandbox_child import _FinalAnswer

        report = verifier.verify(str(answer), quotes or [])
        if report["passed"] and quotes:
            raise _FinalAnswer(answer, is_var=False, verification=report)

        rejections["n"] += 1
        if rejections["n"] >= 3:   # stop the spiral; record the failure
            raise _FinalAnswer(answer, is_var=False, verification=report)
        if not quotes:
            raise ValueError(
                "FINAL(answer, quotes=[...]) requires 1-3 short verbatim "
                "quotes from the source supporting the answer (copy exactly "
                "from search hits or doc text). For a purely computed value "
                "from SQL, return the variable with FINAL_VAR(...) instead."
            )
        failed = [q["quote"] for q in report["quotes"] if not q["matched"]]
        raise ValueError(
            f"FINAL rejected — these quotes do not match the source text "
            f"(after normalization): {failed}. Copy the document text "
            "verbatim, or reconsider the answer."
        )

    return {
        "db": conn,
        "doc": doc,
        "manifest": manifest,
        "semantic_annotate": semantic_annotate,
        "search": ladder.search,
        "verify": verifier.verify,
        "schema_map": schema_map,
        "FINAL": FINAL,  # overrides the unverified classic-mode FINAL
    }
