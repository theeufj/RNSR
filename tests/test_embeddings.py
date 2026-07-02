"""Rung 4: int8 store + fp32 rescore, write-back caching, ablation recall."""

import sqlite3

import numpy as np
import pytest

pytest.importorskip("sqlite_vec")

from rnsr.env.embeddings import EmbeddingStore  # noqa: E402
from rnsr.eval.ablation import run_ablation  # noqa: E402
from rnsr.ingest.model import Element, ParsedDocument  # noqa: E402
from rnsr.ingest.pipeline import ingest  # noqa: E402


def _parse_many_chunks(path):
    words = ["revenue", "liability", "cashflow", "inventory", "goodwill",
             "amortization", "leases", "derivatives", "segments", "taxes"]
    elements = [
        Element("text", f"Section about {words[i % 10]} topic number {i}. " * 30, 1)
        for i in range(40)
    ]
    return ParsedDocument(doc_id="big", source_path=str(path), sha256="c" * 64,
                          n_pages=1, parser="fake", elements=elements)


@pytest.fixture
def corpus(tmp_path):
    out = tmp_path / "corpus.db"
    ingest([tmp_path / "big.pdf"], out, parse=_parse_many_chunks)
    return out


def _seeded_embed(dim=64, seed=13):
    """Deterministic pseudo-embeddings: seeded gaussian keyed by text hash."""

    def embed(texts):
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash((seed, t))) % (2**32))
            out.append(rng.standard_normal(dim).astype(np.float32).tolist())
        return out

    return embed


class TestStore:
    def test_lazy_build_and_cache(self, corpus):
        conn = sqlite3.connect(corpus)
        store = EmbeddingStore(conn)
        embed = _seeded_embed()
        first = store.ensure(embed, model="toy")
        assert first["embedded"] > 0

        calls = {"n": 0}

        def counting_embed(texts):
            calls["n"] += 1
            return embed(texts)

        second = EmbeddingStore(conn).ensure(counting_embed, model="toy")
        assert second["embedded"] == 0          # fully cached
        assert calls["n"] == 0                  # zero embed calls on re-run
        conn.close()

    def test_knn_returns_relevant_chunk(self, corpus):
        conn = sqlite3.connect(corpus)
        store = EmbeddingStore(conn)
        embed = _seeded_embed()
        store.ensure(embed, model="toy")
        row = conn.execute("SELECT chunk_id, text FROM chunks LIMIT 1").fetchone()
        # querying with a chunk's own embedding must return that chunk first
        hits = store.knn(embed([row[1]])[0], k=3)
        assert hits[0][0] == row[0]
        assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
        conn.close()

    def test_manifest_records_rung4_meta(self, corpus):
        conn = sqlite3.connect(corpus)
        EmbeddingStore(conn).ensure(_seeded_embed(), model="toy")
        import json

        meta = json.loads(conn.execute(
            "SELECT value FROM manifest WHERE key='rung4_meta'").fetchone()[0])
        assert meta["quantization"] == "int8" and meta["dim"] == 64
        conn.close()


class TestAblation:
    def test_int8_recall_on_gaussian_vectors(self, corpus):
        embed = _seeded_embed(dim=64)
        conn = sqlite3.connect(corpus)
        texts = [r[0] for r in conn.execute("SELECT text FROM chunks LIMIT 10")]
        conn.close()
        report = run_ablation(corpus, embed, texts, ks=(5, 10))
        # rescore pool (4000) >> corpus (~40 chunks): recall must be perfect
        assert report["recall"]["@5"] == 1.0
        assert report["recall"]["@10"] == 1.0
        assert report["quantizer"] == "int8"


class TestLadderRung4:
    async def test_rung4_through_sandbox(self, corpus):
        from rnsr.env.sandbox import SandboxedRepl

        embed = _seeded_embed()

        async def embed_handler(req):
            return {"vectors": embed(req["texts"])}

        async def log(req):
            return {}

        repl = SandboxedRepl(rpc_handlers={"embed": embed_handler, "log": log})
        await repl.start(mode="docdb", corpus_db=str(corpus))
        try:
            res = await repl.exec_cell(
                "hits = search('goodwill topic', rung=4)\n"
                "print(hits[0]['kind'], 'chunk_id' in hits[0]['provenance'])"
            )
            assert res.ok, res.error
            assert res.stdout == "chunk True\n"
        finally:
            await repl.close()
