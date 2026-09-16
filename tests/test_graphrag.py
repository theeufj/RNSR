import sqlite3

import pytest

from rnsr.eval.graphrag import _parse_extraction, build_graph_index, graph_ready, graph_retrieve
from rnsr.llm.mock import MockLLM


@pytest.mark.parametrize("text", ["no JSON", "{bad json}", '{"entities": null}',
                                   '{"entities": [], "relations": "bad"}'])
def test_bad_extractions_are_ignored(text):
    assert _parse_extraction(text) is None


async def test_graph_build_connects_entities_and_retrieves_source():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE chunks(chunk_id INTEGER, doc_id TEXT, text TEXT)")
    conn.executemany("INSERT INTO chunks VALUES (?, ?, ?)", [
        (1, "contract", "Alice founded ACME."), (2, "letter", "ACME hired Bob.")])
    client = MockLLM().script(
        '{"entities":[{"name":"Alice"},{"name":"ACME"}],'
        '"relations":[{"src":"Alice","rel":"founded","dst":"ACME"}]}',
        '{"entities":[{"name":"acme"},{"name":"Bob"}],'
        '"relations":[{"src":"ACME","rel":"hired","dst":"Bob"}]}',
        "Alice founded ACME and ACME hired Bob.")
    try:
        assert not graph_ready(conn)
        stats = await build_graph_index(conn, client, "mock", concurrency=1)
        assert stats["entities"] == 3 and stats["communities"] == 1
        summaries, chunks = graph_retrieve(conn, "Who founded ACME?")
        assert summaries == ["Alice founded ACME and ACME hired Bob."]
        assert {row[0] for row in chunks} == {"contract", "letter"}
        prior_calls = len(client.calls)
        assert await build_graph_index(conn, client, "mock") == {"cached": True}
        assert len(client.calls) == prior_calls
    finally:
        conn.close()
