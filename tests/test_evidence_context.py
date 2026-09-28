"""Source/period boundaries must survive retrieval and quote verification."""
import sqlite3

import pytest

from rnsr.db.artifact import CorpusDB
from rnsr.env.evidence import SourceContext
from rnsr.env.lazydoc import LazyDoc
from rnsr.env.search import Ladder, terms
from rnsr.env.verify import Verifier
from rnsr.ingest.model import Element, ParsedDocument, RawTable
from rnsr.ingest.pipeline import ingest


@pytest.fixture
def scoped_corpus(tmp_path):
    def parse(path):
        elements, tables = [], []
        # Put the requested year after more than the old k*4 candidate cap.
        for i, year in enumerate([2024, 2023, 2022, 2020, 2019, 2018, 2021]):
            page = i * 2 + 1
            elements += [Element('heading', f'Revenue by division, fiscal {year}', page, heading_level=1),
                         Element('text', 'Routine operations commentary. ' * 100, page),
                         Element('table', f'Division | Revenue\nCameras | {100 + i}', page + 1)]
            tables.append(RawTable(page=page + 1, header=['Division', 'Revenue'],
                                   rows=[['Cameras', str(100 + i)]], extractor='test'))
        return ParsedDocument(doc_id='report', source_path=str(path), sha256='a' * 64,
                              n_pages=14, parser='test', elements=elements, tables=tables)
    path = tmp_path / 'source.db'
    ingest([tmp_path / 'report.pdf'], path, parse=parse)
    with CorpusDB(path) as db:
        manifest = db.manifest_dict()
    conn = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    try:
        yield conn, LazyDoc(conn), manifest
    finally:
        conn.close()


def ladder(env, **kwargs):
    conn, doc, manifest = env
    return Ladder(conn=conn, doc=doc, manifest=manifest,
                  rpc=lambda req: {'results': ['NONE']}, enable_embeddings=False, **kwargs)


def test_sql_prefers_requested_section_before_candidate_limit(scoped_corpus):
    for rebuild in (False, True):
        hit = ladder(scoped_corpus, rebuild_cells=rebuild).search('Cameras revenue fiscal 2021', rung=0, k=1)[0]
        assert hit['rows']['revenue'] == 106
        assert hit['query_scope']['year_alignment'] == 'match'
        assert hit['doc_id'] == 'report'
        assert 'fiscal 2021' in hit['text']
        context = hit['source_context']
        assert context['row_location'] == 'exact'
        assert context['page'] == 14
        source = scoped_corpus[1]['report']
        assert source[context['char_start']:context['char_end']] == 'Cameras | 106'
        assert source[context['text_char_start']:context['text_char_end']] == context['text']


def test_following_heading_does_not_relabel_previous_row(scoped_corpus):
    conn, doc, _ = scoped_corpus
    text = doc['report']
    offset = text.index('Cameras | 100')
    context = SourceContext(conn)('report', char_start=offset, char_end=offset + len('Cameras | 100'))
    assert context['heading_paths'] == ['Revenue by division, fiscal 2024']
    assert '2023' not in context['text']  # next section starts immediately after this row
    hits = ladder(scoped_corpus).search('Cameras revenue fiscal 2021', rung=0, k=20)
    old = next(hit for hit in hits if hit['rows']['revenue'] == 100)
    assert old['query_scope']['year_alignment'] == 'conflict'
    assert old['query_scope']['section_years'] == ['2024']


def test_numeric_only_query_reaches_grep_and_fts(scoped_corpus):
    assert terms('2021') == ['2021']
    assert '2021' in terms(' '.join(f'keyword{i}' for i in range(20)) + ' 2021')
    for rung in (1, 2):
        hit = ladder(scoped_corpus).search('2021', rung=rung, k=1)[0]
        assert '2021' in hit['text']
        assert hit['source_context']['heading_paths'] == ['Revenue by division, fiscal 2021']


def test_grep_ranks_later_joint_match_above_early_partial_matches(scoped_corpus):
    hit = ladder(scoped_corpus).search('Cameras revenue fiscal 2021', rung=1, k=1)[0]
    assert hit['source_context']['heading_paths'] == ['Revenue by division, fiscal 2021']
    assert 'Cameras | 106' in hit['text']


def test_quote_context_is_from_exact_source_span(scoped_corpus):
    _, doc, _ = scoped_corpus
    verifier = Verifier(doc)
    try:
        result = verifier.verify('100', ['Cameras | 100'])
        assert result['passed']  # lexical match does not certify requested fiscal year
        quote = result['quotes'][0]
        assert quote['source_context']['heading_paths'] == ['Revenue by division, fiscal 2024']
        assert result['check'] == 'lexical_source_match'
        assert not verifier.verify('100', ['Cameras | 100'], doc_id='other')['passed']
    finally:
        verifier.close()


def test_repeated_quotes_keep_source_ambiguity_visible():
    verifier = Verifier({'a': 'Revenue 100. Revenue 100.', 'b': 'Revenue 100.'})
    try:
        quote = verifier.verify('100', ['Revenue 100'])['quotes'][0]
        assert [(m['doc_id'], m['char_start']) for m in quote['matches']] == [('a', 0), ('a', 13), ('b', 0)]
        assert not quote['matches_truncated']
        assert verifier.verify('100', ['Revenue 100'], doc_id='b')['quotes'][0]['doc_id'] == 'b'
    finally:
        verifier.close()


def test_duplicate_quote_results_are_bounded():
    verifier = Verifier({'a': 'Revenue 100. ' * 100})
    try:
        quote = verifier.verify('100', ['Revenue 100'])['quotes'][0]
        assert len(quote['matches']) == 8
        assert quote['matches_truncated']
    finally:
        verifier.close()


def test_missing_canonical_row_is_page_context_not_invented_location(scoped_corpus):
    conn, _, manifest = scoped_corpus
    table = manifest['tables'][0]['table_name']
    # A table-level request cannot establish a particular row's source span.
    context = SourceContext(conn)(table=table)
    assert context['row_location'] == 'page_only'
    assert context['row_locations'] == []
    assert context['page_start'] == 2
    with pytest.raises(KeyError):
        SourceContext(conn)(table=table, rowid=999)
    with pytest.raises(ValueError):
        SourceContext(conn)('report', char_start=-1)
    with pytest.raises(ValueError):
        SourceContext(conn)('report', page=999)


@pytest.mark.parametrize('canonical', [
    # Row's amount cannot be found as a substring of a different amount.
    'Division | Revenue\nCameras | 5198',
    # An identical row in two sections has no unique governing year.
    'Division | Revenue\nCameras | 51',
])
def test_ambiguous_page_headings_never_establish_sql_year(tmp_path, canonical):
    def parse(path):
        return ParsedDocument(
            doc_id='ambiguous', source_path=str(path), sha256='b' * 64, n_pages=1, parser='test',
            elements=[Element('heading', 'Revenue 2023', 1, heading_level=1),
                      Element('table', canonical, 1),
                      Element('heading', 'Revenue 2024', 1, heading_level=1),
                      Element('table', canonical, 1)],
            tables=[RawTable(page=1, header=['Division', 'Revenue'],
                             rows=[['Cameras', '51']], extractor='test')])
    path = tmp_path / 'ambiguous.db'
    ingest([tmp_path / 'ambiguous.pdf'], path, parse=parse)
    with CorpusDB(path) as db:
        manifest = db.manifest_dict()
    with sqlite3.connect(path) as conn:
        hit = ladder((conn, LazyDoc(conn), manifest)).search('Cameras revenue 2024', rung=0)[0]
        assert hit['query_scope']['year_alignment'] == 'unknown'
        assert hit['source_context']['row_location'] == 'page_only'
        assert 'row-to-section unverified' in hit['text']
        if '5198' in canonical:
            assert hit['source_context']['row_locations'] == []
        else:
            assert len(hit['source_context']['row_locations']) == 2


async def test_source_context_is_available_in_sandbox(scoped_corpus, tmp_path):
    from rnsr.env.sandbox import SandboxedRepl

    path = scoped_corpus[0].execute('PRAGMA database_list').fetchone()[2]
    async with SandboxedRepl() as repl:
        await repl.start(mode='docdb', corpus_db=path)
        result = await repl.exec_cell(
            "print(source_context(table='t_report_007', rowid=1)['heading_paths'])\n"
            "print(verify('100', ['Cameras | 100'])['quotes'][0]['source_context']['heading_paths'])")
        assert result.ok, result.error
        assert "fiscal 2021" in result.stdout
        assert "fiscal 2024" in result.stdout


def test_grep_keeps_distinct_short_sections_in_same_offset_bucket(tmp_path):
    def parse(path):
        return ParsedDocument(doc_id='short', source_path=str(path), sha256='c' * 64,
                              n_pages=1, parser='test', elements=[
                                  Element('heading', 'Revenue fiscal 2023', 1, heading_level=1),
                                  Element('text', 'Cameras revenue 100.', 1),
                                  Element('heading', 'Revenue fiscal 2024', 1, heading_level=1),
                                  Element('text', 'Cameras revenue 200.', 1)])
    path = tmp_path / 'short.db'
    ingest([tmp_path / 'short.pdf'], path, parse=parse)
    with CorpusDB(path) as db:
        manifest = db.manifest_dict()
    with sqlite3.connect(path) as conn:
        hits = ladder((conn, LazyDoc(conn), manifest)).search('Cameras revenue fiscal 2024', rung=1)
        assert '200' in hits[0]['text']
        assert hits[0]['query_scope']['year_alignment'] == 'match'
        assert len(hits) == 2
