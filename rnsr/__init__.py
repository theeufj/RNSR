"""DocDB-RLM: typed-environment recursive language model system.

Public API (filled in as phases land):
    ingest(sources, out_db, ...)  -> IngestReport   (Phase A)
    open_corpus(path)             -> CorpusDB       (Phase A)
    answer(question, corpus, ...) -> QueryResult    (Phase B/C)
"""

__version__ = "1.0.0a0"


def ingest(*args, **kwargs):
    from rnsr.ingest.pipeline import ingest as _ingest

    return _ingest(*args, **kwargs)


def open_corpus(path, mode: str = "ro"):
    from rnsr.db.artifact import CorpusDB

    return CorpusDB(path, mode=mode)
