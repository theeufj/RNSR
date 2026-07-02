"""DocDB-RLM: typed-environment recursive language model system.

Public API (filled in as phases land):
    ingest(sources, out_db, ...)  -> IngestReport   (Phase A)
    open_corpus(path)             -> CorpusDB       (Phase A)
    answer(question, corpus, ...) -> QueryResult    (Phase B/C)
"""

__version__ = "1.0.0a0"
