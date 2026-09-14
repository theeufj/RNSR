"""DocDB-RLM: typed-environment recursive language model system.

Public API (imported lazily — heavy modules load on first use):
    open_corpus(path, mode="ro")            -> CorpusDB
    answer(question, corpus_db, ...)        -> QueryResult       (async)
    answer_batch(questions, corpus_db, ...) -> list[BatchAnswer] (async)
    answer_sync / answer_batch_sync         — wrappers for sync workers
    make_runner(settings)                   -> RootRunner (reusable across calls)

Ingestion is ``rnsr.sdk.ingest`` (or ``rnsr.ingest.pipeline.ingest``): the
``rnsr.ingest`` subpackage shadows any root-level function of that name, so
the callable deliberately does not live here. See rnsr.sdk for full
signatures and worker-integration notes.
"""

# Single source of truth for the package version: pyproject declares
# `dynamic = ["version"]` and hatch reads it from here at build time.
# (Reading installed metadata instead goes stale in editable installs,
# and this value is stamped into every corpus manifest.)
__version__ = "1.0.0a5"

__all__ = [
    "BatchAnswer",
    "__version__",
    "answer",
    "answer_batch",
    "answer_batch_sync",
    "answer_sync",
    "build_questions",
    "corpus_env",
    "fan_out",
    "make_runner",
    "open_corpus",
    "score_answers",
    "score_answers_sync",
]

_SDK_ATTRS = frozenset({
    "BatchAnswer", "answer", "answer_batch", "answer_batch_sync",
    "answer_sync", "build_questions", "corpus_env", "fan_out",
    "make_runner", "open_corpus", "score_answers", "score_answers_sync",
})


def __getattr__(name: str):
    if name in _SDK_ATTRS:
        from rnsr import sdk

        return getattr(sdk, name)
    raise AttributeError(f"module 'rnsr' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | {"sdk"})
