"""RNSR inside Celery tasks.

    pip install -e . celery[redis]
    celery -A docs.examples.celery_worker worker --concurrency 4

Notes that matter in production:
- One RootRunner per worker process (built lazily below): provider
  resolution is cheap but not free, and the runner is safe to reuse.
- The in-memory governor caps THIS process. Workers on many machines that
  must share one RPM/spend envelope implement GovernorProtocol over Redis
  and `rnsr.llm.governor.install()` it at worker startup — see the class
  docstring in rnsr/llm/governor.py for the contract.
- corpus.db is a read-only artifact at query time: putting it on shared
  storage (EFS/NFS) or copying it to each worker are both fine.
"""

from celery import Celery

import rnsr

app = Celery("rnsr_worker", broker="redis://localhost:6379/0",
             backend="redis://localhost:6379/1")

_runner = None


def runner():
    global _runner
    if _runner is None:
        _runner = rnsr.make_runner()   # reads Settings.from_env()
    return _runner


@app.task(bind=True, max_retries=2)
def answer_question(self, corpus_db: str, question: str) -> dict:
    result = rnsr.answer_sync(question, corpus_db, runner=runner())
    if result.status == "error":
        raise self.retry(countdown=30)
    return {
        "answer": result.answer,
        "status": result.status,
        "spend_usd": result.ledger["spend_usd"],
        "trajectory": result.trajectory_path,   # the audit record
    }


@app.task
def answer_form(corpus_db: str, questions: list[str]) -> list[dict]:
    """Many related questions: shared exploration, ~4x cheaper than solo."""
    answers = rnsr.answer_batch_sync(questions, corpus_db,
                                     runner=runner(), batch_size=8)
    return [{"question": a.question, "answer": a.answer,
             "status": a.status, "error": a.error} for a in answers]
