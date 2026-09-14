"""RNSR in an AWS Lambda handler (serverless workers).

Package rnsr (query-time core only — a prebuilt corpus.db needs no
parsing stack) in the deployment image, set provider keys via Lambda
environment/Secrets Manager, and point EVENT["corpus_key"] at an
artifact in S3.

The single-file corpus.db is what makes this shape work: one GetObject,
then everything — typed tables, FTS, retained text — is local to /tmp.
Warm invocations reuse both the downloaded artifact and the runner.
"""

import os
from pathlib import Path

import boto3

import rnsr

_s3 = boto3.client("s3")
_runner = None
BUCKET = os.environ["CORPUS_BUCKET"]


def _corpus(key: str) -> Path:
    local = Path("/tmp") / key.replace("/", "_")
    if not local.exists():                       # cold start only
        _s3.download_file(BUCKET, key, str(local))
    return local


def handler(event, context):
    global _runner
    if _runner is None:
        _runner = rnsr.make_runner()

    corpus_db = _corpus(event["corpus_key"])
    questions = event.get("questions") or [event["question"]]

    answers = rnsr.answer_batch_sync(
        questions, corpus_db, runner=_runner,
        batch_size=int(event.get("batch_size", 8)),
        run_dir="/tmp/runs",                     # trajectories, if kept
    )
    return {
        "answers": [{"question": a.question, "answer": a.answer,
                     "status": a.status, "error": a.error}
                    for a in answers],
    }
