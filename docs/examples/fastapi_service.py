"""RNSR behind FastAPI — an async REST surface over one shared runner.

    pip install -e ".[service]"
    uvicorn docs.examples.fastapi_service:app --workers 1

This is the pattern for a service that owns its corpora and answers on
request. (rnsr's built-in `rnsr serve` is a job-queue surface; this
example is the request/response shape.) The RLM loop is provider-I/O
bound, so one uvicorn worker multiplexes many in-flight questions; the
governor's in-flight cap is the backpressure.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import rnsr

RUNNER = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    RUNNER["r"] = rnsr.make_runner()   # resolve providers once, at startup
    yield


app = FastAPI(lifespan=lifespan)


class AnswerRequest(BaseModel):
    corpus_db: str          # path to a prebuilt corpus.db artifact
    question: str


class BatchRequest(BaseModel):
    corpus_db: str
    questions: list[str]
    batch_size: int = 8
    consensus: int = 1


@app.post("/answer")
async def answer(req: AnswerRequest):
    result = await rnsr.answer(req.question, req.corpus_db, runner=RUNNER["r"])
    if result.status == "error":
        raise HTTPException(502, detail="loop error — see trajectory")
    return {"answer": result.answer, "status": result.status,
            "spend_usd": result.ledger["spend_usd"],
            "trajectory": result.trajectory_path}


@app.post("/answer-batch")
async def answer_batch(req: BatchRequest):
    answers = await rnsr.answer_batch(
        req.questions, req.corpus_db, runner=RUNNER["r"],
        batch_size=req.batch_size, consensus=req.consensus)
    return [{"question": a.question, "answer": a.answer, "status": a.status,
             "agreement": a.agreement, "contested": a.contested,
             "error": a.error} for a in answers]
