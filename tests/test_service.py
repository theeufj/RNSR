"""HTTP service surface: health, readiness, metrics, job lifecycle."""

import pytest

from rnsr.config import Settings

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from rnsr.service import JobStore  # noqa: E402


@pytest.fixture
def corpus_db(tmp_path):
    """A real (tiny) corpus.db so /jobs validation and ingest paths work."""
    from rnsr.ingest.model import Element, ParsedDocument
    from rnsr.ingest.pipeline import ingest

    def parse(path):
        return ParsedDocument(
            doc_id="doc", source_path=str(path), sha256="d" * 64, n_pages=1,
            parser="fake",
            elements=[Element("text", "The secret number is 7714.", 1)],
            tables=[])

    out = tmp_path / "corpus.db"
    ingest([tmp_path / "doc.pdf"], out, parse=parse)
    return out


@pytest.fixture
def client(tmp_path, monkeypatch):
    from rnsr.service import create_app

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    app = create_app(Settings(provider="anthropic"), run_dir=tmp_path / "runs")
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_healthz(self, client):
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_readyz_reports_providers(self, client):
        r = client.get("/readyz")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ready"
        assert "anthropic" in body["providers"]
        assert body["run_dir_writable"] is True

    def test_readyz_503_without_a_provider_key(self, tmp_path, monkeypatch):
        from rnsr.service import create_app

        for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.delenv(env, raising=False)
        app = create_app(Settings(provider="anthropic"), run_dir=tmp_path / "runs")
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get("/readyz")
        assert r.status_code == 503
        assert r.json()["detail"]["status"] == "not ready"

    def test_metrics_endpoint_exposes_counters(self, client):
        from rnsr.obs import metrics

        metrics().incr("queries_finished", status="final")
        body = client.get("/metrics").json()
        assert body["metrics"]["counters"]["queries_finished{status=final}"] >= 1
        assert "spend_usd" in body["provider"]


class TestJobs:
    def test_rejects_unknown_corpus(self, client):
        r = client.post("/jobs", json={"questions": ["q?"],
                                       "corpus_db": "/nope/corpus.db"})
        assert r.status_code == 400
        assert "corpus_db not found" in r.json()["detail"]

    def test_rejects_empty_question_list(self, client, corpus_db):
        r = client.post("/jobs", json={"questions": [],
                                       "corpus_db": str(corpus_db)})
        assert r.status_code == 422

    def test_job_runs_and_returns_answers(self, client, corpus_db, monkeypatch):
        import rnsr.service as svc

        async def fake_run_job(job, settings, run_dir):
            job.state = "done"
            job.answers = ["7714"] * len(job.questions)
            job.statuses = ["final"] * len(job.questions)
            job.agreement = [1.0] * len(job.questions)

        monkeypatch.setattr(svc, "run_job", fake_run_job)
        r = client.post("/jobs", json={"questions": ["What is the number?"],
                                       "corpus_db": str(corpus_db)})
        assert r.status_code == 202
        job_id = r.json()["id"]

        got = client.get(f"/jobs/{job_id}").json()
        assert got["state"] == "done"
        assert got["results"][0]["answer"] == "7714"
        assert got["results"][0]["question"] == "What is the number?"

    def test_failed_job_reports_the_error(self, client, corpus_db, monkeypatch):
        import rnsr.service as svc

        async def boom(job, settings, run_dir):
            job.state, job.error = "failed", "RuntimeError: provider down"

        monkeypatch.setattr(svc, "run_job", boom)
        job_id = client.post("/jobs", json={"questions": ["q?"],
                                            "corpus_db": str(corpus_db)}
                             ).json()["id"]
        got = client.get(f"/jobs/{job_id}").json()
        assert got["state"] == "failed"
        assert "provider down" in got["error"]

    def test_unknown_job_is_404(self, client):
        assert client.get("/jobs/deadbeef").status_code == 404

    def test_job_listing_omits_answers(self, client, corpus_db, monkeypatch):
        import rnsr.service as svc

        async def fake_run_job(job, settings, run_dir):
            job.state = "done"
            job.answers, job.statuses, job.agreement = ["a"], ["final"], [1.0]

        monkeypatch.setattr(svc, "run_job", fake_run_job)
        client.post("/jobs", json={"questions": ["q?"],
                                   "corpus_db": str(corpus_db)})
        listing = client.get("/jobs").json()["jobs"]
        assert len(listing) == 1
        assert "results" not in listing[0]


class TestJobStore:
    def test_evicts_oldest_finished_job(self):
        from rnsr.service import Job

        store = JobStore(max_jobs=2)
        for i in range(3):
            store.add(Job(id=f"j{i}", questions=["q"], corpus_db="x",
                          state="done", submitted_at=float(i)))
        assert store.get("j0") is None
        assert store.get("j2") is not None

    def test_never_evicts_running_work(self):
        from rnsr.service import Job

        store = JobStore(max_jobs=1)
        store.add(Job(id="running", questions=["q"], corpus_db="x",
                      state="running", submitted_at=0.0))
        store.add(Job(id="new", questions=["q"], corpus_db="x",
                      state="queued", submitted_at=1.0))
        assert store.get("running") is not None


class TestRunJob:
    async def test_uses_the_batched_runner(self, tmp_path, corpus_db,
                                          monkeypatch):
        """The service must answer through the same loop as the CLI, batching
        included — otherwise the validated accuracy path is not what ships."""
        import rnsr.llm.router as router_mod
        from rnsr.harness.loop import BatchQueryResult, QueryResult
        from rnsr.service import Job, run_job

        seen = {}

        class FakeRunner:
            def __init__(self, **kwargs):
                pass

            async def run_batch(self, questions, env, run_dir=None, query_id=None):
                seen["questions"] = questions
                seen["corpus_db"] = env.corpus_db
                return BatchQueryResult(
                    answers={qid: "7714" for qid, _ in questions},
                    result=QueryResult(answer=None, status="final", final=None,
                                       ledger={"spend_usd": 0.0, "sub_calls": 0},
                                       trajectory_path="", iterations=1))

        class FakeResolved:
            client, model = object(), "m"

        monkeypatch.setattr(router_mod.Router, "__init__",
                            lambda self, settings=None: None)
        monkeypatch.setattr(router_mod.Router, "resolve",
                            lambda self, role: FakeResolved())
        monkeypatch.setattr("rnsr.harness.loop.RootRunner", FakeRunner)

        job = Job(id="j1", questions=["a?", "b?"], corpus_db=str(corpus_db))
        await run_job(job, Settings(), tmp_path / "runs")
        assert job.state == "done", job.error
        assert job.answers == ["7714", "7714"]
        assert [q for _, q in seen["questions"]] == ["a?", "b?"]
        assert seen["corpus_db"] == str(corpus_db)

    async def test_missing_corpus_marks_job_failed(self, tmp_path):
        from rnsr.service import Job, run_job

        job = Job(id="j2", questions=["q?"], corpus_db=str(tmp_path / "nope.db"))
        await run_job(job, Settings(), tmp_path / "runs")
        assert job.state == "failed"
        assert job.error
