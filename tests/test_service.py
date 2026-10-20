"""HTTP service surface: health, readiness, metrics, job lifecycle."""

import time

import pytest

from rnsr.config import Settings

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from rnsr.service import JobStore  # noqa: E402


def wait_for_job(client, job_id):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["state"] in ("done", "failed"):
            return job
        time.sleep(0.005)
    pytest.fail("job did not finish within 2 seconds")


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
    app = create_app(Settings(provider="anthropic", service_token="test-token",
                              service_corpus_root=tmp_path), run_dir=tmp_path / "runs")
    with TestClient(app, headers={"Authorization": "Bearer test-token"}) as c:
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
        assert set(body["roles"]) == {"root", "sub"}
        assert body["corpus_root_readable"] is True

    def test_readyz_rejects_a_key_for_the_wrong_configured_provider(self, tmp_path, monkeypatch):
        from rnsr.service import create_app

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        app = create_app(Settings(provider="anthropic", service_token="test",
            service_corpus_root=tmp_path), run_dir=tmp_path / "runs")
        with TestClient(app, headers={"Authorization": "Bearer test"}) as client:
            response = client.get("/readyz")
        assert response.status_code == 503
        assert response.json()["detail"]["providers"] == ["openai"]
        assert response.json()["detail"]["roles"] == {}

    def test_readyz_requires_both_roles_and_never_calls_a_provider(self, client, monkeypatch):
        import httpx

        from rnsr.llm.router import Router

        async def no_live_requests(*args, **kwargs):
            pytest.fail("readiness must not make a provider request")

        monkeypatch.setattr(httpx.AsyncClient, "send", no_live_requests)
        assert client.get("/readyz").status_code == 200
        resolve = Router.resolve

        def unavailable_sub(self, role):
            if role == "sub":
                raise RuntimeError("private provider configuration details")
            return resolve(self, role)

        monkeypatch.setattr(Router, "resolve", unavailable_sub)
        response = client.get("/readyz")
        assert response.status_code == 503
        assert set(response.json()["detail"]["roles"]) == {"root"}
        assert "private provider" not in response.text

    def test_readyz_503_without_a_provider_key(self, tmp_path, monkeypatch):
        from rnsr.service import create_app

        for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.delenv(env, raising=False)
        app = create_app(Settings(provider="anthropic", service_token="test-token",
                              service_corpus_root=tmp_path), run_dir=tmp_path / "runs")
        with TestClient(app, raise_server_exceptions=False,
                        headers={"Authorization": "Bearer test-token"}) as c:
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
        assert "corpus_db unavailable" in r.json()["detail"]

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

        got = wait_for_job(client, job_id)
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
        got = wait_for_job(client, job_id)
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
    def test_running_oldest_does_not_block_eviction(self):
        from rnsr.service import Job

        store = JobStore(max_jobs=2)
        store.add(Job(id="running", questions=["q"], corpus_db="x",
                      state="running", submitted_at=0))
        for i in range(10):
            store.add(Job(id=f"done{i}", questions=["q"], corpus_db="x",
                          state="done", submitted_at=i + 1))
        assert len(store.recent()) == 2
        assert store.get("running") is not None
        assert store.get("done9") is not None

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
        with pytest.raises(OverflowError):
            store.add(Job(id="new", questions=["q"], corpus_db="x",
                          state="queued", submitted_at=1.0))
        assert store.get("running") is not None


class TestRunJob:
    async def test_uses_the_batched_runner(self, tmp_path, corpus_db,
                                          monkeypatch):
        """The service must answer through the same loop as the CLI, batching
        included — otherwise the validated accuracy path is not what ships."""
        from rnsr.harness.loop import BatchQueryResult, QueryResult
        from rnsr.service import Job, run_job

        seen = {}

        class FakeRunner:
            async def run_batch(self, questions, env, run_dir=None, query_id=None):
                seen["questions"] = questions
                seen["corpus_db"] = env.corpus_db
                return BatchQueryResult(
                    answers={qid: "7714" for qid, _ in questions},
                    result=QueryResult(answer=None, status="final", final=None,
                                       ledger={"spend_usd": 0.0, "sub_calls": 0},
                                       trajectory_path="", iterations=1))

        monkeypatch.setattr("rnsr.sdk.make_runner", lambda settings=None: FakeRunner())

        job = Job(id="j1", questions=["a?", "b?"], corpus_db=str(corpus_db))
        await run_job(job, Settings(), tmp_path / "runs")
        assert job.state == "done", job.error
        assert job.answers == ["7714", "7714"]
        assert [q for _, q in seen["questions"]] == ["a?", "b?"]
        assert seen["corpus_db"] == str(corpus_db)
        assert job.tiers == [None, None]

    async def test_missing_corpus_marks_job_failed(self, tmp_path):
        from rnsr.service import Job, run_job

        job = Job(id="j2", questions=["q?"], corpus_db=str(tmp_path / "nope.db"))
        await run_job(job, Settings(), tmp_path / "runs")
        assert job.state == "failed"
        assert job.error


class TestAuditResults:
    @pytest.mark.parametrize("encrypted", [False, True])
    def test_results_include_evidence_and_authorized_audit_without_host_paths(
            self, client, corpus_db, tmp_path, monkeypatch, encrypted):
        import json

        from rnsr.harness.trajectory import TrajectoryWriter
        from rnsr.sdk import BatchAnswer

        key = ""
        if encrypted:
            pytest.importorskip("cryptography")
            from cryptography.fernet import Fernet

            key = Fernet.generate_key().decode()
            client.app.state.settings.trajectory_key = key
        source_path = str(tmp_path.parent / "private-client-source.pdf")
        outside = tmp_path / "outside.jsonl"
        outside.write_text('{"kind":"outside"}\n')

        async def answers(questions, corpus_db, **kwargs):
            directory = kwargs["run_dir"]
            with TrajectoryWriter(directory, "q", key=key) as writer:
                writer.event("cell", ok=True, stdout="private document content",
                    code=source_path, source_path=source_path)
                writer.event("cell", phase=source_path, status=source_path, n=source_path,
                    ok="true", timeout_s=source_path)
                trajectory = writer.path
            return [BatchAnswer(question=questions[0], answer="7714", status="final",
                evidence={"quotes_total": 1, "quotes_verified": 1, "docs_cited": ["doc"]},
                citations=[{"quote": "The secret number is 7714.", "matched": True,
                    "doc_id": "doc", "char_start": 0, "char_end": 26,
                    "source_context": {"page": 1, "source_path": source_path,
                        "heading_paths": ["Fiscal2021"],
                        "sections": [{"heading_path": "Fiscal2021"}]},
                    "matches": [{"doc_id": "doc", "source_context": {"source_path": source_path}}]}],
                health={"grade": "ok"}, claim_review={"status": "supported", "required": True},
                trajectory_paths=[str(trajectory), str(outside)])]

        monkeypatch.setattr("rnsr.sdk.answer_batch", answers)
        response = client.post("/jobs", json={"questions": ["Number?"], "corpus_db": str(corpus_db)})
        assert response.status_code == 202
        job_id = response.json()["id"]
        job = wait_for_job(client, job_id)
        result = job["results"][0]
        assert result["evidence"]["quotes_verified"] == 1
        assert result["citations"][0]["source_context"] == {"page": 1,
            "heading_paths": ["Fiscal2021"], "sections": [{"heading_path": "Fiscal2021"}]}
        assert result["citations"][0]["char_start"] == 0
        assert result["health"] == {"grade": "ok"}
        assert result["claim_review"] == {"status": "supported", "required": True}
        assert source_path not in json.dumps(job)
        assert str(tmp_path) not in json.dumps(job)
        assert len(result["audit_refs"]) == 1
        reference = result["audit_refs"][0]
        assert reference.startswith(f"/jobs/{job_id}/audit/")
        assert client.get(reference, headers={"Authorization": ""}).status_code == 401
        audit = client.get(reference)
        assert audit.status_code == 200
        assert audit.json()["content"] == "metadata"
        event = audit.json()["events"][0]
        assert set(event) == {"kind", "t", "ok"}
        assert event["kind"] == "cell" and event["ok"] is True
        forged = audit.json()["events"][1]
        assert set(forged) == {"kind", "t"}
        assert source_path not in audit.text and "private document content" not in audit.text
        assert client.get(f"/jobs/other/audit/{reference.rsplit('/', 1)[1]}").status_code == 404
        trajectory = next(iter(client.app.state.store.get(job_id).audit_paths.values()))
        trajectory.unlink()
        trajectory.symlink_to(outside)
        assert client.get(reference).status_code == 404

    def test_service_jobs_apply_retention_to_previous_job_directories(
            self, client, corpus_db, tmp_path, monkeypatch):
        import os

        from rnsr.sdk import BatchAnswer

        stale = tmp_path / "runs" / "service" / "previous-job" / "q.jsonl"
        stale.parent.mkdir(parents=True)
        stale.write_text("{}\n")
        expired = time.time() - 3 * 86400
        os.utime(stale, (expired, expired))
        client.app.state.settings.trajectory_retention_days = 1

        async def answers(questions, corpus_db, **kwargs):
            assert not stale.exists()
            return [BatchAnswer(question=questions[0], answer="7714", status="final")]

        monkeypatch.setattr("rnsr.sdk.answer_batch", answers)
        response = client.post("/jobs", json={"questions": ["Number?"], "corpus_db": str(corpus_db)})
        assert wait_for_job(client, response.json()["id"])["state"] == "done"
        assert not stale.exists()


class TestAuthorization:
    @pytest.mark.parametrize("path", ["/readyz", "/metrics", "/jobs", "/jobs/id", "/jobs/id/audit/ref"])
    def test_protected_reads(self, client, path):
        assert client.get(path, headers={"Authorization": ""}).status_code == 401
        assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401

    def test_protected_submit(self, client, corpus_db):
        response = client.post("/jobs", headers={"Authorization": ""}, json={
            "questions": ["q"], "corpus_db": str(corpus_db)})
        assert response.status_code == 401
        assert client.get("/jobs").json()["jobs"] == []

    def test_liveness_is_public(self, client):
        assert client.get("/healthz", headers={"Authorization": ""}).status_code == 200

    def test_startup_requires_auth_and_root(self, tmp_path):
        from rnsr.service import create_app

        with pytest.raises(ValueError, match="SERVICE_TOKEN"):
            create_app(Settings(), run_dir=tmp_path)
        with pytest.raises(ValueError, match="SERVICE_CORPUS_ROOT"):
            create_app(Settings(service_token="test"), run_dir=tmp_path)

    def test_symlink_escape_and_directory_rejected(self, client, tmp_path):
        outside = tmp_path.parent / "outside.db"
        outside.write_bytes(b"sentinel")
        link = tmp_path / "escape.db"
        link.symlink_to(outside)
        for path in (link, outside, tmp_path):
            response = client.post("/jobs", json={"questions": ["q"], "corpus_db": str(path)})
            assert response.status_code == 400
        assert client.get("/jobs").json()["jobs"] == []

    def test_listing_does_not_expose_question_answer_or_errors(self, client):
        from rnsr.service import Job

        client.app.state.store.add(Job(id="private", questions=["private question"],
                                      corpus_db="private.db", state="failed",
                                      error="private source excerpt"))
        data = client.get("/jobs").json()["jobs"][0]
        assert data["questions"] == 1
        assert "error" not in data
        assert "results" not in data
        assert "private question" not in str(data)

    @pytest.mark.parametrize("limit", [-1, 0, 201])
    def test_listing_limits_are_bounded(self, client, limit):
        assert client.get(f"/jobs?limit={limit}").status_code == 422
