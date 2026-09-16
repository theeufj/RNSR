"""Reusable corpus preparation and resumable answer persistence."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

from rnsr.config import Settings
from rnsr.db.artifact import CorpusDB
from rnsr.sdk import BatchAnswer


def make_ingest_hooks(settings: Settings, *, llm: bool = False,
                      no_transcribe: bool = False) -> dict:
    from rnsr.ingest.cost_estimate import resolve_transcriber

    transcriber, _ = resolve_transcriber(settings, no_transcribe=no_transcribe)
    prose_checker = vision = None
    if llm:
        from rnsr.ingest.llm_hooks import (
            make_page_transcriber,
            make_prose_checker,
            make_vision_extractor,
        )
        from rnsr.llm.router import Router

        router = Router(settings)
        sub, vis = router.resolve("sub"), router.resolve("vision")
        prose_checker = make_prose_checker(sub.client, sub.model,
                                           concurrency=settings.sub_concurrency)
        vision = make_vision_extractor(vis.client, vis.model)
        if transcriber is None and not no_transcribe:
            transcriber = make_page_transcriber(vis.client, vis.model)
    return {"transcriber": transcriber, "prose_checker": prose_checker, "vision": vision}


def prepare_corpus(corpus_dir: Path, work_dir: Path, settings: Settings, *,
                   fast_ingest: bool = False, llm: bool = False,
                   no_transcribe: bool = False,
                   progress: Callable[[str], None] = lambda _: None) -> Path:
    from rnsr.db.schema import validate_frozen
    from rnsr.ingest.dispatch import is_ingestable
    from rnsr.ingest.lifecycle import append, file_index, replace_document
    from rnsr.ingest.parse import content_sha256

    files = sorted(p for p in corpus_dir.rglob("*")
                   if p.is_file() and not p.name.startswith(".") and is_ingestable(p))
    if not files:
        raise ValueError(f"no ingestable files under {corpus_dir}")
    path = work_dir / "corpora" / "corpus.db"
    hooks = None

    def get_hooks():
        nonlocal hooks
        if hooks is None:
            hooks = make_ingest_hooks(settings, llm=llm, no_transcribe=no_transcribe)
        return hooks

    if path.exists():
        # Invalid caches are not deleted: preserve the artifact for diagnosis
        # or explicit migration, instead of silently replacing user data.
        with CorpusDB(path) as corpus:
            validate_frozen(corpus.conn)
        index = file_index(path)
        new_files = []
        for source in files:
            rec = index.get(str(source.resolve())) or index.get(source.name)
            if rec is None:
                new_files.append(source)
            elif content_sha256(source) != rec.get("content_sha256"):
                progress(f"replacing changed {source.name}")
                replace_document(path, rec["doc_id"], source, config=settings,
                                 transcriber=get_hooks()["transcriber"])
        if new_files:
            progress(f"appending {len(new_files)} new document(s)")
            append(new_files, path, config=settings,
                   transcriber=get_hooks()["transcriber"])
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        if fast_ingest:
            from rnsr.ingest.bulk import ingest_bulk
            stats = ingest_bulk(files, path, config=settings,
                                transcriber=get_hooks()["transcriber"], progress=progress)
            progress(f"bulk ingest: {stats}")
        else:
            from rnsr.ingest.pipeline import ingest
            report = ingest(files, path, config=settings, **get_hooks())
            progress(f"ingested: {report.n_chunks} chunks, validation pass rate {report.validation_pass_rate:.0%}")
    return path


def corpus_revision(path: Path) -> str:
    """Checkpoint identity follows source content, unaffected by annotations."""
    with CorpusDB(path) as corpus:
        rows = [tuple(r) for r in corpus.conn.execute(
            "SELECT doc_id, content_sha256, parser, ingested_at FROM documents ORDER BY doc_id")]
        revision = {"format": corpus.manifest_get("format_version"), "documents": rows}
        digest = hashlib.sha256(json.dumps(revision).encode())
        for row in corpus.conn.execute("SELECT doc_id, page, text FROM doc_text ORDER BY doc_id, page"):
            digest.update(json.dumps(tuple(row), ensure_ascii=False).encode())
    return digest.hexdigest()


class AnswerCheckpoint:
    """Persist complete evidence, and only resume successful answers for this corpus.

    A partial last JSONL record can result from interruption and is discarded;
    malformed earlier records are errors. Older checkpoints without corpus/evidence
    identity cannot safely be reused and are deliberately ignored.
    """

    def __init__(self, path: Path, questions: Sequence[str], revision: str):
        self.path, self.questions, self.revision = path, list(questions), revision

    def load(self) -> dict[int, BatchAnswer]:
        out = {}
        if not self.path.exists():
            return out
        raw = self.path.read_text(encoding="utf-8")
        lines = raw.splitlines(keepends=True)
        valid_lines = []
        for offset, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if offset == len(lines) - 1 and not line.endswith("\n"):
                    self.path.write_text("".join(valid_lines), encoding="utf-8")
                    break
                raise ValueError(f"invalid answer checkpoint record {offset + 1}") from None
            valid_lines.append(line if line.endswith("\n") else line + "\n")
            i = record.get("i")
            if (type(i) is not int or not 0 <= i < len(self.questions)
                    or record.get("q") != self.questions[i]
                    or record.get("revision") != self.revision):
                continue
            result = BatchAnswer(**record["result"])
            if result.status in {"final", "recovered"} and result.answer:
                out[i] = result
            else:
                out.pop(i, None)
        if raw and not raw.endswith("\n") and len(valid_lines) == len(lines):
            self.path.write_text("".join(valid_lines), encoding="utf-8")
        return out

    def record(self, i: int, result: BatchAnswer) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"i": i, "q": self.questions[i], "revision": self.revision,
                  "result": asdict(result)}
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record) + "\n")
            output.flush()
