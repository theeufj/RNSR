"""Quoted source text must survive malformed provider JSON and bounded retries."""
import io
import json

import anthropic
import httpx
import openai
import pytest
from PIL import Image

from rnsr.ingest import llm_hooks
from rnsr.llm.base import LLMResponse, Usage
from rnsr.llm.governor import GovernedClient, Governor


class ScriptedVision:
    provider = "mock"

    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    async def vision(self, prompt, image_png, *, model, max_tokens):
        self.calls.append((prompt, max_tokens))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return LLMResponse(result, model, Usage(100, 10, 0.001))


@pytest.fixture
def source(tmp_path, monkeypatch):
    rendered = []

    def render(path, page):
        rendered.append((path, page))
        output = io.BytesIO()
        Image.new("RGB", (32, 32), "white").save(output, format="PNG")
        return output.getvalue()

    monkeypatch.setattr(llm_hooks, "rasterize_page", render)
    return tmp_path / "source.pdf", rendered


VALID = json.dumps({"blocks": [{"kind": "text", "text": 'Employment remains "at will."'}],
                    "tables": []})
INVALID = '{"blocks": [{"text": "Employment remains "at will.""}], "tables": []}'


def test_retry_preserves_quoted_source_and_meters_both_attempts(source):
    path, rendered = source
    client = ScriptedVision(INVALID, VALID)
    governor = Governor()
    hook = llm_hooks.make_page_transcriber(GovernedClient(client, governor), "vision")
    assert hook(path, [110])[110]["blocks"][0]["text"] == 'Employment remains "at will."'
    assert rendered == [(path, 110)]
    assert [tokens for _, tokens in client.calls] == [4096, 8192]
    assert llm_hooks._TRANSCRIBE_RETRY in client.calls[1][0]
    assert '\\"' in client.calls[1][0]
    assert governor.requests == 2
    assert governor.spent_usd == pytest.approx(0.002)
    assert governor.snapshot()["in_flight"] == 0


@pytest.mark.parametrize("failure", [
    RuntimeError("provider overloaded 529"), TimeoutError(),
    anthropic.APITimeoutError(request=httpx.Request("POST", "https://test.invalid")),
    openai.APIConnectionError(request=httpx.Request("POST", "https://test.invalid")),
    httpx.ReadTimeout("slow transport"),
])
def test_transient_failure_can_recover_once(source, failure):
    path, _ = source
    client = ScriptedVision(failure, VALID)
    assert llm_hooks.make_page_transcriber(client, "vision")(path, [1])[1] is not None
    assert len(client.calls) == 2


def test_permanent_provider_failure_does_not_retry(source):
    path, _ = source
    client = ScriptedVision(ValueError("invalid authentication"))
    assert llm_hooks.make_page_transcriber(client, "vision")(path, [1]) == {1: None}
    assert len(client.calls) == 1


def provider_error(provider, status):
    response = httpx.Response(status, request=httpx.Request('POST', 'https://test.invalid'))
    classes = {500: provider.InternalServerError, 400: provider.BadRequestError,
               401: provider.AuthenticationError}
    return classes[status]('offline provider failure', response=response, body={'error': 'fixture'})


@pytest.mark.parametrize('provider', [anthropic, openai])
def test_http_500_recovers_with_one_governed_retry(source, provider):
    path, _ = source
    client = ScriptedVision(provider_error(provider, 500), VALID)
    governor = Governor()
    hook = llm_hooks.make_page_transcriber(GovernedClient(client, governor), 'vision')
    assert hook(path, [7])[7]['blocks'][0]['text'] == 'Employment remains "at will."'
    assert len(client.calls) == governor.attempts == 2
    assert governor.requests == 1
    assert governor.spent_usd == pytest.approx(0.001)  # Only the successful response reports usage.
    assert governor.snapshot()['in_flight'] == 0


@pytest.mark.parametrize('provider', [anthropic, openai])
@pytest.mark.parametrize('status', [400, 401])
def test_http_request_and_authentication_errors_do_not_retry(source, provider, status):
    path, _ = source
    client = ScriptedVision(provider_error(provider, status), VALID)
    assert llm_hooks.make_page_transcriber(client, 'vision')(path, [7]) == {7: None}
    assert len(client.calls) == 1


def test_gemini_http_500_can_recover_once(source):
    from google.genai import errors

    path, _ = source
    failure = errors.ServerError(500, {'error': {'message': 'offline provider failure',
                                               'status': 'INTERNAL'}})
    client = ScriptedVision(failure, VALID)
    assert llm_hooks.make_page_transcriber(client, 'vision')(path, [7])[7] is not None
    assert len(client.calls) == 2


def test_repeated_http_500_remains_a_strict_corpus_health_failure(source, tmp_path):
    from rnsr.config import Settings
    from rnsr.errors import CorpusHealthError
    from rnsr.ingest.model import Element, ParsedDocument
    from rnsr.ingest.pipeline import ingest
    from rnsr.sdk import corpus_env

    path, _ = source
    path.write_bytes(b'Offline parser fixture')
    # Partial native text does not excuse an explicitly required scan transcription.
    parsed = ParsedDocument('scan', str(path), 'a'*64, 1, 'fixture',
                            elements=[Element('text', 'Partial native paragraph.', 1)],
                            scanned_pages=[1])
    client = ScriptedVision(provider_error(anthropic, 500), provider_error(anthropic, 500), VALID)
    governor = Governor()
    hook = llm_hooks.make_page_transcriber(GovernedClient(client, governor), 'vision')
    database = tmp_path/'failed-scan.db'
    report = ingest([path], database, parse=lambda path: parsed, transcriber=hook)
    assert len(client.calls) == governor.attempts == 2
    assert governor.requests == 0
    assert governor.snapshot()['in_flight'] == 0
    assert report.scanned_pages_transcribed == 0
    assert report.scanned_pages_untranscribed[0]['pages'] == [1]
    with pytest.raises(CorpusHealthError, match='scanned page'):
        corpus_env(database, settings=Settings(health_max_untranscribed_pages=0))


def test_invalid_json_exhausts_two_attempts_without_claiming_content(source):
    path, _ = source
    client = ScriptedVision(INVALID, INVALID)
    assert llm_hooks.make_page_transcriber(client, "vision")(path, [1]) == {1: None}
    assert len(client.calls) == 2


def test_retry_respects_existing_spend_ceiling(source):
    path, _ = source
    client = ScriptedVision(INVALID, VALID)
    governor = Governor(spend_ceiling_usd=0.0005)
    hook = llm_hooks.make_page_transcriber(GovernedClient(client, governor), "vision")
    assert hook(path, [1]) == {1: None}
    assert len(client.calls) == governor.requests == 1


def test_valid_transcription_makes_one_call(source):
    path, _ = source
    client = ScriptedVision(VALID)
    assert llm_hooks.make_page_transcriber(client, "vision")(path, [1])[1] is not None
    assert len(client.calls) == 1


def test_empty_transcription_retries_and_still_flags_missing_content(source):
    path, _ = source
    empty = '{"blocks": [], "tables": []}'
    client = ScriptedVision(empty, empty)
    assert llm_hooks.make_page_transcriber(client, "vision")(path, [1]) == {1: None}
    assert len(client.calls) == 2


def test_header_only_text_is_available_for_canonical_merge(source):
    path, _ = source
    payload = {"blocks": [], "tables": [{"header": ["Source-only $731"], "rows": []}]}
    client = ScriptedVision(json.dumps(payload))
    assert llm_hooks.make_page_transcriber(client, "vision")(path, [1])[1] == payload
    assert len(client.calls) == 1


def test_sparse_retry_crop_keeps_every_nonwhite_pixel():
    original = Image.new("RGB", (200, 300), "white")
    original.putpixel((80, 190), (0, 0, 0))
    original.putpixel((90, 200), (254, 254, 254))
    output = io.BytesIO()
    original.save(output, format="PNG")
    trimmed = llm_hooks._trim_blank_margins(output.getvalue())
    with Image.open(io.BytesIO(trimmed)) as image:
        assert image.size == (43, 43)
        assert image.getpixel((16, 16)) == (0, 0, 0)
        assert image.getpixel((26, 26)) == (254, 254, 254)


@pytest.mark.parametrize("payload", [
    {"blocks": ["paragraph"]}, {"blocks": [{"text": ["paragraph"]}]},
    {"blocks": [], "tables": "table"},
    {"blocks": [], "tables": [{"header": [], "rows": ["row"]}]},
    {"blocks": [], "tables": [{"header": [{}], "rows": []}]},
    {"blocks": [], "tables": [{"header": ["A"], "rows": [[{"cell": 1}]]}]},
    {"blocks": [{"text": "Heading"}], "tables": [{"header": [], "rows": [["$731"]]}]},
    {"blocks": [], "tables": [{"header": ["A"], "rows": [["$731", "$42"]]}]},
])
def test_invalid_transcription_shape_is_rejected(payload):
    assert llm_hooks._parse_transcription(json.dumps(payload)) is None
