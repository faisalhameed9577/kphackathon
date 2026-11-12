"""Gemini wrapper: retries, friendly errors, JSON parsing, embeddings (fake SDK client)."""
from __future__ import annotations

import math

import pytest

import services.gemini_service as gem
from services.embedding_service import EmbeddingService, LocalHashingEmbedder
from services.gemini_service import GeminiError, GeminiService, GeminiUnavailableError
from tests.conftest import FakeAPIError, FakeGeminiClient
from utils.helpers import parse_json_loose, redact_secrets


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(gem.time, "sleep", lambda s: None)


def test_unavailable_without_key(settings):
    service = GeminiService(settings)
    assert not service.available
    with pytest.raises(GeminiUnavailableError):
        service.generate_text("hi")


def test_generate_text_retries_temporary_errors(keyed_settings):
    client = FakeGeminiClient([FakeAPIError(503), "hello"])
    assert GeminiService(keyed_settings, client).generate_text("hi") == "hello"
    assert len(client.generate_calls) == 2


def test_rate_limit_gives_friendly_error_after_retries(keyed_settings):
    client = FakeGeminiClient([FakeAPIError(429), FakeAPIError(429)])
    with pytest.raises(GeminiError, match="rate limit"):
        GeminiService(keyed_settings, client).generate_text("hi")
    assert len(client.generate_calls) == keyed_settings.max_retries


@pytest.mark.parametrize("code,fragment", [(401, "API key"), (403, "API key"), (404, "GEMINI_MODEL")])
def test_permanent_errors_are_not_retried(keyed_settings, code, fragment):
    client = FakeGeminiClient([FakeAPIError(code, "secret AIzaFAKEKEYFORTESTS1234567890abcd")])
    with pytest.raises(GeminiError, match=fragment) as info:
        GeminiService(keyed_settings, client).generate_text("hi")
    assert len(client.generate_calls) == 1
    assert keyed_settings.gemini_api_key not in str(info.value)


def test_empty_answer_is_an_error(keyed_settings):
    with pytest.raises(GeminiError, match="empty"):
        GeminiService(keyed_settings, FakeGeminiClient([""])).generate_text("hi")


def test_generate_json_retries_once_on_invalid_json(keyed_settings):
    client = FakeGeminiClient(["not json at all", '```json\n{"ok": true}\n```'])
    assert GeminiService(keyed_settings, client).generate_json("x") == {"ok": True}
    with pytest.raises(GeminiError, match="valid structured output"):
        GeminiService(keyed_settings, FakeGeminiClient(["nope", "still nope"])).generate_json("x")


def test_parse_json_loose():
    assert parse_json_loose('Here you go: {"a": [1, 2]} thanks') == {"a": [1, 2]}
    assert parse_json_loose("[1, 2]") == [1, 2]
    with pytest.raises(ValueError):
        parse_json_loose("no json here")


def test_embeddings_are_batched_and_normalised(keyed_settings):
    client = FakeGeminiClient()
    vectors = GeminiService(keyed_settings, client).embed(["t"] * 230)
    assert client.embed_calls == [100, 100, 30] and len(vectors) == 230
    assert math.isclose(sum(v * v for v in vectors[0]), 1.0, rel_tol=1e-9)


def test_embedding_service_falls_back_to_local_when_gemini_fails(keyed_settings):
    client = FakeGeminiClient()
    client.models.embed_content = lambda **kw: (_ for _ in ()).throw(FakeAPIError(403))
    embedder = EmbeddingService(keyed_settings, GeminiService(keyed_settings, client))
    assert embedder.backend == "gemini" and embedder.collection_name == "papers_gemini_8"
    vectors = embedder.embed_documents(["glycaemic control in rural districts"])
    assert embedder.backend == "local" and len(vectors[0]) == 512
    assert "Gemini embeddings failed" in embedder.fallback_reason
    assert embedder.collection_name == "papers_local_512"  # separate collection, no mixed dimensions


def test_local_embedder_is_deterministic_and_lexical():
    emb = LocalHashingEmbedder()
    a, b, c = emb.embed("diabetes prevalence in rural areas"), emb.embed("rural diabetes prevalence"), emb.embed("quantum gravity lattice")
    assert a == emb.embed("diabetes prevalence in rural areas")
    cos = lambda x, y: sum(i * j for i, j in zip(x, y))  # noqa: E731 - vectors are unit length
    assert cos(a, b) > 0.5 > cos(a, c)


def test_secrets_are_redacted():
    text = "GET https://x/y?api_key=abc123&tool=t failed with AIzaSyA1234567890123456789012 and mykey"
    cleaned = redact_secrets(text, ["mykey"])
    assert "abc123" not in cleaned and "AIzaSy" not in cleaned and "mykey" not in cleaned


def test_settings_repr_hides_keys(keyed_settings):
    assert "AIza" not in repr(keyed_settings) and "ncbi-secret" not in repr(keyed_settings)
