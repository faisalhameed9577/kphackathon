"""Shared test fixtures. No network and no paid API is needed by any test."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List, Optional

import pytest

from config.settings import Settings
from database.models import PaperRecord
from services.container import Services, build_services

FIXTURES = Path(__file__).parent / "fixtures"


# ------------------------------------------------------------- fake HTTP layer
class FakeResponse:
    def __init__(self, status_code: int = 200, json_data: Any = None, text: str = "", headers: Optional[dict] = None):
        self.status_code = status_code
        self._json = json_data
        self.text = text if text or json_data is None else json.dumps(json_data)
        self.headers = headers or {}

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class FakeSession:
    """requests.Session stand-in. ``handler(url, params)`` returns a FakeResponse (or raises)."""

    def __init__(self, handler: Callable[[str, Optional[dict]], FakeResponse]):
        self.handler = handler
        self.calls: List[tuple] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return self.handler(url, params)


# ------------------------------------------------------------- fake Gemini SDK
class FakeGeminiClient:
    """Mimics the parts of google-genai used by GeminiService."""

    def __init__(self, text_replies: Optional[List[Any]] = None, embed_dim: int = 8):
        self.text_replies = list(text_replies or [])
        self.generate_calls: List[str] = []
        self.embed_calls: List[int] = []
        self.embed_dim = embed_dim
        self.models = SimpleNamespace(generate_content=self._generate, embed_content=self._embed)

    def _generate(self, model, contents, config):
        self.generate_calls.append(contents)
        if not self.text_replies:
            raise RuntimeError("no scripted reply")
        reply = self.text_replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            reply = reply(contents)
        return SimpleNamespace(text=reply if isinstance(reply, str) else json.dumps(reply))

    def _embed(self, model, contents, config):
        self.embed_calls.append(len(contents))
        vectors = [SimpleNamespace(values=[float((i + j) % 5 + 1) for j in range(self.embed_dim)])
                   for i, _ in enumerate(contents)]
        return SimpleNamespace(embeddings=vectors)


class FakeAPIError(Exception):
    """Looks like google.genai.errors.APIError (has a numeric ``code``)."""

    def __init__(self, code: int, message: str = "boom"):
        super().__init__(message)
        self.code = code


# ------------------------------------------------------------------ settings/services
@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings.for_testing(tmp_path / "data")


@pytest.fixture
def keyed_settings(tmp_path) -> Settings:
    return Settings.for_testing(tmp_path / "data_keyed", gemini_api_key="AIzaFAKEKEYFORTESTS1234567890abcd",
                                ncbi_api_key="ncbi-secret-key", ncbi_email="test@example.org",
                                embedding_dimension=8)


@pytest.fixture
def services(settings) -> Services:
    return build_services(settings)


# ------------------------------------------------------------------ sample data
@pytest.fixture
def pubmed_xml() -> str:
    return (FIXTURES / "pubmed_sample.xml").read_text(encoding="utf-8")


@pytest.fixture
def europepmc_json() -> dict:
    return json.loads((FIXTURES / "europepmc_sample.json").read_text(encoding="utf-8"))


@pytest.fixture
def jats_xml() -> str:
    return (FIXTURES / "jats_sample.xml").read_text(encoding="utf-8")


def make_paper(i: int = 1, **overrides) -> PaperRecord:
    """A SYNTHETIC paper for tests (clearly fake title/DOI)."""
    base = dict(
        paper_id=i, title=f"[TEST FIXTURE] Synthetic study number {i} on fictional glycaemic outcomes",
        abstract=("Background: Synthetic background text for testing purposes only. "
                  "Methods: A cross-sectional survey of 300 participants was simulated. "
                  "Results: Mean HbA1c was higher in the fictional group (p < 0.05). "
                  "Conclusions: More simulated research is needed. A limitation is the small sample size."),
        authors=["Testauthor A", "Fixture B", "Sample G"], journal="Journal of Synthetic Testing",
        year=2015 + i, doi=f"10.0000/test.{i:03d}", pmid=str(90000000 + i), sources=["PubMed"], url="")
    base.update(overrides)
    return PaperRecord(**base)


@pytest.fixture
def sample_papers() -> List[PaperRecord]:
    return [
        make_paper(1),
        make_paper(2, abstract="Methods: This randomized controlled trial enrolled 120 women in a simulated clinic. "
                               "Results: Outcomes improved in the intervention arm. Conclusions: Promising but preliminary."),
        make_paper(3, abstract="A prospective cohort followed 1,200 simulated patients. Findings showed higher rates in "
                               "older adults. Results cannot be generalized beyond this simulated setting."),
        make_paper(4, abstract="Cross-sectional survey of 450 participants in a fictional city. Prevalence was 12%."),
        make_paper(5, abstract=""),
    ]


@pytest.fixture
def saved_topic(services, sample_papers):
    """A topic with the sample papers saved in SQLite (no Chroma indexing). Returns (topic_id, papers)."""
    topic_id = services.topics.get_or_create("Fictional glycaemic outcomes", region="Fictionland", year_from=2010, year_to=2025)
    saved = []
    for p in sample_papers:
        p = p.model_copy(update={"paper_id": None})
        pid = services.papers.upsert(p)
        services.papers.link_to_topic(topic_id, pid)
        saved.append(services.papers.get(pid))
    return topic_id, saved
