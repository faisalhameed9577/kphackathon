"""ChromaDB storage, RAG retrieval, answer safety rules and citation helpers."""
from __future__ import annotations

import pytest

from config.settings import Settings
from services.container import build_services
from tests.conftest import FakeAPIError, FakeGeminiClient, make_paper
from utils.citation_utils import (extract_citation_numbers, filter_paper_labels, format_inline, format_reference,
                                  humanize_labels, parse_paper_label, strip_invalid_citations)

pytest.importorskip("chromadb")

ABSTRACTS = {
    "glyc": "Background: Glycaemic control matters. Methods: Rural farmers were surveyed about HbA1c monitoring and insulin use. "
            "Results: HbA1c levels were poorly controlled among rural farmers.",
    "htn": "Background: Hypertension is common. Methods: Urban adults were screened. "
           "Results: Hypertension prevalence among urban adults was high and linked to salt intake.",
    "hep": "Background: Vaccines prevent hepatitis. Methods: Household visits recorded immunisation cards. "
           "Results: Hepatitis B vaccination coverage in children under five was incomplete.",
}


@pytest.fixture
def indexed(services):
    topic = services.topics.get_or_create("Mixed fictional topics")
    papers = [make_paper(i, paper_id=None, abstract=text, title=f"[TEST FIXTURE] {key} paper", doi=f"10.0000/{key}",
                         pmid=str(91000000 + i)) for i, (key, text) in enumerate(ABSTRACTS.items(), start=1)]
    report = services.paper_service.save_and_index(papers, topic_id=topic, fetch_fulltext=False)
    return services, topic, report


def test_indexing_stores_chunks_in_chroma_and_metadata_in_sqlite(indexed):
    services, _, report = indexed
    assert report.papers_saved == 3 and report.chunks_total > 0 and report.chunks_new == report.chunks_total
    assert report.embedding_backend == "local"
    assert services.vectors.count() == report.chunks_total == services.chunks.count()
    assert services.chunks.indexed_paper_count() == 3
    with services.db.connection() as conn:  # chunk TEXT must not be duplicated in SQLite
        columns = {r[1] for r in conn.execute("PRAGMA table_info(document_chunk)")}
    assert "text" not in columns and "content" not in columns


def test_indexing_again_reuses_embeddings(indexed, monkeypatch):
    services, topic, first = indexed
    calls = []
    original = services.embedder.embed_documents
    monkeypatch.setattr(services.embedder, "embed_documents", lambda t: calls.append(len(t)) or original(t))
    papers = services.papers.list_for_topic(topic)
    second = services.paper_service.save_and_index(papers, topic_id=topic, fetch_fulltext=False)
    assert second.chunks_new == 0 and second.chunks_reused == first.chunks_total and calls == []
    assert services.vectors.count() == first.chunks_total  # nothing duplicated


def test_retrieval_returns_most_relevant_chunk_first(indexed):
    services, topic, _ = indexed
    hits = services.vectors.query("hypertension prevalence among urban adults", k=3)
    assert "htn" in hits[0].metadata["title"] and hits[0].score > hits[-1].score
    hep_id = next(p.paper_id for p in services.papers.list_for_topic(topic) if "hep" in p.title)
    scoped = services.vectors.query("hypertension prevalence among urban adults", k=3, paper_ids=[hep_id])
    assert scoped and all(h.metadata["paper_id"] == hep_id for h in scoped)
    assert services.vectors.query("anything", paper_ids=[]) == []


def test_rag_answer_without_gemini_is_extractive_with_citations(indexed):
    services, _, _ = indexed
    result = services.rag.answer("What was the hepatitis B vaccination coverage in children?")
    assert result.sufficient and not result.used_llm
    assert "[S1]" in result.answer and result.sources[0].title.endswith("hep paper")
    assert result.sources[0].url.startswith("https://doi.org/10.0000/hep")
    assert any("Gemini is not configured" in n for n in result.notices)


def test_rag_reports_insufficient_evidence(indexed):
    services, _, _ = indexed
    result = services.rag.answer("quantum gravity lattice renormalisation theory")
    assert not result.sufficient and "do not contain enough evidence" in result.answer and result.sources == []


def test_rag_empty_database_and_short_question(services):
    assert not services.rag.answer("What is the prevalence of diabetes?").sufficient
    assert "complete research question" in services.rag.answer("hi").answer


def gemini_services(tmp_path, replies):
    settings = Settings.for_testing(tmp_path / "g", gemini_api_key="AIzaFAKEKEYFORTESTS1234567890abcd",
                                    force_local_embeddings=True)
    client = FakeGeminiClient(replies)
    services = build_services(settings, gemini_client=client)
    papers = [make_paper(i, paper_id=None, abstract=text, title=f"[TEST FIXTURE] {key}", doi=f"10.0000/{key}",
                         pmid=str(92000000 + i)) for i, (key, text) in enumerate(ABSTRACTS.items(), start=1)]
    services.paper_service.save_and_index(papers, fetch_fulltext=False)
    return services, client


def test_gemini_answer_strips_invented_citations(tmp_path):
    services, client = gemini_services(tmp_path, ["HbA1c control was poor among rural farmers [S1][S9]."])
    result = services.rag.answer("How well controlled was HbA1c among rural farmers?")
    assert result.used_llm and result.sufficient
    assert "S9" not in result.answer and "[S1]" in result.answer and result.cited_numbers == [1]
    assert "HbA1c" in client.generate_calls[0] and "INSUFFICIENT_EVIDENCE" in client.generate_calls[0]  # grounded prompt


def test_gemini_can_declare_insufficient_evidence(tmp_path):
    services, _ = gemini_services(tmp_path, ["INSUFFICIENT_EVIDENCE: the passages never mention drug dosage."])
    result = services.rag.answer("What do the papers say about HbA1c monitoring?")
    assert not result.sufficient and "never mention drug dosage" in result.answer


def test_gemini_failure_falls_back_to_passages(tmp_path):
    services, _ = gemini_services(tmp_path, [FakeAPIError(503), FakeAPIError(503)])
    result = services.rag.answer("What do the papers say about HbA1c monitoring?")
    assert result.sufficient and not result.used_llm and "Most relevant passages" in result.answer
    assert any("temporarily unavailable" in n for n in result.notices)


def test_answer_without_citations_is_flagged(tmp_path):
    services, _ = gemini_services(tmp_path, ["HbA1c was poorly controlled."])
    result = services.rag.answer("What do the papers say about HbA1c monitoring?")
    assert any("no citations" in n for n in result.notices)


@pytest.mark.parametrize("question", ["Should I take insulin for my high blood sugar?",
                                      "Can you diagnose me, my symptoms are thirst and fatigue?",
                                      "What dose of metformin do I need?"])
def test_medical_advice_requests_are_declined_without_calling_gemini(tmp_path, question):
    services, client = gemini_services(tmp_path, ["should never be used"])
    result = services.rag.answer(question)
    assert not result.sufficient and "cannot provide that" in result.answer and client.generate_calls == []


# ------------------------------------------------------------------ citations
def test_citation_extraction_and_validation():
    text = "A [S1], B [S2, S5] and C [S7]."
    assert extract_citation_numbers(text) == [1, 2, 5, 7]
    assert strip_invalid_citations(text, {1, 2}) == "A [S1], B [S2] and C."
    assert strip_invalid_citations("Nothing cited.", {1}) == "Nothing cited."


def test_reference_and_inline_formatting():
    paper = make_paper(1)
    assert format_inline(paper) == "Testauthor et al., 2016"
    ref = format_reference(paper)
    assert ref.startswith("Testauthor A, Fixture B, Sample G.") and "doi:10.0000/test.001" in ref and "2016." in ref
    no_doi = make_paper(2, doi=None, pmid="55")
    assert "PMID:55" in format_reference(no_doi) and "pubmed.ncbi.nlm.nih.gov/55/" in format_reference(no_doi)
    assert format_inline(make_paper(3, authors=[])) == "Anonymous, 2018"


def test_label_validation_blocks_invented_papers():
    assert parse_paper_label("P12") == 12 and parse_paper_label("p3") == 3 and parse_paper_label("Paper 3") is None
    assert filter_paper_labels(["P1", "P2", "P99", "junk", "P1"], [1, 2, 3]) == [1, 2]
    text = humanize_labels("P1 and P99 differ", [make_paper(1)])
    assert text == "Testauthor et al., 2016 and P99 differ"
