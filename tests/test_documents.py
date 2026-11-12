"""Document processing: abstract sections, full-text parsing, chunking and chunk metadata."""
from __future__ import annotations

import pytest

from database.models import PaperRecord
from services.document_processor import DocumentProcessor, parse_jats_sections, split_abstract_sections
from tests.conftest import make_paper
from utils.helpers import abstract_html_to_text, clean_text, split_sentences


def test_clean_text_and_html_abstract():
    assert clean_text("A <i>very</i>   spaced &amp; tagged title") == "A very spaced & tagged title"
    text = abstract_html_to_text("<h4>Methods</h4><p>We did X.</p><h4>Results</h4><p>Y &gt; Z</p>")
    assert text == "Methods: We did X.\nResults: Y > Z"
    assert split_sentences("First one. Second one! Third?") == ["First one.", "Second one!", "Third?"]


def test_split_abstract_sections_structured_and_plain():
    parts = split_abstract_sections("Background: Why it matters.\nMethods: What we did.\nIt took a year.\nConclusions: Done.")
    assert [label for label, _ in parts] == ["Background", "Methods", "Conclusions"]
    assert parts[1][1] == "What we did. It took a year."
    assert split_abstract_sections("Just an unstructured abstract.") == [(None, "Just an unstructured abstract.")]
    assert split_abstract_sections("Note: this is not a known label.") == [(None, "Note: this is not a known label.")]
    assert split_abstract_sections("") == []


def test_chunk_metadata_is_complete():
    paper = make_paper(7)
    docs = DocumentProcessor().build_documents(paper)
    assert docs, "a paper with an abstract must produce chunks"
    required = {"paper_id", "title", "authors", "year", "doi", "source", "section", "chunk_id"}
    for doc in docs:
        assert required <= set(doc.metadata)
        assert doc.metadata["paper_id"] == 7 and doc.metadata["doi"] == "10.0000/test.007"
        assert doc.metadata["year"] == 2022 and doc.metadata["source"] == "PubMed"
        assert "page" not in doc.metadata  # abstracts have no page numbers; key omitted, not faked
    sections = {d.metadata["section"] for d in docs}
    assert {"Abstract - Background", "Abstract - Methods", "Abstract - Results"} <= sections


def test_chunk_ids_are_unique_and_stable():
    paper = make_paper(3)
    first = [d.metadata["chunk_id"] for d in DocumentProcessor().build_documents(paper)]
    second = [d.metadata["chunk_id"] for d in DocumentProcessor().build_documents(paper)]
    assert first == second and len(set(first)) == len(first)  # same ids -> embeddings can be reused


def test_long_text_is_split_into_overlapping_chunks():
    paper = make_paper(1, abstract="Methods: " + " ".join(f"Sentence number {i} about synthetic methods." for i in range(120)))
    docs = DocumentProcessor(chunk_size=500, chunk_overlap=50).build_documents(paper)
    assert len(docs) > 3 and all(len(d.page_content) <= 500 for d in docs)


def test_full_text_sections_are_added_and_marked():
    paper = make_paper(1)
    sections = {"Methods": "Full text methods " * 10, "Results": "Full text results " * 10}
    docs = DocumentProcessor().build_documents(paper, sections)
    origins = {d.metadata["text_origin"] for d in docs}
    assert origins == {"abstract", "full text"}
    assert {"Methods", "Results"} <= {d.metadata["section"] for d in docs}


def test_missing_abstract_yields_no_chunks_and_unsaved_paper_is_rejected():
    assert DocumentProcessor().build_documents(make_paper(1, abstract="")) == []
    with pytest.raises(ValueError):
        DocumentProcessor().build_documents(PaperRecord(title="No id yet", abstract="text " * 20))


def test_parse_jats_sections(jats_xml):
    sections = parse_jats_sections(jats_xml)
    assert list(sections) == ["Introduction", "Methods", "Results", "Discussion", "Conclusions"]
    assert "small sample size" in sections["Discussion"]  # nested Limitations subsection text kept
    joined = " ".join(sections.values())
    assert "funding statement" not in joined and "Invented reference" not in joined
    assert parse_jats_sections("<broken") == {} and parse_jats_sections("") == {}
    assert parse_jats_sections("<article><front/></article>") == {}  # no body
