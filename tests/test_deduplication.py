"""Duplicate removal across PubMed and Europe PMC."""
from __future__ import annotations

from database.models import PaperRecord
from services.deduplication_service import deduplicate, identity_keys, interleave, merge_records
from services.europe_pmc_service import parse_europepmc_result
from services.pubmed_service import parse_pubmed_xml


def test_pubmed_and_europepmc_fixtures_deduplicate(pubmed_xml, europepmc_json):
    pubmed = parse_pubmed_xml(pubmed_xml)  # 3 papers
    epmc = [parse_europepmc_result(r) for r in europepmc_json["resultList"]["result"]]  # 4 papers, 2 overlap
    unique, removed = deduplicate(interleave([pubmed, epmc]))
    assert removed == 2 and len(unique) == 5  # 7 records - 2 duplicates

    by_pmid = {p.pmid: p for p in unique if p.pmid}
    merged = by_pmid["90000001"]  # matched by DOI (different letter case)
    assert merged.sources == ["PubMed", "Europe PMC"]
    assert merged.is_open_access  # flag learnt from Europe PMC
    assert merged.europepmc_id == "90000001"
    assert by_pmid["90000003"].sources == ["PubMed", "Europe PMC"]  # matched by PMID only


def test_duplicate_by_doi_ignores_case_and_url_prefix():
    a = PaperRecord(title="Paper A", doi="10.1000/ABC")
    b = PaperRecord(title="Different title", doi="https://doi.org/10.1000/abc")
    unique, removed = deduplicate([a, b])
    assert len(unique) == 1 and removed == 1


def test_duplicate_by_pmcid_and_long_title():
    a = PaperRecord(title="A sufficiently long and unique study title about synthetic things", pmcid="PMC1")
    b = PaperRecord(title="A sufficiently long and unique study title about synthetic things!", pmid="5")
    c = PaperRecord(title="Other", pmcid="PMC1")
    unique, removed = deduplicate([a, b, c])
    assert len(unique) == 1 and removed == 2
    assert unique[0].pmid == "5"  # identifier learnt from the merged duplicate


def test_short_titles_are_not_merged_by_title():
    a, b = PaperRecord(title="Editorial"), PaperRecord(title="Editorial")
    assert len(deduplicate([a, b])[0]) == 2
    assert not any(k.startswith("title:") for k in identity_keys(a))


def test_merge_keeps_longer_abstract_and_fills_blanks():
    a = PaperRecord(title="T", abstract="short", sources=["PubMed"])
    b = PaperRecord(title="T", abstract="a much longer abstract text", journal="J", year=2020, doi="10.1/x",
                    pmcid="PMC7", is_open_access=True, sources=["Europe PMC"])
    merged = merge_records(a, b)
    assert merged.abstract == "a much longer abstract text" and merged.journal == "J" and merged.year == 2020
    assert merged.doi == "10.1/x" and merged.pmcid == "PMC7" and merged.is_open_access
    assert merged.sources == ["PubMed", "Europe PMC"]
    assert a.abstract == "short"  # inputs are not mutated


def test_interleave_is_round_robin():
    a = [PaperRecord(title=f"A{i}") for i in range(3)]
    b = [PaperRecord(title=f"B{i}") for i in range(1)]
    assert [p.title for p in interleave([a, b])] == ["A0", "B0", "A1", "A2"]
