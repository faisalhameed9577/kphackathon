"""Research summaries, comparison, limitations, gap discovery and the analysis pipeline."""
from __future__ import annotations

import pytest

from database.models import PaperSummary
from services.container import build_services
from services.gap_detection import GapDetectionService, coverage_gaps, template_questions
from services.gemini_service import GeminiService
from services.research_analysis import (ResearchAnalysisService, detect_sample_size, detect_study_design,
                                        extractive_comparison, extractive_limitations, extractive_summary)
from config.settings import Settings
from tests.conftest import FakeAPIError, FakeGeminiClient, make_paper


# ---------------------------------------------------------- pattern helpers
def test_detect_study_design_and_sample_size():
    assert detect_study_design("A cross-sectional survey was done") == "cross-sectional study; survey / questionnaire"
    assert detect_study_design("This systematic review and meta-analysis") == "systematic review / meta-analysis"
    assert detect_study_design("We describe things") == "Not reported"
    assert detect_study_design("A retrospective cohort study") == "cohort study; retrospective study"
    assert detect_sample_size("n = 1,200 patients") == "n = 1,200"
    assert detect_sample_size("We enrolled 85 women") == "85 women"
    assert detect_sample_size("No numbers here") == "Not reported"


def test_extractive_summary_from_structured_abstract():
    s = extractive_summary(make_paper(1))
    assert s.generated_by == "extractive" and s.paper_id == 1
    assert "cross-sectional survey of 300 participants" in s.methods
    assert s.key_findings and "HbA1c" in s.key_findings[0]
    assert s.study_design.startswith("cross-sectional") and s.sample_size == "300 participants"
    assert s.conclusion.startswith("More simulated research")


def test_extractive_summary_without_abstract_reports_nothing_instead_of_guessing():
    s = extractive_summary(make_paper(1, abstract=""))
    assert s.objective == s.methods == s.conclusion == "Not reported" and s.key_findings == []


def test_extractive_limitations_only_returns_explicit_statements():
    paper = make_paper(1)
    items = extractive_limitations(paper, paper.abstract)
    assert len(items) == 1 and "small sample size" in items[0].text and items[0].method == "keyword-match"
    assert extractive_limitations(make_paper(2, abstract="Plain results with nothing negative."), "Plain results.") == []


def test_extractive_comparison(sample_papers):
    summaries = {p.paper_id: extractive_summary(p) for p in sample_papers}
    result = extractive_comparison(sample_papers, summaries)
    assert len(result.rows) == 5 and result.generated_by == "extractive"
    assert any("cross-sectional" in s for s in result.similarities)
    assert any("Study designs differ" in d for d in result.differences)
    assert any("Publication years range" in d for d in result.differences)
    assert "no abstract" in result.overall_assessment  # paper 5 has none
    assert "Not reported" not in result.rows[0].design


# ------------------------------------------------------- Gemini-backed modules
def llm_service(tmp_path, replies):
    settings = Settings.for_testing(tmp_path / "llm", gemini_api_key="AIzaFAKEKEYFORTESTS1234567890abcd",
                                    force_local_embeddings=True)
    client = FakeGeminiClient(replies)
    return build_services(settings, gemini_client=client), client


def test_gemini_summaries_ignore_invented_papers(tmp_path, sample_papers):
    reply = {"papers": [
        {"label": "P1", "objective": "Obj one", "study_design": "Survey", "methods": "M", "key_findings": ["F1", "F2"],
         "conclusion": "C", "population_setting": "Farmers"},
        {"label": "P999", "objective": "Invented paper", "key_findings": ["x"]}]}
    services, client = llm_service(tmp_path, [reply])
    papers = sample_papers[:2]
    out = services.pipeline.analysis.summarize_papers(papers)
    assert out[1].generated_by == "gemini" and out[1].objective == "Obj one" and out[1].key_findings == ["F1", "F2"]
    assert 999 not in out
    assert out[2].generated_by == "extractive"  # Gemini skipped paper 2 -> rule-based fallback, never blank
    assert "=== P1 |" in client.generate_calls[0] and "=== P2 |" in client.generate_calls[0]


def test_gemini_failure_falls_back_and_is_reported(tmp_path, sample_papers):
    services, client = llm_service(tmp_path, [FakeAPIError(403)])
    notes = []
    out = services.pipeline.analysis.summarize_papers(sample_papers, notes)
    assert all(s.generated_by == "extractive" for s in out.values()) and len(out) == 5
    assert len(notes) == 1 and "API key" in notes[0]
    assert len(client.generate_calls) == 1  # no hammering after the first failure


def test_gemini_limitations_verify_quotes(tmp_path):
    paper = make_paper(1)
    reply = {"papers": [{"label": "P1", "limitations": [
        {"text": "Small sample", "evidence": "A limitation is the small sample size"},
        {"text": "Self-selected volunteers", "evidence": "participants volunteered themselves"},
        {"text": "", "evidence": "ignored"}]}, {"label": "P77", "limitations": [{"text": "invented", "evidence": ""}]}]}
    services, _ = llm_service(tmp_path, [reply])
    # get_paper_text needs a paper with abstract only (no stored full text)
    out = services.pipeline.analysis.extract_limitations([paper])
    assert set(out) == {1}
    verified, unverified = out[1]
    assert verified.method == "gemini" and verified.evidence
    assert unverified.method == "gemini-unverified" and unverified.evidence == ""  # quote not found -> flagged


def test_gemini_comparison_fills_missing_rows(tmp_path, sample_papers):
    reply = {"rows": [{"label": "P1", "objective": "O1", "design": "D1", "population": "Pop", "sample_size": "300",
                       "key_findings": "K", "conclusion": "C"}, {"label": "P404", "objective": "ghost"}],
             "similarities": ["Both use surveys (P1, P2)."], "differences": ["Different years."],
             "overall_assessment": "Evidence is weak."}
    services, _ = llm_service(tmp_path, [reply])
    papers = sample_papers[:2]
    summaries = {p.paper_id: extractive_summary(p) for p in papers}
    result = services.pipeline.analysis.compare(papers, summaries)
    assert result.generated_by == "gemini" and [r.paper_id for r in result.rows] == [1, 2]
    assert result.rows[0].objective == "O1" and result.rows[1].design != "D1"  # row for P2 rebuilt from its summary
    with pytest.raises(ValueError):
        services.pipeline.analysis.compare(papers[:1], summaries)


# ----------------------------------------------------------------- gap discovery
def test_coverage_gaps_explain_themselves_with_numbers(sample_papers):
    summaries = {p.paper_id: extractive_summary(p) for p in sample_papers}
    gaps = coverage_gaps(sample_papers, summaries, "glycaemic outcomes", region="Khyber Pakhtunkhwa")
    by_type = {g.gap_type: g for g in gaps}
    assert "0 of 5 retrieved papers mention the region" in by_type["location"].rationale
    assert "reflect the limits of the search" in by_type["location"].rationale  # never claims true absence
    assert "population" in by_type and "evidence_quality" in by_type
    assert all(g.needs_verification and g.origin == "coverage-analysis" for g in gaps)
    # a randomized trial exists in the sample, so no "no randomized studies" gap is raised
    assert not any("No randomized" in g.description for g in gaps)


def test_no_rct_gap_when_none_present():
    papers = [make_paper(i) for i in range(1, 5)]
    summaries = {p.paper_id: extractive_summary(p) for p in papers}
    gaps = coverage_gaps(papers, summaries, "topic")
    descriptions = " ".join(g.description for g in gaps)
    assert "No randomized or interventional" in descriptions and "Most retrieved studies are cross-sectional" in descriptions
    assert coverage_gaps([], {}, "topic") == []


def test_region_gap_not_raised_when_region_is_covered():
    papers = [make_paper(i, abstract="Survey of 300 adults in Peshawar, Khyber Pakhtunkhwa.") for i in range(1, 5)]
    gaps = coverage_gaps(papers, {}, "topic", region="Khyber Pakhtunkhwa")
    assert not any(g.gap_type == "location" for g in gaps)


def test_llm_gaps_validate_supporting_papers(tmp_path, sample_papers):
    reply = {"gaps": [{"gap_type": "population", "description": "Women are under-represented",
                       "rationale": "Only P2 includes women.", "supporting_papers": ["P2", "P55", "junk"]},
                      {"gap_type": "weird-type", "description": "Another possible gap", "rationale": "r", "supporting_papers": []},
                      {"description": "", "rationale": "dropped because it has no description"}]}
    services, _ = llm_service(tmp_path, [reply])
    summaries = {p.paper_id: extractive_summary(p) for p in sample_papers}
    gaps = GapDetectionService(services.gemini).llm_gaps("topic", "", sample_papers, summaries, {})
    assert len(gaps) == 2 and gaps[0].supporting_paper_ids == [2] and gaps[0].origin == "gemini"
    assert gaps[1].gap_type == "other"


def test_questions_are_template_based_without_gemini(settings, sample_papers):
    service = GapDetectionService(GeminiService(settings))
    summaries = {p.paper_id: extractive_summary(p) for p in sample_papers}
    gaps = service.detect_gaps("glycaemic outcomes", "Fictionland", sample_papers, summaries, {})
    questions = service.generate_questions("glycaemic outcomes", "Fictionland", gaps)
    assert questions and all(q.origin == "template" and q.suggested_design for q in questions)
    assert all(q.gap_index is not None for q in questions) and "Fictionland" in questions[0].question
    assert template_questions("t", "", []) == []


def test_gemini_questions(tmp_path, sample_papers):
    reply = {"questions": [{"gap_index": 0, "question": "What is X?", "rationale": "because", "suggested_design": "cohort"},
                           {"gap_index": 42, "question": "Orphan question?"}]}
    services, _ = llm_service(tmp_path, [reply])
    gaps = coverage_gaps(sample_papers, {}, "topic")
    questions = GapDetectionService(services.gemini).generate_questions("topic", "", gaps)
    assert [q.origin for q in questions] == ["gemini", "gemini"] and questions[0].gap_index == 0 and questions[1].gap_index is None


# --------------------------------------------------------------------- pipeline
def test_pipeline_runs_stores_and_reloads(services, saved_topic):
    topic_id, papers = saved_topic
    steps = []
    bundle = services.pipeline.run_gap_analysis(topic_id, progress=lambda f, m: steps.append(f))
    assert len(bundle.summaries) == 5 and len(bundle.limitations) == 5 and bundle.gaps and bundle.questions
    assert steps[-1] == 1.0 and all(b >= a for a, b in zip(steps, steps[1:]))

    reloaded = services.pipeline.load_bundle(topic_id)
    assert set(reloaded.summaries) == set(bundle.summaries) and len(reloaded.gaps) == len(bundle.gaps)
    assert isinstance(next(iter(reloaded.summaries.values())), PaperSummary)
    assert reloaded.gaps[0].gap_id is not None

    again = services.pipeline.run_gap_analysis(topic_id)  # cached summaries reused, gaps replaced not doubled
    assert len(again.gaps) == len(bundle.gaps) and len(services.gaps.list_gaps(topic_id)) == len(bundle.gaps)


def test_pipeline_comparison_stored_and_validated(services, saved_topic):
    topic_id, papers = saved_topic
    ids = [p.paper_id for p in papers[:3]]
    result = services.pipeline.run_comparison(topic_id, ids)
    assert len(result.rows) == 3 and services.pipeline.stored_comparison_paper_ids(topic_id) == ids
    assert services.pipeline.load_bundle(topic_id).comparison.generated_by == "extractive"
    with pytest.raises(ValueError, match="at least two"):
        services.pipeline.run_comparison(topic_id, ids[:1])


def test_pipeline_errors_are_clear(services):
    topic = services.topics.get_or_create("Empty topic")
    with pytest.raises(ValueError, match="no saved papers"):
        services.pipeline.run_gap_analysis(topic)
    with pytest.raises(ValueError, match="Topic not found"):
        services.pipeline.run_gap_analysis(424242)


def test_rule_based_results_are_upgraded_when_gemini_arrives(tmp_path, saved_topic, services):
    topic_id, papers = saved_topic
    services.pipeline.run_gap_analysis(topic_id)  # rule-based first (no key)
    assert {s.generated_by for s in services.pipeline.load_bundle(topic_id).summaries.values()} == {"extractive"}
    reply = {"papers": [{"label": f"P{p.paper_id}", "objective": "AI objective", "key_findings": ["f"]} for p in papers]}
    upgraded = build_services(Settings.for_testing(services.settings.data_dir, gemini_api_key="AIzaFAKEKEYFORTESTS1234567890abcd",
                                                   force_local_embeddings=True),
                              gemini_client=FakeGeminiClient([reply, reply, {"papers": []}, {"papers": []}, {"gaps": []}, {"questions": []}]))
    summaries = upgraded.pipeline.ensure_summaries(topic_id, upgraded.papers.list_for_topic(topic_id))
    assert {s.generated_by for s in summaries.values()} == {"gemini"}
