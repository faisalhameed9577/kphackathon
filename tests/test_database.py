"""SQLite schema, constraints and repositories."""
from __future__ import annotations

import sqlite3

import pytest

from database.db import Database, DatabaseError
from database.models import GapItem, ResearchQuestionItem
from tests.conftest import make_paper

EXPECTED_TABLES = {
    "research_topic", "research_paper", "paper_author", "document_chunk", "research_query", "research_analysis",
    "research_gap", "research_question", "generated_report", "topic_paper", "gap_supporting_paper"}


def test_schema_contains_all_entities(services):
    with services.db.connection() as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert EXPECTED_TABLES <= tables
    services.db.init_schema()  # idempotent: safe to run on every start


def test_upsert_merges_duplicates_by_identifier(services):
    repo = services.papers
    first = repo.upsert(make_paper(1, paper_id=None, pmid="111", doi="10.0000/dup", abstract="short"))
    second = repo.upsert(make_paper(1, paper_id=None, pmid=None, doi="10.0000/DUP", abstract="a longer abstract text",
                                    pmcid="PMC5", sources=["Europe PMC"], is_open_access=True))
    assert first == second and repo.count() == 1
    stored = repo.get(first)
    assert stored.abstract == "a longer abstract text" and stored.pmid == "111" and stored.pmcid == "PMC5"
    assert stored.sources == ["Europe PMC", "PubMed"] or set(stored.sources) == {"PubMed", "Europe PMC"}
    assert stored.is_open_access and stored.authors == ["Testauthor A", "Fixture B", "Sample G"]


def test_unique_identifier_constraint(services):
    services.papers.upsert(make_paper(1, paper_id=None, doi="10.0000/one", pmid="1"))
    with pytest.raises(DatabaseError):
        with services.db.connection() as conn:
            conn.execute("INSERT INTO research_paper (title,title_key,doi) VALUES ('x','x','10.0000/one')")


def test_check_constraints_and_foreign_keys(services):
    with pytest.raises(DatabaseError):  # unknown analysis type
        with services.db.connection() as conn:
            conn.execute("INSERT INTO research_topic (title,title_key) VALUES ('t','t')")
            conn.execute("INSERT INTO research_analysis (topic_id,analysis_type,content_json,generated_by) "
                         "VALUES (1,'nonsense','{}','x')")
    with pytest.raises(DatabaseError):  # foreign key to a topic that does not exist
        with services.db.connection() as conn:
            conn.execute("INSERT INTO generated_report (topic_id,title,file_path) VALUES (999,'r','p')")


def test_cascade_delete_removes_dependents(services):
    pid = services.papers.upsert(make_paper(1, paper_id=None))
    topic = services.topics.get_or_create("Topic A")
    services.papers.link_to_topic(topic, pid)
    services.chunks.upsert_many(pid, [{"chunk_uid": "c1", "section": "Abstract", "chunk_index": 0, "char_count": 10}], "local")
    with services.db.connection() as conn:
        conn.execute("DELETE FROM research_paper WHERE paper_id=?", (pid,))
        for table in ("paper_author", "document_chunk", "topic_paper"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_topics_are_reused_and_counted(services):
    a = services.topics.get_or_create("Diabetes in KP", region="KP")
    b = services.topics.get_or_create("diabetes in kp", region="kp", keywords="HbA1c")
    c = services.topics.get_or_create("Diabetes in KP", region="Peshawar")
    assert a == b and a != c and services.topics.count() == 2
    assert services.topics.get(a)["keywords"] == "HbA1c"


def test_list_all_filters(services):
    for i in range(1, 4):
        pid = services.papers.upsert(make_paper(i, paper_id=None, is_open_access=(i == 2), year=2010 + i * 5))
    topic = services.topics.get_or_create("Only topic")
    services.papers.link_to_topic(topic, pid)
    repo = services.papers
    assert len(repo.list_all()) == 3
    assert len(repo.list_all(open_access_only=True)) == 1
    assert len(repo.list_all(year_from=2020)) == 2
    assert len(repo.list_all(text="number 2")) == 1
    assert [p.paper_id for p in repo.list_all(topic_id=topic)] == [pid]
    assert repo.list_all()[0].year >= repo.list_all()[-1].year  # newest first


def test_analysis_replace_and_lookup(services):
    topic = services.topics.get_or_create("T")
    pid = services.papers.upsert(make_paper(1, paper_id=None))
    repo = services.analyses
    repo.save(topic, "summary", {"a": 1}, "extractive", paper_id=pid)
    repo.save(topic, "summary", {"a": 2}, "gemini", paper_id=pid)  # replaces, does not duplicate
    assert repo.get(topic, "summary", pid)["content"] == {"a": 2}
    assert len(repo.get_for_papers(topic, "summary")) == 1
    repo.save(topic, "comparison", {"x": 1}, "gemini")
    assert repo.get(topic, "comparison")["generated_by"] == "gemini" and repo.get(topic, "limitations") is None


def test_gaps_questions_and_supporting_papers(services):
    topic = services.topics.get_or_create("T")
    pid = services.papers.upsert(make_paper(1, paper_id=None))
    gaps = [GapItem(description="Gap one", gap_type="location", rationale="why", supporting_paper_ids=[pid, 9999]),
            GapItem(description="Gap two", gap_type="other")]
    questions = [ResearchQuestionItem(question="Q for gap two?", gap_index=1, suggested_design="cohort"),
                 ResearchQuestionItem(question="General question?")]
    services.gaps.replace_for_topic(topic, gaps, questions)
    stored = services.gaps.list_gaps(topic)
    assert [g.description for g in stored] == ["Gap one", "Gap two"]
    assert stored[0].supporting_paper_ids == [pid]  # unknown paper id 9999 silently dropped
    qs = services.gaps.list_questions(topic)
    assert qs[0].gap_index == 1 and qs[1].gap_index is None

    services.gaps.replace_for_topic(topic, gaps[:1], [])  # replacing cleans up old rows
    assert len(services.gaps.list_gaps(topic)) == 1 and services.gaps.list_questions(topic) == []


def test_queries_reports_and_stats(services):
    topic = services.topics.get_or_create("T")
    services.queries.add_search(topic, "diabetes", {"year_from": 2010}, ["PubMed"], 5)
    services.queries.add_question(topic, "Q?", "A.", True, 3)
    assert services.queries.count("search") == 1 and services.queries.count("question") == 1
    assert services.queries.latest_search(topic)["filters"]["year_from"] == 2010
    rid = services.reports.add(topic, "Report", "/tmp/x.pdf", 123)
    assert services.reports.list()[0]["report_id"] == rid and services.reports.count() == 1
    services.papers.upsert(make_paper(1, paper_id=None, year=2020))
    assert services.stats.year_counts() == [(2020, 1)] and services.stats.source_counts() == {"PubMed": 1}


def test_database_error_is_friendly(tmp_path):
    db = Database(tmp_path / "x.db")
    with pytest.raises(DatabaseError, match="Database error"):
        with db.connection() as conn:
            conn.execute("SELECT * FROM table_that_does_not_exist")
    assert issubclass(DatabaseError, Exception) and not issubclass(sqlite3.Error, DatabaseError)
