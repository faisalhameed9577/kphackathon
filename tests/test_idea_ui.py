"""UI smoke test for the Idea Check page (headless Streamlit, fake PubMed / Europe PMC)."""
from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("chromadb")

from tests.test_ui_smoke import click, open_page, texts, ui  # noqa: E402,F401  (ui is a fixture)


def test_idea_check_page_renders_empty(ui):
    at = open_page("Idea Check")
    assert not at.exception, [e.value for e in at.exception]
    assert "Check my idea" in [b.label for b in at.button]
    assert "Type an idea and press" in texts(at)
    assert "does not provide medical" in texts(at)


def test_idea_check_runs_and_shows_a_verdict(ui):
    at = open_page("Idea Check")
    at.text_area(key="idea_text").set_value("Effect of school-based exercise on childhood obesity in Peshawar")
    at = _submit(at)
    assert not at.exception, [e.value for e in at.exception]
    body = texts(at)
    # the fake PubMed returns 3 for every count: more than 2 papers and no review threshold -> MODIFY
    assert "MODIFY" in body
    assert "Closest existing studies" in " ".join(t.label for t in at.tabs)
    assert any("planning aid" in c.value for c in at.caption)


def test_idea_check_rejects_a_too_short_idea(ui):
    at = open_page("Idea Check")
    at.text_area(key="idea_text").set_value("obesity")
    at = _submit(at)
    assert not at.exception
    assert any("at least one full sentence" in e.value for e in at.error)


def test_idea_check_save_and_analyse_creates_a_topic(ui):
    at = open_page("Idea Check")
    at.text_area(key="idea_text").set_value("Effect of school-based exercise on childhood obesity in Peshawar")
    at = _submit(at)
    at = click(at, "Save the closest papers and run the full gap analysis")
    assert not at.exception, [e.value for e in at.exception]
    assert ui.papers.count() >= 1
    assert any("Papers saved and analysed" in s.value for s in at.success)


def _submit(at):
    button = next(b for b in at.button if getattr(b, "label", "") == "Check my idea")
    return button.click().run()
