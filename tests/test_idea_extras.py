"""Evidence badges, publication-type merging and the Idea Check PDF."""
from __future__ import annotations

import shutil
import subprocess

from services.deduplication_service import deduplicate
from services.evidence_labels import evidence_badge, is_retracted
from services.idea_pdf import IdeaPdfService
from tests.conftest import make_paper
from tests.test_idea import IDEA, make_service


def test_badges_pick_the_strongest_design():
    assert evidence_badge(["Journal Article", "Meta-Analysis", "Review"]) == "Meta-analysis"
    assert evidence_badge(["Randomized Controlled Trial"]) == "RCT"
    assert evidence_badge(["Journal Article"]) == "Study (type not indexed)"
    assert evidence_badge([], is_preprint=True) == "Preprint"


def test_retracted_publication_is_detected():
    assert is_retracted(["Journal Article", "Retracted Publication"])
    assert not is_retracted(["Journal Article", "Retraction of Publication"])


def test_dedup_merges_publication_types():
    a = make_paper(1, pub_types=["Journal Article"])
    b = make_paper(1, pub_types=["Randomized Controlled Trial"], sources=["Europe PMC"])
    merged, removed = deduplicate([a, b])
    assert removed == 1 and merged[0].pub_types == ["Journal Article", "Randomized Controlled Trial"]


def test_idea_pdf_is_created_with_the_verdict(settings, tmp_path):
    papers = [make_paper(1, pub_types=["Randomized Controlled Trial"]),
              make_paper(2, pub_types=["Journal Article", "Retracted Publication"])]
    service, _ = make_service(settings, lambda q, y: 3, papers=papers)
    report = service.run(IDEA, "")
    path = IdeaPdfService().generate_idea(report, tmp_path / "idea.pdf")
    data = path.read_bytes()
    assert data.startswith(b"%PDF") and len(data) > 2000
    if shutil.which("pdftotext"):
        text = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True).stdout
        assert report.verdict in text and "does not provide medical" in text and "RETRACTED" in text
