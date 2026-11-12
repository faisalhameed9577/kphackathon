"""
Data models (Pydantic) shared by the services, the database layer and the UI.

Large text (full articles) is NOT stored in these models or in SQLite:
abstracts live in SQLite, chunks live in ChromaDB, full text is cached as a file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from utils.helpers import clean_text, normalize_doi


class PaperRecord(BaseModel):
    """One research paper (bibliographic metadata + abstract)."""

    model_config = ConfigDict(validate_assignment=True)

    paper_id: Optional[int] = None  # SQLite id, set once the paper is saved
    title: str
    abstract: str = ""
    authors: List[str] = Field(default_factory=list)
    journal: str = ""
    year: Optional[int] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    europepmc_id: Optional[str] = None
    europepmc_source: Optional[str] = None  # MED, PMC, PPR (preprint), ...
    sources: List[str] = Field(default_factory=list)  # e.g. ["PubMed", "Europe PMC"]
    url: str = ""
    is_open_access: bool = False
    fulltext_path: Optional[str] = None
    # PubMed publication types (e.g. "Randomized Controlled Trial"). Transient: used for
    # evidence badges in Idea Check and not stored in SQLite.
    pub_types: List[str] = Field(default_factory=list)

    @field_validator("title", mode="before")
    @classmethod
    def _clean_title(cls, value: Any) -> str:
        return clean_text(value) or "Untitled"

    @field_validator("abstract", mode="before")
    @classmethod
    def _clean_abstract(cls, value: Any) -> str:
        return (value or "").strip()

    @field_validator("doi", mode="before")
    @classmethod
    def _clean_doi(cls, value: Any) -> Optional[str]:
        return normalize_doi(value)

    @field_validator("pmid", "europepmc_id", mode="before")
    @classmethod
    def _clean_id(cls, value: Any) -> Optional[str]:
        value = str(value).strip() if value not in (None, "") else ""
        return value or None

    @field_validator("pmcid", mode="before")
    @classmethod
    def _clean_pmcid(cls, value: Any) -> Optional[str]:
        value = str(value).strip().upper() if value not in (None, "") else ""
        if not value:
            return None
        return value if value.startswith("PMC") else f"PMC{value}"

    # ---------------------------------------------------------- convenience
    @property
    def has_abstract(self) -> bool:
        return bool(self.abstract.strip())

    @property
    def is_preprint(self) -> bool:
        return (self.europepmc_source or "").upper() == "PPR"

    @property
    def best_url(self) -> str:
        """Most stable link to the paper: DOI > PubMed > Europe PMC > stored URL."""
        if self.doi:
            return f"https://doi.org/{self.doi}"
        if self.pmid:
            return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"
        if self.europepmc_id and self.europepmc_source:
            return f"https://europepmc.org/article/{self.europepmc_source}/{self.europepmc_id}"
        return self.url or ""

    @property
    def source_label(self) -> str:
        return " + ".join(self.sources) if self.sources else "Unknown"


# --------------------------------------------------------------- search results


@dataclass
class SourceSearchResult:
    """What one research API returned for a search (kept for transparency/reports)."""

    source: str
    query: str = ""
    total_found: int = 0
    papers: List[PaperRecord] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class SearchOutcome:
    """Combined result of searching all sources."""

    papers: List[PaperRecord]
    per_source: List[SourceSearchResult]
    duplicates_removed: int = 0
    retrieved_before_dedup: int = 0


@dataclass
class IndexReport:
    """Summary of saving + indexing a set of papers."""

    topic_id: Optional[int] = None
    papers_saved: int = 0
    papers_without_abstract: int = 0
    fulltext_fetched: int = 0
    chunks_total: int = 0
    chunks_new: int = 0
    chunks_reused: int = 0
    embedding_backend: str = ""
    warnings: List[str] = field(default_factory=list)


# --------------------------------------------------------------- analysis models


class PaperSummary(BaseModel):
    paper_id: Optional[int] = None
    objective: str = "Not reported"
    study_design: str = "Not reported"
    population_setting: str = "Not reported"
    methods: str = "Not reported"
    key_findings: List[str] = Field(default_factory=list)
    conclusion: str = "Not reported"
    generated_by: str = "extractive"  # "gemini" or "extractive"
    sample_size: str = "Not reported"


class LimitationItem(BaseModel):
    paper_id: Optional[int] = None
    text: str
    evidence: str = ""
    method: str = "gemini"  # "gemini" or "keyword-match"


class ComparisonRow(BaseModel):
    paper_id: Optional[int] = None
    objective: str = "Not reported"
    design: str = "Not reported"
    population: str = "Not reported"
    sample_size: str = "Not reported"
    key_findings: str = "Not reported"
    conclusion: str = "Not reported"


class ComparisonResult(BaseModel):
    rows: List[ComparisonRow] = Field(default_factory=list)
    similarities: List[str] = Field(default_factory=list)
    differences: List[str] = Field(default_factory=list)
    overall_assessment: str = ""
    generated_by: str = "extractive"


class GapItem(BaseModel):
    gap_id: Optional[int] = None
    gap_type: str = "other"  # population|location|methodology|outcome|temporal|evidence_quality|other
    description: str
    rationale: str = ""
    supporting_paper_ids: List[int] = Field(default_factory=list)
    origin: str = "coverage-analysis"  # "coverage-analysis" or "gemini"
    needs_verification: bool = True


class ResearchQuestionItem(BaseModel):
    question_id: Optional[int] = None
    gap_index: Optional[int] = None  # index into the gap list when generated
    gap_id: Optional[int] = None
    question: str
    rationale: str = ""
    suggested_design: str = ""
    origin: str = "template"  # "gemini" or "template"


@dataclass
class AnalysisBundle:
    """Everything the analysis pipeline produces for one topic."""

    topic_id: int
    papers: List[PaperRecord] = field(default_factory=list)
    summaries: Dict[int, PaperSummary] = field(default_factory=dict)
    limitations: Dict[int, List[LimitationItem]] = field(default_factory=dict)
    comparison: Optional[ComparisonResult] = None
    gaps: List[GapItem] = field(default_factory=list)
    questions: List[ResearchQuestionItem] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)  # warnings shown to the user
