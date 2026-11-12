"""Europe PMC retrieval (REST API): search + open-access full text."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from config.settings import Settings
from database.models import PaperRecord, SourceSearchResult
from utils.helpers import abstract_html_to_text, clean_text, get_logger, parse_year
from utils.http import ResearchAPIError, request_with_retry
from utils.validators import SearchRequest

log = get_logger(__name__)

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"


def build_europepmc_query(request: SearchRequest) -> str:
    """
    Build the Europe PMC query.

    Region terms are restricted to title/abstract so that a passing mention deep inside
    a full text does not count as "research about" that region.
    """
    parts: List[str] = []
    if request.topic_terms:
        parts.append("(" + " AND ".join(request.topic_terms) + ")")
    if request.keyword_terms:
        kw = [f'"{k}"' if " " in k else k for k in request.keyword_terms]
        parts.append("(" + " OR ".join(kw) + ")")
    if request.region_terms:
        fields = [f'TITLE:"{r}" OR ABSTRACT:"{r}"' for r in request.region_terms]
        parts.append("(" + " OR ".join(fields) + ")")
    parts.append(f"(PUB_YEAR:[{request.year_from} TO {request.year_to}])")
    return " AND ".join(parts)


def parse_europepmc_result(item: Dict[str, Any]) -> Optional[PaperRecord]:
    """Convert one Europe PMC ``result`` JSON object into a PaperRecord."""
    title = clean_text(item.get("title"))
    if not title:
        return None

    authors: List[str] = []
    for author in (item.get("authorList") or {}).get("author", []) or []:
        name = clean_text(author.get("fullName") or author.get("collectiveName"))
        if name:
            authors.append(name)
    if not authors and item.get("authorString"):
        authors = [a.strip(" .") for a in str(item["authorString"]).split(",") if a.strip(" .")]

    journal = clean_text(((item.get("journalInfo") or {}).get("journal") or {}).get("title")
                         or item.get("journalTitle") or "")
    year = parse_year(item.get("pubYear")) or parse_year((item.get("journalInfo") or {}).get("yearOfPublication"))

    source, ext_id = item.get("source"), item.get("id")
    return PaperRecord(
        title=title,
        abstract=abstract_html_to_text(item.get("abstractText")),
        authors=authors, journal=journal, year=year,
        doi=item.get("doi"), pmid=item.get("pmid"), pmcid=item.get("pmcid"),
        europepmc_id=ext_id, europepmc_source=source, sources=["Europe PMC"],
        url=f"https://europepmc.org/article/{source}/{ext_id}" if source and ext_id else "",
        is_open_access=str(item.get("isOpenAccess", "N")).upper() == "Y",
    )


class EuropePMCService:
    def __init__(self, settings: Settings, session: Optional[requests.Session] = None):
        self.settings = settings
        self.session = session or requests.Session()

    def _get(self, url: str, params: Optional[dict] = None, accept: tuple = ()) -> requests.Response:
        return request_with_retry(
            self.session, url, params, source="Europe PMC", timeout=self.settings.http_timeout,
            max_retries=self.settings.max_retries, backoff=self.settings.retry_backoff_seconds,
            accept_statuses=accept)

    def search(self, request: SearchRequest) -> SourceSearchResult:
        """Search Europe PMC. Errors are returned in the result, not raised."""
        result = SourceSearchResult(source="Europe PMC", query=build_europepmc_query(request))
        try:
            response = self._get(f"{BASE_URL}/search", {
                "query": result.query, "format": "json", "resultType": "core",
                "pageSize": min(request.max_papers, 100),
            })
            try:
                data = response.json()
            except ValueError as exc:
                raise ResearchAPIError("Europe PMC returned an unreadable response.") from exc
            result.total_found = int(data.get("hitCount", 0) or 0)
            for item in (data.get("resultList") or {}).get("result", []) or []:
                paper = parse_europepmc_result(item)
                if paper:
                    result.papers.append(paper)
        except ResearchAPIError as exc:
            result.error = str(exc)
            log.warning("Europe PMC search failed: %s", exc)
        return result

    def fetch_fulltext_xml(self, pmcid: str) -> Optional[str]:
        """
        Download open-access full text (JATS XML) for a PMC id.
        Returns None when no open-access full text exists (HTTP 404/403/204).
        """
        pmcid = pmcid.upper()
        response = self._get(f"{BASE_URL}/{pmcid}/fullTextXML", accept=(204, 403, 404))
        if response.status_code != 200 or not response.text.strip():
            return None
        return response.text
