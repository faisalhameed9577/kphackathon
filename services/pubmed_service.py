"""PubMed retrieval through the NCBI E-utilities API (esearch + efetch)."""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import List, Optional

import requests

from config.settings import TOOL_NAME, Settings
from database.models import PaperRecord, SourceSearchResult
from utils.helpers import chunked, clean_text, get_logger, parse_year
from utils.http import ResearchAPIError, request_with_retry
from utils.validators import SearchRequest

log = get_logger(__name__)

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
FETCH_BATCH = 100


def build_pubmed_query(request: SearchRequest) -> str:
    """
    Build the PubMed query string.

    topic terms  -> AND-ed (generic words like "research" are already removed)
    keywords     -> OR-ed together, then AND-ed with the topic
    region       -> OR of its spellings, searched in title/abstract
    """
    parts: List[str] = []
    if request.topic_terms:
        parts.append("(" + " AND ".join(request.topic_terms) + ")")
    if request.keyword_terms:
        kw = [f'"{k}"' if " " in k else k for k in request.keyword_terms]
        parts.append("(" + " OR ".join(kw) + ")")
    if request.region_terms:
        parts.append("(" + " OR ".join(f'"{r}"[tiab]' for r in request.region_terms) + ")")
    return " AND ".join(parts)


def _text(element: Optional[ET.Element]) -> str:
    """All text inside an element, including nested tags such as <i>."""
    return clean_text("".join(element.itertext())) if element is not None else ""


def parse_pubmed_xml(xml_text: str) -> List[PaperRecord]:
    """Convert an efetch XML document into PaperRecord objects."""
    try:
        root = ET.fromstring(xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text)
    except ET.ParseError as exc:
        raise ResearchAPIError("PubMed returned a response that could not be read.") from exc

    papers: List[PaperRecord] = []
    for article in root.findall(".//PubmedArticle"):
        citation = article.find("MedlineCitation")
        if citation is None:
            continue
        art = citation.find("Article")
        if art is None:
            continue

        pmid = _text(citation.find("PMID"))
        title = _text(art.find("ArticleTitle"))

        # Abstract: keep structured labels (BACKGROUND, METHODS, ...) as "LABEL: text" lines.
        abstract_lines = []
        for node in art.findall("Abstract/AbstractText"):
            body = _text(node)
            if not body:
                continue
            label = (node.get("Label") or "").strip()
            abstract_lines.append(f"{label.title()}: {body}" if label and label.upper() != "UNLABELLED" else body)
        abstract = "\n".join(abstract_lines)

        authors: List[str] = []
        for author in art.findall("AuthorList/Author"):
            last = _text(author.find("LastName"))
            initials = _text(author.find("Initials"))
            collective = _text(author.find("CollectiveName"))
            name = f"{last} {initials}".strip() if last else collective
            if name:
                authors.append(name)

        journal = _text(art.find("Journal/Title")) or _text(art.find("Journal/ISOAbbreviation"))

        year = parse_year(_text(art.find("Journal/JournalIssue/PubDate/Year")))
        if year is None:
            year = parse_year(_text(art.find("Journal/JournalIssue/PubDate/MedlineDate")))
        if year is None:
            year = parse_year(_text(art.find("ArticleDate/Year")))

        doi = pmcid = None
        for aid in article.findall("PubmedData/ArticleIdList/ArticleId"):
            kind = (aid.get("IdType") or "").lower()
            if kind == "doi" and not doi:
                doi = _text(aid)
            elif kind == "pmc" and not pmcid:
                pmcid = _text(aid)
        if not doi:
            for eloc in art.findall("ELocationID"):
                if (eloc.get("EIdType") or "").lower() == "doi":
                    doi = _text(eloc)
                    break

        pub_types = [t for t in (_text(n) for n in art.findall("PublicationTypeList/PublicationType")) if t]

        papers.append(PaperRecord(
            title=title, abstract=abstract, authors=authors, journal=journal, year=year,
            doi=doi, pmid=pmid, pmcid=pmcid, sources=["PubMed"], pub_types=pub_types,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
            is_open_access=False,  # PubMed does not state licence; Europe PMC supplies this
        ))
    return papers


class PubMedService:
    def __init__(self, settings: Settings, session: Optional[requests.Session] = None):
        self.settings = settings
        self.session = session or requests.Session()
        # NCBI allows 3 requests/s without a key and 10/s with one.
        self._min_interval = 0.11 if settings.ncbi_api_key else 0.34
        self._last_call = 0.0

    def _common_params(self) -> dict:
        params = {"tool": TOOL_NAME}
        if self.settings.ncbi_email:
            params["email"] = self.settings.ncbi_email
        if self.settings.ncbi_api_key:
            params["api_key"] = self.settings.ncbi_api_key
        return params

    def _get(self, endpoint: str, params: dict) -> requests.Response:
        wait = self._min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        merged = {**self._common_params(), **params}
        try:
            return request_with_retry(
                self.session, f"{BASE_URL}/{endpoint}", merged, source="PubMed",
                timeout=self.settings.http_timeout, max_retries=self.settings.max_retries,
                backoff=self.settings.retry_backoff_seconds, secrets=self.settings.secrets)
        finally:
            self._last_call = time.monotonic()

    # ------------------------------------------------------------------ steps
    def _esearch(self, query: str, year_from: int, year_to: int, retmax: int) -> tuple:
        response = self._get("esearch.fcgi", {
            "db": "pubmed", "term": query, "retmode": "json", "retmax": retmax,
            "sort": "relevance", "datetype": "pdat", "mindate": year_from, "maxdate": year_to,
        })
        try:
            data = response.json()["esearchresult"]
        except (ValueError, KeyError) as exc:
            raise ResearchAPIError("PubMed returned an unreadable search response.") from exc
        if data.get("ERROR"):
            raise ResearchAPIError("PubMed could not understand the search. Try simpler words.")
        return list(data.get("idlist", [])), int(data.get("count", 0) or 0)

    def _efetch(self, pmids: List[str]) -> List[PaperRecord]:
        papers: List[PaperRecord] = []
        for batch in chunked(pmids, FETCH_BATCH):
            response = self._get("efetch.fcgi", {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"})
            papers.extend(parse_pubmed_xml(response.text))
        return papers

    def count(self, query: str, year_from: int, year_to: int) -> int:
        """How many PubMed records match ``query`` (nothing is downloaded). Used by Idea Check."""
        _, total = self._esearch(query, year_from, year_to, retmax=0)
        return total

    # ----------------------------------------------------------------- public
    def search(self, request: SearchRequest) -> SourceSearchResult:
        """Search PubMed. Errors are returned in the result, not raised."""
        result = SourceSearchResult(source="PubMed", query=build_pubmed_query(request))
        try:
            pmids, total = self._esearch(result.query, request.year_from, request.year_to, request.max_papers)
            result.total_found = total
            if pmids:
                result.papers = self._efetch(pmids)
        except ResearchAPIError as exc:
            result.error = str(exc)
            log.warning("PubMed search failed: %s", exc)
        return result
