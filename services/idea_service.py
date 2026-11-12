"""
Idea Check: "Should I do this research?"

The user writes a research idea (English, Urdu or Roman Urdu). The service answers:
has this been done, which part is still untouched, and should I GO, MODIFY or LEAVE it.

Flow
1. parse_idea()      idea -> parts (topic, population, intervention, outcome, region, design)
2. count queries     PubMed counts for the whole idea and for variations
                     (drop one part, other study designs, region -> country -> world)
3. nearest papers    the closest existing studies (PubMed + Europe PMC, ranked by similarity)
4. decide_verdict()  FIXED RULES on the counts -> GO / MODIFY / LEAVE / INCONCLUSIVE.
                     No AI is involved in the verdict, so it is repeatable and explainable.
5. explanation       Gemini (optional) explains the numbers in the user's language. Any number it
                     invents is detected and the rule-based explanation is used instead.

Important: a low count is a hint, not proof. PubMed does not index every regional journal and
keyword matching misses papers that use other words. The report always says so.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from config.prompts import IDEA_EXPLAIN_PROMPT, IDEA_PARSE_PROMPT, SYSTEM_INSTRUCTION, render
from database.models import PaperRecord, SearchOutcome
from services.embedding_service import EmbeddingService
from services.gemini_service import GeminiError, GeminiService
from services.paper_service import PaperService
from utils.helpers import expand_region_terms, get_logger, topic_to_terms
from utils.http import ResearchAPIError
from utils.validators import SearchRequest, validate_search_input

log = get_logger(__name__)

ProgressCallback = Callable[[float, str], None]

# ------------------------------------------------------------------ tunable rules
LEAVE_MIN_PAPERS = 20   # whole-idea matches at or above this (plus recent reviews) -> LEAVE
GO_MAX_PAPERS = 2       # whole-idea matches at or below this -> GO
RECENT_YEARS = 5
FIRST_YEAR = 1990
MAX_TERMS = 5
NEAREST_FETCH = 20
NEAREST_SHOW = 5
MAX_ANGLES = 6

VERDICT_HEADLINES = {
    "GO": "GO - this combination looks under-studied",
    "MODIFY": "MODIFY - it has been studied, but there are openings",
    "LEAVE": "LEAVE - this idea is already well covered",
    "INCONCLUSIVE": "INCONCLUSIVE - the search found nothing to judge",
}

# Study designs: key -> (display name, PubMed filter)
DESIGNS: Dict[str, Tuple[str, str]] = {
    "rct": ("randomized controlled trial", '"Randomized Controlled Trial"[pt]'),
    "systematic_review": ("systematic review / meta-analysis", '("Systematic Review"[pt] OR "Meta-Analysis"[pt])'),
    "cohort": ("cohort study", '"Cohort Studies"[mh]'),
    "cross_sectional": ("cross-sectional study", '"Cross-Sectional Studies"[mh]'),
    "qualitative": ("qualitative study", '"Qualitative Research"[mh]'),
}

FACET_ORDER = ["topic", "population", "intervention", "outcome", "region"]
FACET_NAMES = {"topic": "the topic", "population": "the population", "intervention": "the intervention",
               "outcome": "the outcome", "region": "the region"}

# Edit these to add places. Key = lower-case name the user may type.
_PROVINCE_TERMS = {
    "punjab": ["Lahore", "Rawalpindi", "Faisalabad", "Multan", "Punjab, Pakistan"],
    "sindh": ["Sindh", "Karachi", "Sukkur", "Hyderabad, Sindh"],
    "balochistan": ["Balochistan", "Quetta"],
    "islamabad": ["Islamabad"],
}
_COUNTRY_OF = {name: "Pakistan" for name in (
    "khyber pakhtunkhwa", "kp", "kpk", "nwfp", "peshawar", "abbottabad", "mardan", "swat", "kohat", "bannu",
    "dera ismail khan", "punjab", "sindh", "balochistan", "islamabad", "lahore", "karachi", "quetta")}
_REGION_WORDS = ["khyber pakhtunkhwa", "dera ismail khan", "north west frontier", "peshawar", "abbottabad", "mardan",
                 "swat", "kohat", "bannu", "kpk", "nwfp", "punjab", "sindh", "balochistan", "islamabad", "lahore",
                 "karachi", "quetta", "pakistan"]

_POPULATIONS = {  # rule-based fallback (used only when Gemini is not available)
    "children": (r"\bchild|\bpaediatric|\bpediatric|\binfant|\bneonat", ["children", "child", "paediatric", "pediatric"]),
    "women": (r"\bwom[ae]n\b|\bfemale|\bmaternal|\bpregnan", ["women", "female", "maternal", "pregnant women"]),
    "adolescents": (r"\badolescen|\bteen|\bstudents?\b", ["adolescents", "adolescent", "teenagers"]),
    "older adults": (r"\belderly|\bolder adult|\bgeriatric", ["elderly", "older adults", "geriatric"]),
    "rural populations": (r"\brural|\bvillage", ["rural", "village"]),
}
_DESIGN_WORDS = [
    ("rct", r"\brandomi[sz]ed|\brct\b|\bclinical trial|\btrial\b"),
    ("systematic_review", r"systematic review|meta-?analysis"),
    ("cohort", r"\bcohort\b"),
    ("cross_sectional", r"cross[- ]sectional"),
    ("qualitative", r"qualitative|focus group|interviews?\b"),
]
_RULES_STRIP = {"effect", "effects", "impact", "role", "using", "based", "among", "different", "new",
                "randomized", "randomised", "controlled", "trial", "trials", "cohort", "cross-sectional",
                "qualitative", "systematic", "review", "meta-analysis", "meta", "analysis", "survey"}

_NON_LATIN = re.compile(r"[؀-ۿݐ-ݿऀ-ॿ]")
_BAD_CHARS = re.compile(r"[\"\[\]\(\)\{\}:;<>]")


class IdeaError(Exception):
    """A problem with the idea text or with PubMed. The message is safe to show to users."""


# ------------------------------------------------------------------------ data
@dataclass
class Facet:
    name: str
    label: str
    terms: List[str]
    fields: Tuple[str, ...] = ("tiab",)  # PubMed fields: title/abstract, affiliation
    match: str = "any"                   # "any" = OR the terms, "all" = AND the terms


@dataclass
class IdeaFacets:
    original: str
    english: str
    language: str = "en"
    facets: Dict[str, Facet] = field(default_factory=dict)
    design: str = ""        # key of DESIGNS, or ""
    country: str = ""
    parsed_by: str = "ai"   # "ai" or "rules"

    @property
    def region_label(self) -> str:
        facet = self.facets.get("region")
        return facet.label if facet else ""

    @property
    def topic_label(self) -> str:
        facet = self.facets.get("topic")
        return facet.label if facet else self.english


@dataclass
class CountRow:
    label: str
    count: Optional[int]    # None = the PubMed request failed
    query: str
    year_from: int = FIRST_YEAR
    note: str = ""


@dataclass
class Angle:
    kind: str               # design | narrowing | region | combination
    text: str
    count: int


@dataclass
class NearestPaper:
    paper: PaperRecord
    similarity: float       # cosine similarity (0-1); only meaningful for ranking
    in_region: bool = False


@dataclass
class IdeaReport:
    idea: str
    facets: IdeaFacets
    verdict: str
    headline: str
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Optional[int]] = field(default_factory=dict)
    counts: List[CountRow] = field(default_factory=list)
    designs: List[CountRow] = field(default_factory=list)
    angles: List[Angle] = field(default_factory=list)
    advice: List[str] = field(default_factory=list)
    nearest: List[NearestPaper] = field(default_factory=list)
    explanation: str = ""
    explanation_by: str = "rules"
    notes: List[str] = field(default_factory=list)
    search_request: Optional[SearchRequest] = None
    search_outcome: Optional[SearchOutcome] = None
    recent_from: int = 0
    created: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))


# ----------------------------------------------------------------- text helpers
def _clean_term(value: object) -> str:
    text = re.sub(r"\s+", " ", _BAD_CHARS.sub(" ", str(value or ""))).strip()
    return text[:80]


def _clean_terms(values: object, limit: int = MAX_TERMS) -> List[str]:
    if isinstance(values, str):
        values = [values]
    out: List[str] = []
    seen: Set[str] = set()
    for value in values if isinstance(values, (list, tuple)) else []:
        term = _clean_term(value)
        if len(term) < 2 or term.upper() in {"AND", "OR", "NOT"} or term.lower() in seen:
            continue
        seen.add(term.lower())
        out.append(term)
    return out[:limit]


def region_terms(label: str) -> List[str]:
    """Spellings to search for a place name (Khyber Pakhtunkhwa is expanded)."""
    key = (label or "").lower().strip()
    if key in _PROVINCE_TERMS:
        return list(_PROVINCE_TERMS[key])
    return expand_region_terms(label) or ([_clean_term(label)] if _clean_term(label) else [])


def country_of(label: str) -> str:
    return _COUNTRY_OF.get((label or "").lower().strip(), "")


def region_facet(label: str, country: str = "") -> Optional[Facet]:
    label = _clean_term(label)
    terms = _clean_terms(region_terms(label), limit=12)
    if not terms:
        return None
    return Facet("region", label, terms, fields=("tiab", "ad"))


def _group(facet: Facet) -> str:
    per_term = []
    for term in facet.terms:
        parts = [f'"{term}"[{f}]' for f in facet.fields]
        per_term.append(parts[0] if len(parts) == 1 else "(" + " OR ".join(parts) + ")")
    if not per_term:
        return ""
    return "(" + (" AND " if facet.match == "all" else " OR ").join(per_term) + ")"


def compose_query(facets: IdeaFacets, include: Sequence[str], design: str = "",
                  region: Optional[Facet] = None) -> str:
    """PubMed query for the chosen parts of the idea (all parts are AND-ed)."""
    groups: List[str] = []
    for name in FACET_ORDER:
        if name not in include:
            continue
        facet = region if (name == "region" and region is not None) else facets.facets.get(name)
        if facet and facet.terms:
            groups.append(_group(facet))
    if design and design in DESIGNS:
        groups.append(DESIGNS[design][1])
    return " AND ".join(g for g in groups if g)


def decide_verdict(full: int, topic_only: int, recent_full: int, sr_recent: int,
                   leave_min: int = LEAVE_MIN_PAPERS, go_max: int = GO_MAX_PAPERS) -> str:
    """The rules behind GO / MODIFY / LEAVE. Pure function so it is easy to test and to explain."""
    if topic_only == 0:
        return "INCONCLUSIVE"
    if full >= leave_min and (sr_recent >= 1 or recent_full >= 10):
        return "LEAVE"
    if full <= go_max:
        return "GO"
    return "MODIFY"


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _numbers(text: str) -> Set[int]:
    found: Set[int] = set()
    for token in re.findall(r"\d+", (text or "").replace(",", "")):
        try:
            found.add(int(token))
        except ValueError:
            continue
    return found


# ---------------------------------------------------------------- count helper
class _Counter:
    """Runs each PubMed count query once, keeps a record of every query for transparency."""

    def __init__(self, pubmed, year_to: int, progress: Optional[ProgressCallback], expected: int):
        self.pubmed = pubmed
        self.year_to = year_to
        self.progress = progress
        self.expected = max(1, expected)
        self.rows: List[CountRow] = []
        self.errors: List[str] = []
        self._cache: Dict[Tuple[str, int], Optional[int]] = {}

    def get(self, label: str, query: str, year_from: int = FIRST_YEAR, note: str = "") -> Optional[int]:
        key = (query, year_from)
        if key in self._cache:
            return self._cache[key]
        if self.progress:
            done = len(self._cache)
            self.progress(0.10 + 0.60 * min(done / self.expected, 1.0), f"Counting papers: {label}")
        try:
            value: Optional[int] = int(self.pubmed.count(query, year_from, self.year_to))
        except ResearchAPIError as exc:
            value = None
            self.errors.append(str(exc))
        self._cache[key] = value
        self.rows.append(CountRow(label, value, query, year_from, note))
        return value


# --------------------------------------------------------------------- service
class IdeaService:
    def __init__(self, gemini: GeminiService, paper_service: PaperService, embedder: EmbeddingService):
        self.gemini = gemini
        self.paper_service = paper_service
        self.embedder = embedder

    # ---------------------------------------------------------------- parsing
    def parse_idea(self, idea: str, region: str = "", notes: Optional[List[str]] = None) -> IdeaFacets:
        notes = notes if notes is not None else []
        idea = re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f]+", " ", idea or "")).strip()
        if len(idea) < 8:
            raise IdeaError("Please describe your research idea in at least one full sentence.")
        if len(idea) > 600:
            raise IdeaError("The idea is too long (maximum 600 characters). Describe the core idea only.")
        if self.gemini.available:
            try:
                return self._parse_with_ai(idea, region)
            except GeminiError as exc:
                notes.append(f"AI parsing unavailable: {exc} Simple word-matching was used instead.")
        if _NON_LATIN.search(idea):
            raise IdeaError("Ideas written in Urdu or other scripts need Gemini for translation. Add GEMINI_API_KEY "
                            "to your .env file, or write the idea in English.")
        return self._parse_with_rules(idea, region)

    def _parse_with_ai(self, idea: str, region: str) -> IdeaFacets:
        data = self.gemini.generate_json(render(IDEA_PARSE_PROMPT, IDEA=idea, REGION=region or "not given"),
                                         SYSTEM_INSTRUCTION)
        if not isinstance(data, dict):
            raise GeminiError("Gemini returned an unexpected format for the idea.")
        english = re.sub(r"\s+", " ", str(data.get("english_idea") or idea)).strip()[:600]
        facets: Dict[str, Facet] = {}
        for name in ("topic", "population", "intervention", "outcome"):
            facet = self._facet_from(data.get(name), name)
            if facet:
                facets[name] = facet
        if "topic" not in facets:  # the topic is required: rebuild it from the English idea
            words = [w for w in topic_to_terms(english) if w.lower() not in _RULES_STRIP][:4]
            if not words:
                raise IdeaError("Could not find a medical topic in the idea. Please name a condition or subject.")
            facets["topic"] = Facet("topic", " ".join(words), words, match="all")
        region_data = data.get("region") if isinstance(data.get("region"), dict) else {}
        label = _clean_term(region or region_data.get("label"))
        country = _clean_term(region_data.get("country")) or country_of(label)
        facet = region_facet(label)
        if facet:
            facets["region"] = facet
        if country.lower() == label.lower():
            country = ""
        design = str(data.get("design") or "").strip().lower()
        return IdeaFacets(original=idea, english=english, language=str(data.get("language") or "en").lower()[:5],
                          facets=facets, design=design if design in DESIGNS else "", country=country, parsed_by="ai")

    @staticmethod
    def _facet_from(data: object, name: str) -> Optional[Facet]:
        if not isinstance(data, dict):
            return None
        label = _clean_term(data.get("label"))
        terms = _clean_terms(data.get("terms"))
        if label and label.lower() not in [t.lower() for t in terms]:
            terms.insert(0, label)
        terms = terms[:MAX_TERMS]
        return Facet(name, label or terms[0], terms) if terms else None

    def _parse_with_rules(self, idea: str, region: str) -> IdeaFacets:
        text = idea.lower()
        facets: Dict[str, Facet] = {}
        strip: Set[str] = set()

        region_label = _clean_term(region)
        if not region_label:
            for word in _REGION_WORDS:
                if re.search(rf"\b{re.escape(word)}\b", text):
                    region_label = word.title()
                    break
        if region_label:
            strip.update(re.findall(r"[a-z]+", region_label.lower()))
            facet = region_facet(region_label)
            if facet:
                facets["region"] = facet

        for label, (pattern, terms) in _POPULATIONS.items():
            if re.search(pattern, text):
                facets["population"] = Facet("population", label, list(terms))
                strip.update(re.findall(r"[a-z]+", " ".join(terms).lower()))
                strip.update({"child", "childhood", "woman", "adolescent", "elderly", "student", "students"})
                break

        design = ""
        for key, pattern in _DESIGN_WORDS:
            if re.search(pattern, text):
                design = key
                break

        words = [w for w in topic_to_terms(idea) if w.lower() not in _RULES_STRIP and w.lower() not in strip
                 and len(w) > 2][:4]
        if not words:
            raise IdeaError("Could not find a medical topic in the idea. Please name a condition or subject.")
        facets["topic"] = Facet("topic", " ".join(words), words, match="all")
        label = facets["region"].label if "region" in facets else ""
        return IdeaFacets(original=idea, english=idea, language="en", facets=facets, design=design,
                          country=country_of(label), parsed_by="rules")

    # -------------------------------------------------------------- main entry
    def run(self, idea: str, region: str = "", progress: Optional[ProgressCallback] = None) -> IdeaReport:
        notify = progress or (lambda fraction, message: None)
        notes: List[str] = []
        year_to = datetime.now().year
        recent_from = year_to - RECENT_YEARS + 1

        notify(0.02, "Understanding your idea...")
        facets = self.parse_idea(idea, region, notes)
        include = [n for n in FACET_ORDER if n in facets.facets]
        extra = [n for n in include if n != "topic"]
        counter = _Counter(self.paper_service.pubmed, year_to, progress, 6 + 2 * len(extra) + len(DESIGNS))

        full_query = compose_query(facets, include, facets.design)
        full = counter.get("Your whole idea", full_query, note="all parts of the idea together")
        if full is None:
            raise IdeaError(counter.errors[0] if counter.errors else "PubMed could not be reached. Check your internet connection.")
        topic_only = counter.get("The topic alone", compose_query(facets, ["topic"]))
        if topic_only is None:
            topic_only = 0
        recent_full = counter.get(f"Whole idea since {recent_from}", full_query, recent_from)
        sr_filter = DESIGNS["systematic_review"][1]
        sr_recent = counter.get(f"Reviews / meta-analyses since {recent_from}",
                                f"({full_query}) AND {sr_filter}" if full_query else sr_filter, recent_from)

        # Variations: drop one part of the idea at a time, and topic + one part.
        without: Dict[str, Optional[int]] = {}
        single: Dict[str, Optional[int]] = {}
        if topic_only > 0:
            for name in extra:
                rest = [n for n in include if n != name]
                without[name] = counter.get(f"Without {FACET_NAMES[name]}", compose_query(facets, rest, facets.design))
                single[name] = counter.get(f"Topic + {FACET_NAMES[name]} only", compose_query(facets, ["topic", name]))
            if facets.design:
                without["design"] = counter.get("Without the study design", compose_query(facets, include, ""))

        # Study-design coverage of the idea (the user's own design is left out on purpose).
        design_rows: List[CountRow] = []
        if topic_only > 0:
            for key, (label, _) in DESIGNS.items():
                value = counter.get(f"{label}", compose_query(facets, include, key))
                design_rows.append(CountRow(label, value, compose_query(facets, include, key), note=key))

        # Region ladder: region -> country -> worldwide.
        country_count: Optional[int] = None
        if topic_only > 0 and "region" in facets.facets and facets.country:
            country_facet = Facet("region", facets.country, [facets.country], fields=("tiab", "ad"))
            country_count = counter.get(f"Same idea in {facets.country}",
                                        compose_query(facets, include, facets.design, region=country_facet))

        if counter.errors:
            notes.append(f"{len(counter.errors)} PubMed count request(s) failed, so some numbers are missing.")

        verdict = decide_verdict(full, topic_only, recent_full or 0, sr_recent or 0)
        metrics = {"whole_idea": full, "topic_only": topic_only, "recent": recent_full, "reviews_recent": sr_recent,
                   "region_local": full if "region" in facets.facets else None,
                   "region_country": country_count,
                   "region_world": without.get("region")}
        angles = self._angles(facets, full, topic_only, without, single, design_rows, country_count)
        reasons = self._reasons(facets, metrics, single, recent_from)
        advice = self._advice(verdict, angles)
        notes.extend(self._caveats(verdict, topic_only))

        report = IdeaReport(idea=idea, facets=facets, verdict=verdict, headline=VERDICT_HEADLINES[verdict],
                            reasons=reasons, metrics=metrics, counts=counter.rows, designs=design_rows,
                            angles=angles, advice=advice, notes=notes, recent_from=recent_from)

        notify(0.72, "Finding the closest existing studies...")
        self._attach_nearest(report)
        notify(0.92, "Writing the explanation...")
        self._attach_explanation(report)
        notify(1.0, "Done")
        return report

    # ------------------------------------------------------------ rule outputs
    @staticmethod
    def _angles(facets: IdeaFacets, full: int, topic_only: int, without: Dict[str, Optional[int]],
                single: Dict[str, Optional[int]], designs: List[CountRow], country_count: Optional[int]) -> List[Angle]:
        angles: List[Angle] = []
        region = facets.region_label
        world = without.get("region")

        # 1. Local evidence is thin while the wider evidence is strong (the main "untouched" signal).
        if region and world is not None and world >= 10 and full <= 2:
            where = f" ({country_count} in {facets.country})" if country_count is not None and facets.country else ""
            angles.append(Angle("region", f"Worldwide there are {world} papers on this idea, but only {full} mention "
                                f"{region}{where}. A local study or replication would be new.", full))

        # 2. A part of the idea that cuts the evidence sharply.
        for name, value in without.items():
            if value is None or name == "region":
                continue
            if value >= max(10, 5 * max(full, 1)) and full < LEAVE_MIN_PAPERS:
                what = "the study design" if name == "design" else (facets.facets[name].label if name in facets.facets else name)
                angles.append(Angle("narrowing", f"Without '{what}' there are {value} papers; with it only {full}. "
                                    f"This is the thin part, so work focused on it would be new.", full))

        # 3. Study designs that are missing for this exact idea.
        if full >= 3:
            for row in designs:
                if row.count is not None and row.count <= 2:
                    angles.append(Angle("design", f"Only {row.count} {row.label} paper(s) among the {full} on this idea. "
                                        f"A {row.label} would add something new.", row.count))

        # 4. A rare topic + part combination although the topic itself is common.
        for name, value in single.items():
            if value is not None and value <= 5 and topic_only >= 20:
                label = facets.facets[name].label if name in facets.facets else name
                angles.append(Angle("combination", f"'{facets.topic_label}' together with '{label}' appears in only "
                                    f"{value} paper(s), against {topic_only} on the topic alone.", value))

        angles.sort(key=lambda a: a.count)
        return angles[:MAX_ANGLES]

    @staticmethod
    def _reasons(facets: IdeaFacets, m: Dict[str, Optional[int]], single: Dict[str, Optional[int]],
                 recent_from: int) -> List[str]:
        reasons = [f"{m['whole_idea']:,} paper(s) in PubMed match your whole idea ({m['recent'] if m['recent'] is not None else '?'}"
                   f" published since {recent_from}).",
                   f"{m['reviews_recent'] if m['reviews_recent'] is not None else '?'} systematic review(s) or "
                   f"meta-analysis on it since {recent_from}.",
                   f"The topic alone ('{facets.topic_label}') matches {m['topic_only']:,} paper(s)."]
        for name, value in single.items():
            if value is not None and name in facets.facets:
                reasons.append(f"Topic + {facets.facets[name].label}: {value:,} paper(s).")
        if m.get("region_world") is not None and facets.region_label:
            line = f"Region: {m['region_local']} mentioning {facets.region_label}"
            if m.get("region_country") is not None and facets.country:
                line += f", {m['region_country']} in {facets.country}"
            reasons.append(line + f", {m['region_world']} worldwide.")
        return reasons

    @staticmethod
    def _advice(verdict: str, angles: List[Angle]) -> List[str]:
        top = [a.text for a in angles[:3]]
        if verdict == "GO":
            return ["Few papers match this exact idea. Read the closest studies below to confirm it is really new, "
                    "then refine your research question."] + top
        if verdict == "MODIFY":
            return ["The idea has been studied. Keep it, but change the part that is thin:"] + (top or [
                "No single thin part was found. Try a different population, place, outcome or study design."])
        if verdict == "LEAVE":
            return ["Repeating this adds little. Change the idea instead; these are the least-covered angles:"] + (top or [
                "Try a different population, place, outcome or study design."])
        return ["Rephrase the idea with simpler English words (for example the disease name and the population)."]

    @staticmethod
    def _caveats(verdict: str, topic_only: int) -> List[str]:
        caveats = ["Counts come from PubMed titles and abstracts using keyword matching. A low count can also mean the "
                   "topic is poorly indexed or described with other words. Read the closest studies before deciding.",
                   "Europe PMC, Scopus, regional journals and unpublished work are not part of the counts."]
        if verdict == "GO" and topic_only < 5:
            caveats.append("Even the topic alone has very few papers. Check the spelling or broaden the wording.")
        return caveats

    # ---------------------------------------------------------- nearest papers
    def _attach_nearest(self, report: IdeaReport) -> None:
        facets = report.facets
        try:
            keywords = ", ".join(facets.facets[n].label for n in ("population", "intervention", "outcome")
                                 if n in facets.facets)
            request, error = validate_search_input(facets.topic_label, keywords, "", 2000, 0, NEAREST_FETCH)
            if request is None:
                request, error = validate_search_input(facets.english[:300], "", "", 2000, 0, NEAREST_FETCH)
            if request is None:
                report.notes.append(f"Closest studies could not be searched: {error}")
                return
            outcome = self.paper_service.search_papers(request)
            report.search_request, report.search_outcome = request, outcome
            for source in outcome.per_source:
                if source.error:
                    report.notes.append(source.error)
            report.nearest = self._rank(facets, outcome.papers)
        except Exception as exc:  # noqa: BLE001 - the verdict is still useful without this part
            log.warning("Nearest-paper search failed: %s", exc)
            report.notes.append("The closest existing studies could not be loaded.")

    def _rank(self, facets: IdeaFacets, papers: Sequence[PaperRecord]) -> List[NearestPaper]:
        if not papers:
            return []
        docs = [f"{p.title}. {p.abstract[:1200]}" for p in papers]
        try:
            vectors = self.embedder.embed_documents(docs)
            query = self.embedder.embed_query(facets.english)
            if len(vectors) != len(docs) or any(len(v) != len(query) for v in vectors):
                raise ValueError("embedding size mismatch")
        except Exception:  # noqa: BLE001 - fall back to the offline word-based embedder
            vectors = [self.embedder.local.embed(d) for d in docs]
            query = self.embedder.local.embed(facets.english)
        terms = [t.lower() for t in (facets.facets["region"].terms if "region" in facets.facets else [])]
        ranked = []
        for paper, vector in zip(papers, vectors):
            text = f"{paper.title} {paper.abstract}".lower()
            ranked.append(NearestPaper(paper, round(_cosine(query, vector), 3), any(t.lower() in text for t in terms)))
        ranked.sort(key=lambda n: n.similarity, reverse=True)
        return ranked[:NEAREST_SHOW]

    # ------------------------------------------------------------- explanation
    def _attach_explanation(self, report: IdeaReport) -> None:
        rule_text = self._rule_explanation(report)
        report.explanation, report.explanation_by = rule_text, "rules"
        if not self.gemini.available:
            return
        facts = "\n".join(
            ["Verdict: " + report.verdict] + [f"- {r}" for r in report.reasons] +
            [f"- Untouched angle: {a.text}" for a in report.angles] +
            [f"- Closest study: {n.paper.title} ({n.paper.year or 'n.d.'})" for n in report.nearest[:3]])
        language = {"ur": "Urdu (Arabic script)", "en": "English"}.get(report.facets.language, "English")
        try:
            text = self.gemini.generate_text(render(IDEA_EXPLAIN_PROMPT, VERDICT=report.verdict, LANGUAGE=language,
                                                    FACTS=facts), SYSTEM_INSTRUCTION, temperature=0.2).strip()
        except GeminiError as exc:
            report.notes.append(f"AI explanation unavailable: {exc} A rule-based explanation is shown.")
            return
        allowed = _numbers(facts) | set(range(1990, datetime.now().year + 2))
        invented = _numbers(text) - allowed
        if not text or invented:
            report.notes.append("The AI explanation contained numbers that are not in the data, so it was discarded. "
                                "A rule-based explanation is shown.")
            return
        report.explanation, report.explanation_by = text, "gemini"

    @staticmethod
    def _rule_explanation(report: IdeaReport) -> str:
        m = report.metrics
        full, topic = m.get("whole_idea") or 0, m.get("topic_only") or 0
        first = report.angles[0].text if report.angles else ""
        if report.verdict == "GO":
            return (f"Only {full} paper(s) match your whole idea, although the topic itself has {topic}. This suggests the "
                    f"combination is under-studied. Confirm by reading the closest studies, then refine your question.")
        if report.verdict == "MODIFY":
            return (f"{full} paper(s) match your idea, so it has been studied, but not completely. {first} "
                    f"Consider keeping the idea and changing that part.").strip()
        if report.verdict == "LEAVE":
            return (f"{full} paper(s) match your idea, including recent reviews, so it is already well covered. "
                    f"Repeating it would add little. {first}").strip()
        return "The search did not find papers on this topic. Try simpler English words and check the spelling."
