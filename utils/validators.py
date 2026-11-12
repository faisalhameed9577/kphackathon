"""Input validation for the search form (Pydantic)."""
from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional, Tuple

from pydantic import BaseModel, ValidationError, field_validator, model_validator

from utils.helpers import expand_region_terms, split_keywords, topic_to_terms

MIN_YEAR = 1900
MAX_PAPERS_LIMIT = 100


def _current_year() -> int:
    return datetime.now().year


class SearchRequest(BaseModel):
    """A validated research search."""

    topic: str
    keywords: str = ""
    region: str = ""
    year_from: int = 2010
    year_to: int = 0  # 0 means "this year"
    max_papers: int = 20

    @field_validator("topic")
    @classmethod
    def _check_topic(cls, value: str) -> str:
        value = re.sub(r"[\x00-\x1f]+", " ", value or "").strip()
        if len(value) < 4:
            raise ValueError("Please enter a research topic of at least 4 characters.")
        if len(value) > 300:
            raise ValueError("The topic is too long (maximum 300 characters).")
        if len(re.findall(r"[A-Za-z]", value)) < 3:
            raise ValueError("The topic must contain real words, for example 'diabetes in rural Pakistan'.")
        if not topic_to_terms(value):
            raise ValueError("The topic only contains generic words. Please name a condition, population or intervention.")
        return value

    @field_validator("keywords", "region")
    @classmethod
    def _trim(cls, value: str) -> str:
        return re.sub(r"[\x00-\x1f]+", " ", value or "").strip()[:300]

    @field_validator("max_papers")
    @classmethod
    def _check_max(cls, value: int) -> int:
        if value < 1 or value > MAX_PAPERS_LIMIT:
            raise ValueError(f"Maximum number of papers must be between 1 and {MAX_PAPERS_LIMIT}.")
        return value

    @model_validator(mode="after")
    def _check_years(self) -> "SearchRequest":
        if self.year_to == 0:
            self.year_to = _current_year()
        if self.year_from < MIN_YEAR or self.year_to > _current_year() + 1:
            raise ValueError(f"Years must be between {MIN_YEAR} and {_current_year() + 1}.")
        if self.year_from > self.year_to:
            raise ValueError("'From year' cannot be later than 'To year'.")
        return self

    # --------------------------------------------------------- derived values
    @property
    def topic_terms(self) -> List[str]:
        return topic_to_terms(self.topic)

    @property
    def keyword_terms(self) -> List[str]:
        return split_keywords(self.keywords)

    @property
    def region_terms(self) -> List[str]:
        return expand_region_terms(self.region)

    def as_filters(self) -> dict:
        return {
            "keywords": self.keywords,
            "region": self.region,
            "year_from": self.year_from,
            "year_to": self.year_to,
            "max_papers": self.max_papers,
        }


def validate_search_input(
    topic: str,
    keywords: str = "",
    region: str = "",
    year_from: int = 2010,
    year_to: int = 0,
    max_papers: int = 20,
) -> Tuple[Optional[SearchRequest], Optional[str]]:
    """Return ``(request, None)`` if valid, otherwise ``(None, friendly_error_message)``."""
    try:
        request = SearchRequest(
            topic=topic, keywords=keywords, region=region,
            year_from=year_from, year_to=year_to, max_papers=max_papers,
        )
        return request, None
    except ValidationError as exc:
        first = exc.errors()[0]
        message = str(first.get("msg", "Invalid input")).replace("Value error, ", "")
        return None, message
