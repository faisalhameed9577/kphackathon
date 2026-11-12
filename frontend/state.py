"""Streamlit helpers: cached services, session-state defaults and small shared widgets."""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import streamlit as st

from config.settings import get_settings
from services.container import Services, build_services
from utils.helpers import get_logger, redact_secrets

log = get_logger(__name__)

PAGES = [
    "Dashboard", "Idea Check", "Research Discovery", "Research Library", "Research Assistant",
    "Research Comparison", "Research Gap Analysis", "Reports",
]

_DEFAULTS = {
    "nav_page": "Dashboard",
    "search_outcome": None,      # SearchOutcome of the latest search
    "search_request": None,      # validated SearchRequest of the latest search
    "search_topic_id": None,     # topic_id created for the latest search
    "search_id": 0,              # bumps on every search (resets the selection table)
    "index_report": None,
    "chat_history": [],          # RagAnswer objects asked in this session
    "analysis_notes": {},        # topic_id -> list of warning strings
    "auto_search": False,
}


@st.cache_resource(show_spinner="Starting ZETESIS AI...")
def get_services() -> Services:
    """Build all services once per Streamlit server process."""
    return build_services(get_settings())


def init_state() -> None:
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value if not isinstance(value, (list, dict)) else type(value)()


def go_to(page: str) -> None:
    """Callback for buttons that switch page (callbacks run before widgets are created)."""
    st.session_state["nav_page"] = page


def friendly_error(exc: Exception) -> str:
    """Log the full error, return a short secret-free message for the user."""
    log.exception("Unhandled error in UI action")
    return redact_secrets(str(exc) or type(exc).__name__, get_settings().secrets)


def make_progress(initial: str = "Starting...") -> Tuple[object, Callable[[float, str], None]]:
    bar = st.progress(0.0, text=initial)

    def update(fraction: float, message: str) -> None:
        bar.progress(min(max(float(fraction), 0.0), 1.0), text=message)

    return bar, update


def topic_selector(services: Services, key: str, label: str = "Research topic",
                   only_with_papers: bool = True) -> Optional[int]:
    """Dropdown of saved topics. Returns the chosen topic_id or None when there are none."""
    topics = services.topics.list(limit=100, only_with_papers=only_with_papers)
    if not topics:
        st.info("No research topics with saved papers yet. Go to **Research Discovery**, search, and index some papers first.")
        st.button("Go to Research Discovery", on_click=go_to, args=("Research Discovery",), key=f"{key}_go")
        return None
    ids = [t["topic_id"] for t in topics]
    default = st.session_state.get("search_topic_id")
    index = ids.index(default) if default in ids else 0
    labels = {t["topic_id"]: f"{t['title']}" + (f" - {t['region']}" if t.get("region") else "")
              + f"  ({t['paper_count']} papers)" for t in topics}
    return st.selectbox(label, ids, index=index, format_func=lambda i: labels[i], key=key)


def show_notes(notes: List[str]) -> None:
    for note in notes:
        st.warning(note)
