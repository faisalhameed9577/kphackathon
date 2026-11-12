"""Page 2 - Research Discovery: search PubMed + Europe PMC, select papers, save and index."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from config.prompts import SEARCH_LIMITATION_NOTE
from database.db import DatabaseError
from frontend.state import friendly_error, get_services, go_to, make_progress
from frontend.theme import hero
from utils.validators import validate_search_input

EXAMPLE_TOPIC = "Diabetes research in rural Khyber Pakhtunkhwa"
EXAMPLE_REGION = "Khyber Pakhtunkhwa"


def _load_example() -> None:
    st.session_state["disc_topic"] = EXAMPLE_TOPIC
    st.session_state["disc_region"] = EXAMPLE_REGION
    st.session_state["disc_keywords"] = ""


def _run_search(topic, keywords, region, year_from, year_to, max_papers) -> None:
    request, error = validate_search_input(topic, keywords, region, year_from, year_to, max_papers)
    if error:
        st.error(error)
        return
    services = get_services()
    try:
        with st.spinner("Searching PubMed and Europe PMC..."):
            outcome = services.paper_service.search_papers(request)
            topic_id = services.paper_service.record_search(request, outcome)
    except DatabaseError as exc:
        st.error(f"Could not record the search: {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"The search failed: {friendly_error(exc)}")
        return
    st.session_state.update(search_outcome=outcome, search_request=request, search_topic_id=topic_id,
                            search_id=st.session_state["search_id"] + 1, index_report=None)


def _show_source_status(outcome) -> None:
    rows = [{"Source": r.source, "Records found": r.total_found, "Retrieved": len(r.papers),
             "Status": "Failed (see message below)" if r.error else "OK"} for r in outcome.per_source]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    for r in outcome.per_source:
        if r.error:
            st.warning(r.error)  # the message already names the source
    with st.expander("Queries sent to each source (search methodology)"):
        for r in outcome.per_source:
            st.markdown(f"**{r.source}**")
            st.code(r.query, language="text")
    if outcome.duplicates_removed:
        st.caption(f"{outcome.duplicates_removed} duplicate record(s) found in both sources were merged.")


def _results_table(outcome):
    rows = []
    for p in outcome.papers:
        rows.append({
            "Select": True, "Title": p.title, "Year": p.year, "Journal": p.journal,
            "Authors": ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else ""),
            "Sources": p.source_label, "Open access": "Yes" if p.is_open_access else "No",
            "Type": "Preprint" if p.is_preprint else "Article",
            "Abstract": "Yes" if p.has_abstract else "Missing", "Link": p.best_url})
    frame = pd.DataFrame(rows)
    return st.data_editor(
        frame, hide_index=True, width="stretch", key=f"disc_editor_{st.session_state['search_id']}",
        disabled=[c for c in frame.columns if c != "Select"],
        column_config={"Select": st.column_config.CheckboxColumn("Select", default=True),
                       "Link": st.column_config.LinkColumn("DOI / source link"),
                       "Year": st.column_config.NumberColumn("Year", format="%d")})


def render() -> None:
    services = get_services()
    hero("Research Discovery", "Search PubMed and Europe PMC, choose the papers you trust, and index them for analysis.")

    st.button("Use the example topic", on_click=_load_example, key="disc_example")
    this_year = datetime.now().year

    with st.form("search_form"):
        topic = st.text_input("Medical research topic *", key="disc_topic",
                              placeholder="e.g. Diabetes research in rural Khyber Pakhtunkhwa")
        c1, c2 = st.columns(2)
        keywords = c1.text_input("Optional keywords (comma separated)", key="disc_keywords",
                                 help="Keywords are OR-ed together and combined (AND) with the topic.")
        region = c2.text_input("Country or region", key="disc_region", placeholder="e.g. Khyber Pakhtunkhwa, Pakistan",
                               help="Searched in title/abstract. Khyber Pakhtunkhwa is expanded to KPK, NWFP, Peshawar and other common spellings.")
        c3, c4 = st.columns([2, 1])
        years = c3.slider("Publication years", 1990, this_year, (2010, this_year), key="disc_years")
        max_papers = c4.slider("Maximum papers", 5, 100, 20, step=5, key="disc_max")
        st.checkbox("Fetch open-access full text when available (Europe PMC) when I index papers", value=True, key="disc_fulltext")
        submitted = st.form_submit_button("Search PubMed + Europe PMC", type="primary")

    auto = st.session_state.pop("auto_search", False)
    if submitted or auto:
        _run_search(st.session_state.get("disc_topic", topic), keywords, region, years[0], years[1], max_papers)

    outcome = st.session_state.get("search_outcome")
    if outcome is None:
        st.info("Enter a topic and press **Search**. Tip: fewer, specific words work best, for example "
                "'diabetes' plus a region.")
        return

    st.subheader("Search results")
    _show_source_status(outcome)

    if all(r.error for r in outcome.per_source):
        st.error("Neither source could be reached. Check your internet connection and try again.")
        return
    if not outcome.papers:
        st.info("No papers matched these filters. Try broader words, remove the region, or widen the year range.")
        st.caption(SEARCH_LIMITATION_NOTE)
        return

    st.markdown(f"**{len(outcome.papers)} unique papers** (from {outcome.retrieved_before_dedup} records). "
                "Untick papers you do not want to keep.")
    edited = _results_table(outcome)
    selected = [outcome.papers[i] for i, flag in enumerate(edited["Select"].tolist()) if flag]

    preview = st.selectbox("Preview an abstract", list(range(len(outcome.papers))),
                           format_func=lambda i: outcome.papers[i].title[:110], key=f"disc_preview_{st.session_state['search_id']}")
    paper = outcome.papers[preview]
    with st.expander("Abstract", expanded=False):
        st.markdown(paper.abstract or "_This record has no abstract. Only its metadata can be saved._")
        if paper.best_url:
            st.markdown(f"[Open source link]({paper.best_url})")

    st.caption(f"{len(selected)} of {len(outcome.papers)} papers selected.")
    if st.button("Save and index selected papers", type="primary", disabled=not selected, key="disc_index"):
        bar, progress = make_progress("Preparing...")
        try:
            report = services.paper_service.save_and_index(
                selected, topic_id=st.session_state["search_topic_id"],
                fetch_fulltext=st.session_state.get("disc_fulltext", True), progress=progress)
            st.session_state["index_report"] = report
        except DatabaseError as exc:
            st.error(f"Could not save papers: {exc}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Indexing failed: {friendly_error(exc)}")
        finally:
            bar.empty()

    report = st.session_state.get("index_report")
    if report:
        st.success(f"Saved {report.papers_saved} papers. {report.chunks_new} new text chunks were embedded "
                   f"and {report.chunks_reused} were reused from earlier indexing ({report.embedding_backend} embeddings).")
        m1, m2, m3 = st.columns(3)
        m1.metric("Open-access full texts fetched", report.fulltext_fetched)
        m2.metric("Papers without abstract", report.papers_without_abstract)
        m3.metric("Chunks indexed", report.chunks_total)
        for warning in report.warnings:
            st.warning(warning)
        st.button("Ask questions about these papers", on_click=go_to, args=("Research Assistant",), key="disc_next_ask")
