"""Page - Idea Check: "Should I do this research?" (GO / MODIFY / LEAVE)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from database.db import DatabaseError
from frontend.state import friendly_error, get_services, go_to, make_progress
from frontend.theme import hero, neutral_badge
from services.evidence_labels import evidence_badge, is_retracted
from services.idea_pdf import IdeaPdfService
from services.idea_service import (GO_MAX_PAPERS, IdeaError, IdeaReport, LEAVE_MIN_PAPERS)

EXAMPLES = {
    "School exercise and obesity (Peshawar)": "Effect of school-based exercise programs on childhood obesity in Peshawar",
    "Diabetes in rural KP": "Prevalence and risk factors of type 2 diabetes among adults in rural Khyber Pakhtunkhwa",
    "Anaemia in pregnancy": "Iron supplementation for anaemia in pregnant women in Khyber Pakhtunkhwa, randomized trial",
}

_STYLE = {  # verdict -> (icon, colour, one-line meaning)
    "GO": ("🟢", "success", "Few papers match this exact idea. There is room for new work."),
    "MODIFY": ("🟡", "warning", "The idea has been studied. Change the thin part to make it new."),
    "LEAVE": ("🔴", "error", "The idea is already well covered. Change it instead of repeating it."),
    "INCONCLUSIVE": ("⚪", "info", "The search found nothing to judge. Rephrase the idea."),
}


def _load_example(text: str) -> None:
    st.session_state["idea_text"] = text


def _count_label(value) -> str:
    return "n/a" if value is None else f"{value:,}"


def _run(idea: str, region: str) -> None:
    services = get_services()
    bar, progress = make_progress("Starting...")
    try:
        st.session_state["idea_report"] = services.idea.run(idea, region, progress)
        st.session_state["idea_saved_topic"] = None
    except IdeaError as exc:
        st.session_state["idea_report"] = None
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        st.session_state["idea_report"] = None
        st.error(f"The idea check failed: {friendly_error(exc)}")
    finally:
        bar.empty()


def _save_and_analyse(report: IdeaReport) -> None:
    """Save the closest papers to the library and run the full gap analysis on them."""
    services = get_services()
    outcome, request = report.search_outcome, report.search_request
    if not (outcome and request and outcome.papers):
        st.warning("There are no papers to save for this idea.")
        return
    bar, progress = make_progress("Saving papers...")
    try:
        request = request.model_copy(update={"region": report.facets.region_label})
        topic_id = services.paper_service.record_search(request, outcome)
        index_report = services.paper_service.save_and_index(
            outcome.papers, topic_id=topic_id, fetch_fulltext=True, progress=lambda f, m: progress(0.5 * f, m))
        services.pipeline.run_gap_analysis(topic_id, progress=lambda f, m: progress(0.5 + 0.5 * f, m))
        st.session_state.update(search_topic_id=topic_id, idea_saved_topic=topic_id)
        for warning in index_report.warnings[:3]:
            st.warning(warning)
    except (DatabaseError, ValueError) as exc:
        st.error(f"Could not save and analyse the papers: {exc}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Saving or analysis failed: {friendly_error(exc)}")
    finally:
        bar.empty()


def _pdf_bytes(report: IdeaReport) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = IdeaPdfService().generate_idea(report, Path(tmp) / "idea_check.pdf")
        return path.read_bytes()


def _show_report(report: IdeaReport) -> None:
    icon, colour, meaning = _STYLE.get(report.verdict, _STYLE["INCONCLUSIVE"])
    box = getattr(st, colour)
    box(f"### {icon} {report.headline}\n\n{meaning}")

    if report.explanation:
        st.markdown(report.explanation)
        st.caption("Explanation written by Gemini from the numbers below." if report.explanation_by == "gemini"
                   else "Rule-based explanation built from the numbers below.")

    m = report.metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Papers on your whole idea", _count_label(m.get("whole_idea")))
    c2.metric(f"Since {report.recent_from}", _count_label(m.get("recent")))
    c3.metric("Recent reviews", _count_label(m.get("reviews_recent")))
    c4.metric("Topic alone", _count_label(m.get("topic_only")))

    if m.get("region_world") is not None and report.facets.region_label:
        st.markdown(f"**Where the evidence comes from** (region: {report.facets.region_label})")
        r1, r2, r3 = st.columns(3)
        r1.metric(f"Mentioning {report.facets.region_label}", _count_label(m.get("region_local")))
        r2.metric(report.facets.country or "Country", _count_label(m.get("region_country")))
        r3.metric("Worldwide", _count_label(m.get("region_world")))

    st.markdown("#### What to do next")
    for line in report.advice:
        st.markdown(f"- {line}")

    if report.angles:
        st.markdown("#### Untouched or thin angles")
        for angle in report.angles:
            st.markdown(f"- {neutral_badge(angle.kind)} {angle.text}", unsafe_allow_html=True)

    tab_near, tab_numbers, tab_design = st.tabs(["Closest existing studies", "Numbers behind the verdict", "Study designs"])

    with tab_near:
        if not report.nearest:
            st.info("No closest studies could be loaded.")
        for item in report.nearest:
            p = item.paper
            with st.container(border=True):
                link = f"[{p.title}]({p.best_url})" if p.best_url else p.title
                st.markdown(f"**{link}**")
                if is_retracted(p.pub_types):
                    st.error("Retracted publication. Do not cite this paper.")
                st.markdown(neutral_badge(evidence_badge(p.pub_types, p.is_preprint)), unsafe_allow_html=True)
                st.caption(f"{', '.join(p.authors[:3])}{' et al.' if len(p.authors) > 3 else ''} · {p.journal or 'n/a'} · "
                           f"{p.year or 'n.d.'} · similarity {item.similarity:.2f}"
                           + (f" · mentions {report.facets.region_label}" if item.in_region else ""))
                if p.abstract:
                    st.write(p.abstract[:420] + ("..." if len(p.abstract) > 420 else ""))
        st.caption("Similarity is only for ranking these papers. Read them to decide whether your idea is really covered.")

    with tab_numbers:
        st.markdown("\n".join(f"- {r}" for r in report.reasons))
        frame = pd.DataFrame([{"Check": r.label, "Papers": r.count, "From year": r.year_from} for r in report.counts])
        st.dataframe(frame, hide_index=True, width="stretch")
        with st.expander("PubMed queries used (search methodology)"):
            for r in report.counts:
                st.markdown(f"**{r.label}**")
                st.code(r.query or "(empty)", language="text")
        st.caption(f"Verdict rules: GO when the whole idea has {GO_MAX_PAPERS} or fewer papers; LEAVE when it has "
                   f"{LEAVE_MIN_PAPERS} or more and there is a recent review; otherwise MODIFY.")

    with tab_design:
        rows = [{"Study design": d.label, "Papers on your idea": d.count} for d in report.designs]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("A design with very few papers is a possible opening for your study.")
        else:
            st.info("Design counts are not available for this idea.")

    for note in report.notes:
        st.caption(f"ℹ️ {note}")

    st.divider()
    try:
        st.download_button("Download this Idea Check as a PDF", _pdf_bytes(report),
                           file_name=f"zetesis_idea_check_{report.created[:10]}.pdf", mime="application/pdf",
                           key="idea_pdf")
    except Exception as exc:  # noqa: BLE001 - the on-screen report stays usable
        st.caption(f"The PDF could not be created: {friendly_error(exc)}")
    can_save = bool(report.search_outcome and report.search_outcome.papers)
    if st.button("Save the closest papers and run the full gap analysis", type="primary", disabled=not can_save,
                 key="idea_save"):
        _save_and_analyse(report)
    if st.session_state.get("idea_saved_topic"):
        st.success("Papers saved and analysed. Open the Gap Analysis page to see limitations, gaps and research questions.")
        st.button("Open Research Gap Analysis", on_click=go_to, args=("Research Gap Analysis",), key="idea_open_gaps")


def render() -> None:
    hero("Idea Check", "Write your research idea. Find out if it has been done, what is still untouched, "
                       "and whether to go ahead, change it or leave it.")
    st.caption("You can write in English or Urdu (Urdu needs a Gemini key). Include the topic, the population and the place.")

    cols = st.columns(len(EXAMPLES))
    for col, (label, text) in zip(cols, EXAMPLES.items()):
        col.button(label, on_click=_load_example, args=(text,), key=f"idea_ex_{label}")

    with st.form("idea_form"):
        idea = st.text_area("Your research idea *", key="idea_text", height=110,
                            placeholder="e.g. Effect of a school exercise program on obesity in children in Peshawar")
        region = st.text_input("Place (optional)", key="idea_region", placeholder="e.g. Khyber Pakhtunkhwa",
                               help="Leave empty if the place is already in your idea.")
        submitted = st.form_submit_button("Check my idea", type="primary")

    if submitted:
        _run(idea, region)

    report = st.session_state.get("idea_report")
    if report is None:
        st.info("Type an idea and press **Check my idea**. It takes about 10-20 seconds.")
        return
    _show_report(report)
    st.caption("Idea Check is a planning aid, not a guarantee. Always read the closest studies and ask your supervisor.")
