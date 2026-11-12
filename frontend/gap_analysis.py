"""Page 6 - Research Gap Analysis: limitations, possible gaps and research questions."""
from __future__ import annotations

import streamlit as st

from frontend.state import friendly_error, get_services, make_progress, show_notes, topic_selector
from frontend.theme import badge, hero, neutral_badge
from utils.citation_utils import format_inline, humanize_labels

TYPE_NAMES = {"population": "Population", "location": "Location", "methodology": "Methodology", "outcome": "Outcome",
              "temporal": "Recent evidence", "evidence_quality": "Evidence depth", "other": "Other"}


def render() -> None:
    services = get_services()
    hero("Research Gap Analysis",
         "Limitations stated by the authors, possible research gaps, and suggested research questions - kept clearly separate.")

    topic_id = topic_selector(services, "gap_topic")
    if topic_id is None:
        return

    c1, c2 = st.columns([1, 3])
    force = c2.checkbox("Re-run everything (ignore saved results)", key="gap_force")
    if c1.button("Run analysis", type="primary", key="gap_run"):
        bar, progress = make_progress("Starting analysis...")
        try:
            bundle = services.pipeline.run_gap_analysis(topic_id, force=force, progress=progress)
            st.session_state["analysis_notes"][topic_id] = bundle.notes
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Analysis failed: {friendly_error(exc)}")
        finally:
            bar.empty()

    bundle = services.pipeline.load_bundle(topic_id)
    if not (bundle.summaries or bundle.gaps):
        st.info("No analysis has been run for this topic yet. Press **Run analysis**. "
                + ("Gemini will be used." if services.gemini.available else
                   "Gemini is not configured, so rule-based methods will be used."))
        return

    show_notes(st.session_state["analysis_notes"].get(topic_id, []))
    by_id = {p.paper_id: p for p in bundle.papers}
    tab_lim, tab_gap, tab_q = st.tabs(["Limitations", "Possible research gaps", "Suggested research questions"])

    with tab_lim:
        st.markdown(badge("limitation"), unsafe_allow_html=True)
        st.caption("Only limitations that the authors themselves state in the available text. Many abstracts state none.")
        none_stated = 0
        for p in bundle.papers:
            items = bundle.limitations.get(p.paper_id, [])
            if not items:
                none_stated += 1
                continue
            with st.expander(f"{format_inline(p)} - {p.title[:100]}"):
                for lim in items:
                    flag = {"keyword-match": " _(keyword match - please verify)_",
                            "gemini-unverified": " _(AI-reported; quote could not be verified in the text)_"}.get(lim.method, "")
                    st.markdown(f"- {lim.text}{flag}")
                    if lim.evidence:
                        st.caption(f"Evidence: “{lim.evidence}”")
        if none_stated:
            st.caption(f"{none_stated} paper(s) had no explicit limitations in the available text. "
                       "This does not mean those studies have no limitations.")

    with tab_gap:
        st.markdown(badge("gap"), unsafe_allow_html=True)
        st.caption("Hypotheses suggested by the retrieved papers. They do not prove that a topic is unresearched.")
        if not bundle.gaps:
            st.info("No gaps were identified.")
        for i, gap in enumerate(bundle.gaps, start=1):
            with st.container(border=True):
                st.markdown(f"**Gap {i}: {humanize_labels(gap.description, by_id.values())}**")
                st.markdown(neutral_badge(TYPE_NAMES.get(gap.gap_type, gap.gap_type))
                            + neutral_badge("Gemini" if gap.origin == "gemini" else "Coverage analysis (rules)")
                            + badge("verify"), unsafe_allow_html=True)
                st.markdown(f"**Why this was identified:** {humanize_labels(gap.rationale, by_id.values()) or 'Not provided.'}")
                if gap.supporting_paper_ids:
                    st.markdown("**Supporting references:**")
                    for pid in gap.supporting_paper_ids:
                        p = by_id.get(pid)
                        if p:
                            link = f"[{p.title}]({p.best_url})" if p.best_url else p.title
                            st.markdown(f"- {format_inline(p)} - {link}")
                else:
                    st.caption("No single paper supports this gap; it is derived from what the retrieved set does not contain.")

    with tab_q:
        st.markdown(badge("verify"), unsafe_allow_html=True)
        st.caption("Suggestions for researchers. They need expert review and ethical approval before use.")
        if not bundle.questions:
            st.info("No research questions were generated.")
        for i, q in enumerate(bundle.questions, start=1):
            with st.container(border=True):
                st.markdown(f"**Q{i}. {q.question}**")
                if q.rationale:
                    st.markdown(f"**Rationale:** {humanize_labels(q.rationale, by_id.values())}")
                if q.suggested_design:
                    st.markdown(f"**Possible design:** {q.suggested_design}")
                if q.gap_index is not None and q.gap_index < len(bundle.gaps):
                    st.caption(f"Linked to Gap {q.gap_index + 1}  ·  "
                               + ("generated by Gemini" if q.origin == "gemini" else "template-based"))
