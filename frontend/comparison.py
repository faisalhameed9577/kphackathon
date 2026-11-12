"""Page 5 - Research Comparison: compare selected papers side by side."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.state import friendly_error, get_services, make_progress, show_notes, topic_selector
from frontend.theme import badge, hero
from utils.citation_utils import format_inline, humanize_labels


def _show_result(result, papers_by_id) -> None:
    labels = {pid: format_inline(p) for pid, p in papers_by_id.items()}
    frame = pd.DataFrame([{
        "Paper": labels.get(r.paper_id, str(r.paper_id)), "Objective": r.objective, "Design": r.design,
        "Population / setting": r.population, "Sample size": r.sample_size,
        "Key findings": r.key_findings, "Conclusion": r.conclusion} for r in result.rows])
    st.dataframe(frame, hide_index=True, width="stretch")

    c1, c2 = st.columns(2, gap="large")
    humanize = lambda t: humanize_labels(t, papers_by_id.values())  # noqa: E731
    with c1:
        st.markdown("#### Similarities")
        for item in result.similarities or ["No clear similarities were detected."]:
            st.markdown(f"- {humanize(item)}")
    with c2:
        st.markdown("#### Differences")
        for item in result.differences or ["No clear differences were detected."]:
            st.markdown(f"- {humanize(item)}")
    if result.overall_assessment:
        st.markdown("#### Overall assessment")
        st.markdown(humanize(result.overall_assessment))
    st.markdown(badge("finding") + badge("verify"), unsafe_allow_html=True)
    how = "AI-generated with Gemini" if result.generated_by == "gemini" else "rule-based comparison (no AI)"
    st.caption(f"Method: {how}. Comparisons use titles and abstracts (plus open-access text when stored). "
               "They do not pool statistics or judge which result is correct.")


def render() -> None:
    services = get_services()
    hero("Research Comparison", "Compare objectives, methods, populations, findings and conclusions of selected papers.")

    topic_id = topic_selector(services, "cmp_topic")
    if topic_id is None:
        return
    papers = services.papers.list_for_topic(topic_id)
    by_id = {p.paper_id: p for p in papers}
    default = [p.paper_id for p in papers[:3]]
    chosen = st.multiselect("Papers to compare (2 to 12)", list(by_id), default=default, key=f"cmp_papers_{topic_id}",
                            format_func=lambda i: f"{format_inline(by_id[i])} - {by_id[i].title[:90]}")

    if st.button("Compare selected papers", type="primary", disabled=len(chosen) < 2, key="cmp_run"):
        bar, progress = make_progress("Preparing comparison...")
        notes: list = []
        try:
            services.pipeline.run_comparison(topic_id, chosen, notes, progress)
            st.session_state["analysis_notes"][f"cmp_{topic_id}"] = notes
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Comparison failed: {friendly_error(exc)}")
        finally:
            bar.empty()
    if len(chosen) < 2:
        st.caption("Select at least two papers.")

    stored = services.analyses.get(topic_id, "comparison")
    if not stored:
        return
    used_ids = stored["content"].get("paper_ids", [])
    used = {pid: by_id[pid] for pid in used_ids if pid in by_id}
    st.subheader("Latest comparison for this topic")
    st.caption("Papers compared: " + "; ".join(format_inline(p) for p in used.values()))
    show_notes(st.session_state["analysis_notes"].get(f"cmp_{topic_id}", []))
    from database.models import ComparisonResult  # local import keeps module header light

    _show_result(ComparisonResult(**stored["content"]["result"]), used)
