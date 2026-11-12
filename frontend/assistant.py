"""Page 4 - Research Assistant: RAG question answering with citations."""
from __future__ import annotations

import streamlit as st

from frontend.state import friendly_error, get_services, go_to
from frontend.theme import badge, hero, neutral_badge


def _scope(services):
    """Let the user limit which papers are searched. Returns (paper_ids or None, label)."""
    mode = st.radio("Search within", ["All indexed papers", "One research topic", "Choose papers"],
                    horizontal=True, key="asst_scope")
    if mode == "One research topic":
        topics = services.topics.list(limit=100, only_with_papers=True)
        if not topics:
            return None, "all"
        names = {t["topic_id"]: f"{t['title']} ({t['paper_count']} papers)" for t in topics}
        tid = st.selectbox("Topic", list(names), format_func=lambda i: names[i], key="asst_topic")
        ids = [p.paper_id for p in services.papers.list_for_topic(tid)]
        return ids, f"topic {tid}"
    if mode == "Choose papers":
        papers = services.papers.list_all(limit=500)
        by_id = {p.paper_id: p for p in papers}
        chosen = st.multiselect("Papers", list(by_id), format_func=lambda i: by_id[i].title[:100], key="asst_papers")
        return (chosen or []), "selection"
    return None, "all"


def _render_answer(result, nested: bool = False) -> None:
    """Show one answer. ``nested=True`` avoids st.expander (Streamlit forbids expanders inside expanders)."""
    if result.sufficient:
        st.success("Evidence found in the indexed papers.")
    else:
        st.warning("Insufficient evidence in the indexed papers.")
    st.markdown(result.answer)
    st.markdown(badge("finding") if result.sufficient and result.used_llm else badge("verify"), unsafe_allow_html=True)
    for note in result.notices:
        st.caption(f"ℹ️ {note}")

    if result.sources:
        shown = result.cited_sources or result.sources
        title = "Evidence cited in the answer" if result.cited_numbers and result.used_llm else "Retrieved evidence passages"
        container = st.container(border=True) if nested else st.expander(f"{title} ({len(shown)})", expanded=result.sufficient)
        with container:
            if nested:
                st.markdown(f"**{title} ({len(shown)})**")
            for s in shown:
                link = f"[{s.title}]({s.url})" if s.url else s.title
                st.markdown(f"**[{s.label}]** {s.inline_citation} - {link}")
                st.markdown(neutral_badge(s.section) + neutral_badge(f"{s.text_origin}") + neutral_badge(f"relevance {s.score}"),
                            unsafe_allow_html=True)
                st.caption(s.excerpt)
                st.divider()


def render() -> None:
    services = get_services()
    hero("Research Assistant", "Ask questions about the papers you indexed. Answers cite their sources and say so when evidence is thin.")

    if services.chunks.count() == 0:
        st.info("No papers are indexed yet. Search and click **Save and index selected papers** first.")
        st.button("Go to Research Discovery", on_click=go_to, args=("Research Discovery",), key="asst_go")
        return

    embedder = services.embedder
    st.caption(f"Retrieval: {'Gemini semantic embeddings' if embedder.backend == 'gemini' else 'local keyword embeddings'}  ·  "
               f"Answers: {'Gemini' if services.gemini.available else 'extractive (Gemini not configured)'}")
    if embedder.backend == "local" and services.gemini.available and embedder.fallback_reason:
        st.warning(f"Using local embeddings: {embedder.fallback_reason}. Papers indexed earlier with Gemini embeddings "
                   "may need to be re-indexed (Research Library).")

    paper_ids, _label = _scope(services)
    with st.form("ask_form"):
        question = st.text_area("Your research question", key="asst_question", height=90,
                                placeholder="e.g. What risk factors for type 2 diabetes are reported in rural populations?")
        asked = st.form_submit_button("Ask", type="primary")

    if asked:
        if paper_ids is not None and not paper_ids:
            st.error("Choose at least one paper, or search within all indexed papers.")
        else:
            try:
                with st.spinner("Retrieving evidence and writing a cited answer..."):
                    result = services.rag.answer(question, paper_ids=paper_ids)
                services.queries.add_question(None, result.question, result.answer, result.sufficient, len(result.sources))
                st.session_state["chat_history"].insert(0, result)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not answer: {friendly_error(exc)}")

    history = st.session_state["chat_history"]
    if history:
        st.subheader("Answer")
        st.markdown(f"**Q:** {history[0].question}")
        _render_answer(history[0])
        if len(history) > 1:
            st.subheader("Earlier questions in this session")
            for i, old in enumerate(history[1:], start=1):
                with st.expander(old.question):
                    _render_answer(old, nested=True)
