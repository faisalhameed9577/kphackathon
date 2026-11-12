"""Page 1 - Dashboard: overview numbers, recent topics, quick search, charts."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.state import get_services, go_to
from frontend.theme import badge, hero


def _quick_search() -> None:
    """Callback: copy the quick-search text into the Discovery form and run it there."""
    topic = (st.session_state.get("dash_quick_topic") or "").strip()
    st.session_state["disc_topic"] = topic
    st.session_state["auto_search"] = bool(topic)
    go_to("Research Discovery")


def render() -> None:
    services = get_services()
    hero("ZETESIS AI",
         "Before you start a study, find out if it has already been done. Write your research idea and get "
         "a GO / MODIFY / LEAVE answer backed by PubMed counts, the closest studies and untouched angles.")

    with st.container(border=True):
        st.markdown("#### New here? Start with your research idea")
        st.caption("Idea Check tells you how crowded your idea is, which part is still untouched, and what to change.")
        st.button("Check my research idea", type="primary", on_click=go_to, args=("Idea Check",), key="dash_idea_btn")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Papers saved", services.papers.count())
    c2.metric("Papers indexed (RAG)", services.chunks.indexed_paper_count())
    c3.metric("Research searches", services.queries.count("search"))
    c4.metric("Topics", services.topics.count())
    c5.metric("Reports", services.reports.count())

    left, right = st.columns([3, 2], gap="large")
    with right:
        st.subheader("Quick search")
        st.text_input("Medical research topic", key="dash_quick_topic",
                      placeholder="e.g. Diabetes research in rural Khyber Pakhtunkhwa")
        st.button("Search PubMed + Europe PMC", type="primary", on_click=_quick_search, key="dash_quick_btn")
        st.caption("Opens Research Discovery and runs the search with default filters.")

        st.subheader("How statements are labelled")
        for kind in ("finding", "limitation", "gap", "verify"):
            st.markdown(badge(kind), unsafe_allow_html=True)
        st.caption("The platform never claims a topic is unresearched just because a search found few papers.")

    with left:
        st.subheader("Recent research topics")
        topics = services.topics.list(limit=8)
        if topics:
            frame = pd.DataFrame([{"Topic": t["title"], "Region": t["region"] or "-", "Papers": t["paper_count"],
                                   "Last updated (UTC)": t["updated_at"]} for t in topics])
            st.dataframe(frame, hide_index=True, width="stretch")
        else:
            st.info("No research yet. Use **Quick search** or open **Research Discovery** to begin.")

    years = services.stats.year_counts()
    sources = services.stats.source_counts()
    if years or sources:
        st.subheader("Your library at a glance")
        ca, cb = st.columns([3, 2], gap="large")
        if years:
            fig = px.bar(pd.DataFrame(years, columns=["Year", "Papers"]), x="Year", y="Papers",
                         color_discrete_sequence=["#0f766e"], title="Papers by publication year")
            fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=300)
            ca.plotly_chart(fig, width="stretch")
        if sources:
            fig = px.pie(pd.DataFrame({"Source": list(sources), "Papers": list(sources.values())}),
                         names="Source", values="Papers", hole=0.5, title="Papers by source",
                         color_discrete_sequence=["#0f766e", "#5eead4", "#99f6e4"])
            fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=300)
            cb.plotly_chart(fig, width="stretch")
