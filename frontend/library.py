"""Page 3 - Research Library: browse, search and inspect saved papers."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from frontend.state import friendly_error, get_services
from frontend.theme import hero


def render() -> None:
    services = get_services()
    hero("Research Library", "Every paper you have saved, with its abstract, identifiers and indexing status.")

    if services.papers.count() == 0:
        st.info("The library is empty. Search and index papers in **Research Discovery** first.")
        return

    this_year = datetime.now().year
    topics = services.topics.list(limit=100, only_with_papers=True)
    c1, c2, c3, c4 = st.columns([3, 3, 3, 2])
    text = c1.text_input("Search title, abstract or journal", key="lib_text")
    topic_ids = [None] + [t["topic_id"] for t in topics]
    names = {None: "All topics", **{t["topic_id"]: t["title"] for t in topics}}
    topic_id = c2.selectbox("Topic", topic_ids, format_func=lambda i: names[i], key="lib_topic")
    years = c3.slider("Years", 1900, this_year, (1990, this_year), key="lib_years")
    oa_only = c4.checkbox("Open access only", key="lib_oa")

    papers = services.papers.list_all(text=text, year_from=years[0], year_to=years[1],
                                      open_access_only=oa_only, topic_id=topic_id)
    chunk_counts = services.chunks.counts_by_paper()
    st.caption(f"{len(papers)} paper(s) match.")
    if not papers:
        st.info("No papers match these filters.")
        return

    frame = pd.DataFrame([{
        "ID": p.paper_id, "Title": p.title, "Year": p.year, "Journal": p.journal,
        "Authors": ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else ""),
        "Sources": p.source_label, "Open access": "Yes" if p.is_open_access else "No",
        "Indexed chunks": chunk_counts.get(p.paper_id, 0), "Link": p.best_url} for p in papers])
    st.dataframe(frame, hide_index=True, width="stretch",
                 column_config={"Link": st.column_config.LinkColumn("DOI / source link")})
    st.download_button("Download this list (CSV)", frame.to_csv(index=False).encode("utf-8"),
                       file_name="zetesis_library.csv", mime="text/csv", key="lib_csv")

    st.subheader("Paper details")
    by_id = {p.paper_id: p for p in papers}
    chosen = st.selectbox("Choose a paper", list(by_id), format_func=lambda i: f"{by_id[i].title[:120]} ({by_id[i].year or 'n.d.'})",
                          key="lib_choice")
    paper = by_id[chosen]
    st.markdown(f"### {paper.title}")
    st.markdown(f"**Authors:** {', '.join(paper.authors) or 'Not available'}")
    st.markdown(f"**Journal:** {paper.journal or 'Not available'}  ·  **Year:** {paper.year or 'n.d.'}  ·  "
                f"**Source(s):** {paper.source_label}" + ("  ·  **Preprint (not peer reviewed)**" if paper.is_preprint else ""))
    ids = []
    if paper.doi:
        ids.append(f"DOI: [{paper.doi}](https://doi.org/{paper.doi})")
    if paper.pmid:
        ids.append(f"PMID: [{paper.pmid}](https://pubmed.ncbi.nlm.nih.gov/{paper.pmid}/)")
    if paper.pmcid:
        ids.append(f"PMCID: [{paper.pmcid}](https://pmc.ncbi.nlm.nih.gov/articles/{paper.pmcid}/)")
    st.markdown("  ·  ".join(ids) if ids else f"[Source link]({paper.best_url})" if paper.best_url else "No identifiers available.")

    full = services.paper_service.has_fulltext(paper)
    st.markdown(f"**Open access:** {'Yes' if paper.is_open_access else 'Not confirmed'}  ·  "
                f"**Full text stored locally:** {'Yes' if full else 'No (abstract only)'}  ·  "
                f"**Indexed chunks:** {chunk_counts.get(paper.paper_id, 0)}")
    with st.expander("Abstract", expanded=True):
        st.markdown(paper.abstract or "_No abstract available for this record._")

    if st.button("Re-index this paper", key="lib_reindex", help="Re-create its text chunks and embeddings if missing."):
        try:
            with st.spinner("Indexing..."):
                report = services.paper_service.reindex_paper(paper.paper_id)
            st.success(f"Done: {report.chunks_new} new chunks, {report.chunks_reused} reused.")
            for warning in report.warnings:
                st.warning(warning)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Indexing failed: {friendly_error(exc)}")
