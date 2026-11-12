"""Page 7 - Reports: generate, download and revisit PDF reports."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from database.db import DatabaseError
from frontend.state import friendly_error, get_services, make_progress, topic_selector
from frontend.theme import hero


def render() -> None:
    services = get_services()
    hero("Reports", "Generate a professional PDF evidence report: methodology, papers, comparison, limitations, gaps and references.")

    topic_id = topic_selector(services, "rep_topic")
    if topic_id is not None:
        st.caption("If no analysis exists for the topic yet, it is created first (this can take a minute with Gemini).")
        if st.button("Generate PDF report", type="primary", key="rep_generate"):
            bar, progress = make_progress("Preparing report...")
            try:
                path, _report_id = services.report_service.create_report(topic_id, progress)
                st.session_state["last_report_path"] = str(path)
                st.success("Report created.")
            except ValueError as exc:
                st.error(str(exc))
            except DatabaseError as exc:
                st.error(f"Could not record the report: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Report generation failed: {friendly_error(exc)}")
            finally:
                bar.empty()

        last = st.session_state.get("last_report_path")
        if last and Path(last).exists():
            st.download_button("⬇️ Download the report you just created", Path(last).read_bytes(),
                               file_name=Path(last).name, mime="application/pdf", key="rep_download_latest")

    st.subheader("Previous reports")
    reports = services.reports.list(limit=15)
    if not reports:
        st.info("No reports yet.")
        return
    for r in reports:
        path = Path(r["file_path"])
        c1, c2, c3 = st.columns([5, 2, 2])
        c1.markdown(f"**{r['title']}**  \n{r['created_at']} UTC · {r['file_size_bytes'] / 1024:.0f} KB")
        if path.exists():
            c3.download_button("Download", path.read_bytes(), file_name=path.name, mime="application/pdf",
                               key=f"rep_dl_{r['report_id']}")
        else:
            c3.caption("File no longer on disk")
