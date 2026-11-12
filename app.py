"""
ZETESIS AI - AI-powered medical research intelligence platform.

Start with:   streamlit run app.py
"""
import streamlit as st

# set_page_config must be the first Streamlit call.
st.set_page_config(page_title="ZETESIS AI", page_icon="🩺", layout="wide",
                   initial_sidebar_state="expanded")

from config.prompts import MEDICAL_DISCLAIMER  # noqa: E402
from config.settings import APP_NAME, APP_VERSION, get_settings  # noqa: E402
from frontend import (assistant, comparison, dashboard, discovery, gap_analysis,  # noqa: E402
                      idea_check, library, reports)
from frontend.state import PAGES, friendly_error, get_services, init_state  # noqa: E402
from frontend.theme import disclaimer, inject_css  # noqa: E402

RENDERERS = {
    "Dashboard": dashboard.render,
    "Idea Check": idea_check.render,
    "Research Discovery": discovery.render,
    "Research Library": library.render,
    "Research Assistant": assistant.render,
    "Research Comparison": comparison.render,
    "Research Gap Analysis": gap_analysis.render,
    "Reports": reports.render,
}


def main() -> None:
    inject_css()
    init_state()
    settings = get_settings()
    issues = settings.validate()

    with st.sidebar:
        st.markdown(f"## 🩺 {APP_NAME}")
        st.caption(f"Medical research intelligence · v{APP_VERSION}")
        st.radio("Navigation", PAGES, key="nav_page", label_visibility="collapsed")
        st.divider()
        st.markdown("**System status**")
        st.markdown(f"- Gemini AI: {'✅ connected' if settings.has_gemini else '⚠️ not configured'}")
        st.markdown(f"- NCBI email: {'✅ set' if settings.ncbi_email else 'ℹ️ not set (optional)'}")
        st.divider()
        st.caption(MEDICAL_DISCLAIMER)

    errors = [i for i in issues if i.level == "error"]
    for issue in errors:
        st.error(issue.message)
    if errors:
        st.stop()  # nothing else can work without a writable data folder

    warnings = [i for i in issues if i.level == "warning"]
    if warnings:
        with st.expander("⚠️ Configuration notice", expanded=st.session_state["nav_page"] == "Dashboard"):
            for issue in issues:
                (st.warning if issue.level == "warning" else st.info)(issue.message)
            st.caption("Copy `.env.example` to `.env`, add your keys and restart the app. Keys are never shown in the app.")

    try:
        get_services()  # builds services / creates the database on first run
        RENDERERS[st.session_state["nav_page"]]()
    except Exception as exc:  # noqa: BLE001 - last-resort guard so users never see a raw traceback
        st.error(f"Something went wrong: {friendly_error(exc)}")
        st.caption("Your saved data is safe. Try again, or choose another page.")

    st.divider()
    disclaimer(MEDICAL_DISCLAIMER)


main()
