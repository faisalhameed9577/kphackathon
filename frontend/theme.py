"""Visual theme: healthcare / scientific look implemented with a little CSS."""
from __future__ import annotations

from html import escape

import streamlit as st

CSS = """
<style>
html, body, [class*="css"] { font-family: "Segoe UI", "Inter", system-ui, sans-serif; }
.block-container { padding-top: 1.6rem; max-width: 1250px; }
.se-hero { background: linear-gradient(120deg,#0f766e 0%,#115e59 55%,#134e4a 100%); color:#fff;
           padding: 1.4rem 1.8rem; border-radius: 14px; margin-bottom: 1.1rem; }
.se-hero h1 { color:#fff; margin:0 0 .25rem 0; font-size: 1.9rem; }
.se-hero p { color:#d1fae5; margin:0; font-size: .98rem; }
.se-badge { display:inline-block; padding:2px 10px; border-radius: 999px; font-size:.72rem; font-weight:600;
            margin-right:6px; vertical-align:middle; }
.se-finding { background:#ccfbf1; color:#115e59; }
.se-limitation { background:#dbeafe; color:#1e40af; }
.se-gap { background:#fef3c7; color:#92400e; }
.se-verify { background:#ffedd5; color:#9a3412; }
.se-neutral { background:#e2e8f0; color:#334155; }
.se-card { border:1px solid #d1e7e3; border-radius:12px; padding: .8rem 1rem; margin-bottom:.7rem; background:#f8fffd; }
.se-disclaimer { border-left: 4px solid #b45309; background:#fffbeb; padding:.6rem .9rem; border-radius:6px;
                 font-size:.86rem; color:#78350f; }
section[data-testid="stSidebar"] { background: #f0fdfa; }
div[data-testid="stMetric"] { background:#f0fdfa; border:1px solid #ccebe6; padding:.6rem .8rem; border-radius:12px; }
</style>
"""

_BADGES = {
    "finding": ("Finding supported by research", "se-finding"),
    "limitation": ("Limitation stated by authors", "se-limitation"),
    "gap": ("AI-inferred possible gap", "se-gap"),
    "verify": ("Needs further verification", "se-verify"),
}


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="se-hero"><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>', unsafe_allow_html=True)


def badge(kind: str) -> str:
    """HTML for an evidence badge. ``kind`` is one of finding/limitation/gap/verify."""
    text, css = _BADGES[kind]
    return f'<span class="se-badge {css}">{escape(text)}</span>'


def neutral_badge(text: str) -> str:
    return f'<span class="se-badge se-neutral">{escape(text)}</span>'


def show_badge(kind: str, extra: str = "") -> None:
    st.markdown(badge(kind) + (neutral_badge(extra) if extra else ""), unsafe_allow_html=True)


def disclaimer(text: str) -> None:
    st.markdown(f'<div class="se-disclaimer">{escape(text)}</div>', unsafe_allow_html=True)
