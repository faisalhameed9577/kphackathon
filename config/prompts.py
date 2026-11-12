"""
Prompt templates and fixed safety wording.

Templates use ``<<NAME>>`` placeholders (not ``{}``) so JSON examples can contain
braces freely. Use ``render(template, NAME=value)`` to fill them in.
"""
from __future__ import annotations

MEDICAL_DISCLAIMER = (
    "ZETESIS AI supports literature research only. It does not provide medical "
    "diagnosis or treatment advice and does not replace professional medical judgment."
)

# The four kinds of statements the platform keeps apart (shown as badges in the UI).
EVIDENCE_LABELS = {
    "finding": "Finding supported by research",
    "limitation": "Limitation stated by authors",
    "gap": "AI-inferred possible gap",
    "verify": "Needs further verification",
}

SEARCH_LIMITATION_NOTE = (
    "Finding few or no papers does not prove a topic has never been studied. The search "
    "covers only PubMed and Europe PMC, uses keyword matching, and may miss regional "
    "journals, grey literature, non-English papers and papers indexed under other terms."
)

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The indexed papers do not contain enough evidence to answer this question reliably. "
    "Try indexing more papers (Research Discovery), widening the scope, or rephrasing the question."
)

CLINICAL_ADVICE_MESSAGE = (
    "This looks like a request for personal medical advice, a diagnosis or a treatment "
    "decision. ZETESIS AI cannot provide that. Please consult a qualified health "
    "professional. You can instead ask what the indexed research reports about the topic."
)

SYSTEM_INSTRUCTION = """You are the analysis engine of ZETESIS AI, a tool that supports medical literature research.
Strict rules:
1. Use ONLY the material supplied in the prompt. Never use outside knowledge to add facts.
2. Never invent papers, authors, numbers, DOIs, statistics or quotations.
3. Cite only with the labels supplied in the prompt (for example [S1] or P12). Never create new labels.
4. Never give medical diagnosis, treatment, dosing or personal health advice. If asked, say this tool supports research only.
5. Keep three things clearly apart: what papers report (findings), limitations the authors themselves state, and possible research gaps that YOU infer. Inferred gaps are hypotheses that need verification.
6. A small or empty set of papers is NOT proof that a topic is unresearched. Never claim a topic "has not been studied".
7. If the material is insufficient, say so plainly instead of guessing.
8. Write in clear, neutral, scientific English."""

RAG_PROMPT = """Answer the research question using ONLY the numbered evidence passages below.

Rules:
- Put the passage label(s) in square brackets right after each claim, e.g. [S1] or [S2, S3].
- Use only labels that appear below. Do not cite anything else.
- Report what the papers say; do not add outside knowledge and do not give medical advice.
- If studies disagree or the evidence is weak (for example abstract-only, small samples), say so.
- If the passages do not contain enough information, begin your reply with the exact words
  INSUFFICIENT_EVIDENCE: and then explain briefly what is missing.
- Keep the answer under 250 words.

EVIDENCE PASSAGES:
<<CONTEXT>>

QUESTION: <<QUESTION>>
"""

SUMMARY_PROMPT = """Summarise each research paper below using ONLY its text.
Return JSON exactly in this shape (no other text):
{"papers": [{"label": "P12", "objective": "...", "study_design": "...", "population_setting": "...",
  "methods": "...", "key_findings": ["...", "..."], "conclusion": "..."}]}
Rules:
- "label" must be copied exactly from the paper header.
- Use "Not reported" when the text does not say. Do not guess numbers.
- key_findings: at most 4 short items that are actually reported in the text.
- Do not add recommendations or medical advice.

PAPERS:
<<PAPERS>>
"""

LIMITATIONS_PROMPT = """For each paper below, list ONLY the limitations that the authors explicitly state in the text
(for example "small sample", "single centre", "cross-sectional design", "self-reported data").
Do not infer limitations the authors did not state. If none are stated, return an empty list.
Return JSON exactly in this shape (no other text):
{"papers": [{"label": "P12", "limitations": [{"text": "short description", "evidence": "a short phrase (max 20 words) copied from the text"}]}]}

PAPERS:
<<PAPERS>>
"""

COMPARISON_PROMPT = """Compare the research papers below using ONLY the per-paper summaries provided.
Return JSON exactly in this shape (no other text):
{"rows": [{"label": "P12", "objective": "...", "design": "...", "population": "...", "sample_size": "...",
  "key_findings": "...", "conclusion": "..."}],
 "similarities": ["..."], "differences": ["..."], "overall_assessment": "..."}
Rules:
- One row per paper; copy labels exactly.
- Similarities and differences must refer to objectives, methods, populations, findings or conclusions, and mention paper labels.
- Do not pool or recalculate statistics and do not state which treatment or approach is best.
- overall_assessment: 2-3 sentences on how consistent the evidence looks and how much weight it can bear (note abstract-only evidence, small samples, differing designs).

PAPER SUMMARIES:
<<SUMMARIES>>
"""

GAPS_PROMPT = """Topic: <<TOPIC>>
Region of interest: <<REGION>>

Below are summaries and author-stated limitations of the retrieved papers. Identify POSSIBLE research gaps:
under-studied populations, locations, outcomes, study designs, follow-up periods or unanswered questions.
Return JSON exactly in this shape (no other text):
{"gaps": [{"gap_type": "population|location|methodology|outcome|temporal|evidence_quality|other",
  "description": "...", "rationale": "why the supplied papers suggest this gap", "supporting_papers": ["P12", "P15"]}]}
Rules:
- Maximum 6 gaps. Each must be grounded in the supplied papers; cite them in supporting_papers.
- These are hypotheses ("possible gap"), not facts. Never say the topic has not been studied.
- The paper set is small and search-limited; say in the rationale when a gap may simply reflect that.
- No medical advice.

PAPERS:
<<PAPERS>>
"""

QUESTIONS_PROMPT = """Topic: <<TOPIC>>
Region of interest: <<REGION>>

Possible research gaps (numbered from 0):
<<GAPS>>

Suggest at most <<MAX_QUESTIONS>> concrete future research questions, each tied to one gap.
Return JSON exactly in this shape (no other text):
{"questions": [{"gap_index": 0, "question": "...", "rationale": "...", "suggested_design": "..."}]}
Rules: questions must be answerable by research (not clinical advice); suggested_design is a short study-design suggestion
(e.g. "community-based cross-sectional survey", "prospective cohort"). These are suggestions that need expert review.
"""


IDEA_PARSE_PROMPT = """Read the research idea below. It may be written in English, Urdu (Arabic or Devanagari script),
Roman Urdu or another language. Translate it into English and split it into parts.
Return JSON exactly in this shape (no other text):
{"language": "en", "english_idea": "...",
 "topic": {"label": "main condition or subject, 2-6 words", "terms": ["spelling variants or synonyms used in medical papers"]},
 "population": {"label": "", "terms": []},
 "intervention": {"label": "intervention or exposure being studied", "terms": []},
 "outcome": {"label": "", "terms": []},
 "region": {"label": "place named in the idea", "country": "country of that place"},
 "design": "none"}
Rules:
- "language" is the language the idea was written in: en, ur, ps or other.
- Use only what the idea says. Use "" and [] for any part the idea does not mention.
- terms are English words that authors use in titles and abstracts (for example "children", "paediatric"). Maximum 5 per part.
- Place given separately by the user: <<REGION>> (use it for region.label when it is not "not given").
- design is one of: rct, cohort, cross_sectional, systematic_review, qualitative, none. Use none unless the idea clearly names a study design.
- Do not add medical advice.

RESEARCH IDEA:
<<IDEA>>
"""

IDEA_EXPLAIN_PROMPT = """You explain the result of a research-idea check to a student.
The verdict (<<VERDICT>>) was decided by fixed rules from PubMed paper counts. Do NOT change it.
Write 4 to 6 short sentences in <<LANGUAGE>>. Use ONLY the facts below. Never invent numbers, papers or claims.
Explain what the numbers mean, which part looks untouched, and what the student could do next (keep, change or drop the idea).
Say once that a low count can also mean the topic is poorly indexed, and never say a topic "has not been studied".

FACTS:
<<FACTS>>
"""


def render(template: str, **values: str) -> str:
    """Fill ``<<NAME>>`` placeholders."""
    out = template
    for key, value in values.items():
        out = out.replace(f"<<{key}>>", str(value))
    return out
