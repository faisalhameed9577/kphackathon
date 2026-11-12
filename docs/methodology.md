# Methodology

## Search
* Generic words are stripped; topic terms are AND-ed; keywords are OR-ed; the region is expanded
  into synonyms. PubMed uses `[tiab]`; Europe PMC uses `TITLE:` / `ABSTRACT:` and `PUB_YEAR:[a TO b]`.
* Results from both sources are merged and de-duplicated by DOI, PMID, PMCID or a long
  normalised title. Records are merged so the richest metadata wins.

## Retrieval-augmented generation (RAG)
* Chunks: `RecursiveCharacterTextSplitter`, 1200 characters, 150 overlap, section metadata kept.
* Embeddings: Gemini `gemini-embedding-001` (768 dims) with separate document/query task types;
  otherwise a deterministic hashing embedder (keyword matching only). Each backend has its own
  Chroma collection, so vectors are never mixed.
* Retrieval is scoped to the selected papers. A relevance threshold (0.35 Gemini / 0.08 local)
  decides whether evidence exists. Below it the app answers *insufficient evidence* without
  calling the LLM.
* The model must cite sources as `[S#]` or reply `INSUFFICIENT_EVIDENCE:`. Citations that do not
  match supplied sources are removed.

## Analysis (pipeline functions, not agents)
* **Summaries** – batches of 4 papers per Gemini call, JSON output validated with Pydantic;
  paper labels (`P<id>`) are checked against the supplied papers. Fallback: extractive summary
  with study-design and sample-size detection.
* **Limitations** – Gemini quotes must appear verbatim in the source text, else flagged
  "unverified". Fallback: keyword cues ("limitation", "small sample", …).
* **Comparison** – structured table; fallback is rule-based.
* **Gaps** – coverage analysis (region mentions, absence of RCTs, dominance of cross-sectional
  designs, subgroups, recency, abstract-only evidence) plus optional Gemini suggestions. All gaps
  carry a numeric rationale and *needs verification*.
* **Research questions** – Gemini or templates, linked to gaps.

## Safety rules
* No fabricated references: every reference comes from a retrieved record.
* No diagnosis, dosing or treatment advice; such requests are declined.
* Few search results never mean a topic is unresearched. This is stated in the app and PDF.
* Keys are read from the environment, never displayed, and removed from error messages.

## Idea Check verdict
1. The idea is split into facets (topic, population, intervention, outcome, place, design) by Gemini, or by
   simple rules when no key is set.
2. PubMed is counted (`esearch`, `retmax=0`) for the whole idea, the topic alone, each part removed in turn,
   recent papers (last 5 years), recent systematic reviews / meta-analyses, each study design, and the place
   ladder (place -> country -> worldwide).
3. Verdict (rules, not AI): INCONCLUSIVE if the topic alone has 0 papers; LEAVE if the whole idea has >= 20
   papers and a recent review (or >= 10 recent papers); GO if it has <= 2 papers; otherwise MODIFY.
4. Angles are the parts or designs with the fewest papers. The closest studies are ranked by embedding
   similarity and labelled by PubMed publication type; retracted papers are flagged.
