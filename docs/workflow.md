# Workflow

End-to-end flow of a typical session.

1. **Dashboard** – shows counts (papers, indexed papers, searches, topics, reports) and any
   configuration notices (missing Gemini key, NCBI email/key).
2. **Research Discovery**
   1. Enter topic, optional keywords, region (e.g. *Khyber Pakhtunkhwa*), year range, max papers.
   2. Input is validated (`utils/validators.py`). Generic words such as "research" or "studies"
      are removed; topic terms are AND-ed, keywords OR-ed and the region is expanded
      (KPK, NWFP, Peshawar, …).
   3. PubMed and Europe PMC are searched. Each source reports its own status; one failing
      source never hides the other's results.
   4. Records are de-duplicated (DOI → PMID → PMCID → long normalised title) and interleaved.
   5. You tick the papers to keep and press **Save and index selected papers**.
   6. Papers and authors go into SQLite. For open-access papers with a PMCID the JATS full text
      is downloaded from Europe PMC and cached under `data/papers/`; otherwise the abstract is used.
   7. Text is split into sections and chunks (1200 chars, 150 overlap), embedded
      (Gemini, or the offline fallback) and stored in ChromaDB. Chunks already embedded are reused.
3. **Research Library** – browse, filter and open saved papers; see whether each has full text
   or only an abstract; re-index a paper.
4. **Research Assistant** – ask a question over all papers, one topic, or chosen papers.
   Retrieval → relevance check → Gemini answer with `[S#]` citations → citation validation.
   If nothing relevant is found the app says **insufficient evidence** and does not call Gemini.
   Requests for diagnosis, dosing or treatment are declined.
5. **Research Comparison** – pick 2–12 papers; a table of design, population, sample size,
   methods, findings and limitations is produced (Gemini JSON, or rule-based extraction).
6. **Research Gap Analysis** – three tabs: limitations stated by authors (with verified quotes),
   possible research gaps (AI/rule-inferred, always "needs verification") and suggested research
   questions.
7. **Reports** – generate a PDF with ten sections, evidence labels and a reference list;
   download it from the page.

## Evidence labels

| Label | Meaning |
|---|---|
| Finding supported by research | Taken from the retrieved paper text and cited. |
| Limitation stated by authors | Quoted (and verified as a substring) from the paper. |
| AI-inferred possible gap | A hypothesis from coverage analysis or Gemini. Not proof. |
| Needs further verification | Recommendations / gaps to be checked by a human against the literature. |

## Fallback behaviour

| Situation | Behaviour |
|---|---|
| No `GEMINI_API_KEY` | Local keyword embeddings; extractive answers/summaries; rule-based gaps; template questions. |
| Gemini call fails | Friendly message; automatic fallback where possible. |
| PubMed or Europe PMC down | Per-source error status, other source continues. |
| Paper has no abstract / no open-access text | Shown as limited evidence; nothing is invented. |
