# Known limitations

1. **Coverage** – only PubMed and Europe PMC are searched; other databases (Scopus, Google
   Scholar, local Pakistani journals not indexed in these sources) are not covered.
2. **Search is query-based** – few results mean the *query* found few papers, not that a topic is
   unresearched. Gap analysis is therefore hypothesis generation, always marked "needs verification".
3. **Abstracts vs full text** – full text is only retrieved for open-access papers with a PMCID.
   Many answers are based on abstracts alone and the app says so.
4. **Local embedder is lexical** – without a Gemini key, retrieval matches words, not meaning.
5. **LLM limits** – Gemini can still misread a passage. Citations are validated, limitation quotes
   are checked against the source, but all output should be verified against the original paper.
6. **Study-design and sample-size detection** in the fallback mode uses regular expressions and can
   be wrong or "Not reported".
7. **PDF fonts** – the PDF supports Western-European characters; other scripts (e.g. Urdu, Pashto)
   are replaced with `?`.
8. **Single user, local app** – SQLite and ChromaDB are local files; there is no login or multi-user
   concurrency.
9. **Model names change** – set `GEMINI_MODEL` in `.env` if the default model is retired.
10. **Not a medical device** – literature support only; not for diagnosis or treatment decisions.
11. **Live API behaviour** – the automated tests use mocks and synthetic fixtures; API response
    formats can change over time.
12. **Idea Check verdict is a planning aid** – it counts PubMed titles/abstracts by keyword. A low count can
    mean the topic is not indexed or worded differently, and a high count can include loosely related
    papers. Thresholds (`LEAVE_MIN_PAPERS`, `GO_MAX_PAPERS`, `RECENT_YEARS` in `services/idea_service.py`)
    are tunable. Always read the closest studies before deciding.
13. **Idea Check queries are not yet validated live** – field tags such as `[pt]`, `[mh]` and `[ad]` follow the
    PubMed documentation but were tested against mocks. Check the "PubMed queries used" panel on first use.
14. **Urdu ideas need Gemini** – without a key, write the idea in English. The PDF cannot print Urdu script.
