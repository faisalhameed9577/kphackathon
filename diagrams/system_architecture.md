# System architecture

ZETESIS AI is a layered application. Each layer only calls the layer below it, so
every piece can be tested (and replaced) on its own.

```mermaid
flowchart TB
    User([Researcher / student / clinician-researcher])

    subgraph UI["Streamlit frontend (app.py + frontend/)"]
        P1[Dashboard]
        P2[Research Discovery]
        P3[Research Library]
        P4[Research Assistant]
        P5[Research Comparison]
        P6[Research Gap Analysis]
        P7[Reports]
    end

    subgraph SVC["Service layer (services/)"]
        PS[PaperService<br/>search, save, index]
        RAG[RagService<br/>retrieve, answer, cite]
        AP[AnalysisPipeline<br/>summary, limitations,<br/>comparison, gaps, questions]
        RS[ReportService + PdfReportService]
        DEDUP[DeduplicationService]
        DP[DocumentProcessor<br/>sections + chunking]
        EMB[EmbeddingService]
        GEM[GeminiService]
        PUB[PubMedService]
        EPMC[EuropePMCService]
        VS[VectorStore]
    end

    subgraph DATA["Local storage (data/)"]
        SQL[(SQLite<br/>metadata, analyses,<br/>gaps, reports)]
        CH[(ChromaDB<br/>chunk text + vectors)]
        FT[/papers/*.json<br/>cached open-access full text/]
        PDF[/reports/*.pdf/]
    end

    subgraph EXT["External services"]
        NCBI[(PubMed E-utilities)]
        EU[(Europe PMC REST)]
        GAI[(Google Gemini API)]
    end

    User --> UI
    P2 --> PS
    P3 --> PS
    P4 --> RAG
    P5 --> AP
    P6 --> AP
    P7 --> RS
    P1 --> PS

    PS --> PUB --> NCBI
    PS --> EPMC --> EU
    PS --> DEDUP
    PS --> DP
    PS --> EMB
    PS --> VS
    PS --> SQL
    PS --> FT

    RAG --> VS
    RAG --> EMB
    RAG --> GEM
    RAG --> SQL

    AP --> GEM
    AP --> SQL
    AP --> FT

    RS --> AP
    RS --> PDF
    RS --> SQL

    EMB --> GEM
    EMB -. offline fallback .-> LOCAL[Local hashing embedder]
    GEM --> GAI
    VS --> CH
```

## Component responsibilities

| Layer / component | Responsibility |
|---|---|
| `app.py`, `frontend/*.py` | Streamlit pages, forms, tables, charts, progress bars. No business logic, only calls services and shows results and friendly error messages. |
| `PaperService` | Orchestrates search (PubMed + Europe PMC), de-duplication, saving to SQLite, full-text retrieval (open access only), chunking and indexing. Reuses chunks that are already embedded. |
| `PubMedService` | Builds the PubMed query, calls `esearch` + `efetch`, parses XML into `PaperRecord`s. Rate limited and retried. |
| `EuropePMCService` | Builds the Europe PMC query, parses `core` results, downloads JATS full text for open-access papers. |
| `DeduplicationService` | Merges records with the same DOI / PMID / PMCID / long normalised title, keeping the richest data. |
| `DocumentProcessor` | Splits abstracts and JATS full text into sections, then into overlapping chunks (1200 chars / 150 overlap) with metadata. |
| `EmbeddingService` | Gemini embeddings, or an offline keyword (hashing) embedder when Gemini is not available. |
| `VectorStore` | ChromaDB persistent collection (cosine). Scoped queries by paper id. |
| `RagService` | Retrieval, relevance threshold, prompt, Gemini answer, citation validation, "insufficient evidence" handling, clinical-advice refusal, extractive fallback. |
| `ResearchAnalysisService` | Per-paper summaries, author-stated limitations, structured comparison (Gemini with rule-based fallback). |
| `GapDetectionService` | Rule-based coverage analysis plus Gemini suggestions for possible gaps; generates research questions. |
| `AnalysisPipeline` | Runs the analysis functions above in order and caches results in SQLite. These are pipeline functions, not independent agents. |
| `PdfReportService` / `ReportService` | Assemble a 10-section PDF report with ReportLab and register it in SQLite. |
| `GeminiService` | The only place that talks to Gemini: retries, JSON mode, friendly errors, no key leakage. |
| SQLite | Source of truth for metadata, analyses, gaps, questions and report records. |
| ChromaDB | Chunk text and embedding vectors (not duplicated in SQL). |
