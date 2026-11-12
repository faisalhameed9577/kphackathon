# Entity relationship diagram

The nine required entities are `ResearchTopic`, `ResearchPaper`, `PaperAuthor`,
`DocumentChunk`, `ResearchQuery`, `ResearchAnalysis`, `ResearchGap`, `ResearchQuestion`
and `GeneratedReport`. Two **associative tables** (`topic_paper`, `gap_supporting_paper`)
implement many-to-many relationships.

```mermaid
erDiagram
    RESEARCH_TOPIC ||--o{ TOPIC_PAPER : "contains"
    RESEARCH_PAPER ||--o{ TOPIC_PAPER : "belongs to"
    RESEARCH_PAPER ||--o{ PAPER_AUTHOR : "has"
    RESEARCH_PAPER ||--o{ DOCUMENT_CHUNK : "split into"
    RESEARCH_TOPIC |o--o{ RESEARCH_QUERY : "searched / asked about"
    RESEARCH_TOPIC ||--o{ RESEARCH_ANALYSIS : "has"
    RESEARCH_PAPER |o--o{ RESEARCH_ANALYSIS : "analysed in"
    RESEARCH_TOPIC ||--o{ RESEARCH_GAP : "has"
    RESEARCH_GAP ||--o{ GAP_SUPPORTING_PAPER : "motivated by"
    RESEARCH_PAPER ||--o{ GAP_SUPPORTING_PAPER : "motivates"
    RESEARCH_TOPIC ||--o{ RESEARCH_QUESTION : "has"
    RESEARCH_GAP |o--o{ RESEARCH_QUESTION : "inspires"
    RESEARCH_TOPIC ||--o{ GENERATED_REPORT : "has"

    RESEARCH_TOPIC {
        int topic_id PK
        text title
        text title_key UK
        text keywords
        text region
        int year_from
        int year_to
        text created_at
        text updated_at
    }
    RESEARCH_PAPER {
        int paper_id PK
        text title
        text title_key
        text abstract
        text journal
        int pub_year
        text doi UK
        text pmid UK
        text pmcid UK
        text europepmc_id
        text sources
        text url
        int is_open_access
        text fulltext_path
    }
    TOPIC_PAPER {
        int topic_id PK, FK
        int paper_id PK, FK
        text added_at
    }
    PAPER_AUTHOR {
        int author_id PK
        int paper_id FK
        int author_order
        text author_name
    }
    DOCUMENT_CHUNK {
        int chunk_id PK
        text chunk_uid UK
        int paper_id FK
        text section
        int chunk_index
        int page_number
        int char_count
        text embedding_backend
    }
    RESEARCH_QUERY {
        int query_id PK
        int topic_id FK
        text query_type "search or question"
        text query_text
        text filters_json
        text sources
        int result_count
        text answer_text
        int sufficient_evidence
    }
    RESEARCH_ANALYSIS {
        int analysis_id PK
        int topic_id FK
        int paper_id FK
        text analysis_type "summary, limitations, comparison"
        text content_json
        text generated_by
    }
    RESEARCH_GAP {
        int gap_id PK
        int topic_id FK
        text gap_type
        text description
        text rationale
        text origin
        int needs_verification
    }
    GAP_SUPPORTING_PAPER {
        int gap_id PK, FK
        int paper_id PK, FK
    }
    RESEARCH_QUESTION {
        int question_id PK
        int topic_id FK
        int gap_id FK
        text question_text
        text rationale
        text suggested_design
        text origin
    }
    GENERATED_REPORT {
        int report_id PK
        int topic_id FK
        text title
        text file_path
        int file_size_bytes
    }
```

## Relationship explanation

| Relationship | Meaning |
|---|---|
| Topic ↔ Paper (via `topic_paper`) | Many-to-many. The same paper can appear in several topics; saving it twice never duplicates it. |
| Paper → Author | One paper has many ordered authors. |
| Paper → Chunk | One paper is split into many chunks. Only metadata is in SQL; the text and vector are in ChromaDB under `chunk_uid`. |
| Topic → Query | A search run or a question asked in the assistant. `topic_id` is optional (questions may span all papers). |
| Topic / Paper → Analysis | Summaries and limitations are per paper; a comparison belongs to the topic (`paper_id` is NULL). |
| Topic → Gap, Gap ↔ Paper | Gaps belong to a topic and list the papers that motivated them. A gap derived from "what is missing" may have none. |
| Gap → Question | A suggested research question may be linked to the gap that inspired it. |
| Topic → Report | Generated PDFs are registered with their path and size. |

Deleting a topic or a paper cascades to dependent rows (`PRAGMA foreign_keys=ON`).
Large text (chunks, full text) is intentionally **not** duplicated in SQL.
