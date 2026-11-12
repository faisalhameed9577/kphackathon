# Data flow diagrams

## Level 0 (context diagram)

```mermaid
flowchart LR
    U([User])
    S((ZETESIS AI))
    PM[(PubMed)]
    EP[(Europe PMC)]
    GM[(Gemini API)]

    U -- "topic, keywords, region,<br/>year range, max papers, questions" --> S
    S -- "papers, cited answers, comparisons,<br/>gaps, research questions, PDF report" --> U
    S -- "search queries" --> PM
    PM -- "titles, abstracts, metadata" --> S
    S -- "search queries, full-text requests" --> EP
    EP -- "metadata, open-access full text" --> S
    S -- "prompts + retrieved evidence" --> GM
    GM -- "embeddings, generated text" --> S
```

## Level 1

```mermaid
flowchart TB
    U([User])

    P1((1.0<br/>Validate input<br/>and build queries))
    P2((2.0<br/>Search sources))
    P3((3.0<br/>Normalise and<br/>de-duplicate))
    P4((4.0<br/>Save papers and<br/>fetch open-access text))
    P5((5.0<br/>Chunk and embed))
    P6((6.0<br/>Answer questions<br/>with RAG))
    P7((7.0<br/>Analyse: summary,<br/>limitations, comparison,<br/>gaps, questions))
    P8((8.0<br/>Generate PDF report))

    D1[(D1 SQLite)]
    D2[(D2 ChromaDB)]
    D3[/D3 Full-text cache/]
    D4[/D4 Report files/]

    PM[(PubMed)]
    EP[(Europe PMC)]
    GM[(Gemini)]

    U -- "search form" --> P1
    P1 -- "validated SearchRequest" --> P2
    P2 <--> PM
    P2 <--> EP
    P2 -- "raw records" --> P3
    P3 -- "unique records" --> P4
    P3 -- "search log" --> D1
    U -- "selected papers" --> P4
    P4 <-->|"full text (open access only)"| EP
    P4 -- "paper + author metadata" --> D1
    P4 -- "full-text JSON" --> D3
    P4 -- "text to index" --> P5
    P5 -- "embedding requests" --> GM
    P5 -- "chunk text + vectors" --> D2
    P5 -- "chunk metadata" --> D1

    U -- "research question" --> P6
    P6 -- "query vector" --> D2
    D2 -- "relevant chunks" --> P6
    P6 -- "prompt + evidence" --> GM
    P6 -- "cited answer or<br/>insufficient evidence" --> U
    P6 -- "query log" --> D1

    U -- "choose topic" --> P7
    D1 -- "papers, abstracts" --> P7
    D3 -- "full text" --> P7
    P7 -- "analysis prompts" --> GM
    P7 -- "analyses, gaps, questions" --> D1
    P7 -- "labelled results" --> U

    U -- "generate report" --> P8
    D1 -- "topic, papers, analyses" --> P8
    P8 -- "PDF" --> D4
    P8 -- "report record" --> D1
    D4 -- "download" --> U
```

### Notes

* Every Gemini arrow is optional. Without a key the app uses the offline embedder and
  rule-based extraction, and says so on screen.
* Process 4.0 only downloads full text when Europe PMC reports the paper as open access.
* Process 7.0 labels its output: *finding supported by research*, *limitation stated by
  authors*, *AI-inferred possible gap*, *recommendation needing verification*.
