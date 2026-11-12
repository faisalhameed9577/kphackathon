"""
Repositories: all SQL lives here so the rest of the app never writes SQL.

One small class per entity, each receiving the shared ``Database`` object.
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from database.db import Database
from database.models import GapItem, PaperRecord, ResearchQuestionItem
from utils.helpers import normalize_title

# --------------------------------------------------------------------- topics


class TopicRepository:
    def __init__(self, db: Database):
        self.db = db

    def get_or_create(self, title: str, keywords: str = "", region: str = "",
                      year_from: Optional[int] = None, year_to: Optional[int] = None) -> int:
        key = normalize_title(f"{title}|{region}")
        with self.db.connection() as conn:
            row = conn.execute("SELECT topic_id FROM research_topic WHERE title_key = ?", (key,)).fetchone()
            if row:
                conn.execute(
                    "UPDATE research_topic SET keywords=?, year_from=?, year_to=?, updated_at=datetime('now') "
                    "WHERE topic_id=?", (keywords, year_from, year_to, row["topic_id"]))
                return int(row["topic_id"])
            cur = conn.execute(
                "INSERT INTO research_topic (title, title_key, keywords, region, year_from, year_to) "
                "VALUES (?,?,?,?,?,?)", (title, key, keywords, region, year_from, year_to))
            return int(cur.lastrowid)

    def get(self, topic_id: int) -> Optional[Dict[str, Any]]:
        with self.db.connection() as conn:
            row = conn.execute("SELECT * FROM research_topic WHERE topic_id=?", (topic_id,)).fetchone()
            return dict(row) if row else None

    def list(self, limit: int = 50, only_with_papers: bool = False) -> List[Dict[str, Any]]:
        having = "HAVING paper_count > 0" if only_with_papers else ""
        sql = f"""
            SELECT t.*, COUNT(tp.paper_id) AS paper_count
            FROM research_topic t LEFT JOIN topic_paper tp ON tp.topic_id = t.topic_id
            GROUP BY t.topic_id {having}
            ORDER BY t.updated_at DESC, t.topic_id DESC LIMIT ?"""
        with self.db.connection() as conn:
            return [dict(r) for r in conn.execute(sql, (limit,)).fetchall()]

    def touch(self, topic_id: int) -> None:
        with self.db.connection() as conn:
            conn.execute("UPDATE research_topic SET updated_at=datetime('now') WHERE topic_id=?", (topic_id,))

    def count(self) -> int:
        with self.db.connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM research_topic").fetchone()[0])


# --------------------------------------------------------------------- papers


def _row_to_paper(row: sqlite3.Row, authors: List[str]) -> PaperRecord:
    return PaperRecord(
        paper_id=row["paper_id"], title=row["title"], abstract=row["abstract"] or "",
        authors=authors, journal=row["journal"] or "", year=row["pub_year"], doi=row["doi"],
        pmid=row["pmid"], pmcid=row["pmcid"], europepmc_id=row["europepmc_id"],
        europepmc_source=row["europepmc_source"],
        sources=[s for s in (row["sources"] or "").split(",") if s],
        url=row["url"] or "", is_open_access=bool(row["is_open_access"]),
        fulltext_path=row["fulltext_path"],
    )


class PaperRepository:
    def __init__(self, db: Database):
        self.db = db

    # --------------------------------------------------------------- reading
    @staticmethod
    def _authors_for(conn: sqlite3.Connection, paper_ids: Sequence[int]) -> Dict[int, List[str]]:
        if not paper_ids:
            return {}
        marks = ",".join("?" * len(paper_ids))
        rows = conn.execute(
            f"SELECT paper_id, author_name FROM paper_author WHERE paper_id IN ({marks}) "
            "ORDER BY paper_id, author_order", tuple(paper_ids)).fetchall()
        result: Dict[int, List[str]] = {}
        for r in rows:
            result.setdefault(r["paper_id"], []).append(r["author_name"])
        return result

    def _hydrate(self, conn: sqlite3.Connection, rows: Sequence[sqlite3.Row]) -> List[PaperRecord]:
        authors = self._authors_for(conn, [r["paper_id"] for r in rows])
        return [_row_to_paper(r, authors.get(r["paper_id"], [])) for r in rows]

    def get(self, paper_id: int) -> Optional[PaperRecord]:
        found = self.get_many([paper_id])
        return found[0] if found else None

    def get_many(self, paper_ids: Sequence[int]) -> List[PaperRecord]:
        """Papers in the same order as ``paper_ids`` (unknown ids are skipped)."""
        ids = list(dict.fromkeys(paper_ids))
        if not ids:
            return []
        with self.db.connection() as conn:
            marks = ",".join("?" * len(ids))
            rows = conn.execute(f"SELECT * FROM research_paper WHERE paper_id IN ({marks})", ids).fetchall()
            by_id = {p.paper_id: p for p in self._hydrate(conn, rows)}
        return [by_id[i] for i in ids if i in by_id]

    def list_all(self, text: str = "", year_from: Optional[int] = None, year_to: Optional[int] = None,
                 open_access_only: bool = False, topic_id: Optional[int] = None,
                 limit: int = 1000) -> List[PaperRecord]:
        sql = "SELECT p.* FROM research_paper p"
        where, params = [], []
        if topic_id is not None:
            sql += " JOIN topic_paper tp ON tp.paper_id = p.paper_id"
            where.append("tp.topic_id = ?")
            params.append(topic_id)
        if text.strip():
            like = f"%{text.strip()}%"
            where.append("(p.title LIKE ? OR p.abstract LIKE ? OR p.journal LIKE ?)")
            params += [like, like, like]
        if year_from is not None:
            where.append("(p.pub_year IS NULL OR p.pub_year >= ?)")
            params.append(year_from)
        if year_to is not None:
            where.append("(p.pub_year IS NULL OR p.pub_year <= ?)")
            params.append(year_to)
        if open_access_only:
            where.append("p.is_open_access = 1")
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY p.pub_year DESC, p.paper_id DESC LIMIT ?"
        params.append(limit)
        with self.db.connection() as conn:
            return self._hydrate(conn, conn.execute(sql, params).fetchall())

    def list_for_topic(self, topic_id: int) -> List[PaperRecord]:
        return self.list_all(topic_id=topic_id)

    def count(self) -> int:
        with self.db.connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM research_paper").fetchone()[0])

    # --------------------------------------------------------------- writing
    @staticmethod
    def _find_existing(conn: sqlite3.Connection, paper: PaperRecord) -> Optional[sqlite3.Row]:
        for column, value in (("doi", paper.doi), ("pmid", paper.pmid), ("pmcid", paper.pmcid)):
            if value:
                row = conn.execute(f"SELECT * FROM research_paper WHERE {column} = ?", (value,)).fetchone()
                if row:
                    return row
        key = normalize_title(paper.title)
        if len(key) >= 25:
            row = conn.execute(
                "SELECT * FROM research_paper WHERE title_key = ? AND "
                "(pub_year IS NULL OR ? IS NULL OR pub_year = ?)",
                (key, paper.year, paper.year)).fetchone()
            if row:
                return row
        return None

    @staticmethod
    def _free_identifier(conn: sqlite3.Connection, column: str, value: Optional[str], paper_id: int) -> Optional[str]:
        """Return ``value`` unless another row already owns it (UNIQUE constraint guard)."""
        if not value:
            return None
        clash = conn.execute(
            f"SELECT 1 FROM research_paper WHERE {column} = ? AND paper_id != ?", (value, paper_id)).fetchone()
        return None if clash else value

    @staticmethod
    def _replace_authors(conn: sqlite3.Connection, paper_id: int, authors: Iterable[str]) -> None:
        conn.execute("DELETE FROM paper_author WHERE paper_id = ?", (paper_id,))
        conn.executemany(
            "INSERT INTO paper_author (paper_id, author_order, author_name) VALUES (?,?,?)",
            [(paper_id, i, name) for i, name in enumerate(authors) if name])

    def upsert(self, paper: PaperRecord) -> int:
        """Insert a paper or merge it into the existing row; returns the paper_id."""
        with self.db.connection() as conn:
            existing = self._find_existing(conn, paper)
            sources = ",".join(paper.sources)
            if existing is None:
                cur = conn.execute(
                    "INSERT INTO research_paper (title,title_key,abstract,journal,pub_year,doi,pmid,pmcid,"
                    "europepmc_id,europepmc_source,sources,url,is_open_access,fulltext_path) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (paper.title, normalize_title(paper.title), paper.abstract, paper.journal, paper.year,
                     paper.doi, paper.pmid, paper.pmcid, paper.europepmc_id, paper.europepmc_source,
                     sources, paper.url, int(paper.is_open_access), paper.fulltext_path))
                pid = int(cur.lastrowid)
                self._replace_authors(conn, pid, paper.authors)
                return pid

            pid = int(existing["paper_id"])
            merged_sources = sorted(set(filter(None, (existing["sources"] or "").split(","))) | set(paper.sources))
            abstract = paper.abstract if len(paper.abstract) > len(existing["abstract"] or "") else existing["abstract"]
            conn.execute(
                "UPDATE research_paper SET abstract=?, journal=?, pub_year=?, doi=?, pmid=?, pmcid=?, "
                "europepmc_id=?, europepmc_source=?, sources=?, url=?, is_open_access=?, fulltext_path=?, "
                "updated_at=datetime('now') WHERE paper_id=?",
                (abstract, existing["journal"] or paper.journal, existing["pub_year"] or paper.year,
                 existing["doi"] or self._free_identifier(conn, "doi", paper.doi, pid),
                 existing["pmid"] or self._free_identifier(conn, "pmid", paper.pmid, pid),
                 existing["pmcid"] or self._free_identifier(conn, "pmcid", paper.pmcid, pid),
                 existing["europepmc_id"] or paper.europepmc_id,
                 existing["europepmc_source"] or paper.europepmc_source,
                 ",".join(merged_sources), existing["url"] or paper.url,
                 int(bool(existing["is_open_access"]) or paper.is_open_access),
                 existing["fulltext_path"] or paper.fulltext_path, pid))
            has_authors = conn.execute("SELECT 1 FROM paper_author WHERE paper_id=? LIMIT 1", (pid,)).fetchone()
            if not has_authors and paper.authors:
                self._replace_authors(conn, pid, paper.authors)
            return pid

    def link_to_topic(self, topic_id: int, paper_id: int) -> None:
        with self.db.connection() as conn:
            conn.execute("INSERT OR IGNORE INTO topic_paper (topic_id, paper_id) VALUES (?,?)", (topic_id, paper_id))

    def set_fulltext_path(self, paper_id: int, path: Optional[str]) -> None:
        with self.db.connection() as conn:
            conn.execute("UPDATE research_paper SET fulltext_path=?, updated_at=datetime('now') WHERE paper_id=?",
                         (path, paper_id))


# --------------------------------------------------------------------- chunks


class ChunkRepository:
    """Chunk metadata only. Text and vectors are stored in ChromaDB."""

    def __init__(self, db: Database):
        self.db = db

    def upsert_many(self, paper_id: int, chunks: Sequence[Dict[str, Any]], backend: str) -> None:
        with self.db.connection() as conn:
            conn.executemany(
                "INSERT INTO document_chunk (chunk_uid,paper_id,section,chunk_index,page_number,char_count,embedding_backend) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(chunk_uid) DO UPDATE SET "
                "embedding_backend=excluded.embedding_backend, char_count=excluded.char_count",
                [(c["chunk_uid"], paper_id, c.get("section", ""), c.get("chunk_index", 0),
                  c.get("page_number"), c.get("char_count", 0), backend) for c in chunks])

    def count(self) -> int:
        with self.db.connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM document_chunk").fetchone()[0])

    def counts_by_paper(self) -> Dict[int, int]:
        with self.db.connection() as conn:
            rows = conn.execute("SELECT paper_id, COUNT(*) AS n FROM document_chunk GROUP BY paper_id").fetchall()
            return {r["paper_id"]: r["n"] for r in rows}

    def indexed_paper_count(self) -> int:
        with self.db.connection() as conn:
            return int(conn.execute("SELECT COUNT(DISTINCT paper_id) FROM document_chunk").fetchone()[0])


# -------------------------------------------------------------------- queries


class QueryRepository:
    def __init__(self, db: Database):
        self.db = db

    def add_search(self, topic_id: Optional[int], text: str, filters: Dict[str, Any],
                   sources: Sequence[str], result_count: int) -> int:
        with self.db.connection() as conn:
            cur = conn.execute(
                "INSERT INTO research_query (topic_id,query_type,query_text,filters_json,sources,result_count) "
                "VALUES (?,?,?,?,?,?)",
                (topic_id, "search", text, json.dumps(filters, default=str), ",".join(sources), result_count))
            return int(cur.lastrowid)

    def add_question(self, topic_id: Optional[int], question: str, answer: str,
                     sufficient: bool, source_count: int) -> int:
        with self.db.connection() as conn:
            cur = conn.execute(
                "INSERT INTO research_query (topic_id,query_type,query_text,result_count,answer_text,sufficient_evidence) "
                "VALUES (?,?,?,?,?,?)", (topic_id, "question", question, source_count, answer, int(sufficient)))
            return int(cur.lastrowid)

    def count(self, query_type: str = "search") -> int:
        with self.db.connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM research_query WHERE query_type=?",
                                    (query_type,)).fetchone()[0])

    def latest_search(self, topic_id: int) -> Optional[Dict[str, Any]]:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM research_query WHERE topic_id=? AND query_type='search' "
                "ORDER BY query_id DESC LIMIT 1", (topic_id,)).fetchone()
        if not row:
            return None
        data = dict(row)
        try:
            data["filters"] = json.loads(data.get("filters_json") or "{}")
        except json.JSONDecodeError:
            data["filters"] = {}
        return data

    def recent_questions(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM research_query WHERE query_type='question' ORDER BY query_id DESC LIMIT ?",
                (limit,)).fetchall()
            return [dict(r) for r in rows]


# ------------------------------------------------------------------- analyses


class AnalysisRepository:
    def __init__(self, db: Database):
        self.db = db

    def save(self, topic_id: int, analysis_type: str, content: Any, generated_by: str,
             paper_id: Optional[int] = None) -> None:
        """Store an analysis, replacing the previous one of the same kind."""
        with self.db.connection() as conn:
            conn.execute(
                "DELETE FROM research_analysis WHERE topic_id=? AND analysis_type=? AND "
                "((paper_id IS NULL AND ? IS NULL) OR paper_id = ?)",
                (topic_id, analysis_type, paper_id, paper_id))
            conn.execute(
                "INSERT INTO research_analysis (topic_id,paper_id,analysis_type,content_json,generated_by) "
                "VALUES (?,?,?,?,?)",
                (topic_id, paper_id, analysis_type, json.dumps(content, default=str), generated_by))

    def get(self, topic_id: int, analysis_type: str, paper_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        with self.db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM research_analysis WHERE topic_id=? AND analysis_type=? AND "
                "((paper_id IS NULL AND ? IS NULL) OR paper_id = ?) ORDER BY analysis_id DESC LIMIT 1",
                (topic_id, analysis_type, paper_id, paper_id)).fetchone()
        if not row:
            return None
        return {"content": json.loads(row["content_json"]), "generated_by": row["generated_by"],
                "created_at": row["created_at"]}

    def get_for_papers(self, topic_id: int, analysis_type: str) -> Dict[int, Dict[str, Any]]:
        """All per-paper analyses of one type for a topic, keyed by paper_id."""
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM research_analysis WHERE topic_id=? AND analysis_type=? AND paper_id IS NOT NULL "
                "ORDER BY analysis_id", (topic_id, analysis_type)).fetchall()
        return {r["paper_id"]: {"content": json.loads(r["content_json"]), "generated_by": r["generated_by"]}
                for r in rows}


# ----------------------------------------------------------------- gaps + Qs


class GapRepository:
    def __init__(self, db: Database):
        self.db = db

    def replace_for_topic(self, topic_id: int, gaps: Sequence[GapItem],
                          questions: Sequence[ResearchQuestionItem]) -> None:
        """Replace all stored gaps and questions of a topic in one transaction."""
        with self.db.connection() as conn:
            conn.execute("DELETE FROM research_question WHERE topic_id=?", (topic_id,))
            conn.execute("DELETE FROM research_gap WHERE topic_id=?", (topic_id,))
            gap_ids: List[int] = []
            for gap in gaps:
                cur = conn.execute(
                    "INSERT INTO research_gap (topic_id,gap_type,description,rationale,origin,needs_verification) "
                    "VALUES (?,?,?,?,?,?)",
                    (topic_id, gap.gap_type, gap.description, gap.rationale, gap.origin, int(gap.needs_verification)))
                gap_id = int(cur.lastrowid)
                gap_ids.append(gap_id)
                for pid in gap.supporting_paper_ids:
                    conn.execute("INSERT OR IGNORE INTO gap_supporting_paper (gap_id, paper_id) "
                                 "SELECT ?, paper_id FROM research_paper WHERE paper_id=?", (gap_id, pid))
            for q in questions:
                gap_id = gap_ids[q.gap_index] if q.gap_index is not None and 0 <= q.gap_index < len(gap_ids) else None
                conn.execute(
                    "INSERT INTO research_question (topic_id,gap_id,question_text,rationale,suggested_design,origin) "
                    "VALUES (?,?,?,?,?,?)",
                    (topic_id, gap_id, q.question, q.rationale, q.suggested_design, q.origin))

    def list_gaps(self, topic_id: int) -> List[GapItem]:
        with self.db.connection() as conn:
            rows = conn.execute("SELECT * FROM research_gap WHERE topic_id=? ORDER BY gap_id", (topic_id,)).fetchall()
            support = conn.execute(
                "SELECT gs.gap_id, gs.paper_id FROM gap_supporting_paper gs "
                "JOIN research_gap g ON g.gap_id = gs.gap_id WHERE g.topic_id=?", (topic_id,)).fetchall()
        by_gap: Dict[int, List[int]] = {}
        for r in support:
            by_gap.setdefault(r["gap_id"], []).append(r["paper_id"])
        return [GapItem(gap_id=r["gap_id"], gap_type=r["gap_type"], description=r["description"],
                        rationale=r["rationale"], supporting_paper_ids=by_gap.get(r["gap_id"], []),
                        origin=r["origin"], needs_verification=bool(r["needs_verification"])) for r in rows]

    def list_questions(self, topic_id: int) -> List[ResearchQuestionItem]:
        with self.db.connection() as conn:
            rows = conn.execute("SELECT * FROM research_question WHERE topic_id=? ORDER BY question_id",
                                (topic_id,)).fetchall()
            gap_order = [r["gap_id"] for r in conn.execute(
                "SELECT gap_id FROM research_gap WHERE topic_id=? ORDER BY gap_id", (topic_id,)).fetchall()]
        items = []
        for r in rows:
            idx = gap_order.index(r["gap_id"]) if r["gap_id"] in gap_order else None
            items.append(ResearchQuestionItem(
                question_id=r["question_id"], gap_id=r["gap_id"], gap_index=idx, question=r["question_text"],
                rationale=r["rationale"], suggested_design=r["suggested_design"], origin=r["origin"]))
        return items


# -------------------------------------------------------------------- reports


class ReportRepository:
    def __init__(self, db: Database):
        self.db = db

    def add(self, topic_id: int, title: str, file_path: str, size: int) -> int:
        with self.db.connection() as conn:
            cur = conn.execute(
                "INSERT INTO generated_report (topic_id,title,file_path,file_size_bytes) VALUES (?,?,?,?)",
                (topic_id, title, file_path, size))
            return int(cur.lastrowid)

    def list(self, topic_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        sql = ("SELECT r.*, t.title AS topic_title FROM generated_report r "
               "JOIN research_topic t ON t.topic_id = r.topic_id")
        params: List[Any] = []
        if topic_id is not None:
            sql += " WHERE r.topic_id = ?"
            params.append(topic_id)
        sql += " ORDER BY r.report_id DESC LIMIT ?"
        params.append(limit)
        with self.db.connection() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def count(self) -> int:
        with self.db.connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM generated_report").fetchone()[0])


# ---------------------------------------------------------------------- stats


class StatsRepository:
    """Aggregates for the dashboard."""

    def __init__(self, db: Database):
        self.db = db

    def year_counts(self) -> List[Tuple[int, int]]:
        with self.db.connection() as conn:
            rows = conn.execute(
                "SELECT pub_year, COUNT(*) AS n FROM research_paper WHERE pub_year IS NOT NULL "
                "GROUP BY pub_year ORDER BY pub_year").fetchall()
        return [(int(r["pub_year"]), int(r["n"])) for r in rows]

    def source_counts(self) -> Dict[str, int]:
        with self.db.connection() as conn:
            rows = conn.execute("SELECT sources FROM research_paper").fetchall()
        counter: Counter = Counter()
        for r in rows:
            for source in (r["sources"] or "").split(","):
                if source:
                    counter[source] += 1
        return dict(counter)
