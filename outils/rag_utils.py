import json
import logging
import os
import time
from typing import Any, Optional

from psycopg import connect
from psycopg.rows import dict_row
from sentence_transformers import CrossEncoder, SentenceTransformer

from schemas.models import CVData
from outils.metrics import observe_rag_embedding, observe_rag_index

logger = logging.getLogger("rag_utils")

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_BATCH_SIZE = 16
DEFAULT_PGVECTOR_COLLECTION = "cv_collection"
DEFAULT_PGVECTOR_TABLE_PREFIX = "rag_vector"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANKER_BATCH_SIZE = 8
DEFAULT_RERANKER_CANDIDATE_POOL = 20
DEFAULT_HYBRID_DENSE_CANDIDATE_POOL = 30
DEFAULT_HYBRID_LEXICAL_CANDIDATE_POOL = 30
DEFAULT_SQL_PREFILTER_CANDIDATE_POOL = 120
DEFAULT_HYBRID_RRF_K = 60


def list_recent_analyses(
    limit: int = 10, collection_name: str = DEFAULT_PGVECTOR_COLLECTION
) -> list[dict[str, Any]]:
    """Requête légère sur pgvector (pas de chargement du modèle d'embedding) pour
    afficher l'historique des dernières analyses, ex. au chargement d'une page."""
    dsn = os.getenv("PGVECTOR_DSN") or os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN")
    if not dsn:
        return []
    table_prefix = os.getenv("PGVECTOR_TABLE_PREFIX", DEFAULT_PGVECTOR_TABLE_PREFIX)
    document_table = f"{table_prefix}_documents"
    with connect(dsn, autocommit=True, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT doc_id, metadata, created_at
            FROM {document_table}
            WHERE collection_name = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (collection_name, limit),
        )
        rows = cur.fetchall()
    return [
        {
            "doc_id": row["doc_id"],
            "nom": row["metadata"].get("nom", ""),
            "prenom": row["metadata"].get("prenom", ""),
            "titre": row["metadata"].get("titre", ""),
            "source": row["metadata"].get("source", row["doc_id"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _build_like_patterns(terms: list[str]) -> list[str]:
    patterns: list[str] = []
    for term in _dedupe_keep_order(terms):
        escaped = term.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        patterns.append(f"%{escaped}%")
    return patterns


def _build_websearch_query(terms: list[str]) -> str:
    tokens: list[str] = []
    for term in _dedupe_keep_order(terms):
        token = term.replace('"', " ").strip()
        if not token:
            continue
        if " " in token:
            tokens.append(f'"{token}"')
        else:
            tokens.append(token)
    return " OR ".join(tokens)


def cv_to_searchable_text(cv: CVData) -> str:
    """
    Transforme un objet CVData en une description textuelle riche pour le RAG.
    Extraction déterministe sans LLM.
    """
    sections = []

    identite = f"Candidat : {cv.identite.prenom} {cv.identite.nom}"
    if cv.titre_professionnel:
        identite += f" ({cv.titre_professionnel})"
    sections.append(identite)

    if cv.profil:
        sections.append(f"Profil résumé : {cv.profil}")

    if cv.competences:
        skills = []
        for skill in cv.competences:
            if isinstance(skill, dict):
                skills.append(f"{skill.get('nom')} ({skill.get('niveau', 'Intermédiaire')})")
            else:
                skills.append(skill.nom if hasattr(skill, "nom") else str(skill))
        sections.append(f"Compétences techniques : {', '.join(skills)}.")

    if cv.experiences:
        exp_list = []
        for exp in cv.experiences[:5]:
            exp_text = f"{exp.titre} chez {exp.entreprise} ({exp.date_debut} à {exp.date_fin or 'Présent'})"
            if exp.projets:
                for proj in exp.projets:
                    proj_text = f" — Projet {proj.nom}" if proj.nom else ""
                    if proj.description:
                        proj_text += f" : {proj.description}"
                    if proj.missions:
                        proj_text += f". Missions : {' '.join(proj.missions[:5])}"
                    if proj.technologies:
                        proj_text += f". Technologies : {', '.join(proj.technologies)}"
                    exp_text += proj_text
            elif exp.missions:
                exp_text += f". Missions : {' '.join(exp.missions[:3])}"
            exp_list.append(exp_text)
        sections.append(f"Parcours professionnel : {' | '.join(exp_list)}")

    if cv.certifications:
        certs = [f"{cert.nom} ({cert.organisme or 'N/A'})" for cert in cv.certifications]
        sections.append(f"Certifications : {', '.join(certs)}.")

    if cv.langues:
        langs = [f"{lang.langue} ({lang.niveau})" for lang in cv.langues]
        sections.append(f"Langues : {', '.join(langs)}.")

    return "\n".join(sections)


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


class _PgVectorCollection:
    def __init__(self, manager: "VectorStoreManager") -> None:
        self._manager = manager

    def count(self) -> int:
        return self._manager.count()

    def get(self, include: Optional[list[str]] = None) -> dict[str, list[Any]]:
        return self._manager.get(include=include)


class VectorStoreManager:
    def __init__(self, db_path: Optional[str] = None, collection_name: str = DEFAULT_PGVECTOR_COLLECTION):
        del db_path  # Conservé pour compatibilité d'interface; pgvector utilise un DSN PostgreSQL.

        self.collection_name = collection_name
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", str(DEFAULT_EMBEDDING_BATCH_SIZE)))
        self.table_prefix = os.getenv("PGVECTOR_TABLE_PREFIX", DEFAULT_PGVECTOR_TABLE_PREFIX)
        self.collection_table = f"{self.table_prefix}_collections"
        self.document_table = f"{self.table_prefix}_documents"
        self.reranker_enabled = _env_flag("RERANKER_ENABLED", False)
        self.reranker_model_name = os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
        self.reranker_batch_size = int(os.getenv("RERANKER_BATCH_SIZE", str(DEFAULT_RERANKER_BATCH_SIZE)))
        self.reranker_candidate_pool = int(
            os.getenv("RERANKER_CANDIDATE_POOL", str(DEFAULT_RERANKER_CANDIDATE_POOL))
        )
        self.hybrid_dense_candidate_pool = int(
            os.getenv("HYBRID_DENSE_CANDIDATE_POOL", str(DEFAULT_HYBRID_DENSE_CANDIDATE_POOL))
        )
        self.hybrid_lexical_candidate_pool = int(
            os.getenv("HYBRID_LEXICAL_CANDIDATE_POOL", str(DEFAULT_HYBRID_LEXICAL_CANDIDATE_POOL))
        )
        self.sql_prefilter_candidate_pool = int(
            os.getenv("SQL_PREFILTER_CANDIDATE_POOL", str(DEFAULT_SQL_PREFILTER_CANDIDATE_POOL))
        )
        self.hybrid_rrf_k = int(os.getenv("HYBRID_RRF_K", str(DEFAULT_HYBRID_RRF_K)))
        self.dsn = (
            os.getenv("PGVECTOR_DSN")
            or os.getenv("DATABASE_URL")
            or os.getenv("POSTGRES_DSN")
        )

        if not self.dsn:
            raise RuntimeError("PGVECTOR_DSN ou DATABASE_URL est requis pour le vector store PostgreSQL.")

        logger.info("Chargement du modèle d'embedding local : %s", self.embedding_model_name)
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            trust_remote_code=self.embedding_model_name.lower() == "baai/bge-m3",
        )
        warmup = self.embedding_model.encode(
            ["warmup"],
            normalize_embeddings=True,
            batch_size=1,
            show_progress_bar=False,
        )
        self.embedding_dim = len(warmup[0])
        self.reranker: Optional[CrossEncoder] = None
        if self.reranker_enabled:
            logger.info("Chargement du reranker local : %s", self.reranker_model_name)
            self.reranker = CrossEncoder(self.reranker_model_name)

        self.collection = _PgVectorCollection(self)
        self._ensure_schema()
        self._ensure_collection_registration()
        logger.info(
            "Vector store PostgreSQL prêt : collection=%s dim=%s",
            self.collection_name,
            self.embedding_dim,
        )

    def _connect(self):
        conn = connect(self.dsn, autocommit=True, row_factory=dict_row)
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.collection_table} (
                    name TEXT PRIMARY KEY,
                    embedding_model TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.document_table} (
                    collection_name TEXT NOT NULL REFERENCES {self.collection_table}(name) ON DELETE CASCADE,
                    doc_id TEXT NOT NULL,
                    document TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({self.embedding_dim}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (collection_name, doc_id)
                )
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.document_table}_collection_idx ON {self.document_table} (collection_name)"
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.document_table}_document_fts_idx
                ON {self.document_table}
                USING gin (to_tsvector('simple', document))
                """
            )
            try:
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self.document_table}_embedding_hnsw_idx
                    ON {self.document_table}
                    USING hnsw (embedding vector_cosine_ops)
                    """
                )
            except Exception as exc:
                logger.warning("Impossible de créer l'index HNSW pgvector: %s", exc)

    def _ensure_collection_registration(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT embedding_model, embedding_dim FROM {self.collection_table} WHERE name = %s",
                (self.collection_name,),
            )
            row = cur.fetchone()
            if row:
                if row["embedding_model"] != self.embedding_model_name or row["embedding_dim"] != self.embedding_dim:
                    raise ValueError(
                        f"La collection '{self.collection_name}' existe déjà avec "
                        f"embedding_model={row['embedding_model']} et embedding_dim={row['embedding_dim']}. "
                        "Réinitialise la collection avant de changer de modèle d'embedding."
                    )
                cur.execute(
                    f"UPDATE {self.collection_table} SET updated_at = NOW() WHERE name = %s",
                    (self.collection_name,),
                )
                return

            cur.execute(
                f"""
                INSERT INTO {self.collection_table} (name, embedding_model, embedding_dim)
                VALUES (%s, %s, %s)
                """,
                (self.collection_name, self.embedding_model_name, self.embedding_dim),
            )

    def _encode_documents(self, texts: list[str]) -> list[list[float]]:
        start = time.perf_counter()
        try:
            embeddings = self.embedding_model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=self.embedding_batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:
            observe_rag_embedding(
                model=self.embedding_model_name,
                operation="document",
                status="error",
                error_type=exc.__class__.__name__,
                latency_ms=round((time.perf_counter() - start) * 1000, 1),
            )
            raise
        observe_rag_embedding(
            model=self.embedding_model_name,
            operation="document",
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        return embeddings.tolist()

    def _encode_queries(self, texts: list[str]) -> list[list[float]]:
        start = time.perf_counter()
        try:
            embeddings = self.embedding_model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=self.embedding_batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:
            observe_rag_embedding(
                model=self.embedding_model_name,
                operation="query",
                status="error",
                error_type=exc.__class__.__name__,
                latency_ms=round((time.perf_counter() - start) * 1000, 1),
            )
            raise
        observe_rag_embedding(
            model=self.embedding_model_name,
            operation="query",
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        return embeddings.tolist()

    def _rerank_rows(self, query: str, rows: list[dict[str, Any]], n_results: int) -> list[dict[str, Any]]:
        if not self.reranker or not rows:
            return rows[:n_results]

        scores = self.reranker.predict(
            [(query, row["document"]) for row in rows],
            batch_size=self.reranker_batch_size,
            show_progress_bar=False,
        )
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if not isinstance(scores, list):
            scores = [float(scores)]

        for row, score in zip(rows, scores):
            row["rerank_score"] = float(score)

        rows.sort(key=lambda row: row.get("rerank_score", float("-inf")), reverse=True)
        return rows[:n_results]

    def _fetch_dense_rows(
        self,
        query: str,
        *,
        limit: int,
        candidate_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        query_embedding = _vector_literal(self._encode_queries([query])[0])
        sql = f"""
            SELECT
                doc_id,
                document,
                metadata,
                (embedding <=> %s::vector) AS distance
            FROM {self.document_table}
            WHERE collection_name = %s
        """
        params: list[Any] = [query_embedding, self.collection_name]
        if candidate_ids:
            sql += " AND doc_id = ANY(%s)"
            params.append(candidate_ids)
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([query_embedding, limit])

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def _prefilter_doc_ids(self, terms: list[str], limit: int) -> list[str]:
        patterns = _build_like_patterns(terms)
        if not patterns:
            return []

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT doc_id
                FROM {self.document_table}
                WHERE collection_name = %s
                  AND EXISTS (
                      SELECT 1
                      FROM unnest(%s::text[]) AS pattern
                      WHERE LOWER(document) LIKE pattern ESCAPE '\\'
                         OR LOWER(COALESCE(metadata->>'titre', '')) LIKE pattern ESCAPE '\\'
                  )
                ORDER BY (
                    SELECT COALESCE(SUM(
                        CASE
                            WHEN LOWER(COALESCE(metadata->>'titre', '')) LIKE pattern ESCAPE '\\' THEN 2
                            WHEN LOWER(document) LIKE pattern ESCAPE '\\' THEN 1
                            ELSE 0
                        END
                    ), 0)
                    FROM unnest(%s::text[]) AS pattern
                ) DESC, doc_id
                LIMIT %s
                """,
                (self.collection_name, patterns, patterns, limit),
            )
            rows = cur.fetchall()
        return [row["doc_id"] for row in rows]

    def _fetch_lexical_rows(
        self,
        terms: list[str],
        *,
        limit: int,
        candidate_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        patterns = _build_like_patterns(terms)
        lexical_query = _build_websearch_query(terms)
        if not patterns and not lexical_query:
            return []

        sql = f"""
            SELECT
                doc_id,
                document,
                metadata,
                (
                    SELECT COALESCE(SUM(
                        CASE
                            WHEN LOWER(COALESCE(metadata->>'titre', '')) LIKE pattern ESCAPE '\\' THEN 2
                            WHEN LOWER(document) LIKE pattern ESCAPE '\\' THEN 1
                            ELSE 0
                        END
                    ), 0)
                    FROM unnest(%s::text[]) AS pattern
                ) AS lexical_hits,
                CASE
                    WHEN %s = '' THEN 0.0
                    ELSE ts_rank_cd(
                        to_tsvector('simple', document),
                        websearch_to_tsquery('simple', %s)
                    )
                END AS lexical_score
            FROM {self.document_table}
            WHERE collection_name = %s
        """
        params: list[Any] = [patterns, lexical_query, lexical_query, self.collection_name]
        if candidate_ids:
            sql += " AND doc_id = ANY(%s)"
            params.append(candidate_ids)

        sql += """
            AND (
                EXISTS (
                    SELECT 1
                    FROM unnest(%s::text[]) AS pattern
                    WHERE LOWER(document) LIKE pattern ESCAPE '\\'
                       OR LOWER(COALESCE(metadata->>'titre', '')) LIKE pattern ESCAPE '\\'
                )
        """
        params.append(patterns)

        if lexical_query:
            sql += """
                OR to_tsvector('simple', document) @@ websearch_to_tsquery('simple', %s)
            )
            """
            params.append(lexical_query)
        else:
            sql += ")"

        sql += " ORDER BY lexical_hits DESC, lexical_score DESC, doc_id LIMIT %s"
        params.append(limit)

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def _rrf_fuse(
        self,
        *,
        query: str,
        dense_rows: list[dict[str, Any]],
        lexical_rows: list[dict[str, Any]],
        n_results: int,
    ) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}

        for rank, row in enumerate(dense_rows, start=1):
            entry = fused.setdefault(
                row["doc_id"],
                {
                    "doc_id": row["doc_id"],
                    "document": row["document"],
                    "metadata": row["metadata"],
                    "distance": float(row.get("distance", 2.0)),
                    "dense_rank": None,
                    "lexical_rank": None,
                    "dense_score": 0.0,
                    "lexical_score": 0.0,
                    "lexical_hits": 0,
                    "hybrid_score": 0.0,
                },
            )
            entry["dense_rank"] = rank
            entry["distance"] = float(row.get("distance", entry["distance"]))
            entry["dense_score"] = max(0.0, 1.0 - entry["distance"] / 2.0)
            entry["hybrid_score"] += 1.0 / (self.hybrid_rrf_k + rank)

        for rank, row in enumerate(lexical_rows, start=1):
            entry = fused.setdefault(
                row["doc_id"],
                {
                    "doc_id": row["doc_id"],
                    "document": row["document"],
                    "metadata": row["metadata"],
                    "distance": 2.0,
                    "dense_rank": None,
                    "lexical_rank": None,
                    "dense_score": 0.0,
                    "lexical_score": 0.0,
                    "lexical_hits": 0,
                    "hybrid_score": 0.0,
                },
            )
            entry["lexical_rank"] = rank
            entry["lexical_score"] = float(row.get("lexical_score", 0.0) or 0.0)
            entry["lexical_hits"] = int(row.get("lexical_hits", 0) or 0)
            entry["hybrid_score"] += 1.0 / (self.hybrid_rrf_k + rank)

        rows = list(fused.values())
        rows.sort(
            key=lambda row: (
                -row["hybrid_score"],
                -row["lexical_hits"],
                -row["lexical_score"],
                -row["dense_score"],
            )
        )
        return self._rerank_rows(query, rows, n_results)

    def reset_collection(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self.collection_table} WHERE name = %s",
                (self.collection_name,),
            )
        self._ensure_collection_registration()

    def count(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) AS total FROM {self.document_table} WHERE collection_name = %s",
                (self.collection_name,),
            )
            row = cur.fetchone()
            return int(row["total"]) if row else 0

    def get(self, include: Optional[list[str]] = None) -> dict[str, list[Any]]:
        include = include or ["ids", "documents", "metadatas"]
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT doc_id, document, metadata
                FROM {self.document_table}
                WHERE collection_name = %s
                ORDER BY doc_id
                """,
                (self.collection_name,),
            )
            rows = cur.fetchall()

        result: dict[str, list[Any]] = {}
        if "ids" in include:
            result["ids"] = [row["doc_id"] for row in rows]
        if "documents" in include:
            result["documents"] = [row["document"] for row in rows]
        if "metadatas" in include:
            result["metadatas"] = [row["metadata"] for row in rows]
        return result

    def list_recent(self, limit: int = 10) -> list[dict[str, Any]]:
        """Retourne les CV indexés les plus récents (pour l'historique des analyses)."""
        return list_recent_analyses(limit=limit, collection_name=self.collection_name)

    def add_cv(
        self,
        cv: CVData,
        doc_id: str,
        extra_metadata: Optional[dict] = None,
        index_source: str = "unknown",
    ):
        """Ajoute ou met à jour un CV dans le vector store PostgreSQL/pgvector."""
        start = time.perf_counter()
        text = cv_to_searchable_text(cv)
        try:
            embedding = self._encode_documents([text])[0]

            metadata = {
                "nom": cv.identite.nom,
                "prenom": cv.identite.prenom,
                "titre": cv.titre_professionnel or "",
                "source": doc_id,
                "embedding_model": self.embedding_model_name,
                "skills": [
                    skill.get("nom") if isinstance(skill, dict) else getattr(skill, "nom", str(skill))
                    for skill in (cv.competences or [])
                ],
                "languages": [lang.langue for lang in (cv.langues or [])],
            }
            if extra_metadata:
                metadata.update(extra_metadata)

            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.document_table} (
                        collection_name, doc_id, document, metadata, embedding, updated_at
                    )
                    VALUES (%s, %s, %s, %s::jsonb, %s::vector, NOW())
                    ON CONFLICT (collection_name, doc_id) DO UPDATE SET
                        document = EXCLUDED.document,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        updated_at = NOW()
                    """,
                    (
                        self.collection_name,
                        doc_id,
                        text,
                        json.dumps(metadata, ensure_ascii=False),
                        _vector_literal(embedding),
                    ),
                )
        except Exception:
            observe_rag_index(
                collection=self.collection_name,
                source=index_source,
                status="error",
                latency_ms=round((time.perf_counter() - start) * 1000, 1),
            )
            raise
        observe_rag_index(
            collection=self.collection_name,
            source=index_source,
            status="success",
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            chunks_indexed=1,
        )
        logger.info("CV de %s indexé avec succès dans PostgreSQL.", cv.identite.nom)

    def _rows_to_result(self, rows: list[dict[str, Any]]) -> dict[str, list[list[Any]]]:
        return {
            "ids": [[row["doc_id"] for row in rows]],
            "documents": [[row["document"] for row in rows]],
            "metadatas": [[row["metadata"] for row in rows]],
            "distances": [[float(row.get("distance", 2.0)) for row in rows]],
            "rerank_scores": [[row.get("rerank_score") for row in rows]],
            "hybrid_scores": [[row.get("hybrid_score") for row in rows]],
            "lexical_scores": [[row.get("lexical_score") for row in rows]],
            "lexical_hits": [[row.get("lexical_hits") for row in rows]],
            "dense_scores": [[row.get("dense_score") for row in rows]],
        }

    def search(self, query: str, n_results: int = 3):
        """Recherche les CV les plus pertinents pour une requête donnée."""
        dense_limit = n_results
        if self.reranker_enabled:
            dense_limit = max(n_results, self.reranker_candidate_pool)
        rows = self._fetch_dense_rows(query, limit=dense_limit)

        rows = self._rerank_rows(query, rows, n_results)
        return self._rows_to_result(rows)

    def search_job(
        self,
        *,
        query: str,
        job_terms: list[str],
        n_results: int = 5,
    ) -> dict[str, list[list[Any]]]:
        candidate_ids = self._prefilter_doc_ids(job_terms, limit=self.sql_prefilter_candidate_pool)
        if candidate_ids and len(candidate_ids) < n_results:
            candidate_ids = None

        dense_limit = max(n_results, self.hybrid_dense_candidate_pool)
        if self.reranker_enabled:
            dense_limit = max(dense_limit, self.reranker_candidate_pool)

        dense_rows = self._fetch_dense_rows(
            query,
            limit=dense_limit,
            candidate_ids=candidate_ids,
        )
        lexical_rows = self._fetch_lexical_rows(
            job_terms,
            limit=max(n_results, self.hybrid_lexical_candidate_pool),
            candidate_ids=candidate_ids,
        )

        if lexical_rows:
            rows = self._rrf_fuse(
                query=query,
                dense_rows=dense_rows,
                lexical_rows=lexical_rows,
                n_results=n_results,
            )
        else:
            rows = self._rerank_rows(query, dense_rows, n_results)

        return self._rows_to_result(rows)
