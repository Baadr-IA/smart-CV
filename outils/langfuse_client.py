"""
Client Langfuse optionnel pour l'observabilite du pipeline CV.
Compatible avec langfuse SDK v2.x (API trace/score).
Si LANGFUSE_SECRET_KEY et LANGFUSE_PUBLIC_KEY ne sont pas definis,
toutes les fonctions sont des no-ops transparents (pas d'erreur).

Configuration dans .env :
  LANGFUSE_SECRET_KEY=sk-lf-...
  LANGFUSE_PUBLIC_KEY=pk-lf-...
  LANGFUSE_HOST=http://localhost:3000
"""

import os
import logging
from typing import Optional, Any

logger = logging.getLogger("langfuse_client")

_langfuse: Optional[Any] = None
_enabled = False


def init_langfuse():
    global _langfuse, _enabled
    secret = os.getenv("LANGFUSE_SECRET_KEY", "")
    public = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    if not secret or not public:
        logger.info("Langfuse desactive (cles manquantes)")
        return
    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(secret_key=secret, public_key=public, host=host)
        _enabled = True
        logger.info("Langfuse initialise -> %s", host)
    except ImportError:
        logger.warning("Package langfuse non installe")
    except Exception as e:
        logger.warning("Langfuse init echoue : %s", e)


def is_enabled() -> bool:
    return _enabled


def get_client():
    return _langfuse


def trace_rag_search(query: str, job_title: str, candidates: list, required_skills: list,
                     context_relevance: float = 0.0, latency_ms: float = 0.0):
    if not _enabled or _langfuse is None:
        return None
    try:
        trace = _langfuse.trace(
            name="rag_job_search",
            input={"query": query, "job_title": job_title, "required_skills": required_skills},
            output={"n_candidates": len(candidates), "candidates": [c.get("name", "?") for c in candidates]},
            metadata={"context_relevance_avg": context_relevance, "latency_ms": latency_ms},
            tags=["rag", "job-search"],
        )
        _langfuse.score(trace_id=trace.id, name="context_relevance", value=context_relevance)
        return trace.id
    except Exception as e:
        logger.debug("trace_rag_search failed: %s", e)
        return None


def trace_cv_analysis(filename: str, cv_data: dict, latency_ms: float = 0.0, indexed: bool = False):
    if not _enabled or _langfuse is None:
        return None
    try:
        identite = cv_data.get("identite", {})
        competences = cv_data.get("competences", [])
        completude = cv_data.get("metadata", {}).get("score_completude", 0.0)
        trace = _langfuse.trace(
            name="cv_analysis",
            input={"filename": filename},
            output={
                "candidat": f"{identite.get('prenom', '')} {identite.get('nom', '')}".strip(),
                "n_competences": len(competences),
                "indexed_in_rag": indexed,
            },
            metadata={"score_completude": completude, "latency_ms": latency_ms},
            tags=["cv-analysis"],
        )
        _langfuse.score(trace_id=trace.id, name="completude", value=completude)
        return trace.id
    except Exception as e:
        logger.debug("trace_cv_analysis failed: %s", e)
        return None


def send_rag_eval_scores(poste: str, context_relevance: float, groundedness: float,
                         answer_relevance: float, hit_rate: float, n_cvs_indexed: int):
    if not _enabled or _langfuse is None:
        return None
    try:
        trace = _langfuse.trace(
            name="rag_evaluation",
            input={"poste": poste, "n_cvs_indexed": n_cvs_indexed},
            output={
                "context_relevance": context_relevance,
                "groundedness": groundedness,
                "answer_relevance": answer_relevance,
                "hit_rate": hit_rate,
            },
            tags=["evaluation", "rag-triad"],
        )
        for score_name, value in [
            ("context_relevance", context_relevance),
            ("groundedness", groundedness),
            ("answer_relevance", answer_relevance),
            ("hit_rate", hit_rate),
        ]:
            _langfuse.score(trace_id=trace.id, name=score_name, value=value)
        return trace.id
    except Exception as e:
        logger.debug("send_rag_eval_scores failed: %s", e)
        return None


def flush():
    if _enabled and _langfuse is not None:
        try:
            _langfuse.flush()
        except Exception:
            pass