from __future__ import annotations

import time
from typing import Any, Callable

from outils.llm_client import create_client, llm_call

CHATBOT_SEARCH_SYSTEM_PROMPT = (
    "Tu es un assistant expert en recrutement pour Finaxys. Tu aides des RH à évaluer rapidement "
    "des consultants par rapport à un besoin précis (poste, appel d'offre, mission).\n\n"
    "RÈGLES :\n"
    "1. Ne parle QUE des profils réellement pertinents pour la demande. Si un profil ne correspond pas, "
    "ignore-le silencieusement — n'explique pas pourquoi il est écarté, le RH n'a pas besoin de lire une "
    "comparaison de chaque candidat trouvé, seulement des profils qui correspondent.\n"
    "2. Pour chaque profil pertinent, sois DÉTAILLÉ et concret : cite les expériences, missions, projets "
    "et technologies précises du candidat qui répondent au besoin exprimé. Ne résume pas le CV en général, "
    "explique le lien direct entre son parcours réel et la demande.\n"
    "3. Mets en avant ce qui n'est pas visible dans un simple titre de poste : contexte des missions, "
    "résultats obtenus, spécificités techniques — tout ce qu'un RH ne devinerait pas sans lire le CV en détail.\n"
    "4. Va droit au but dans la forme (pas de remplissage ni de formules creuses), mais complet sur le fond.\n"
    "5. Termine chaque profil mentionné par la mention de son fichier source, pour que le RH puisse "
    "télécharger son CV complet."
)


def _compute_relevance(distance: float) -> float:
    return round(max(0.0, 1.0 - float(distance) / 2.0), 3)


def _answer_mentions_sources(answer: str, sources: list[str]) -> bool:
    normalized_answer = answer.lower()
    return any(source.lower() in normalized_answer for source in sources if source)


_GREETING_WORDS = {
    "bonjour", "salut", "hello", "hi", "coucou", "bonsoir", "merci",
    "ça va", "ca va", "au revoir", "bye", "ok", "d'accord", "test",
}


def is_greeting_or_chitchat(query: str) -> bool:
    """Detecte les messages de politesse/hors-sujet pour eviter de lancer une
    recherche RAG complete sur un simple 'bonjour'."""
    normalized = query.strip().lower().rstrip("!?. ")
    return normalized in _GREETING_WORDS or len(normalized) < 3


def run_chatbot_search(
    query: str,
    *,
    vdb: Any,
    n_results: int = 3,
    create_client_fn: Callable[..., tuple[Any, str]] = create_client,
    llm_call_fn: Callable[..., str] = llm_call,
) -> dict[str, Any]:
    start = time.perf_counter()
    results = vdb.search(query, n_results=n_results)

    documents = results.get("documents", [[]])
    metadatas = results.get("metadatas", [[]])
    distances = results.get("distances", [[]])
    docs = documents[0] if documents else []
    metas = metadatas[0] if metadatas else []
    dists = distances[0] if distances else []

    if not docs:
        return {
            "status": "empty",
            "query": query,
            "answer": "",
            "sources": [],
            "candidates": [],
            "candidate_count": 0,
            "avg_relevance": 0.0,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "answer_mentions_source": False,
        }

    traced_candidates: list[dict[str, Any]] = []
    context_parts: list[str] = []
    sources: list[str] = []
    for index, doc in enumerate(docs):
        meta = metas[index] if index < len(metas) else {}
        source = str(meta.get("source", "Inconnu"))
        distance = dists[index] if index < len(dists) else 1.0
        relevance = _compute_relevance(distance)
        traced_candidates.append(
            {
                "name": source,
                "source": source,
                "relevanceScore": relevance,
                "metadata": meta,
            }
        )
        sources.append(source)
        context_parts.append(f"CANDIDAT {index + 1} (Source: {source}):\n{doc}")

    client, provider = create_client_fn()
    answer = llm_call_fn(
        client,
        provider,
        CHATBOT_SEARCH_SYSTEM_PROMPT,
        f"Recherche RH: {query}\n\nProfils trouvés:\n" + "\n---\n".join(context_parts),
        operation="telegram_rag_answer",
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    avg_relevance = round(
        sum(candidate["relevanceScore"] for candidate in traced_candidates) / len(traced_candidates),
        3,
    )
    # Seuls les candidats reellement cites dans la reponse sont proposes au
    # telechargement -- sinon on affiche des profils que le LLM a ecartes.
    cited_candidates = [
        c for c in traced_candidates if c["source"].lower() in answer.lower()
    ] or traced_candidates
    return {
        "status": "success",
        "query": query,
        "answer": answer,
        "sources": sources,
        "candidates": traced_candidates,
        "cited_candidates": cited_candidates,
        "candidate_count": len(traced_candidates),
        "avg_relevance": avg_relevance,
        "latency_ms": latency_ms,
        "answer_mentions_source": _answer_mentions_sources(answer, sources),
    }
