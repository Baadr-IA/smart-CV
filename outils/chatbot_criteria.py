"""
Chatbot RH conversationnel avec état : contrairement a `run_chatbot_search`
(outils/chatbot_search.py) qui traite chaque message independamment, ce
module fait decider le LLM, a partir de l'HISTORIQUE COMPLET de la
conversation, s'il a assez d'information pour lancer une recherche libre
sur l'appel d'offre accumule, ou s'il doit demander une precision.

Le moteur de matching (search_job / search, dans outils/rag_utils.py) reste
seul responsable du classement -- le LLM ne fait ni ranking ni score. Sa
phase finale produit uniquement des justifications ancrees sur les
competences trouvees/manquantes et le texte reel du CV, jamais inventees.

Perimetre volontairement limite au role recherche + aux competences : la
seniorite, la localisation, la disponibilite, le secteur d'activite et les
contraintes commerciales ne sont pas des champs exploitables par le moteur
actuel -- les mentionner est ignore par construction (voir schemas/models.py
et outils/rag_utils.py).
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Optional

from outils.job_referentiel import compute_skill_gap
from outils.llm_client import create_client, llm_call
from outils.metrics import observe_chatbot_turn

DECISION_SYSTEM_PROMPT = (
    "Tu es l'orchestrateur d'un chatbot de recherche de consultants pour Finaxys. Tu recois "
    "l'historique complet d'une conversation avec un RH ou un commercial. Ton role est de "
    "decider, a partir de CET HISTORIQUE COMPLET (pas seulement le dernier message), si tu as "
    "assez d'information pour lancer une recherche, ou s'il te manque un element essentiel.\n\n"
    "CRITERES EXPLOITABLES PAR LE MOTEUR DE RECHERCHE, UNIQUEMENT :\n"
    "- le role ou l'intitule de poste recherche ;\n"
    "- les competences techniques obligatoires ou souhaitees.\n\n"
    "Le moteur ne gere PAS la seniorite, la localisation, la disponibilite, le secteur "
    "d'activite ni les contraintes commerciales/operationnelles : si l'utilisateur en "
    "mentionne, ignore-les silencieusement -- ne demande jamais de precision sur ces points, "
    "ne les fais jamais apparaitre dans ta reponse.\n\n"
    "REGLES :\n"
    "1. Si un role et au moins une competence sont identifiables dans l'historique (meme "
    "approximativement, par exemple un appel d'offre colle en une fois), action='search'.\n"
    "2. Sinon, action='ask_clarification', avec UNE seule question ciblee sur le role ou les "
    "competences manquantes -- jamais plusieurs questions a la fois, jamais sur un critere non "
    "gere par le moteur.\n"
    "3. Si action='search', reformule une requete de recherche concise a partir de "
    "l'historique (search_query), et liste les competences identifiees (skills).\n\n"
    "Reponds UNIQUEMENT avec un objet JSON de cette forme, sans texte autour ni bloc de code :\n"
    '{"action": "search", "search_query": "...", "skills": ["...", "..."]}\n'
    "ou\n"
    '{"action": "ask_clarification", "clarification_question": "..."}'
)

SHORTLIST_EXPLANATION_SYSTEM_PROMPT = (
    "Tu es un assistant expert en recrutement pour Finaxys. On te donne une liste de "
    "candidats deja classes par un moteur de matching (dense + lexical + fusion), avec pour "
    "chacun les competences trouvees, les competences manquantes, et le texte integral de son "
    "CV indexe. Ton role est de justifier ce classement au RH, jamais de le recalculer.\n\n"
    "REGLES :\n"
    "1. Presente les candidats dans l'ordre donne (deja classe par le moteur) -- ne les "
    "reordonne jamais, n'attribue aucun score toi-meme.\n"
    "2. Pour chaque candidat, justifie sa presence avec les competences trouvees ET des "
    "extraits CONCRETS de son CV (missions, projets, technologies precises) qui montrent le "
    "lien reel avec le besoin -- pas un resume generique.\n"
    "3. Mentionne aussi les competences manquantes, sans dramatiser : ce sont des ecarts, pas "
    "des disqualifications.\n"
    "4. N'invente et ne deduis RIEN qui ne figure pas explicitement dans le texte du CV fourni.\n"
    "5. Termine chaque candidat mentionne par son fichier source."
)


def _parse_json_block(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = None
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None
    return parsed if isinstance(parsed, dict) else {}


def _format_history(history: list[dict[str, str]]) -> str:
    role_labels = {"user": "RH", "assistant": "Assistant"}
    lines = [f"{role_labels.get(m['role'], m['role'])}: {m['content']}" for m in history]
    return "\n".join(lines)


def decide_next_step(
    history: list[dict[str, str]],
    *,
    create_client_fn: Callable[..., tuple[Any, str]] = create_client,
    llm_call_fn: Callable[..., str] = llm_call,
) -> dict[str, Any]:
    """Decide, a partir de l'historique complet, s'il faut chercher ou demander une precision."""
    client, provider = create_client_fn()
    raw = llm_call_fn(
        client,
        provider,
        DECISION_SYSTEM_PROMPT,
        _format_history(history),
        max_tokens=400,
        temperature=0.0,
        operation="chatbot_orchestration_decision",
    )
    decision = _parse_json_block(raw)
    action = decision.get("action")
    if action == "ask_clarification" and decision.get("clarification_question"):
        return {"action": "ask_clarification", "clarification_question": str(decision["clarification_question"])}
    if action == "search":
        skills = decision.get("skills") or []
        return {
            "action": "search",
            "search_query": str(decision.get("search_query") or _format_history(history)),
            "skills": [str(s) for s in skills if str(s).strip()],
        }
    # Reponse LLM non exploitable (JSON invalide, champ attendu absent...) -- on redemande
    # plutot que de chercher a l'aveugle. Marque "fallback=True" pour distinguer, cote metriques,
    # ce cas d'echec du modele d'une vraie clarification decidee par le LLM (cf. observe_chatbot_turn).
    return {
        "action": "ask_clarification",
        "clarification_question": (
            "Peux-tu preciser le poste recherche et les principales competences attendues ?"
        ),
        "fallback": True,
    }


def _build_candidates(required_skills: list[str], docs: list[str], metas: list[dict]) -> list[dict[str, Any]]:
    candidates = []
    for doc, meta in zip(docs, metas):
        matched, missing = compute_skill_gap(required_skills, doc)
        candidates.append(
            {
                "name": f"{meta.get('prenom', '')} {meta.get('nom', '')}".strip() or meta.get("source", "Inconnu"),
                "source": meta.get("source", ""),
                "matchedSkills": matched,
                "missingSkills": missing,
                "document": doc,
            }
        )
    return candidates


def explain_shortlist(
    skills: list[str],
    candidates: list[dict[str, Any]],
    *,
    create_client_fn: Callable[..., tuple[Any, str]] = create_client,
    llm_call_fn: Callable[..., str] = llm_call,
) -> str:
    client, provider = create_client_fn()
    payload = {
        "competences_recherchees": skills,
        "candidats": [
            {
                "nom": c["name"],
                "source": c["source"],
                "competences_trouvees": c["matchedSkills"],
                "competences_manquantes": c["missingSkills"],
                "cv": c["document"],
            }
            for c in candidates
        ],
    }
    return llm_call_fn(
        client,
        provider,
        SHORTLIST_EXPLANATION_SYSTEM_PROMPT,
        json.dumps(payload, ensure_ascii=False),
        max_tokens=2000,
        temperature=0.0,
        operation="chatbot_shortlist_explanation",
    )


def run_conversational_search(
    history: list[dict[str, str]],
    *,
    vdb: Any,
    n_results: int = 5,
    create_client_fn: Callable[..., tuple[Any, str]] = create_client,
    llm_call_fn: Callable[..., str] = llm_call,
) -> dict[str, Any]:
    """Un tour de conversation complet : decide (chercher / demander une precision),
    puis si recherche -- interroge le moteur de matching et fait justifier le
    classement par le LLM a partir des competences trouvees/manquantes et du CV reel."""
    start = time.perf_counter()
    decision = decide_next_step(history, create_client_fn=create_client_fn, llm_call_fn=llm_call_fn)

    if decision["action"] == "ask_clarification":
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        turn_status = "decision_fallback" if decision.get("fallback") else "needs_clarification"
        observe_chatbot_turn(status=turn_status, latency_ms=latency_ms)
        return {
            "status": "needs_clarification",
            "answer": decision["clarification_question"],
            "candidates": [],
            "latency_ms": latency_ms,
        }

    query = decision["search_query"]
    skills = decision["skills"]

    search_results = (
        vdb.search_job(query=query, job_terms=skills, n_results=n_results)
        if skills
        else vdb.search(query, n_results=n_results)
    )

    docs = search_results["documents"][0] if search_results.get("documents") else []
    metas = search_results["metadatas"][0] if search_results.get("metadatas") else []

    if not docs:
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        observe_chatbot_turn(status="empty", latency_ms=latency_ms)
        return {
            "status": "empty",
            "answer": (
                "Aucun profil ne correspond a cette recherche. Essaie de reformuler le poste "
                "ou de retirer une competence trop specifique."
            ),
            "candidates": [],
            "latency_ms": latency_ms,
        }

    candidates = _build_candidates(skills, docs, metas)
    answer = explain_shortlist(skills, candidates, create_client_fn=create_client_fn, llm_call_fn=llm_call_fn)

    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    observe_chatbot_turn(status="success", latency_ms=latency_ms)
    return {
        "status": "success",
        "answer": answer,
        "candidates": candidates,
        "latency_ms": latency_ms,
    }
