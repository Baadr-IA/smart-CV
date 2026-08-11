"""
Interface web SmartCV — remplace le bot Telegram par une interface Streamlit.

Suit le même pattern d'intégration que bot.py : appel direct des fonctions du
pipeline (process_cv_pipeline, VectorStoreManager, run_chatbot_search), sans
passer par l'API HTTP.
"""
import logging
import os
import shutil
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from service import process_cv_pipeline
from outils.rag_utils import VectorStoreManager, list_recent_analyses
from outils.chatbot_search import run_chatbot_search, is_greeting_or_chitchat
from outils.chatbot_criteria import run_conversational_search
from outils.fs_utils import sanitize_client_filename, ensure_extension
from outils.metrics import observe_cv_analysis, observe_rag_search, push_bot_metrics
from outils.langfuse_client import init_langfuse, trace_cv_analysis, trace_rag_search, flush as langfuse_flush
from schemas.models import CVData as CVDataModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit_app")

init_langfuse()
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "http://smartcv-pushgateway:9091")


def _push_metrics() -> None:
    try:
        push_bot_metrics(pushgateway_url=PUSHGATEWAY_URL)
    except Exception:
        logger.warning("push_bot_metrics échoué (non bloquant)", exc_info=True)

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
TEMP_DIR = Path(os.getenv("TEMP_UPLOAD_DIR", "temp_uploads"))
CV_ORIGINALS_DIR = Path(os.getenv("CV_ORIGINALS_DIR", "cv_originals"))
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
CV_ORIGINALS_DIR.mkdir(exist_ok=True)


def _find_candidate_files(source: str) -> dict:
    """Retrouve le CV original et la fiche Finaxys d'un candidat déjà analysé, par nom de fichier source."""
    files = {}
    original = CV_ORIGINALS_DIR / source
    if original.exists():
        files["original"] = original
    fiche = OUTPUT_DIR / f"{Path(source).stem}_finaxys.docx"
    if fiche.exists():
        files["fiche"] = fiche
    return files

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_FILE_SIZE = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024

st.set_page_config(page_title="SmartCV — Finaxys", page_icon="📄", layout="wide")

FINAXYS_LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 91.97 61.22" width="46" height="31">
<style>.st0{fill-rule:evenodd;clip-rule:evenodd;fill:#0f172a;}.st1{fill-rule:evenodd;clip-rule:evenodd;fill:#ff4d00;}</style>
<path class="st0" d="M90.55,23.48L80.72,0c-0.06,0-0.11,0-0.17,0c-2.34,0-4.63,0-6.97,0l-0.04,0.09c-0.05,0.11-0.11,0.23-0.16,0.34 c-3.59,8.51-7.18,17.03-10.76,25.55c-0.12,0.28-0.23,0.56-0.35,0.84c0.07,0.02,0.09,0.04,0.12,0.04c2.45,0,4.91,0,7.36,0.01 c0.27,0,0.32-0.14,0.4-0.33c0.54-1.36,1.08-2.71,1.62-4.07c0.09-0.24,0.2-0.35,0.48-0.35c3.19,0.01,6.37,0.01,9.56,0 c0.25,0,0.36,0.08,0.46,0.32c0.54,1.36,1.11,2.7,1.65,4.06c0.11,0.27,0.24,0.38,0.54,0.38c2.35-0.02,4.71-0.01,7.06-0.01 c0.15,0,0.3,0.03,0.46,0.04c0-0.06,0-0.11,0-0.17C91.5,25.65,91.01,24.57,90.55,23.48z M74.09,16.34c0.98-2.48,1.96-4.93,2.96-7.47 c0.99,2.53,1.95,4.98,2.93,7.47C78.01,16.34,76.08,16.34,74.09,16.34z"></path>
<path class="st0" d="M76.28,36.63c-1.25,1.77-2.47,3.49-3.68,5.2c-0.78-0.44-1.53-0.9-2.31-1.29c-1.65-0.81-3.39-1.32-5.24-1.28 c-0.54,0.01-1.09,0.15-1.6,0.33c-0.58,0.21-0.93,0.67-0.97,1.32c-0.04,0.66,0.21,1.22,0.8,1.46c1.08,0.43,2.18,0.82,3.3,1.1 c2.05,0.51,4.11,0.96,6.02,1.88c1.1,0.53,2.11,1.19,2.9,2.14c1.21,1.45,1.55,3.15,1.45,4.98c-0.22,3.98-2.67,6.63-6.39,7.54 c-5.95,1.45-11.47,0.36-16.37-3.43c-0.12-0.09-0.23-0.2-0.35-0.31c-0.02-0.02-0.03-0.04-0.07-0.09c1.35-1.61,2.7-3.23,4.08-4.88 c0.3,0.22,0.59,0.43,0.89,0.64c2.4,1.67,5.04,2.61,7.98,2.53c0.58-0.02,1.17-0.15,1.72-0.34c0.65-0.22,1.03-0.72,1.06-1.43 c0.03-0.69-0.31-1.2-0.9-1.45c-0.92-0.38-1.86-0.73-2.81-0.98c-1.96-0.51-3.94-0.91-5.82-1.7c-1.07-0.45-2.08-1-2.93-1.8 c-1.68-1.57-2.17-3.57-1.99-5.77c0.29-3.61,2.54-6.23,6.25-7.33c1.75-0.52,3.54-0.61,5.35-0.51c2.81,0.15,5.47,0.83,7.9,2.3 C75.14,35.82,75.7,36.23,76.28,36.63z"></path>
<path class="st0" d="M27.77,33.58c2.77,0,5.46,0.01,8.15-0.01c0.26,0,0.33,0.14,0.43,0.31c1.7,3.01,3.4,6.02,5.1,9.03 c0.05,0.09,0.1,0.17,0.17,0.29c0.09-0.14,0.17-0.25,0.24-0.37c1.7-2.98,3.39-5.96,5.09-8.94c0.08-0.14,0.27-0.31,0.41-0.31 c2.65-0.02,5.3-0.01,8-0.01c-0.1,0.17-0.18,0.31-0.26,0.45c-3.19,5.22-6.38,10.43-9.56,15.66c-0.16,0.26-0.25,0.6-0.25,0.9 c-0.02,3.03-0.01,6.06-0.01,9.09c0,0.14,0,0.28,0,0.44c-2.48,0-4.93,0-7.43,0c0-0.16,0-0.31,0-0.47c0-3,0.01-6-0.01-9.01 c0-0.28-0.08-0.59-0.22-0.82c-3.2-5.29-6.42-10.58-9.63-15.87C27.92,33.85,27.86,33.75,27.77,33.58z"></path>
<path class="st0" d="M23.41,34.3c1.14,1.14,2.31,2.31,3.51,3.5c-3.04,2.98-6.09,5.98-9.17,9.01c3.06,3.01,6.12,6.01,9.14,8.97 c-1.8,1.8-3.59,3.59-5.41,5.41c-2.95-3-5.95-6.06-8.97-9.14c-3.02,3.07-6.02,6.13-9,9.16c-1.19-1.19-2.36-2.36-3.5-3.51 C7.79,49.92,15.61,42.1,23.41,34.3z"></path>
<path class="st0" d="M60.46,26.41c0,0.13,0,0.29,0,0.45c-0.12,0.01-0.23,0-0.34,0c-1.32,0-2.65,0.01-3.97-0.01 c-0.18,0-0.42-0.1-0.52-0.24c-3.56-4.53-7.1-9.07-10.65-13.6c-0.05-0.06-0.1-0.13-0.12-0.17c1.57-1.57,3.12-3.12,4.7-4.7 c1.09,1.34,2.21,2.7,3.41,4.16c0-0.23,0-0.34,0-0.46c0-2.27,0.01-4.54-0.01-6.81c0-0.29,0.09-0.49,0.29-0.69 c1.44-1.41,2.86-2.84,4.29-4.26c0.96,0,1.93,0,2.89,0"></path>
<path class="st0" d="M41.96,15.64c0,3.77,0,7.48,0,11.2c-2.42,0-4.83,0-7.27,0c-0.01-0.13-0.03-0.25-0.03-0.37 c0-1.07-0.01-2.14,0.01-3.21c0-0.18,0.07-0.4,0.19-0.52C37.2,20.39,39.55,18.04,41.96,15.64z"></path>
<path class="st1" d="M4.52,15.21c0,3.93,0,7.8,0,11.7c-0.94,0-1.85,0-2.78,0c0-8.93,0-17.85,0-26.8c6.63,0,13.25,0,19.9,0 c0,0.94,0,1.86,0,2.82c-5.7,0-11.38,0-17.09,0c0,3.15,0,6.26,0,9.43c5.08,0,10.16,0,15.26,0c0,0.98,0,1.9,0,2.85 C14.72,15.21,9.65,15.21,4.52,15.21z"></path>
<path class="st1" d="M34.68,0.1c1.2,0,2.37-0.01,3.54,0.01c0.12,0,0.27,0.09,0.35,0.18c2.23,2.76,4.45,5.53,6.67,8.3 c0.02,0.02,0.02,0.05,0.01,0.04c-0.7,0.7-1.38,1.38-2.08,2.08c-1.86-2.31-3.74-4.65-5.62-6.98c-0.03,0-0.07,0.01-0.1,0.01 c0,0.16,0,0.31,0,0.47c0,3.96,0,7.92,0.01,11.88c0,0.31-0.08,0.53-0.3,0.74c-0.72,0.69-1.42,1.41-2.13,2.12 c-0.1,0.1-0.2,0.19-0.35,0.33C34.68,12.85,34.68,6.51,34.68,0.1z"></path>
<path class="st1" d="M26.69,27.28c0-9.11,0-18.11,0-27.15c0.92,0,1.84,0,2.81,0c0,0.16,0,0.31,0,0.46c0,7.82,0,15.65,0.01,23.47 c0,0.29-0.09,0.5-0.29,0.7C28.39,25.57,27.58,26.4,26.69,27.28z"></path>
<path class="st1" d="M11.25,42.66c-0.59,0.59-1.24,1.24-1.86,1.85c-3.04-3.04-6.11-6.11-9.14-9.14c0.61-0.61,1.25-1.25,1.86-1.86 C5.12,36.53,8.18,39.59,11.25,42.66z"></path>
</svg>
"""

FINAXYS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

.finaxys-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 4px 0 2px 0;
}
.finaxys-brand .finaxys-divider {
    width: 1px;
    height: 22px;
    background-color: #e2e8f0;
}
.finaxys-brand span {
    font-size: 20px;
    font-weight: 700;
    color: #0f172a;
}

section[data-testid="stSidebar"] {
    border-right: 1px solid #e2e8f0;
}

div.stButton > button[kind="primary"] {
    border-radius: 8px;
    font-weight: 500;
    transition: all 120ms ease;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #e44200;
    transform: translateY(-1px);
}

div[data-testid="stMetric"] {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
}

div[data-baseweb="notification"] {
    border-radius: 8px;
}

hr {
    border-color: #e2e8f0;
}
</style>
"""


def _inject_finaxys_style() -> None:
    st.markdown(FINAXYS_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_vdb() -> VectorStoreManager:
    return VectorStoreManager()


def _format_experiences(cv_data: dict) -> None:
    experiences = cv_data.get("experiences") or []
    if not experiences:
        st.caption("Aucune expérience détectée.")
        return
    for exp in experiences:
        titre = exp.get("titre") or "Poste non précisé"
        entreprise = exp.get("entreprise", "")
        debut = exp.get("date_debut", "")
        fin = "En cours" if exp.get("en_cours") else exp.get("date_fin", "")
        st.markdown(f"**{titre}** @ {entreprise} _({debut} → {fin})_")

        projets = exp.get("projets") or []
        if not projets and exp.get("projet"):
            projets = [{"nom": exp.get("projet"), "missions": exp.get("missions") or []}]

        for proj in projets:
            if proj.get("nom") or proj.get("description"):
                st.markdown(f"*Projet : {proj.get('nom') or ''}* — {proj.get('description') or ''}")
            for m in proj.get("missions") or []:
                st.markdown(f"- {m}")


def _render_cv_data(cv_data: dict) -> None:
    identite = cv_data.get("identite", {})
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"{identite.get('prenom', '')} {identite.get('nom', '')}".strip() or "Candidat")
        st.write(cv_data.get("titre_professionnel") or cv_data.get("profil") or "")
        st.caption(f"{identite.get('email', '')}  ·  {identite.get('telephone', '')}")
    with col2:
        metadata = cv_data.get("metadata", {})
        st.metric("Score de complétude", f"{float(metadata.get('score_completude', 0.0)):.0%}")

    tab_exp, tab_skills, tab_formation, tab_raw = st.tabs(
        ["Expériences", "Compétences", "Formations", "JSON brut"]
    )
    with tab_exp:
        _format_experiences(cv_data)
    with tab_skills:
        skills = cv_data.get("competences") or []
        noms = [s.get("nom") if isinstance(s, dict) else str(s) for s in skills]
        st.write(", ".join(noms) if noms else "Aucune compétence détectée.")
    with tab_formation:
        for f in cv_data.get("formations") or []:
            st.markdown(f"- **{f.get('diplome', '')}** — {f.get('etablissement', '')} ({f.get('annee', '')})")
    with tab_raw:
        st.json(cv_data)


def _render_recent_analyses() -> None:
    try:
        recents = list_recent_analyses(limit=10)
    except Exception:
        logger.warning("Impossible de charger l'historique des analyses (non bloquant)", exc_info=True)
        return
    if not recents:
        return
    with st.expander(f"🕓 Analyses récentes ({len(recents)})", expanded=False):
        st.dataframe(
            [
                {
                    "Candidat": f"{r['prenom']} {r['nom']}".strip() or r["source"],
                    "Poste": r["titre"],
                    "Fichier source": r["source"],
                    "Analysé le": r["created_at"].strftime("%d/%m/%Y %H:%M") if r["created_at"] else "",
                }
                for r in recents
            ],
            hide_index=True,
        )


def page_analyse() -> None:
    st.title("📄 Analyser un CV")
    st.caption("Upload un PDF ou DOCX — extraction, structuration Finaxys et indexation RAG.")

    _render_recent_analyses()

    col_opts1, col_opts2 = st.columns(2)
    generate_word = col_opts1.checkbox("Générer la fiche Word Finaxys", value=True)
    index_rag = col_opts2.checkbox("Indexer dans le moteur de recherche", value=True)

    uploaded = st.file_uploader("Fichier CV", type=["pdf", "docx"])
    if uploaded is None:
        return

    try:
        safe_name = sanitize_client_filename(uploaded.name)
        ensure_extension(safe_name, ALLOWED_EXTENSIONS)
    except ValueError as e:
        st.error(f"Fichier invalide : {e}")
        return

    if uploaded.size and uploaded.size > MAX_FILE_SIZE:
        st.error(f"Fichier trop volumineux ({uploaded.size // (1024*1024)} Mo). Maximum : {MAX_FILE_SIZE // (1024*1024)} Mo.")
        return

    if not st.button("Analyser", type="primary"):
        return

    temp_path = TEMP_DIR / safe_name
    temp_path.write_bytes(uploaded.getvalue())

    try:
        with st.spinner("Analyse en cours (30 à 90s selon la complexité du CV)..."):
            pipeline_start = time.perf_counter()
            cv_data = process_cv_pipeline(
                file_path=temp_path,
                output_dir=OUTPUT_DIR,
                generate_word_doc=generate_word,
                source="streamlit",
            )
            latency_ms = round((time.perf_counter() - pipeline_start) * 1000, 1)

            indexed = False
            if index_rag:
                try:
                    vdb = get_vdb()
                    cv_obj = CVDataModel(**cv_data)
                    vdb.add_cv(cv_obj, safe_name, index_source="streamlit")
                    indexed = True
                except Exception:
                    logger.exception("Indexation RAG échouée pour %s (non bloquant)", safe_name)

            try:
                shutil.copyfile(temp_path, CV_ORIGINALS_DIR / safe_name)
            except OSError:
                logger.warning("Copie du CV original échouée pour %s (non bloquant)", safe_name)

        observe_cv_analysis(endpoint="streamlit", indexed=indexed, status="success", latency_ms=latency_ms)
        try:
            trace_cv_analysis(filename=safe_name, cv_data=cv_data, latency_ms=latency_ms, indexed=indexed)
            langfuse_flush()
        except Exception:
            logger.warning("Trace Langfuse échouée (non bloquant)", exc_info=True)
        _push_metrics()

        st.success(f"Analyse terminée en {latency_ms/1000:.1f}s" + (" — indexé dans le RAG." if indexed else ""))
        _render_cv_data(cv_data)

        if generate_word:
            word_name = f"{temp_path.stem}_finaxys.docx"
            word_path = OUTPUT_DIR / word_name
            if word_path.exists():
                st.download_button(
                    "📥 Télécharger la fiche Word Finaxys",
                    data=word_path.read_bytes(),
                    file_name=f"{safe_name.rsplit('.', 1)[0]}_finaxys.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            else:
                st.info("Le fichier Word n'a pas pu être généré.")

    except ValueError as e:
        st.error(f"Fichier invalide : {e}")
    except Exception:
        logger.exception("Erreur pipeline pour %s", safe_name)
        observe_cv_analysis(
            endpoint="streamlit",
            indexed=False,
            status="error",
            latency_ms=round((time.perf_counter() - pipeline_start) * 1000, 1),
        )
        _push_metrics()
        st.error("Une erreur est survenue lors de l'analyse. Vérifie que le CV est lisible et réessaie.")
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _render_candidate_sources(candidates: list, key_prefix: str) -> None:
    candidates = [c for c in (candidates or []) if c.get("source")]
    if not candidates:
        return
    st.caption("Sources :")
    for i, cand in enumerate(candidates):
        source = cand["source"]
        files = _find_candidate_files(source)
        col_name, col_cv, col_fiche = st.columns([3, 1, 1])
        col_name.markdown(f"📄 {source}")
        if files.get("original"):
            col_cv.download_button(
                "CV original",
                data=files["original"].read_bytes(),
                file_name=files["original"].name,
                key=f"{key_prefix}_orig_{i}",
            )
        if files.get("fiche"):
            col_fiche.download_button(
                "Fiche Finaxys",
                data=files["fiche"].read_bytes(),
                file_name=files["fiche"].name,
                key=f"{key_prefix}_fiche_{i}",
            )


def _report_rag_search(*, query: str, endpoint: str, result: dict) -> None:
    observe_rag_search(
        endpoint=endpoint,
        mode="dense",
        status="success",
        latency_ms=result["latency_ms"],
        candidates_returned=result["candidate_count"],
    )
    try:
        trace_rag_search(
            query=query,
            job_title=query,
            candidates=result["candidates"],
            required_skills=[],
            context_relevance=result["avg_relevance"],
            latency_ms=result["latency_ms"],
        )
        langfuse_flush()
    except Exception:
        logger.warning("Trace Langfuse échouée (non bloquant)", exc_info=True)
    _push_metrics()


def page_recherche_libre() -> None:
    st.title("🔍 Recherche libre")
    st.caption("Recherche sémantique en langage naturel parmi les CV indexés.")

    query = st.text_input("Requête", placeholder="Ex : développeur Python senior avec expérience Kafka")
    n_results = st.slider("Nombre de résultats", 1, 10, 3)

    if not st.button("Rechercher", type="primary") or not query:
        return

    with st.spinner("Recherche en cours..."):
        vdb = get_vdb()
        result = run_chatbot_search(query, vdb=vdb, n_results=n_results)

    _report_rag_search(query=query, endpoint="streamlit:/recherche_libre", result=result)

    if result["status"] == "empty":
        st.warning("Aucun profil trouvé pour cette recherche.")
        return

    st.markdown(result["answer"])
    _render_candidate_sources(result["cited_candidates"], key_prefix="libre")


def page_chatbot() -> None:
    st.title("🤖 Chatbot RH")
    st.caption("Pose tes questions en langage naturel sur les profils indexés, en mode conversation.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if st.session_state.chat_messages and st.button("🗑️ Effacer la conversation"):
        st.session_state.chat_messages = []
        st.rerun()

    for i, msg in enumerate(st.session_state.chat_messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            _render_candidate_sources(msg.get("candidates", []), key_prefix=f"hist_{i}")

    query = st.chat_input("Ex : développeur Python senior avec expérience Kafka")
    if not query:
        return

    st.session_state.chat_messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if is_greeting_or_chitchat(query):
            answer = (
                "Bonjour ! Pose-moi une question sur un poste ou des compétences recherchées "
                "(ex : *\"développeur React senior avec expérience micro-frontends\"*) et je te "
                "proposerai les profils les plus pertinents parmi les CV indexés."
            )
            st.markdown(answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer, "candidates": []})
            return

        with st.spinner("Recherche en cours..."):
            vdb = get_vdb()
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_messages]
            result = run_conversational_search(history, vdb=vdb, n_results=5)

        _push_metrics()

        if result["status"] in ("empty", "needs_clarification"):
            st.markdown(result["answer"])
            st.session_state.chat_messages.append({"role": "assistant", "content": result["answer"], "candidates": []})
        else:
            st.markdown(result["answer"])
            _render_candidate_sources(result["candidates"], key_prefix=f"live_{len(st.session_state.chat_messages)}")
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": result["answer"], "candidates": result["candidates"]}
            )


def main() -> None:
    _inject_finaxys_style()
    st.sidebar.markdown(
        f'<div class="finaxys-brand">{FINAXYS_LOGO_SVG}<div class="finaxys-divider"></div><span>SmartCV</span></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Analyse et recherche de CV — Finaxys")
    st.sidebar.divider()
    page = st.sidebar.radio(
        "Navigation",
        ["Analyser un CV", "Recherche libre", "Chatbot"],
    )
    if page == "Analyser un CV":
        page_analyse()
    elif page == "Chatbot":
        page_chatbot()
    else:
        page_recherche_libre()


if __name__ == "__main__":
    main()
