"""
Bot Telegram pour l'analyse de CV Finaxys.
Remplace OpenClaw — utilise directement les modules du projet.

Commandes :
  /start         → message de bienvenue
  /help          → aide
  /search <query> → recherche sémantique RAG
  Envoyer un fichier PDF/DOCX → analyse complète
"""

import logging
import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from telegram import Update, Document
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from service import process_cv_pipeline
from outils.rag_utils import VectorStoreManager
from outils.fs_utils import sanitize_client_filename, ensure_extension
from schemas.models import CVData as CVDataModel

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO"),
)
logger = logging.getLogger("bot")

TEMP_DIR = Path("temp_api")
OUTPUT_DIR = Path("output")
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


def _parse_allowed_chat_ids() -> set[int]:
    raw = os.getenv("TELEGRAM_ALLOWED_CHATS", "")
    ids: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            ids.add(int(token))
        except ValueError:
            logger.warning("TELEGRAM_ALLOWED_CHATS invalide: %s", token)
    return ids


ALLOWED_CHAT_IDS = _parse_allowed_chat_ids()


def _is_allowed(update: Update) -> bool:
    if not ALLOWED_CHAT_IDS:
        return True
    chat = update.effective_chat
    if not chat:
        return False
    return chat.id in ALLOWED_CHAT_IDS


async def _reply_unauthorized(update: Update) -> None:
    message = update.effective_message
    if message:
        await message.reply_text("â›” AccÃ¨s non autorisÃ©.")

# ---------------------------------------------------------------------------
# Formatage du résultat CV
# ---------------------------------------------------------------------------

def _format_cv_summary(cv: dict) -> str:
    """Formate le dictionnaire CVData en message Telegram lisible."""
    lines = []

    identite = cv.get("identite", {})
    prenom = identite.get("prenom", "")
    nom = identite.get("nom", "")
    titre = cv.get("titre_professionnel", "")
    email = identite.get("email", "")
    telephone = identite.get("telephone", "")
    localisation = identite.get("localisation", "")

    lines.append(f"✅ *Analyse terminée !*\n")
    lines.append(f"👤 *{prenom} {nom}*" + (f" — {titre}" if titre else ""))
    if email:
        lines.append(f"📧 {email}")
    if telephone:
        lines.append(f"📞 {telephone}")
    if localisation:
        lines.append(f"📍 {localisation}")

    # Compétences (groupées par catégorie)
    competences = cv.get("competences", [])
    if competences:
        lines.append("\n🔧 *Compétences :*")
        by_cat: dict = {}
        for c in competences:
            if isinstance(c, dict):
                cat = c.get("categorie", "Général")
                by_cat.setdefault(cat, []).append(c.get("nom", ""))
            else:
                by_cat.setdefault("Général", []).append(str(c))
        for cat, skills in list(by_cat.items())[:6]:  # max 6 catégories
            lines.append(f"  • *{cat}* : {', '.join(skills[:8])}")

    # Expériences
    experiences = cv.get("experiences", [])
    if experiences:
        lines.append("\n💼 *Expériences :*")
        for exp in experiences[:4]:  # max 4 expériences
            titre_exp = exp.get("titre", "")
            entreprise = exp.get("entreprise", "")
            debut = exp.get("date_debut", "")
            fin = exp.get("date_fin", "En cours" if exp.get("en_cours") else "")
            lines.append(f"  • *{titre_exp}* @ {entreprise} ({debut} → {fin})")
            missions = exp.get("missions", [])
            if missions:
                lines.append(f"    _{missions[0]}_")

    # Formations
    formations = cv.get("formations", [])
    if formations:
        lines.append("\n🎓 *Formations :*")
        for f in formations[:3]:
            diplome = f.get("diplome", "")
            etablissement = f.get("etablissement", "")
            annee = f.get("annee", "")
            lines.append(f"  • {diplome} — {etablissement}" + (f" ({annee})" if annee else ""))

    # Langues
    langues = cv.get("langues", [])
    if langues:
        lang_str = ", ".join(
            f"{l.get('langue', '')} ({l.get('niveau', '')})" for l in langues
        )
        lines.append(f"\n🌍 *Langues :* {lang_str}")

    # Certifications
    certifications = cv.get("certifications", [])
    if certifications:
        cert_str = ", ".join(c.get("nom", "") for c in certifications[:5])
        lines.append(f"\n🏅 *Certifications :* {cert_str}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        await _reply_unauthorized(update)
        return
    await update.message.reply_text(
        "👋 *Bonjour ! Je suis l'assistant RH Finaxys.*\n\n"
        "📄 Envoie-moi un CV en *PDF* ou *DOCX* et je l'analyse automatiquement.\n"
        "🔍 Utilise `/search <recherche>` pour trouver un profil dans la base.\n\n"
        "Tape /help pour plus d'infos.",
        parse_mode="Markdown",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        await _reply_unauthorized(update)
        return
    await update.message.reply_text(
        "*Commandes disponibles :*\n\n"
        "📄 *Envoyer un fichier* (PDF/DOCX) → analyse complète du CV\n"
        "🔍 `/search <query>` → recherche sémantique dans les CV indexés\n"
        "   _Exemple : /search développeur Java 5 ans d'expérience_\n\n"
        "ℹ️ `/start` → message de bienvenue\n"
        "ℹ️ `/help` → cette aide",
        parse_mode="Markdown",
    )


async def cmd_search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        await _reply_unauthorized(update)
        return
    query = " ".join(context.args) if context.args else ""
    if not query:
        await update.message.reply_text(
            "⚠️ Précise ta recherche.\n_Exemple : /search développeur Python senior_",
            parse_mode="Markdown",
        )
        return

    await update.message.reply_text("🔍 Recherche en cours...")
    try:
        vdb = VectorStoreManager()
        results = vdb.search(query, n_results=3)

        if not results or not results["documents"][0]:
            await update.message.reply_text("❌ Aucun profil trouvé pour cette recherche.")
            return

        from outils.llm_client import create_client, llm_call

        context_parts = []
        for i, doc in enumerate(results["documents"][0]):
            source = results["metadatas"][0][i].get("source", "Inconnu")
            context_parts.append(f"CANDIDAT {i+1} (Source: {source}):\n{doc}")

        system_prompt = (
            "Tu es un assistant expert en recrutement pour Finaxys. "
            "Analyse les profils trouvés et réponds de manière synthétique en français. "
            "Cite les sources."
        )
        user_msg = f"Recherche RH: {query}\n\nProfils trouvés:\n" + "\n---\n".join(context_parts)

        client, provider = create_client()
        answer = llm_call(client, provider, system_prompt, user_msg)

        sources = [m.get("source", "") for m in results["metadatas"][0]]
        sources_str = "\n".join(f"  • {s}" for s in sources if s)

        await update.message.reply_text(
            f"🔍 *Résultats pour :* _{query}_\n\n{answer}\n\n📁 *Sources :*\n{sources_str}",
            parse_mode="Markdown",
        )

    except Exception as e:
        logger.exception("Erreur recherche RAG")
        await update.message.reply_text("❌ Erreur lors de la recherche. Réessaie plus tard.")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reçoit un PDF ou DOCX, l'analyse et renvoie le résultat."""
    if not _is_allowed(update):
        await _reply_unauthorized(update)
        return
    doc: Document = update.message.document
    original_name = doc.file_name or "cv_upload"

    # --- Validation nom de fichier ---
    try:
        safe_name = sanitize_client_filename(original_name)
        ensure_extension(safe_name, ALLOWED_EXTENSIONS)
    except ValueError as e:
        await update.message.reply_text(f"⚠️ {e}\nSeuls les fichiers PDF et DOCX sont acceptés.")
        return

    # --- Validation taille ---
    if doc.file_size and doc.file_size > MAX_FILE_SIZE:
        await update.message.reply_text(
            f"⚠️ Fichier trop volumineux ({doc.file_size // (1024*1024)} MB). Maximum : 50 MB."
        )
        return

    temp_path = TEMP_DIR / f"{uuid4().hex}_{safe_name}"

    status_msg = await update.message.reply_text(
        f"⏳ Analyse de *{safe_name}* en cours...\n"
        "_Cette opération peut prendre 30 à 90 secondes selon la complexité du CV._",
        parse_mode="Markdown",
    )

    try:
        # Télécharger le fichier depuis Telegram
        tg_file = await doc.get_file()
        await tg_file.download_to_drive(str(temp_path))
        logger.info("Fichier reçu et téléchargé : %s", temp_path)

        # Lancer le pipeline d'analyse
        cv_data = process_cv_pipeline(
            file_path=temp_path,
            output_dir=OUTPUT_DIR,
            generate_word_doc=True,
        )

        # Indexation RAG automatique (silencieuse)
        try:
            vdb = VectorStoreManager()
            cv_obj = CVDataModel(**cv_data)
            vdb.add_cv(cv_obj, safe_name)
            logger.info("CV indexé dans le RAG : %s", safe_name)
        except Exception:
            logger.warning("Indexation RAG échouée pour %s (non bloquant)", safe_name)

        # Envoyer le résumé texte
        summary = _format_cv_summary(cv_data)
        await status_msg.edit_text(summary, parse_mode="Markdown")

        # Envoyer le fichier Word généré
        word_name = f"{temp_path.stem}_finaxys.docx"
        word_path = OUTPUT_DIR / word_name
        if word_path.exists():
            await update.message.reply_document(
                document=open(word_path, "rb"),
                filename=word_name,
                caption="📄 Fiche Finaxys générée",
            )
        else:
            await update.message.reply_text("ℹ️ Le fichier Word n'a pas pu être généré.")

    except ValueError as e:
        logger.warning("Fichier rejeté : %s", e)
        await status_msg.edit_text(f"⚠️ Fichier invalide : {e}")
    except Exception:
        logger.exception("Erreur pipeline pour %s", safe_name)
        await status_msg.edit_text(
            "❌ Une erreur est survenue lors de l'analyse.\n"
            "Vérifie que le CV est lisible et réessaie."
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Redirige les messages texte vers une recherche RAG."""
    if not _is_allowed(update):
        await _reply_unauthorized(update)
        return
    await update.message.reply_text(
        "💡 Tu peux rechercher un profil avec `/search <ta recherche>`\n"
        "Ou envoie-moi directement un fichier *PDF* ou *DOCX* à analyser.",
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN manquant dans les variables d'environnement.")

    logger.info("Démarrage du bot Telegram Finaxys...")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("search", cmd_search))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot en écoute (polling)...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
