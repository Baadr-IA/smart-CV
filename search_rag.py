import click
import logging
import sys
from pathlib import Path
from outils.rag_utils import VectorStoreManager
from outils.llm_client import create_client, llm_call

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("search_rag")

SYSTEM_PROMPT = """Tu es un assistant expert en recrutement pour la société Finaxys.
Ton rôle est d'analyser les profils de candidats extraits d'une base de données vectorielle et de répondre à la demande d'un recruteur (RH).

CONSIGNES :
1. Sois synthétique et professionnel.
2. Si aucun candidat ne correspond vraiment, explique pourquoi.
3. Pour chaque candidat cité, précise ses points forts par rapport à la recherche.
4. Cite la source (nom du fichier) pour chaque candidat.
5. Réponds en FRANÇAIS.
"""

@click.command()
@click.argument("query")
@click.option("--results", "-n", default=3, help="Nombre de candidats à récupérer")
def search(query, results):
    """
    Recherche RAG : Trouve les meilleurs candidats pour une requête donnée.
    Exemple : python search_rag.py "Développeur avec expérience Scrum et PHP"
    """
    click.echo(f"🔍 Recherche en cours pour : '{query}'...")
    
    try:
        # 1. Initialiser la DB Vectorielle
        vdb = VectorStoreManager()
        
        # 2. Récupérer les documents
        search_results = vdb.search(query, n_results=results)
        
        if not search_results or not search_results['documents'][0]:
            click.echo("❌ Aucun candidat trouvé dans la base de données.")
            return

        # 3. Préparer le contexte pour le LLM
        context_parts = []
        for i, doc in enumerate(search_results['documents'][0]):
            meta = search_results['metadatas'][0][i]
            source = meta.get('source', 'Inconnu')
            context_parts.append(f"--- CANDIDAT {i+1} (Source: {source}) ---\n{doc}\n")
        
        context_text = "\n".join(context_parts)
        
        # 4. Appeler le LLM pour la synthèse (Génération)
        client, provider = create_client()
        
        user_message = (
            f"DEMANDE DU RECRUTEUR : {query}\n\n"
            f"PROFILS TROUVÉS :\n{context_text}\n\n"
            "Analyse ces profils et fais une recommandation courte à la RH."
        )
        
        click.echo("🤖 Analyse des profils par l'IA...")
        response = llm_call(client, provider, SYSTEM_PROMPT, user_message)
        
        click.echo("\n" + "="*50)
        click.echo("📢 RÉPONSE POUR LA RH :")
        click.echo("="*50)
        click.echo(response)
        click.echo("="*50)

    except Exception as e:
        click.echo(f"❌ Erreur lors de la recherche : {e}", err=True)
        logger.exception(e)

if __name__ == "__main__":
    search()
