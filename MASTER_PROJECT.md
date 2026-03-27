# 🚀 Master Project : Finaxys CV Intelligence & RAG

Ce document est le guide suprême de l'architecture, du fonctionnement et de la maintenance de l'écosystème.

---

## 1. 🏗️ Architecture Systèmes & Flux Numérotés (100% Docker)

Cette architecture sépare l'interface conversationnelle (Telegram) du moteur de traitement (Python/IA).

```mermaid
graph TB
    subgraph "🌐 Services Externes"
        TG_API((Telegram API))
        LLM_API((OpenAI API / GPT-4o))
    end

    subgraph "🐳 Docker Desktop (Environnement Isolé)"
        
        subgraph "🤖 Interface : [openclaw-bot]"
            OC[OpenClaw Engine - Node.js]
            OC_CONF[(Vol: .openclaw config)]
        end

        subgraph "🧠 Cœur : [cv-finaxys-api]"
            API[FastAPI Server - Python 3.11]
            
            subgraph "🛠️ Pipeline de Traitement"
                PRSE[4. Parser : PDF/OCR]
                EXTR[5. Extractor : LLM]
                WORD[7. Generator : Word]
            end

            subgraph "📊 Mémoire & RAG"
                VDB[(6. ChromaDB : Vector Store)]
                EMB[Embeddings : Local]
            end
        end

        %% Docker Network
        OC -- "3. Requête HTTP" --> API
        API -- "8. JSON + Chemin Fichier" --> OC
    end

    subgraph "📂 Stockage Local (Volumes Persistants)"
        INPUT[/./input - Fichiers sources/]
        OUTPUT[/./output - Résultats/]
        VDB_DATA[/./vector_db - Vecteurs Binaires/]
    end

    %% Séquence Numérotée
    User((Recruteur RH)) -- "1. Envoi Message/CV" --> TG_API
    TG_API -- "2. Webhook" --> OC
    OC -- "Analyse sémantique" --> LLM_API
    
    API -- "Extraction Sémantique" --> EXTR
    EXTR <--> LLM_API
    
    API -- "Mapping Template" --> WORD
    WORD -- "Sauvegarde" --> OUTPUT
    INPUT -- "Lecture Source" --> API
    
    OC -- "9. Synthèse Finale" --> TG_API
    TG_API -- "10. Recommandation Argumentée" --> User

    %% Styles
    style API fill:#ff9900,stroke:#333,stroke-width:2px,color:#fff
    style OC fill:#00cc66,stroke:#333,stroke-width:2px,color:#fff
    style VDB fill:#3399ff,stroke:#333,stroke-width:2px,color:#fff
```

---

## 2. 📝 Séquence de Fonctionnement (Le Cheminement)

1.  **[1-2] Ingestion** : Le recruteur communique avec le bot Telegram.
2.  **[3] Orchestration** : OpenClaw (Docker) identifie l'intention et sollicite l'API locale.
3.  **[4] Parsing** : L'API extrait le texte (utilise **Tesseract OCR** si le PDF est une image).
4.  **[5] Extraction Sémantique** : GPT-4o transforme le texte en JSON (Focus : **Certifications** et **Barres de langues**).
5.  **[6] Indexation RAG** : Le CV est transformé en vecteurs et stocké dans **ChromaDB**.
6.  **[7] Génération DOCX** : Un document Word officiel est créé via le template `finaxys_template.docx`.
7.  **[8] Feedback API** : L'API renvoie les données et le chemin du fichier Word à OpenClaw.
8.  **[9-10] Recommandation** : Le bot rédige une réponse personnalisée ("Thomas Dubois est un bon profil car...") sur Telegram.

---

## 3. 🛠️ Technologies Clés

| Composant | Technologie | Rôle |
| :--- | :--- | :--- |
| **API** | FastAPI (Python) | Orchestration des services et endpoints. |
| **LLM** | OpenAI (GPT-4o) | Intelligence d'extraction et de synthèse. |
| **OCR** | Tesseract | Lecture des CV scannés. |
| **Vector DB** | ChromaDB | Mémoire sémantique du projet. |
| **Embeddings** | all-MiniLM-L6-v2 | Recherche sémantique locale et gratuite ($0). |
| **Bot** | OpenClaw | Agent conversationnel et interface Telegram. |

---

## 4. 🚀 Commandes Vitales

### Démarrage Rapide (Tout-en-un) :
```bash
docker-compose up -d --build
```

### Indexer un nouveau dossier de CV :
```bash
python main.py process-dir ./input --index --skip-word
```

### Faire une recherche RAG locale :
```bash
python search_rag.py "Cherche un dev PHP avec Scrum"
```

---

## 5. ⚠️ Notes de Maintenance
*   **Données Persistantes** : Ne supprimez pas le dossier `vector_db` ou le volume `.openclaw` si vous voulez garder votre base de CV et votre session Telegram.
*   **Coûts LLM** : Surveillez votre consommation sur OpenAI. L'utilisation de GPT-4o coûte environ **2 centimes par recherche**.
*   **Template Word** : Le modèle se trouve dans `templates/`. Modifiez-le pour changer le look du document final généré.

---