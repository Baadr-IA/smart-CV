# 🏗️ Architecture Systèmes & Flux de Données - Finaxys CV Intelligence

Ce schéma représente l'infrastructure complète **100% conteneurisée** et la séquence logique des opérations.

```mermaid
graph TB
    subgraph "🌐 External Services"
        TG_API((Telegram API))
        LLM_API((OpenAI API / GPT-4o))
    end

    subgraph "🐳 Docker Desktop Environment"
        
        subgraph "🤖 Interface : Container [openclaw-bot]"
            OC[OpenClaw Engine - Node.js]
            OC_CONF[(Vol: .openclaw config)]
        end

        subgraph "🧠 Cœur : Container [cv-finaxys-api]"
            API[FastAPI Server - Python 3.11]
            
            subgraph "🛠️ Processing Pipeline"
                PRSE[4. Parser : PDF/OCR]
                EXTR[5. Extractor : LLM]
                WORD[7. Generator : Word]
            end

            subgraph "📊 Memory & RAG"
                VDB[(6. PostgreSQL + pgvector : Vector Store)]
                EMB[Embeddings : Local]
            end
        end

        %% Docker Network
        OC -- "3. HTTP Request" --> API
        API -- "8. JSON + Path" --> OC
    end

    subgraph "📂 Local File System (Volumes)"
        INPUT[/./input - PDF/DOCX/]
        OUTPUT[/./output - JSON/DOCX/]
    end

    %% Data Flows Numbered
    User((Recruteur RH)) -- "1. Message/Fichier" --> TG_API
    TG_API -- "2. Webhook" --> OC
    OC -- "Prompting" --> LLM_API
    
    API -- "Extraction" --> EXTR
    EXTR <--> LLM_API
    
    API -- "Template Word" --> WORD
    WORD -- "Sauvegarde" --> OUTPUT
    INPUT -- "Source" --> API
    
    OC -- "9. Réponse/Lien" --> TG_API
    TG_API -- "10. Recommandation" --> User

    %% Styles
    style API fill:#ff9900,stroke:#333,stroke-width:2px,color:#fff
    style OC fill:#00cc66,stroke:#333,stroke-width:2px,color:#fff
    style VDB fill:#3399ff,stroke:#333,stroke-width:2px,color:#fff
```

## 📋 Séquence des Opérations (Flux numérotés)

1.  **[1-2] Envoi** : Le recruteur envoie un message ou un fichier sur Telegram. OpenClaw (Docker) le reçoit.
2.  **[3] Requête** : OpenClaw identifie le besoin (Recherche ou Analyse) et appelle l'API Finaxys.
3.  **[4] Parsing** : L'API extrait le texte du CV (utilise l'OCR si nécessaire).
4.  **[5] Extraction** : Le LLM transforme le texte en données structurées (Langues, Certifs, etc.).
5.  **[6] Indexation/Recherche** : 
    *   *Si Analyse* : Le CV est indexé dans PostgreSQL avec pgvector.
    *   *Si Recherche* : Le système cherche les meilleurs profils par proximité sémantique.
6.  **[7] Génération** : Le document Word officiel est créé à partir du template.
7.  **[8] Retour** : L'API renvoie le résultat JSON et le chemin du fichier Word à OpenClaw.
8.  **[9-10] Livraison** : OpenClaw rédige la réponse finale et l'envoie au recruteur sur Telegram.

---

## 🏗️ État de l'Infrastructure
*   **Conteneur `openclaw-bot`** : Isolé, gère la logique conversationnelle.
*   **Conteneur `cv-finaxys-api`** : Gère toute la logique métier et l'IA.
*   **Volumes** : Toutes les données (CV, Vecteurs, Config) sont persistées sur l'hôte.

*Astuce : Appuyez sur `Ctrl+Shift+V` dans VS Code pour voir le schéma interactif.*
