# Architecture Visuelle - Projet Finaxys CV Analysis

Ce schéma représente l'architecture technique complète et le flux de données de l'assistant RH Finaxys.

```mermaid
graph TD
    %% Définition des styles et sous-graphes
    subgraph RH ["🌐 Interface Utilisateur (RH Finaxys)"]
        A[Telegram Bot]
    end

    subgraph HOST ["💻 Windows Host (Bureau de Badr)"]
        B[OpenClaw Gateway<br/>(Local Instance)]
        
        subgraph FOLDERS ["📁 Système de Fichiers"]
            I[(Input Folder<br/>CVs Bruts)]
            O[(Output Folder<br/>JSON & DOCX)]
        end
    end

    subgraph DOCKER ["🐳 Docker Container (cv-api)"]
        C[FastAPI Orchestrator]
        
        subgraph ENGINE ["⚙️ Moteurs de Traitement"]
            D[Docling / OCR<br/>(Extraction Texte)]
            E[ChromaDB<br/>(Vector Store / RAG)]
        end
    end

    subgraph AI ["🧠 Intelligence Artificielle (Cloud)"]
        F[OpenAI GPT-4o<br/>(LLM Reasoning)]
    end

    %% Flux de données (Ordre des étapes)
    A -- "1. Commande / Offre" --> B
    B -- "2. Appel API (localhost:8000)" --> C
    
    %% Flux Analyse de CV
    C -- "3a. Lecture Fichier" --> I
    C -- "4a. Parsing & OCR" --> D
    D -- "5a. Texte Brut" --> F
    F -- "6a. JSON Structuré" --> C
    C -- "7a. Génération Word (.docx)" --> O
    C -- "8a. Indexation Embeedings" --> E

    %% Flux Matching RAG
    C -- "3b. Requête Sémantique" --> E
    E -- "4b. Top Candidats" --> F
    F -- "5b. Justification Matching" --> C

    %% Retour final
    C -- "9. Réponse Structurée" --> B
    B -- "10. Résultat & Lien Word" --> A

    %% Styles Visuels
    style DOCKER fill:#f0f9ff,stroke:#007bff,stroke-width:2px
    style RH fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    style AI fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    style FOLDERS fill:#fff3e0,stroke:#ff9800,stroke-width:1px
```

## Légende du flux (Demo Workflow)

1. **Entrée (1-2) :** Interaction via Telegram. OpenClaw (Windows) transmet la requête à l'API (Docker).
2. **Analyse (3a-7a) :** Extraction du texte via **Docling**, structuration par **GPT-4o**, et génération du **Word** dans `output/`.
3. **RAG (8a-5b) :** Stockage des vecteurs dans **ChromaDB**. Pour une recherche, l'assistant compare les besoins RH avec les profils indexés et justifie son choix.
4. **Réponse (9-10) :** Retour du résultat final et du lien vers le livrable au RH.
