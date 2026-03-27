# 🏗️ Architecture Finale : Pipeline Analyse de CV & RAG Finaxys

Ce document décrit l'écosystème complet mis en place pour l'extraction, la validation et la recherche intelligente de CV.

---

## 1. Vue d'Ensemble du Workflow
Le système transforme un flux de documents non structurés (PDF/Images) en une base de connaissances interrogeable par une IA via Telegram.

1.  **Ingestion** : Dépôt d'un CV (PDF natif ou scanné).
2.  **Parsing & OCR** : Extraction du texte brut via `PyMuPDF` ou `Tesseract`.
3.  **Extraction LLM** : Transformation du texte en JSON structuré (Pydantic) avec focus sur :
    *   **Certifications** (génériques et spécifiques).
    *   **Niveaux de langues** (interprétation des barres de progression ●●●○○).
4.  **Validation** : Double contrôle structurel (Pydantic) et sémantique (LLM) pour éviter les hallucinations.
5.  **Indexation RAG** : Transformation du JSON en texte riche et stockage vectoriel dans `ChromaDB`.
6.  **Interface RH** : Recherche sémantique en langage naturel via **Telegram (OpenClaw)**.

---

## 2. Composants Techniques

### 🐳 Backend (Dockerisé)
L'API centrale est isolée dans un conteneur Docker pour garantir la stabilité des dépendances (Tesseract, Python 3.11, etc.).
*   **Framework** : FastAPI (Python).
*   **Vector DB** : ChromaDB (Stockage local persistant).
*   **Embeddings** : `all-MiniLM-L6-v2` (Sentence-Transformers locaux).
*   **Orchestration** : `docker-compose.yml`.

### 🤖 Intelligence Artificielle
*   **Modèles** : OpenAI (GPT-4o) ou Gemini (via interface unifiée).
*   **Prompts (Skills)** : Dossier `instructions/` contenant les rôles d'expert (Extractor, Validator, Normalizer).

### 📱 Interface Telegram (OpenClaw)
*   **Moteur** : OpenClaw (Node.js).
*   **Liaison** : Script de pont `recherche_cv.py` permettant au bot d'exécuter des requêtes sémantiques sur l'API Dockerisée.

---

## 3. Commandes de Pilotage

### Lancer l'infrastructure (Docker) :
```bash
docker-compose up -d --build
```

### Indexer un dossier complet de CV :
```bash
python main.py process-dir ./input --index --skip-word
```

### Tester la recherche RAG (sans Telegram) :
```bash
python search_rag.py "Cherche un expert PHP certifié Scrum Master"
```

---

## 4. Structure des Données (JSON Finaxys)
Le schéma `CVData` (`schemas/models.py`) garantit l'intégrité :
*   `identite` : Nom, Prénom, LinkedIn, etc.
*   `competences` : Liste normalisée par catégories.
*   `experiences` : Missions détaillées, technos, résultats.
*   `certifications` : Nouveau module pour Scrum, Cloud, etc.
*   `langues` : Niveaux traduits des barres graphiques.

---
*Dernière mise à jour : Mars 2026 - Finaxys Academy*
