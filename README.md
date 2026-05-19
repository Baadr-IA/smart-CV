# 🚀 Smart-CV : Assistant RH Intelligent (RAG & Analyse)

**Smart-CV** est une solution complète d'automatisation du recrutement développée pour **Finaxys**. Elle combine l'extraction de données de CV par IA, la génération de documents au format officiel et la recherche sémantique (RAG) via un bot Telegram.

---

## 🌟 Fonctionnalités Clés

1.  **Analyse Intelligente (Extraction)** : Transforme des CV bruts (PDF/DOCX) en données JSON structurées.
2.  **Standardisation Finaxys** : Génère automatiquement un CV Word (.docx) conforme au template officiel.
3.  **Moteur de Recherche RAG** : Recherche sémantique dans la base de données des candidats (ex: "Cherche un dev Angular motivé").
4.  **Interface Telegram** : Pilotez tout le processus depuis votre téléphone via un bot Telegram natif Python.

---

## 🛠️ Architecture Technique

*   **Backend** : FastAPI (Python 3.11)
*   **IA/LLM** : OpenAI GPT-4o / endpoint OpenAI-compatible local
*   **Parsing** : PyPDF -> Tesseract -> Qwen (fallback)
*   **Base Vectorielle** : PostgreSQL + pgvector
*   **Interface Bot** : `python-telegram-bot`
*   **Conteneurisation** : Docker & Docker Compose

---

## 🚀 Installation et Démarrage

### 1. Prérequis
*   Docker & Docker Compose installés.
*   Une clé API OpenAI.
*   Un bot Telegram (créé via @BotFather).

### 2. Configuration
Copiez `.env_example` vers `.env`, puis renseignez au minimum :
```env
OPENAI_API_KEY=votre_cle_ici
TELEGRAM_BOT_TOKEN=votre_token_bot_ici
API_KEY_DISABLED=true
```

### 3. Lancer la stack locale
Le `docker-compose.yml` principal démarre maintenant :

1. `pgvector` pour le moteur RAG
2. `cv-api` pour l'API FastAPI
3. `cv-bot` pour le bot Telegram

```bash
docker compose up -d
```

Accès locaux :
- API : `http://localhost:8000`
- Swagger : `http://localhost:8000/docs`
- PostgreSQL/pgvector : `localhost:5432`

### 4. Lancer seulement l'API
```bash
docker compose up -d pgvector cv-api
```

### 5. Bot Telegram
Le projet utilise désormais `bot.py` directement. Une première version via OpenClaw a été testée, mais la surcouche consommait trop de tokens pour un gain fonctionnel limité, donc elle n'est plus la voie recommandée ni documentée comme runtime principal.

En Docker :
```bash
docker compose up -d cv-bot
```

En local :
```bash
python bot.py
```

---

## 📖 Utilisation

### Via Telegram
Une fois le bot connecté, vous pouvez lui parler directement :
*   **Analyse** : Envoyez un CV (PDF) et demandez "Analyse ce CV".
*   **Recherche** : Demandez "Trouve-moi un expert Python dans la base".

### Via CLI (Pipeline local)
Vous pouvez aussi utiliser le script `main.py` directement :
```bash
python main.py process input/mon_cv.pdf --output-dir output/
```

### Via API
Deux modes sont disponibles :

1. `POST /analyze` pour une analyse synchrone avec upload direct
2. `POST /trigger` puis `GET /result` pour une analyse asynchrone sur un fichier déjà présent côté serveur

### Évaluer le RAG
Un benchmark synthétique déterministe est disponible pour mesurer :

1. `Context Relevance`
2. `Groundedness` via **LettuceDetect**
3. `Answer Relevance`

Installation de la dépendance d'évaluation :
```bash
pip install -r requirements.txt
```

Lancer le benchmark synthétique isolé :
```bash
python evaluation/eval_rag.py --dataset synthetic --results 5 --json-output evaluation/results_synthetic.json
```

Ce mode :
- génère un corpus synthétique reproductible,
- l'indexe dans une collection pgvector séparée,
- produit un rapport HTML et un JSON de résultats,
- ajoute aussi `Retrieval Hit Rate`, `Precision@K` et `MRR` pour comparer facilement avant/après une migration de moteur vectoriel.

Pour évaluer la collection réelle déjà indexée :
```bash
python evaluation/eval_rag.py --dataset live --collection-name cv_collection
```

### Modèle d'embedding RAG
Le moteur RAG utilise désormais par défaut **`BAAI/bge-m3`**.

Configuration :
```env
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_BATCH_SIZE=16
RERANKER_ENABLED=true
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_BATCH_SIZE=8
RERANKER_CANDIDATE_POOL=20
PGVECTOR_DSN=postgresql://postgres:postgres@localhost:5432/smartcv
PGVECTOR_TABLE_PREFIX=rag_vector
```

Après tout changement de modèle d'embedding, il faut **réindexer entièrement** la base vectorielle existante, car les dimensions et l'espace vectoriel changent.

### Reranker RAG
Le retrieval dense pgvector peut désormais être suivi d'un reranking cross-encoder optionnel avec **`BAAI/bge-reranker-v2-m3`**.

- `RERANKER_ENABLED=true` active le reranking
- `RERANKER_CANDIDATE_POOL` définit combien de candidats denses sont récupérés avant reranking
- les endpoints `/search/job` renvoient alors aussi `rerankScore` et conservent `denseRelevanceScore` pour comparaison

### Recherche hybride RH
L'endpoint `/search/job` utilise désormais un pipeline plus métier :

1. **Préfiltrage SQL** à partir du titre, des alias du poste et des skills principaux
2. **Dense retrieval** avec pgvector + `bge-m3`
3. **Lexical retrieval** PostgreSQL sur le texte des CV
4. **Fusion hybride** par Reciprocal Rank Fusion

Configuration optionnelle :
```env
HYBRID_DENSE_CANDIDATE_POOL=30
HYBRID_LEXICAL_CANDIDATE_POOL=30
SQL_PREFILTER_CANDIDATE_POOL=120
HYBRID_RRF_K=60
```

Exemple de reconstruction simple :
```bash
psql postgresql://postgres:postgres@localhost:5432/smartcv -c "DELETE FROM rag_vector_collections WHERE name = 'cv_collection';"
python main.py process-dir input --index --skip-word
```

### OCR PDF intelligent
Le parseur PDF suit désormais une cascade :

1. `pypdf` pour les PDF natifs
2. `pytesseract` si le texte natif est insuffisant
3. OCR Vision OpenAI-compatible (ex: Qwen local) en dernier recours

L'arrêt anticipé est piloté par un critère composite :
- nombre de caractères / mots,
- densité par page,
- `alpha_ratio`,
- présence de champs-clés (email, téléphone, titres de section).

Exemple de configuration locale pour Qwen :
```env
PDF_PARSE_MODE=smart
OCR_PROVIDER=local_openai
LOCAL_LLM_BASE_URL=http://localhost:8000/v1
LOCAL_LLM_API_KEY=dummy
LOCAL_LLM_MODEL=qwen-cv
OCR_ENABLE_LLM_FALLBACK=true
```

### Observabilité
Le projet expose désormais deux niveaux d'observabilité complémentaires :

1. **Langfuse** pour les traces LLM/RAG (`cv_analysis`, `rag_job_search`, `rag_evaluation`)
2. **Prometheus + Grafana** pour les métriques techniques et produit

L'API expose un endpoint Prometheus sur :
```text
GET /metrics
```

Principales métriques :
- `smartcv_http_requests_total`
- `smartcv_http_request_duration_seconds`
- `smartcv_cv_analysis_total`
- `smartcv_cv_analysis_duration_seconds`
- `smartcv_rag_search_total`
- `smartcv_rag_search_duration_seconds`
- `smartcv_rag_candidates_returned`
- `smartcv_active_jobs`
- `smartcv_langfuse_enabled`

Pour lancer l'observabilité locale avec l'API :
```bash
docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d
```

Accès :
- Langfuse : `http://localhost:3000`
- Grafana : `http://localhost:3001`
- Prometheus : `http://localhost:9090`

Variables utiles :
```env
LANGFUSE_SECRET_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_HOST=http://localhost:3000
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin
```

## 🧪 Qualité et développement

Installation locale :
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

Commandes utiles :
```bash
python -m ruff check api.py bot.py main.py service.py outils schemas tests
python -m pytest tests -v
```

---

## 📁 Structure du Projet
*   `api.py` : Serveur FastAPI.
*   `service.py` : Logique du pipeline d'analyse.
*   `outils/` : Modules de parsing, extraction, RAG et validation.
*   `bot.py` : Bot Telegram natif Python.
*   `templates/` : Modèles Word officiels.
*   `input/` & `output/` : Dossiers de données (ignorés par Git).

---

## 🔒 Sécurité
Les fichiers sensibles (`.env`, `input/`, `output/`) sont exclus du dépôt via `.gitignore`. Ne partagez jamais vos clés API.

---
**Développé par Badr LAMBARKI EL ALLIOUI**

## Security (Production)
- Set `API_KEY` and send header `X-API-Key: <value>` on all API calls.
- For local dev, you can set `API_KEY_DISABLED=true`.
- Configure CORS explicitly with `CORS_ALLOW_ORIGINS=https://app.example.com` (comma-separated).
- Use `RATE_LIMIT_PER_MIN`, `MAX_CONCURRENT_JOBS`, and `MAX_UPLOAD_MB` to protect the API.
- Put the API behind HTTPS (ALB + ACM in AWS).

## Storage (S3 + PostgreSQL/pgvector)
When running on AWS with S3 for `input/` + `output/` and PostgreSQL/pgvector for retrieval:
```env
STORAGE_MODE=s3
S3_BUCKET=your-bucket
S3_INPUT_PREFIX=input/
S3_OUTPUT_PREFIX=output/
S3_REGION=eu-west-3
PGVECTOR_DSN=postgresql://user:password@postgres-host:5432/smartcv
```

## CI/CD (ECR)
Workflow: `.github/workflows/deploy-ecr.yml`
Required GitHub secrets:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `ECR_REGISTRY` (e.g. 123456789012.dkr.ecr.eu-west-3.amazonaws.com)
- `ECR_REPOSITORY_API` (e.g. cv-finaxys-api)
- `ECR_REPOSITORY_BOT` (e.g. cv-finaxys-bot)

## Telegram Bot Access
Whitelist chat IDs with:
```env
TELEGRAM_ALLOWED_CHATS=123456789,987654321
```
If empty, the bot accepts all chats.
