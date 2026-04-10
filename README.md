# 🚀 Smart-CV : Assistant RH Intelligent (RAG & Analyse)

**Smart-CV** est une solution complète d'automatisation du recrutement développée pour **Finaxys**. Elle combine l'extraction de données de CV par IA, la génération de documents au format officiel et la recherche sémantique (RAG) via un bot Telegram.

---

## 🌟 Fonctionnalités Clés

1.  **Analyse Intelligente (Extraction)** : Transforme des CV bruts (PDF/DOCX) en données JSON structurées.
2.  **Standardisation Finaxys** : Génère automatiquement un CV Word (.docx) conforme au template officiel.
3.  **Moteur de Recherche RAG** : Recherche sémantique dans la base de données des candidats (ex: "Cherche un dev Angular motivé").
4.  **Interface Telegram** : Pilotez tout le processus depuis votre téléphone via un bot OpenClaw.

---

## 🛠️ Architecture Technique

*   **Backend** : FastAPI (Python 3.11)
*   **IA/LLM** : OpenAI GPT-4o
*   **Parsing** : Docling & PyPDF
*   **Base Vectorielle** : ChromaDB (RAG local)
*   **Interface Bot** : OpenClaw (Gateway & Telegram)
*   **Conteneurisation** : Docker & Docker Compose

---

## 🚀 Installation et Démarrage

### 1. Prérequis
*   Docker & Docker Compose installés.
*   Une clé API OpenAI.
*   Un bot Telegram (créé via @BotFather).

### 2. Configuration
Créez un fichier `.env` à la racine du projet :
```env
OPENAI_API_KEY=votre_cle_ici
TELEGRAM_BOT_TOKEN=votre_token_bot_ici
```

### 3. Lancer l'API (Le Cerveau)
L'API gère toute l'intelligence et la base vectorielle.
```bash
docker-compose up -d cv-api
```
L'API sera accessible sur `http://localhost:8000`. Vous pouvez voir la documentation Swagger sur `http://localhost:8000/docs`.

### 4. Configurer le Bot Telegram (OpenClaw)
Vous pouvez lancer le bot via Docker ou localement sur Windows.

**Option A : Via Docker (Recommandé)**
```bash
docker-compose up -d openclaw
```

**Option B : En local (Windows)**
1.  Installez OpenClaw : `npm install -g openclaw`
2.  Initialisez la configuration dans `~/.openclaw`.
3.  Copiez les outils du dossier `tools/` de ce projet vers `~/.openclaw/tools/`.
4.  Lancez le gateway : `openclaw gateway`.

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

---

## 📁 Structure du Projet
*   `api.py` : Serveur FastAPI.
*   `service.py` : Logique du pipeline d'analyse.
*   `outils/` : Modules de parsing, extraction, RAG et validation.
*   `tools/` : Définitions des outils pour le bot Telegram.
*   `templates/` : Modèles Word officiels.
*   `input/` & `output/` : Dossiers de données (ignorés par Git).

---

## 🔒 Sécurité
Les fichiers sensibles (`.env`, `vector_db/`, `input/`, `output/`) sont exclus du dépôt via `.gitignore`. Ne partagez jamais vos clés API.

---
**Développé par Badr LAMBARKI EL ALLIOUI**

## Security (Production)
- Set `API_KEY` and send header `X-API-Key: <value>` on all API calls.
- For local dev, you can set `API_KEY_DISABLED=true`.
- Configure CORS explicitly with `CORS_ALLOW_ORIGINS=https://app.example.com` (comma-separated).
- Use `RATE_LIMIT_PER_MIN`, `MAX_CONCURRENT_JOBS`, and `MAX_UPLOAD_MB` to protect the API.
- Put the API behind HTTPS (ALB + ACM in AWS).

## Storage (S3 + EFS)
When running on AWS with S3 for `input/` + `output/` and EFS for ChromaDB:
```env
STORAGE_MODE=s3
S3_BUCKET=your-bucket
S3_INPUT_PREFIX=input/
S3_OUTPUT_PREFIX=output/
S3_REGION=eu-west-3
CHROMA_DB_PATH=/mnt/efs/vector_db
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
