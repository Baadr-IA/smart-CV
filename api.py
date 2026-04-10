import os
import logging
import math
import threading
import time
import zipfile
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

from schemas.models import CVData
from service import process_cv_pipeline
from outils.rag_utils import VectorStoreManager
from outils.llm_client import create_client, llm_call
from outils.fs_utils import (
    ensure_extension,
    ensure_path_within_directory,
    resolve_path_in_directory,
    sanitize_client_filename,
)
from outils.storage import (
    download_json,
    download_to_path,
    s3_bucket,
    s3_enabled,
    s3_input_prefix,
    s3_key,
    s3_output_prefix,
    upload_file,
    upload_json,
)

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CV Finaxys API",
    description="API d'analyse de CV (PDF/DOCX) pour transformation en JSON Finaxys et recherche RAG.",
    version="1.2.0"
)

# CORS (prefer explicit origins via env)
_cors_origins = _parse_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials(_cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dossiers temporaires
TEMP_DIR = Path("temp_api")
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
INPUT_DIR = Path("input")
INPUT_DIR.mkdir(exist_ok=True)

# SystÃ¨mes de prompts pour le RAG
RAG_SYSTEM_PROMPT = """Tu es un assistant expert en recrutement pour la sociÃ©tÃ© Finaxys.
Ton rÃ´le est d'analyser les profils de candidats extraits d'une base de donnÃ©es vectorielle et de rÃ©pondre Ã  la demande d'un recruteur (RH).
RÃ©ponds de maniÃ¨re synthÃ©tique et cite tes sources."""

ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".docx"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

API_KEY = os.getenv("API_KEY")
API_KEY_DISABLED = os.getenv("API_KEY_DISABLED", "false").lower() == "true"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))

MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
if MAX_CONCURRENT_JOBS < 0:
    MAX_CONCURRENT_JOBS = 0
JOB_SEMAPHORE = threading.BoundedSemaphore(MAX_CONCURRENT_JOBS or 1)


def _parse_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _cors_allow_credentials(origins: list[str]) -> bool:
    want = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true"
    if "*" in origins:
        return False
    return want


def _normalize_upload_filename(filename: str) -> str:
    try:
        safe_name = sanitize_client_filename(filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        ensure_extension(safe_name, ALLOWED_UPLOAD_EXTENSIONS)
    except ValueError:
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF et DOCX sont acceptÃ©s.")

    return safe_name


def _prepare_temp_upload_path(filename: str) -> Path:
    safe_name = _normalize_upload_filename(filename)
    temp_candidate = TEMP_DIR / f"{uuid4().hex}_{safe_name}"
    try:
        return ensure_path_within_directory(TEMP_DIR, temp_candidate, strict=False)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="Chemin temporaire invalide.")


def _require_s3_bucket() -> str:
    bucket = s3_bucket()
    if not bucket:
        raise HTTPException(
            status_code=500,
            detail="S3_BUCKET est requis quand STORAGE_MODE=s3."
        )
    return bucket


def _cleanup_temp_file(path: Path) -> None:
    try:
        if path and path.exists():
            temp_root = TEMP_DIR.resolve()
            if path.resolve().is_relative_to(temp_root):
                path.unlink()
    except Exception:
        logger.warning("Impossible de supprimer le fichier temporaire: %s", path)


def _resolve_path_inside(directory: Path, filename: str) -> Path:
    if not filename or not filename.strip():
        raise HTTPException(status_code=422, detail="Le nom de fichier est requis.")
    try:
        return resolve_path_in_directory(directory, filename, strict=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _find_existing_cv(filename: str) -> Path:
    target = _resolve_path_inside(INPUT_DIR, filename)
    if target.exists():
        return target

    target = _resolve_path_inside(TEMP_DIR, filename)
    if target.exists():
        return target

    raise HTTPException(
        status_code=404,
        detail=f"Fichier '{filename}' introuvable dans '{INPUT_DIR}' ou '{TEMP_DIR}'."
    )


def _fetch_input_cv(filename: str) -> Path:
    if s3_enabled():
        safe_name = _normalize_upload_filename(filename)
        bucket = _require_s3_bucket()
        key = s3_key(s3_input_prefix(), safe_name)
        temp_path = _prepare_temp_upload_path(safe_name)
        try:
            download_to_path(bucket, key, temp_path)
        except Exception:
            raise HTTPException(
                status_code=404,
                detail=f"Objet S3 introuvable: s3://{bucket}/{key}"
            )
        return temp_path
    return _find_existing_cv(filename)


def _upload_outputs_to_s3(cv_data: dict, stem: str, generate_word: bool) -> None:
    bucket = _require_s3_bucket()
    output_prefix = s3_output_prefix()

    result_name = f"{stem}_result.json"
    result_key = s3_key(output_prefix, result_name)
    upload_json(bucket, result_key, cv_data)
    cv_data.setdefault("metadata", {})
    cv_data["metadata"]["s3_result_key"] = result_key
    cv_data["metadata"]["s3_result_uri"] = f"s3://{bucket}/{result_key}"

    if generate_word:
        word_name = f"{stem}_finaxys.docx"
        word_path = OUTPUT_DIR / word_name
        if word_path.exists():
            word_key = s3_key(output_prefix, word_name)
            upload_file(
                bucket,
                word_key,
                word_path,
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            cv_data["metadata"]["s3_word_key"] = word_key
            cv_data["metadata"]["fichier_word"] = f"s3://{bucket}/{word_key}"


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


class _RateLimiter:
    def __init__(self, limit_per_window: int, window_seconds: int = 60) -> None:
        self.limit = max(1, limit_per_window)
        self.window = window_seconds
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[float, int]] = {}

    def check(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            window_start, count = self._buckets.get(key, (now, 0))
            if now - window_start >= self.window:
                window_start, count = now, 0
            if count >= self.limit:
                retry_after = max(1, int(self.window - (now - window_start)))
                return False, retry_after
            self._buckets[key] = (window_start, count + 1)
            return True, 0


_rate_limiter = _RateLimiter(RATE_LIMIT_PER_MIN)


def require_api_key(api_key: Optional[str] = Depends(api_key_header)) -> None:
    if API_KEY_DISABLED:
        return
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY manquant.")
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Non autorise.")


def enforce_rate_limit(request: Request) -> None:
    if not RATE_LIMIT_ENABLED:
        return
    ok, retry_after = _rate_limiter.check(_client_ip(request))
    if not ok:
        raise HTTPException(
            status_code=429,
            detail="Trop de requetes, reessaie plus tard.",
            headers={"Retry-After": str(retry_after)},
        )


def _acquire_job_slot() -> None:
    if MAX_CONCURRENT_JOBS == 0:
        return
    if not JOB_SEMAPHORE.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Serveur occupe, reessaie plus tard.")


def _release_job_slot() -> None:
    if MAX_CONCURRENT_JOBS == 0:
        return
    try:
        JOB_SEMAPHORE.release()
    except ValueError:
        pass


def _enforce_upload_size(request: Request) -> None:
    content_length = request.headers.get("content-length")
    if not content_length:
        return
    try:
        if int(content_length) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Fichier trop volumineux.")
    except ValueError:
        return


def _save_upload_file(upload_file: UploadFile, dest: Path) -> int:
    size = 0
    with open(dest, "wb") as buffer:
        while True:
            chunk = upload_file.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Fichier trop volumineux.")
            buffer.write(chunk)
    return size


def _is_pdf(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except Exception:
        return False


def _is_docx(path: Path) -> bool:
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            if "[Content_Types].xml" not in names:
                return False
            return any(name.startswith("word/") for name in names)
    except Exception:
        return False


def _validate_upload_content(path: Path, filename: str) -> None:
    try:
        if path.stat().st_size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Fichier trop volumineux.")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    ext = Path(filename).suffix.lower()
    if ext == ".pdf" and not _is_pdf(path):
        raise HTTPException(status_code=415, detail="Fichier invalide (PDF attendu).")
    if ext == ".docx" and not _is_docx(path):
        raise HTTPException(status_code=415, detail="Fichier invalide (DOCX attendu).")


def _internal_error(context: str, exc: Exception) -> None:
    error_id = uuid4().hex[:8]
    logger.exception("%s (error_id=%s): %s", context, error_id, exc)
    raise HTTPException(status_code=500, detail=f"Erreur interne. Code={error_id}")
@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API CV Finaxys. AccÃ©dez Ã  /docs pour la documentation."}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post(
    "/analyze",
    response_model=CVData,
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)
async def analyze_cv(
    request: Request,
    file: UploadFile = File(...),
    generate_word: bool = True,
    index: bool = True
):
    """
    Analyse un CV (PDF/DOCX), extrait les donnÃ©es et optionnellement l'indexe dans le RAG.
    """
    logger.info("RÃ©ception d'un fichier : %s", file.filename)

    _enforce_upload_size(request)
    _acquire_job_slot()
    temp_path = _prepare_temp_upload_path(file.filename)
    try:
        _save_upload_file(file, temp_path)
        _validate_upload_content(temp_path, file.filename)
        if s3_enabled():
            bucket = _require_s3_bucket()
            original_name = temp_path.name.split("_", 1)[-1]
            input_key = s3_key(s3_input_prefix(), original_name)
            upload_file(bucket, input_key, temp_path)
        
        # ExÃ©cution du pipeline
        cv_data = process_cv_pipeline(
            file_path=temp_path,
            output_dir=OUTPUT_DIR,
            generate_word_doc=generate_word
        )
        
        # Ajouter le chemin du fichier Word dans les mÃ©tadonnÃ©es
        if generate_word:
            word_name = f"{temp_path.stem}_finaxys.docx"
            cv_data["metadata"]["fichier_word"] = word_name

        if s3_enabled():
            _upload_outputs_to_s3(cv_data, temp_path.stem, generate_word)
        
        # Indexation RAG automatique
        if index:
            try:
                from schemas.models import CVData as CVDataModel
                vdb = VectorStoreManager()
                cv_obj = CVDataModel(**cv_data) if isinstance(cv_data, dict) else cv_data
                vdb.add_cv(cv_obj, file.filename)
                logger.info("CV indexÃ© dans le RAG : %s", file.filename)
            except Exception as e:
                logger.error("Erreur indexation RAG : %s", e)

        return cv_data

    except HTTPException:
        raise
    except Exception as e:
        _internal_error("Erreur lors du traitement du CV", e)
    
    finally:
        # Nettoyage fichier temporaire
        _cleanup_temp_file(temp_path)
        _release_job_slot()

@app.api_route(
    "/analyze-local",
    methods=["GET", "POST"],
    response_model=CVData,
    dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)],
)
async def analyze_local_cv(
    request: Request,
    filename: Optional[str] = None,
    generate_word: bool = True,
    index: bool = True
):
    """
    Analyse un CV dÃ©jÃ  prÃ©sent dans le dossier 'input/', extrait les donnÃ©es 
    et optionnellement l'indexe dans le RAG. IdÃ©al pour les agents OpenClaw.
    Accepte filename en query param ou dans le body JSON.
    """
    # Si filename pas en query param, essayer le body JSON
    if filename is None:
        try:
            body = await request.json()
            filename = body.get("filename")
            generate_word = body.get("generate_word", generate_word)
            index = body.get("index", index)
        except Exception:
            pass

    if not filename:
        raise HTTPException(status_code=422, detail="Le paramÃ¨tre 'filename' est requis (query param ou body JSON).")
    _acquire_job_slot()
    file_path = _fetch_input_cv(filename)
    _validate_upload_content(file_path, filename)

    logger.info("Analyse locale demandÃ©e pour : %s", filename)
    
    try:
        # ExÃ©cution du pipeline
        cv_data = process_cv_pipeline(
            file_path=file_path,
            output_dir=OUTPUT_DIR,
            generate_word_doc=generate_word
        )
        
        # Ajouter le chemin du fichier Word dans les mÃ©tadonnÃ©es
        if generate_word:
            word_name = f"{file_path.stem}_finaxys.docx"
            cv_data["metadata"]["fichier_word"] = word_name

        if s3_enabled():
            _upload_outputs_to_s3(cv_data, file_path.stem, generate_word)
        
        # Indexation RAG automatique
        if index:
            try:
                from schemas.models import CVData as CVDataModel
                vdb = VectorStoreManager()
                cv_obj = CVDataModel(**cv_data) if isinstance(cv_data, dict) else cv_data
                vdb.add_cv(cv_obj, filename)
                logger.info("CV indexÃ© dans le RAG : %s", filename)
            except Exception as e:
                logger.error("Erreur indexation RAG : %s", e)

        return cv_data

    except HTTPException:
        raise
    except Exception as e:
        _internal_error("Erreur lors du traitement local du CV", e)
    finally:
        _cleanup_temp_file(file_path)
        _release_job_slot()

@app.get("/search", dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)])
async def search_candidates(query: str, results: int = 3):
    """
    Recherche sÃ©mantique (RAG) parmi les CV indexÃ©s. IdÃ©al pour le bot Telegram.
    """
    logger.info("RequÃªte RAG reÃ§ue : %s", query)
    try:
        vdb = VectorStoreManager()
        search_results = vdb.search(query, n_results=results)
        
        if not search_results or not search_results['documents'][0]:
            return {"answer": "DÃ©solÃ©, je n'ai trouvÃ© aucun candidat correspondant Ã  cette recherche.", "candidates": []}

        # GÃ©nÃ©ration de la rÃ©ponse IA
        context_parts = []
        for i, doc in enumerate(search_results['documents'][0]):
            source = search_results['metadatas'][0][i].get('source', 'Inconnu')
            context_parts.append(f"CANDIDAT {i+1} (Source: {source}):\n{doc}")
        
        client, provider = create_client()
        user_msg = f"Demande RH: {query}\n\nProfils trouvÃ©s:\n" + "\n".join(context_parts)
        
        answer = llm_call(client, provider, RAG_SYSTEM_PROMPT, user_msg)
        
        return {
            "query": query,
            "answer": answer,
            "sources": [m.get('source') for m in search_results['metadatas'][0]]
        }

    except Exception as e:
        _internal_error("Erreur recherche RAG", e)

@app.get("/analyze-wait", dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)])
async def analyze_wait(filename: str, generate_word: bool = True, index: bool = False, timeout: int = 90):
    """
    DÃ©clenche l'analyse et attend le rÃ©sultat complet (bloquant, max timeout secondes).
    IdÃ©al pour les agents qui veulent une seule commande curl.
    """
    import asyncio
    import json as json_lib

    file_path = _resolve_path_inside(INPUT_DIR, filename)
    if s3_enabled():
        file_path = _fetch_input_cv(filename)
    elif not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Fichier '{filename}' non trouvÃ© dans '{INPUT_DIR}'.")
    _validate_upload_content(file_path, filename)
    _acquire_job_slot()

    if timeout <= 0:
        raise HTTPException(status_code=422, detail="Le paramÃ¨tre timeout doit Ãªtre supÃ©rieur Ã  0.")

    job_id = uuid4().hex
    result_path = OUTPUT_DIR / f"{job_id}_result.json"

    # Supprimer l'ancien rÃ©sultat
    if result_path.exists():
        result_path.unlink()

    def run_pipeline():
        try:
            logger.info("[analyze-wait] DÃ©marrage pipeline pour : %s (job_id=%s)", filename, job_id)
            cv_data = process_cv_pipeline(
                file_path=file_path,
                output_dir=OUTPUT_DIR,
                generate_word_doc=generate_word
            )
            if index:
                try:
                    from schemas.models import CVData as CVDataModel
                    vdb = VectorStoreManager()
                    cv_obj = CVDataModel(**cv_data) if isinstance(cv_data, dict) else cv_data
                    vdb.add_cv(cv_obj, filename)
                except Exception as e:
                    logger.error("[analyze-wait] Erreur RAG : %s", e)
            if s3_enabled():
                _upload_outputs_to_s3(cv_data, file_path.stem, generate_word)
                bucket = _require_s3_bucket()
                result_key = s3_key(s3_output_prefix(), f"{job_id}_result.json")
                upload_json(bucket, result_key, cv_data)
                logger.info("[analyze-wait] RÃ©sultat sauvegardÃ© : s3://%s/%s", bucket, result_key)
            else:
                with open(result_path, 'w', encoding='utf-8') as f:
                    json_lib.dump(cv_data, f, ensure_ascii=False, indent=2)
                logger.info("[analyze-wait] RÃ©sultat sauvegardÃ© : %s", result_path)
        except Exception as e:
            error_id = uuid4().hex[:8]
            logger.error("[analyze-wait] Erreur pipeline (error_id=%s): %s", error_id, e)
            if s3_enabled():
                bucket = _require_s3_bucket()
                result_key = s3_key(s3_output_prefix(), f"{job_id}_result.json")
                upload_json(bucket, result_key, {"error": f"Erreur interne. Code={error_id}"})
            else:
                with open(result_path, 'w', encoding='utf-8') as f:
                    json_lib.dump({"error": f"Erreur interne. Code={error_id}"}, f)
        finally:
            _cleanup_temp_file(file_path)
            _release_job_slot()

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    # Attendre le rÃ©sultat (polling toutes les 5s)
    poll_interval = 5
    attempts = max(1, math.ceil(timeout / poll_interval))
    for _ in range(attempts):
        await asyncio.sleep(poll_interval)
        if s3_enabled():
            bucket = _require_s3_bucket()
            result_key = s3_key(s3_output_prefix(), f"{job_id}_result.json")
            try:
                result = download_json(bucket, result_key)
            except Exception:
                result = None
            if result:
                if isinstance(result, dict) and result.get("error"):
                    raise HTTPException(status_code=500, detail=result.get("error"))
                return result
        else:
            if result_path.exists():
                with open(result_path, encoding='utf-8') as f:
                    result = json_lib.load(f)
                if isinstance(result, dict) and result.get("error"):
                    raise HTTPException(status_code=500, detail=result.get("error"))
                return result

    raise HTTPException(
        status_code=504,
        detail=f"Timeout : l'analyse a pris trop de temps (>{timeout}s)."
    )


async def trigger_analysis(filename: str, generate_word: bool = True, index: bool = False):
    """
    DÃ©clenche l'analyse d'un CV en background et rÃ©pond immÃ©diatement.
    Le rÃ©sultat est sauvegardÃ© dans output/{job_id}_result.json une fois terminÃ©.
    """
    file_path = _resolve_path_inside(INPUT_DIR, filename)
    if s3_enabled():
        file_path = _fetch_input_cv(filename)
    elif not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Fichier '{filename}' non trouvÃ© dans '{INPUT_DIR}'.")
    _validate_upload_content(file_path, filename)
    _acquire_job_slot()

    stem = file_path.stem
    job_id = uuid4().hex
    output_file = f"{stem}_finaxys.docx"

    # Supprimer l'ancien rÃ©sultat pour Ã©viter de renvoyer un rÃ©sultat pÃ©rimÃ©
    result_path = OUTPUT_DIR / f"{job_id}_result.json"
    if result_path.exists():
        result_path.unlink()

    def run_pipeline():
        try:
            logger.info("[trigger] DÃ©marrage pipeline background pour : %s (job_id=%s)", filename, job_id)
            cv_data = process_cv_pipeline(
                file_path=file_path,
                output_dir=OUTPUT_DIR,
                generate_word_doc=generate_word
            )
            if index:
                try:
                    from schemas.models import CVData as CVDataModel
                    vdb = VectorStoreManager()
                    cv_obj = CVDataModel(**cv_data) if isinstance(cv_data, dict) else cv_data
                    vdb.add_cv(cv_obj, filename)
                    logger.info("[trigger] CV indexÃ© dans le RAG : %s", filename)
                except Exception as e:
                    logger.error("[trigger] Erreur indexation RAG : %s", e)
            # Sauvegarder le rÃ©sultat pour le endpoint /result
            if s3_enabled():
                _upload_outputs_to_s3(cv_data, file_path.stem, generate_word)
                bucket = _require_s3_bucket()
                result_key = s3_key(s3_output_prefix(), f"{job_id}_result.json")
                upload_json(bucket, result_key, cv_data)
                logger.info("[trigger] RÃ©sultat sauvegardÃ© : s3://%s/%s", bucket, result_key)
            else:
                import json
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(cv_data, f, ensure_ascii=False, indent=2)
                logger.info("[trigger] RÃ©sultat sauvegardÃ© : %s", result_path)
        except Exception as e:
            error_id = uuid4().hex[:8]
            logger.error("[trigger] Erreur pipeline (error_id=%s): %s", error_id, e)
            if s3_enabled():
                bucket = _require_s3_bucket()
                result_key = s3_key(s3_output_prefix(), f"{job_id}_result.json")
                upload_json(bucket, result_key, {"error": f"Erreur interne. Code={error_id}"})
            else:
                import json
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump({"error": f"Erreur interne. Code={error_id}"}, f, ensure_ascii=False, indent=2)
        finally:
            _cleanup_temp_file(file_path)
            _release_job_slot()

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    return {
        "status": "processing",
        "message": f"Analyse de '{filename}' dÃ©marrÃ©e.",
        "output_file": output_file,
        "job_id": job_id,
        "result_key": job_id
    }

@app.post("/trigger", dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)])
async def trigger_endpoint(filename: str, generate_word: bool = True, index: bool = False):
    return await trigger_analysis(filename=filename, generate_word=generate_word, index=index)


@app.get("/result", dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)])
async def get_result(filename: Optional[str] = None, job_id: Optional[str] = None):
    """
    Retourne le rÃ©sultat JSON d'une analyse dÃ©jÃ  dÃ©clenchÃ©e via /trigger.
    Retourne 404 si l'analyse n'est pas encore terminÃ©e.
    """
    if job_id:
        result_name = f"{job_id}_result.json"
    elif filename:
        stem = Path(filename).stem
        result_name = f"{stem}_result.json"
    else:
        raise HTTPException(status_code=422, detail="job_id ou filename requis.")

    if s3_enabled():
        bucket = _require_s3_bucket()
        result_key = s3_key(s3_output_prefix(), result_name)
        try:
            result = download_json(bucket, result_key)
        except Exception:
            raise HTTPException(status_code=404, detail="Resultat pas encore pret, reessaie dans quelques secondes.")
        if isinstance(result, dict) and result.get("error"):
            raise HTTPException(status_code=500, detail=result.get("error"))
        return result

    result_path = OUTPUT_DIR / result_name
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="RÃ©sultat pas encore prÃªt, rÃ©essaie dans quelques secondes.")
    import json
    with open(result_path, encoding='utf-8') as f:
        result = json.load(f)
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(status_code=500, detail=result.get("error"))
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

