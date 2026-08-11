import logging
import time
from typing import Optional

from fastapi import FastAPI, Request, Response

logger = logging.getLogger("metrics")

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

    _PROMETHEUS_ENABLED = True
except ImportError:
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"
    Counter = Gauge = Histogram = None
    generate_latest = None
    _PROMETHEUS_ENABLED = False


def _normalize_path(request: Request) -> str:
    route = request.scope.get("route")
    route_path = getattr(route, "path", None)
    if route_path:
        return route_path
    return request.url.path


if _PROMETHEUS_ENABLED:
    HTTP_REQUESTS_TOTAL = Counter(
        "smartcv_http_requests_total",
        "Total HTTP requests handled by SmartCV.",
        ["method", "path", "status"],
    )
    HTTP_REQUEST_DURATION_SECONDS = Histogram(
        "smartcv_http_request_duration_seconds",
        "HTTP request latency for SmartCV.",
        ["method", "path"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    )
    CV_ANALYSIS_TOTAL = Counter(
        "smartcv_cv_analysis_total",
        "Count of CV analysis executions.",
        ["endpoint", "indexed", "status"],
    )
    CV_ANALYSIS_DURATION_SECONDS = Histogram(
        "smartcv_cv_analysis_duration_seconds",
        "Latency of CV analysis executions.",
        ["endpoint", "indexed"],
        buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )
    CV_STAGE_TOTAL = Counter(
        "smartcv_cv_stage_total",
        "Count of CV pipeline stage executions.",
        ["stage", "status"],
    )
    CV_STAGE_DURATION_SECONDS = Histogram(
        "smartcv_cv_stage_duration_seconds",
        "Latency of CV pipeline stages.",
        ["stage"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )
    CV_STAGE_ERRORS_TOTAL = Counter(
        "smartcv_cv_stage_errors_total",
        "Count of CV pipeline stage failures by error type.",
        ["stage", "error_type"],
    )
    CV_VALIDATION_ISSUES_TOTAL = Counter(
        "smartcv_cv_validation_issues_total",
        "Count of validation issues detected in CV data.",
        ["issue_type"],
    )
    CV_INPUT_FILES_TOTAL = Counter(
        "smartcv_cv_input_files_total",
        "Count of CV files entering the pipeline.",
        ["source", "mime_type"],
    )
    CV_INPUT_SIZE_BYTES = Histogram(
        "smartcv_cv_input_size_bytes",
        "Size of CV files entering the pipeline.",
        ["source", "mime_type"],
        buckets=(16_384, 65_536, 262_144, 524_288, 1_048_576, 2_097_152, 5_242_880, 10_485_760, 20_971_520, 52_428_800),
    )
    CV_PAGES_TOTAL = Histogram(
        "smartcv_cv_pages_total",
        "Number of pages detected in parsed CV files.",
        ["source"],
        buckets=(1, 2, 3, 5, 10, 20, 50),
    )
    CV_EXTRACTION_METHOD_TOTAL = Counter(
        "smartcv_cv_extraction_method_total",
        "Count of CV parses by the extraction method that ultimately won "
        "(pypdf, pytesseract-ocr, llm-vision-api, pymupdf-columns, docling-markdown, python-docx). "
        "Used to track the real-world escalation rate through the pypdf -> Tesseract -> vision cascade.",
        ["method"],
    )
    CHATBOT_TURN_TOTAL = Counter(
        "smartcv_chatbot_turn_total",
        "Count of conversational chatbot turns by outcome "
        "(needs_clarification: not enough info yet, LLM decision was valid; "
        "decision_fallback: LLM response was unparseable/unusable, code fell back to asking "
        "for clarification -- a direct robustness signal on the orchestration LLM; "
        "empty: search ran but found nothing; success: candidates found and justified).",
        ["status"],
    )
    CHATBOT_TURN_DURATION_SECONDS = Histogram(
        "smartcv_chatbot_turn_duration_seconds",
        "End-to-end latency of one conversational chatbot turn (decision + search + justification).",
        ["status"],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0),
    )
    LLM_CALLS_TOTAL = Counter(
        "smartcv_llm_calls_total",
        "Count of LLM calls.",
        ["provider", "model", "operation", "status"],
    )
    LLM_LATENCY_SECONDS = Histogram(
        "smartcv_llm_latency_seconds",
        "Latency of LLM calls.",
        ["provider", "model", "operation"],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )
    LLM_TOKENS_TOTAL = Counter(
        "smartcv_llm_tokens_total",
        "Count of LLM tokens.",
        ["provider", "model", "operation", "token_type"],
    )
    LLM_COST_TOTAL = Counter(
        "smartcv_llm_cost_total",
        "Estimated LLM cost in USD.",
        ["provider", "model", "operation"],
    )
    LLM_ERRORS_TOTAL = Counter(
        "smartcv_llm_errors_total",
        "Count of LLM call failures by error type.",
        ["provider", "model", "operation", "error_type"],
    )
    RAG_SEARCH_TOTAL = Counter(
        "smartcv_rag_search_total",
        "Count of RAG searches.",
        ["endpoint", "mode", "status"],
    )
    RAG_SEARCH_DURATION_SECONDS = Histogram(
        "smartcv_rag_search_duration_seconds",
        "Latency of RAG searches.",
        ["endpoint", "mode"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    RAG_CANDIDATES_RETURNED = Histogram(
        "smartcv_rag_candidates_returned",
        "Number of candidates returned by RAG search.",
        ["endpoint", "mode"],
        buckets=(0, 1, 3, 5, 10, 20, 50),
    )
    RAG_INDEX_TOTAL = Counter(
        "smartcv_rag_index_total",
        "Count of RAG indexing executions.",
        ["collection", "source", "status"],
    )
    RAG_INDEX_DURATION_SECONDS = Histogram(
        "smartcv_rag_index_duration_seconds",
        "Latency of RAG indexing executions.",
        ["collection", "source"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    )
    RAG_INDEX_CHUNKS_TOTAL = Counter(
        "smartcv_rag_index_chunks_total",
        "Count of indexed RAG chunks/documents.",
        ["collection", "source"],
    )
    RAG_EMBEDDING_DURATION_SECONDS = Histogram(
        "smartcv_rag_embedding_duration_seconds",
        "Latency of RAG embedding generation.",
        ["model", "operation"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    )
    RAG_EMBEDDING_ERRORS_TOTAL = Counter(
        "smartcv_rag_embedding_errors_total",
        "Count of RAG embedding failures by error type.",
        ["model", "operation", "error_type"],
    )
    ACTIVE_JOBS_GAUGE = Gauge(
        "smartcv_active_jobs",
        "Number of in-flight CV analysis jobs.",
    )
    JOB_CAPACITY_GAUGE = Gauge(
        "smartcv_job_capacity",
        "Maximum number of concurrent CV analysis jobs.",
    )
    LANGFUSE_ENABLED_GAUGE = Gauge(
        "smartcv_langfuse_enabled",
        "Whether Langfuse tracing is enabled (1) or disabled (0).",
    )
    # --- Métriques d'évaluation LLM (bot + API, via Pushgateway pour le bot) ---
    LLM_EVAL_PRECISION_SCORE = Histogram(
        "smartcv_llm_eval_precision_score",
        "Score de complétude (précision extraction LLM) : fraction des champs CV remplis [0-1].",
        ["source", "model"],
        buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    )
    LLM_EVAL_ROBUSTNESS_SEMANTIC = Histogram(
        "smartcv_llm_eval_robustness_semantic_score",
        "Score de validation sémantique (robustesse LLM) : cohérence globale du CV extrait [0-1].",
        ["source", "model"],
        buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    )
    LLM_EVAL_ROBUSTNESS_FAILED = Counter(
        "smartcv_llm_eval_robustness_failed_total",
        "Nombre de CV dont la validation structurelle ou sémantique a échoué.",
        ["source", "reason"],
    )
    LLM_EVAL_COHERENCE_FIELDS = Histogram(
        "smartcv_llm_eval_coherence_fields_extracted",
        "Nombre de champs clés remplis par le LLM (cohérence de l'extraction) : nom, email, tel, résumé, compétences, expériences.",
        ["source"],
        buckets=(0, 1, 2, 3, 4, 5, 6),
    )
    LLM_EVAL_PIPELINE_LATENCY = Histogram(
        "smartcv_llm_eval_pipeline_latency_seconds",
        "Latence totale du pipeline CV de bout en bout (parse → Word généré).",
        ["source"],
        buckets=(5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 300.0),
    )
else:
    HTTP_REQUESTS_TOTAL = None
    HTTP_REQUEST_DURATION_SECONDS = None
    CV_ANALYSIS_TOTAL = None
    CV_ANALYSIS_DURATION_SECONDS = None
    CV_STAGE_TOTAL = None
    CV_STAGE_DURATION_SECONDS = None
    CV_STAGE_ERRORS_TOTAL = None
    CV_VALIDATION_ISSUES_TOTAL = None
    CV_INPUT_FILES_TOTAL = None
    CV_INPUT_SIZE_BYTES = None
    CV_PAGES_TOTAL = None
    CV_EXTRACTION_METHOD_TOTAL = None
    CHATBOT_TURN_TOTAL = None
    CHATBOT_TURN_DURATION_SECONDS = None
    LLM_CALLS_TOTAL = None
    LLM_LATENCY_SECONDS = None
    LLM_TOKENS_TOTAL = None
    LLM_COST_TOTAL = None
    LLM_ERRORS_TOTAL = None
    RAG_SEARCH_TOTAL = None
    RAG_SEARCH_DURATION_SECONDS = None
    RAG_CANDIDATES_RETURNED = None
    RAG_INDEX_TOTAL = None
    RAG_INDEX_DURATION_SECONDS = None
    RAG_INDEX_CHUNKS_TOTAL = None
    RAG_EMBEDDING_DURATION_SECONDS = None
    RAG_EMBEDDING_ERRORS_TOTAL = None
    ACTIVE_JOBS_GAUGE = None
    JOB_CAPACITY_GAUGE = None
    LANGFUSE_ENABLED_GAUGE = None
    LLM_EVAL_PRECISION_SCORE = None
    LLM_EVAL_ROBUSTNESS_SEMANTIC = None
    LLM_EVAL_ROBUSTNESS_FAILED = None
    LLM_EVAL_COHERENCE_FIELDS = None
    LLM_EVAL_PIPELINE_LATENCY = None


def metrics_enabled() -> bool:
    return _PROMETHEUS_ENABLED


def install_metrics(app: FastAPI) -> None:
    if not _PROMETHEUS_ENABLED:
        logger.info("Prometheus metrics disabled (package missing).")
        return
    if getattr(app.state, "metrics_installed", False):
        return

    @app.middleware("http")
    async def prometheus_http_middleware(request: Request, call_next):
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            path = _normalize_path(request)
            duration = time.perf_counter() - start
            HTTP_REQUESTS_TOTAL.labels(
                method=request.method,
                path=path,
                status=str(status_code),
            ).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=request.method,
                path=path,
            ).observe(duration)

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.state.metrics_installed = True


def set_job_capacity(capacity: int) -> None:
    if JOB_CAPACITY_GAUGE is not None:
        JOB_CAPACITY_GAUGE.set(max(capacity, 0))


def increment_active_jobs() -> None:
    if ACTIVE_JOBS_GAUGE is not None:
        ACTIVE_JOBS_GAUGE.inc()


def decrement_active_jobs() -> None:
    if ACTIVE_JOBS_GAUGE is not None:
        ACTIVE_JOBS_GAUGE.dec()


def set_langfuse_enabled(enabled: bool) -> None:
    if LANGFUSE_ENABLED_GAUGE is not None:
        LANGFUSE_ENABLED_GAUGE.set(1 if enabled else 0)


def observe_cv_analysis(*, endpoint: str, indexed: bool, status: str, latency_ms: float) -> None:
    if CV_ANALYSIS_TOTAL is None or CV_ANALYSIS_DURATION_SECONDS is None:
        return
    indexed_label = "true" if indexed else "false"
    CV_ANALYSIS_TOTAL.labels(endpoint=endpoint, indexed=indexed_label, status=status).inc()
    CV_ANALYSIS_DURATION_SECONDS.labels(endpoint=endpoint, indexed=indexed_label).observe(latency_ms / 1000.0)


def _normalize_error_type(error_type: Optional[str]) -> str:
    if not error_type:
        return "unknown"
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in error_type).strip("_")
    return normalized or "unknown"


def observe_cv_input(*, source: str, mime_type: str, size_bytes: Optional[int] = None) -> None:
    if CV_INPUT_FILES_TOTAL is None:
        return
    CV_INPUT_FILES_TOTAL.labels(source=source, mime_type=mime_type).inc()
    if CV_INPUT_SIZE_BYTES is not None and size_bytes is not None and size_bytes >= 0:
        CV_INPUT_SIZE_BYTES.labels(source=source, mime_type=mime_type).observe(size_bytes)


def observe_cv_pages(*, source: str, page_count: Optional[int]) -> None:
    if CV_PAGES_TOTAL is None or page_count is None or page_count <= 0:
        return
    CV_PAGES_TOTAL.labels(source=source).observe(page_count)


def observe_cv_stage(
    *,
    stage: str,
    status: str,
    latency_ms: float,
    error_type: Optional[str] = None,
) -> None:
    if CV_STAGE_TOTAL is None or CV_STAGE_DURATION_SECONDS is None:
        return
    CV_STAGE_TOTAL.labels(stage=stage, status=status).inc()
    CV_STAGE_DURATION_SECONDS.labels(stage=stage).observe(latency_ms / 1000.0)
    if status != "success" and CV_STAGE_ERRORS_TOTAL is not None:
        CV_STAGE_ERRORS_TOTAL.labels(
            stage=stage,
            error_type=_normalize_error_type(error_type),
        ).inc()


def observe_extraction_method(method: str) -> None:
    """Incrémente le compteur de méthode d'extraction gagnante (pypdf/tesseract/vision/...).

    Permet de calculer, en production, le taux d'escalade réel dans la cascade
    pypdf -> Tesseract -> Qwen2-VL, information qui n'existait auparavant que
    sur le jeu de test offline.
    """
    if CV_EXTRACTION_METHOD_TOTAL is None:
        return
    CV_EXTRACTION_METHOD_TOTAL.labels(method=method or "unknown").inc()


def observe_chatbot_turn(*, status: str, latency_ms: float) -> None:
    """Incrémente le compteur/histogramme d'un tour de chatbot conversationnel.

    ``status`` vaut ``needs_clarification`` (pas assez d'info pour chercher),
    ``empty`` (recherche lancée mais aucun candidat) ou ``success``. Couvre les
    axes "latence" et une partie de "précision" (répartition des issues) de la
    mission d'évaluation du chatbot -- la cohérence des justifications
    générées par le LLM resterait à évaluer via un jugement humain ou un
    LLM-juge, non construit dans ce projet.
    """
    if CHATBOT_TURN_TOTAL is None or CHATBOT_TURN_DURATION_SECONDS is None:
        return
    CHATBOT_TURN_TOTAL.labels(status=status).inc()
    CHATBOT_TURN_DURATION_SECONDS.labels(status=status).observe(latency_ms / 1000.0)


def observe_validation_report(report: dict) -> None:
    if CV_VALIDATION_ISSUES_TOTAL is None:
        return
    for _ in report.get("structural_errors", []):
        CV_VALIDATION_ISSUES_TOTAL.labels(issue_type="structural_error").inc()
    for _ in report.get("semantic_errors", []):
        CV_VALIDATION_ISSUES_TOTAL.labels(issue_type="semantic_error").inc()
    for _ in report.get("semantic_warnings", []):
        CV_VALIDATION_ISSUES_TOTAL.labels(issue_type="semantic_warning").inc()


def observe_llm_call(
    *,
    provider: str,
    model: str,
    operation: str,
    status: str,
    latency_ms: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    cost_usd: Optional[float] = None,
    error_type: Optional[str] = None,
) -> None:
    if LLM_CALLS_TOTAL is None or LLM_LATENCY_SECONDS is None:
        return
    LLM_CALLS_TOTAL.labels(
        provider=provider,
        model=model,
        operation=operation,
        status=status,
    ).inc()
    LLM_LATENCY_SECONDS.labels(
        provider=provider,
        model=model,
        operation=operation,
    ).observe(latency_ms / 1000.0)
    if LLM_TOKENS_TOTAL is not None:
        token_values = {
            "prompt": max(prompt_tokens, 0),
            "completion": max(completion_tokens, 0),
            "total": max(total_tokens, 0),
        }
        for token_type, value in token_values.items():
            if value:
                LLM_TOKENS_TOTAL.labels(
                    provider=provider,
                    model=model,
                    operation=operation,
                    token_type=token_type,
                ).inc(value)
    if cost_usd is not None and cost_usd > 0 and LLM_COST_TOTAL is not None:
        LLM_COST_TOTAL.labels(
            provider=provider,
            model=model,
            operation=operation,
        ).inc(cost_usd)
    if status != "success" and LLM_ERRORS_TOTAL is not None:
        LLM_ERRORS_TOTAL.labels(
            provider=provider,
            model=model,
            operation=operation,
            error_type=_normalize_error_type(error_type),
        ).inc()


def observe_rag_search(
    *,
    endpoint: str,
    mode: str,
    status: str,
    latency_ms: float,
    candidates_returned: Optional[int] = None,
) -> None:
    if RAG_SEARCH_TOTAL is None or RAG_SEARCH_DURATION_SECONDS is None:
        return
    RAG_SEARCH_TOTAL.labels(endpoint=endpoint, mode=mode, status=status).inc()
    RAG_SEARCH_DURATION_SECONDS.labels(endpoint=endpoint, mode=mode).observe(latency_ms / 1000.0)
    if candidates_returned is not None and RAG_CANDIDATES_RETURNED is not None:
        RAG_CANDIDATES_RETURNED.labels(endpoint=endpoint, mode=mode).observe(candidates_returned)


def observe_rag_index(
    *,
    collection: str,
    source: str,
    status: str,
    latency_ms: float,
    chunks_indexed: int = 0,
) -> None:
    if RAG_INDEX_TOTAL is None or RAG_INDEX_DURATION_SECONDS is None:
        return
    RAG_INDEX_TOTAL.labels(collection=collection, source=source, status=status).inc()
    RAG_INDEX_DURATION_SECONDS.labels(collection=collection, source=source).observe(latency_ms / 1000.0)
    if chunks_indexed > 0 and RAG_INDEX_CHUNKS_TOTAL is not None:
        RAG_INDEX_CHUNKS_TOTAL.labels(collection=collection, source=source).inc(chunks_indexed)


def observe_rag_embedding(
    *,
    model: str,
    operation: str,
    latency_ms: float,
    status: str = "success",
    error_type: Optional[str] = None,
) -> None:
    if status == "success":
        if RAG_EMBEDDING_DURATION_SECONDS is not None:
            RAG_EMBEDDING_DURATION_SECONDS.labels(model=model, operation=operation).observe(latency_ms / 1000.0)
        return
    if RAG_EMBEDDING_ERRORS_TOTAL is not None:
        RAG_EMBEDDING_ERRORS_TOTAL.labels(
            model=model,
            operation=operation,
            error_type=_normalize_error_type(error_type),
        ).inc()


def observe_llm_eval(
    *,
    source: str,
    model: str,
    completude_score: float,
    semantic_score: float,
    structural_ok: bool,
    semantic_ok: bool,
    fields_filled: int,
    pipeline_latency_ms: float,
) -> None:
    """Enregistre les métriques d'évaluation LLM après chaque analyse CV.

    - Précision   : completude_score (% champs CV remplis par le LLM)
    - Robustesse  : semantic_score + compteur d'échecs de validation
    - Cohérence   : nombre de champs clés effectivement remplis
    - Latence     : durée totale du pipeline de bout en bout
    """
    if not _PROMETHEUS_ENABLED:
        return
    if LLM_EVAL_PRECISION_SCORE is not None:
        LLM_EVAL_PRECISION_SCORE.labels(source=source, model=model).observe(
            max(0.0, min(1.0, completude_score))
        )
    if LLM_EVAL_ROBUSTNESS_SEMANTIC is not None:
        LLM_EVAL_ROBUSTNESS_SEMANTIC.labels(source=source, model=model).observe(
            max(0.0, min(1.0, semantic_score))
        )
    if LLM_EVAL_ROBUSTNESS_FAILED is not None:
        if not structural_ok:
            LLM_EVAL_ROBUSTNESS_FAILED.labels(source=source, reason="structural").inc()
        if not semantic_ok:
            LLM_EVAL_ROBUSTNESS_FAILED.labels(source=source, reason="semantic").inc()
    if LLM_EVAL_COHERENCE_FIELDS is not None:
        LLM_EVAL_COHERENCE_FIELDS.labels(source=source).observe(fields_filled)
    if LLM_EVAL_PIPELINE_LATENCY is not None:
        LLM_EVAL_PIPELINE_LATENCY.labels(source=source).observe(pipeline_latency_ms / 1000.0)


def push_bot_metrics(pushgateway_url: str = "http://smartcv-pushgateway:9091") -> None:
    """Pousse les métriques du bot vers le Pushgateway Prometheus.

    À appeler après chaque analyse pour que Prometheus puisse scraper
    les métriques du bot (qui n'a pas d'endpoint /metrics HTTP).
    """
    if not _PROMETHEUS_ENABLED:
        return
    try:
        from prometheus_client import push_to_gateway, REGISTRY
        push_to_gateway(pushgateway_url, job="smartcv-bot", registry=REGISTRY)
        logger.debug("Métriques bot poussées vers %s", pushgateway_url)
    except Exception as e:
        logger.warning("push_bot_metrics échoué (non bloquant) : %s", e)
